"""
Training Script for Weather-Aware Lane Detection
"""
import os
import yaml
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from models import build_model
from data.dataset import build_dataloader
from models.losses import LaneDetectionLoss
from utils.metrics import compute_lane_metrics


class Trainer:
    def __init__(self, config: dict, exp_dir: str):
        self.config = config
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(config["hardware"]["device"])
        self.model = build_model(config=config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.criterion = LaneDetectionLoss(config)
        train_cfg = config["training"]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
        
        scheduler_type = train_cfg.get("scheduler", "cosine")
        total_steps = train_cfg["num_epochs"]
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=train_cfg["learning_rate"] * 0.01)
        else:
            self.scheduler = None
        
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.scaler = GradScaler() if self.use_amp else None
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.writer = SummaryWriter(self.exp_dir / "logs")
        self.global_step = 0
        self.best_f1 = 0.0
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        loss_components = {"lane_cls": 0.0, "lane_loc": 0.0, "weather_cls": 0.0, "lane_type": 0.0, "instance": 0.0}
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "image"}
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets)
                self.scaler.scale(losses["total"]).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                losses["total"].backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += losses["total"].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            
            self.global_step += 1
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {avg_loss:.4f}")
                self.writer.add_scalar("train/total_loss", losses["total"].item(), self.global_step)
        
        num_batches = len(dataloader)
        avg_losses = {k: v / num_batches for k, v in loss_components.items()}
        avg_losses["total"] = total_loss / num_batches
        return avg_losses
    
    @torch.no_grad()
    def validate(self, dataloader, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_targets = [], []
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "image"}
            outputs = self.model(images)
            losses = self.criterion(outputs, targets)
            total_loss += losses["total"].item()
            all_predictions.append(outputs)
            all_targets.append(targets)
        
        avg_loss = total_loss / len(dataloader)
        metrics = compute_lane_metrics(all_predictions, all_targets, self.config)
        metrics["loss"] = avg_loss
        
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/f1", metrics["f1"], epoch)
        print(f"\nValidation - Loss: {avg_loss:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        if epoch % self.config["logging"].get("save_interval", 5) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch}.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"  Saved best model with F1: {metrics['f1']:.4f}")
    
    def train(self, train_loader, val_loader):
        num_epochs = self.config["training"]["num_epochs"]
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            train_losses = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            if self.scheduler:
                self.scheduler.step()
            
            is_best = val_metrics["f1"] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics["f1"]
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            print(f"Epoch time: {time.time() - start_time:.2f}s")
        
        print(f"\nTraining complete! Best F1: {self.best_f1:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--exp-dir", type=str, default="experiments/default")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{args.exp_dir}_{timestamp}"
    
    train_loader = build_dataloader(
        data_root=config["data"]["train_root"],
        dataset_type=config["data"]["dataset_type"],
        split="train", batch_size=config["training"]["batch_size"],
        num_workers=config["hardware"]["num_workers"], config=config
    )
    val_loader = build_dataloader(
        data_root=config["data"]["val_root"],
        dataset_type=config["data"]["dataset_type"],
        split="val", batch_size=config["training"]["batch_size"],
        num_workers=config["hardware"]["num_workers"], config=config
    )
    
    trainer = Trainer(config, exp_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()