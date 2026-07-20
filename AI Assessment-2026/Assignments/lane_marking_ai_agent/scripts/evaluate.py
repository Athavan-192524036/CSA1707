"""
Evaluation Script for Lane Detection Model
"""
import argparse
import yaml
import torch
from tqdm import tqdm
import json

from models import build_model
from data.dataset import build_dataloader
from utils.metrics import compute_lane_metrics


class Evaluator:
    def __init__(self, config: dict, checkpoint_path: str, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_model(config=config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "test") -> dict:
        all_predictions, all_targets = [], []
        weather_correct = {w: 0 for w in ["clear", "rain", "fog", "snow", "night", "glare"]}
        weather_total = {w: 0 for w in ["clear", "rain", "fog", "snow", "night", "glare"]}
        
        print(f"\nEvaluating on {split_name} set...")
        for batch in tqdm(dataloader):
            images = batch["image"].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "image"}
            outputs = self.model(images)
            losses = None
            
            pred_weather = torch.argmax(outputs["weather_logits"], dim=1)
            true_weather = targets["weather_targets"]
            for pw, tw in zip(pred_weather, true_weather):
                w_name = ["clear", "rain", "fog", "snow", "night", "glare"][tw.item()]
                weather_total[w_name] += 1
                if pw == tw:
                    weather_correct[w_name] += 1
            
            all_predictions.append(outputs)
            all_targets.append(targets)
        
        metrics = compute_lane_metrics(all_predictions, all_targets, self.config)
        weather_acc = sum(weather_correct.values()) / sum(weather_total.values()) if sum(weather_total.values()) > 0 else 0
        
        return {
            "split": split_name,
            "f1_score": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "false_positive_rate": metrics["fpr"],
            "false_negative_rate": metrics["fnr"],
            "lane_type_accuracy": metrics["type_accuracy"],
            "weather_accuracy": weather_acc,
            "weather_per_class": {w: weather_correct[w] / weather_total[w] if weather_total[w] > 0 else 0 for w in weather_correct.keys()},
            "num_samples": sum(weather_total.values())
        }
    
    def export_results(self, results: dict, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="tusimple")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--weather-stratified", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluator = Evaluator(config, args.checkpoint, args.device)
    
    if args.weather_stratified:
        weather_conditions = ["clear", "rain", "fog", "snow", "night", "glare"]
        all_results = {}
        for weather in weather_conditions:
            try:
                loader = build_dataloader(
                    data_root=args.data_root, dataset_type=args.dataset,
                    split=args.split, batch_size=config["training"]["batch_size"],
                    num_workers=config["hardware"]["num_workers"], config=config
                )
                all_results[weather] = evaluator.evaluate(loader, weather)
            except Exception as e:
                print(f"Skipping {weather}: {e}")
        
        print("\n" + "="*60)
        print("WEATHER-STRATIFIED EVALUATION RESULTS")
        print("="*60)
        for weather, results in all_results.items():
            print(f"\n{weather.upper()}:")
            print(f"  F1 Score: {results['f1_score']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
    else:
        dataloader = build_dataloader(
            data_root=args.data_root, dataset_type=args.dataset,
            split=args.split, batch_size=config["training"]["batch_size"],
            num_workers=config["hardware"]["num_workers"], config=config
        )
        results = evaluator.evaluate(dataloader, args.split)
        print(f"\nF1 Score: {results['f1_score']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
    
    evaluator.export_results(results if not args.weather_stratified else all_results, args.output)


if __name__ == "__main__":
    main()