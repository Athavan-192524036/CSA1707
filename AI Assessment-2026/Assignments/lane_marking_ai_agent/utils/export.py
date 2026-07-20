"""
Model Export Utilities
ONNX, TensorRT, and INT8 Quantization support.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Tuple
import yaml


class ModelExporter:
    """Export trained models to deployment formats."""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.model.eval()
        
    def export_onnx(self, output_path: str, input_shape: Tuple = (1, 3, 288, 800), opset_version: int = 11, simplify: bool = True):
        """Export model to ONNX format."""
        dummy_input = torch.randn(*input_shape).to(next(self.model.parameters()).device)
        
        torch.onnx.export(
            self.model, dummy_input, output_path,
            export_params=True, opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["lane_logits", "lane_types", "confidences", "weather_probs"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "lane_logits": {0: "batch_size"},
                "lane_types": {0: "batch_size"},
                "confidences": {0: "batch_size"},
                "weather_probs": {0: "batch_size"}
            }
        )
        print(f"ONNX model exported to: {output_path}")
        
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                onnx_model = onnx.load(output_path)
                simplified, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(simplified, output_path)
                    print("ONNX model simplified successfully")
            except ImportError:
                pass
        
        self._verify_onnx(output_path, dummy_input)
    
    def _verify_onnx(self, onnx_path: str, dummy_input: torch.Tensor):
        try:
            import onnxruntime as ort
            with torch.no_grad():
                pytorch_out = self.model(dummy_input)
            session = ort.InferenceSession(onnx_path)
            onnx_input = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_out = session.run(None, onnx_input)
            diff = np.abs(pytorch_out["lane_logits"].cpu().numpy() - onnx_out[0]).max()
            print(f"ONNX verification - Max difference: {diff:.6f}")
        except ImportError:
            pass