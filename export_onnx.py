"""Export a trained WikiArt classifier checkpoint to ONNX format.

Usage::

    python export_onnx.py --checkpoint best_model.pth --output model.onnx
"""

import argparse

import torch

from config import Config
from model import get_model


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "model.onnx",
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """Load a ``.pth`` checkpoint and export the model to ONNX.

    Args:
        checkpoint_path: Path to the saved PyTorch checkpoint.
        output_path: Destination ``.onnx`` file path.
        opset_version: ONNX opset version (default 17).
        device: Device to use during export (``"cpu"`` recommended).
    """
    cfg = Config()
    dev = torch.device(device)

    model = get_model(cfg, freeze_backbone=False).to(dev)
    state_dict = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=dev)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[ONNX] Model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export WikiArt model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.opset)
