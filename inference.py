"""Single-image inference with optional Grad-CAM visualisation.

Usage::

    python inference.py --image photo.jpg --checkpoint best_model.pth
    python inference.py --image photo.jpg --checkpoint best_model.pth --gradcam
"""

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from data import get_val_transform
from model import GradCAM, get_model
from utils import overlay_gradcam


def predict(
    image_path: str,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[int, str, torch.Tensor]:
    """Run inference on a single image.

    Args:
        image_path: Path to the input image.
        model: Trained classifier model (already on *device*).
        transform: Preprocessing transform pipeline.
        device: Target device.
        class_names: Ordered class names.

    Returns:
        Tuple of (predicted_index, predicted_class_name, softmax_probabilities).
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    pred_name = class_names[pred_idx]

    return pred_idx, pred_name, probs.squeeze(0)


def predict_with_gradcam(
    image_path: str,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
    save_path: str = "gradcam_output.png",
) -> tuple[int, str, np.ndarray]:
    """Run inference and generate a Grad-CAM overlay.

    Args:
        image_path: Path to the input image.
        model: Trained classifier model (already on *device*).
        transform: Preprocessing transform pipeline.
        device: Target device.
        class_names: Ordered class names.
        save_path: Where to save the Grad-CAM visualisation.

    Returns:
        Tuple of (predicted_index, predicted_class_name, blended_image_array).
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model, target_layer_name="layer4")
    heatmap = cam(input_tensor)

    # Up-sample heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))

    blended = overlay_gradcam(image, heatmap_resized, alpha=0.5, save_path=save_path)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    pred_name = class_names[pred_idx]

    return pred_idx, pred_name, blended


def main() -> None:
    """CLI entry-point for single-image inference."""
    parser = argparse.ArgumentParser(description="WikiArt single-image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualisation")
    parser.add_argument("--gradcam_output", type=str, default="gradcam_output.png")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = get_model(cfg, freeze_backbone=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[INFERENCE] Loaded checkpoint from {args.checkpoint}")

    transform = get_val_transform(cfg)

    if args.gradcam:
        pred_idx, pred_name, _blended = predict_with_gradcam(
            args.image, model, transform, device, cfg.display_class_names, args.gradcam_output,
        )
        print(f"[INFERENCE] Prediction: {pred_name} (index={pred_idx})")
        print(f"[INFERENCE] Grad-CAM saved to {args.gradcam_output}")
    else:
        pred_idx, pred_name, probs = predict(
            args.image, model, transform, device, cfg.display_class_names,
        )
        print(f"[INFERENCE] Prediction: {pred_name} (index={pred_idx})")
        print("[INFERENCE] Class probabilities:")
        for i, name in enumerate(cfg.display_class_names):
            print(f"  {name:20s}: {probs[i].item():.4f}")


if __name__ == "__main__":
    main()
