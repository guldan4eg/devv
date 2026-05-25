"""Evaluation script for the WikiArt style classifier.

Loads a trained checkpoint, evaluates it on the test split, and produces:
  - Per-class classification report (precision / recall / F1).
  - Confusion matrix heatmap.
  - Grid of misclassified examples.

Usage::

    python evaluate.py --data_dir /path/to/wikiart --checkpoint best_model.pth
"""

import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data import build_dataloaders
from model import get_model
from utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_misclassified,
    print_classification_report,
)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """Evaluate the model on the given DataLoader.

    Args:
        model: Trained classifier model.
        loader: Test DataLoader.
        device: Target device.
        class_names: Ordered class names.

    Returns:
        Tuple of (metrics_dict, all_preds, all_labels).
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_preds, all_labels, num_classes=len(class_names))
    return metrics, all_preds, all_labels


def collect_misclassified(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 16,
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """Collect misclassified examples for visualisation.

    Args:
        model: Trained classifier model.
        loader: Test DataLoader.
        device: Target device.
        max_samples: Maximum number of misclassified samples to return.

    Returns:
        Tuple of (images, true_labels, pred_labels).
    """
    model.eval()
    mis_images: List[torch.Tensor] = []
    mis_true: List[int] = []
    mis_pred: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device, non_blocking=True)
            outputs = model(images_dev)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if preds[i].item() != labels[i].item():
                    mis_images.append(images[i].cpu())
                    mis_true.append(labels[i].item())
                    mis_pred.append(preds[i].item())
                    if len(mis_images) >= max_samples:
                        return mis_images, mis_true, mis_pred

    return mis_images, mis_true, mis_pred


def main(cfg: Config, checkpoint: str) -> None:
    """Run full evaluation pipeline on the test set.

    Args:
        cfg: Global configuration object.
        checkpoint: Path to the saved ``.pth`` model checkpoint.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    _train_loader, _val_loader, test_loader, _class_to_idx = build_dataloaders(cfg)

    model = get_model(cfg, freeze_backbone=False).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[EVAL] Loaded checkpoint from {checkpoint}")

    # Metrics
    metrics, all_preds, all_labels = evaluate(model, test_loader, device, cfg.class_names)
    print(f"\n[EVAL] Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"[EVAL] Test Precision: {metrics['precision']:.4f}")
    print(f"[EVAL] Test Recall:    {metrics['recall']:.4f}")
    print(f"[EVAL] Test F1:        {metrics['f1']:.4f}")

    # Per-class report
    print("\n--- Per-class Classification Report ---")
    print_classification_report(all_preds, all_labels, cfg.display_class_names)

    # Confusion matrix
    plot_confusion_matrix(
        all_preds,
        all_labels,
        cfg.display_class_names,
        save_path="confusion_matrix.png",
    )

    # Misclassified examples
    mis_images, mis_true, mis_pred = collect_misclassified(model, test_loader, device)
    plot_misclassified(
        mis_images,
        mis_true,
        mis_pred,
        cfg.display_class_names,
        mean=cfg.mean,
        std=cfg.std,
        save_path="misclassified.png",
    )

    print("\n[EVAL] Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WikiArt classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.data_dir = args.data_dir
    config.device = args.device
    config.batch_size = args.batch_size

    main(config, args.checkpoint)
