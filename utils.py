"""Utility helpers: early stopping, metrics computation, and visualisation.

All plotting functions save figures to disk and optionally display them.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"max"`` (e.g. accuracy) or ``"min"`` (e.g. loss).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Update state and return ``True`` if training should stop.

        Args:
            value: Current epoch metric value.

        Returns:
            ``True`` when the patience budget is exhausted.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int,
) -> Dict[str, float]:
    """Compute accuracy, macro-averaged precision, recall, and F1.

    Args:
        all_preds: Predicted class indices.
        all_labels: Ground-truth class indices.
        num_classes: Total number of classes (unused but kept for API symmetry).

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_classification_report(
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
) -> str:
    """Print and return the sklearn classification report string.

    Args:
        all_preds: Predicted class indices.
        all_labels: Ground-truth class indices.
        class_names: Human-readable class names.

    Returns:
        The full classification report as a string.
    """
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )
    print(report)
    return report


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "training_curves.png",
) -> None:
    """Plot loss and accuracy curves for both training and validation.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        train_accs: Per-epoch training accuracies.
        val_accs: Per-epoch validation accuracies.
        save_path: File path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[VIS] Training curves saved to {save_path}")


def plot_confusion_matrix(
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
) -> None:
    """Plot and save a confusion matrix heatmap using seaborn.

    Args:
        all_preds: Predicted class indices.
        all_labels: Ground-truth class indices.
        class_names: Ordered class names for axis labels.
        save_path: File path to save the figure.
    """
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[VIS] Confusion matrix saved to {save_path}")


def plot_misclassified(
    images: List[torch.Tensor],
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    max_images: int = 16,
    save_path: str = "misclassified.png",
) -> None:
    """Display a grid of misclassified examples.

    Args:
        images: List of normalised image tensors.
        true_labels: Ground-truth label indices.
        pred_labels: Predicted label indices.
        class_names: Ordered class names.
        mean: Channel means used for normalisation (for de-normalisation).
        std: Channel stds used for normalisation (for de-normalisation).
        max_images: Maximum number of images to show.
        save_path: File path to save the figure.
    """
    n = min(len(images), max_images)
    if n == 0:
        print("[VIS] No misclassified images to plot.")
        return

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes).flatten()

    for i in range(n):
        img = images[i].clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}",
            fontsize=9,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[VIS] Misclassified examples saved to {save_path}")


def overlay_gradcam(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto the original image.

    Args:
        image: Original PIL image.
        heatmap: 2-D Grad-CAM heatmap in ``[0, 1]``.
        alpha: Blending factor for the overlay.
        save_path: If provided, save the blended image to this path.

    Returns:
        Blended image as a numpy array (H, W, 3) in ``[0, 255]``.
    """
    img_np = np.array(image.resize((heatmap.shape[1], heatmap.shape[0]))) / 255.0
    cmap = plt.cm.jet(heatmap)[:, :, :3]

    blended = (1 - alpha) * img_np + alpha * cmap
    blended = (blended * 255).astype(np.uint8)

    if save_path:
        Image.fromarray(blended).save(save_path)
        print(f"[VIS] Grad-CAM overlay saved to {save_path}")

    return blended
