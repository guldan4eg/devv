"""Configuration module for WikiArt style classifier.

Contains all hyperparameters, paths, class definitions, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    """Central configuration for the WikiArt classifier pipeline."""

    # --- Dataset ---
    data_dir: str = "data"
    class_names: List[str] = field(default_factory=lambda: [
        "Art_Nouveau",
        "Baroque",
        "Cubism",
        "Expressionism",
        "Impressionism",
        "Realism",
        "Romanticism",
        "Surrealism",
        "Symbolism",
        "Ukiyo_e",
    ])
    num_classes: int = 10
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 42

    # --- Image preprocessing ---
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Data loader ---
    batch_size: int = 32
    num_workers: int = 4

    # --- Stage 1: transfer learning (frozen backbone) ---
    stage1_epochs: int = 15
    stage1_lr: float = 1e-3

    # --- Stage 2: fine-tuning (layer3 + layer4 + fc) ---
    stage2_epochs: int = 25
    stage2_lr: float = 1e-4

    # --- Optimizer & scheduler ---
    weight_decay: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # --- Loss ---
    label_smoothing: float = 0.1

    # --- Early stopping ---
    early_stopping_patience: int = 7

    # --- Checkpoints / logging ---
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    best_model_name: str = "best_model.pth"

    # --- Logging backend: "tensorboard" or "wandb" ---
    logger_backend: str = "tensorboard"
    wandb_project: str = "wikiart-classifier"

    # --- Device ---
    device: str = "cuda"

    @property
    def checkpoint_path(self) -> Path:
        """Return full path to the best model checkpoint."""
        return Path(self.checkpoint_dir) / self.best_model_name

    @property
    def display_class_names(self) -> List[str]:
        """Return human-readable class names (spaces instead of underscores)."""
        return [name.replace("_", " ") for name in self.class_names]
