"""Training script for the WikiArt style classifier.

Implements a two-stage training procedure:
  1. Transfer learning — frozen ResNet50 backbone, only the classifier head is
     trained.
  2. Fine-tuning — ``layer3``, ``layer4``, and ``fc`` are unfrozen and trained
     with a reduced learning rate.

Usage::

    python train.py --data_dir /path/to/wikiart
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data import build_dataloaders
from model import get_model, get_trainable_params, unfreeze_for_finetuning
from utils import EarlyStopping, compute_metrics, plot_training_curves

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch.

    Args:
        model: The classifier model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimiser instance.
        device: Target device (cpu / cuda).

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_labels, num_classes=10)
    return avg_loss, metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model on a validation or test set.

    Args:
        model: The classifier model.
        loader: Validation / test DataLoader.
        criterion: Loss function.
        device: Target device (cpu / cuda).

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_labels, num_classes=10)
    return avg_loss, metrics


def _run_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    num_epochs: int,
    early_stopper: EarlyStopping,
    checkpoint_path: str,
    writer: SummaryWriter,
    global_epoch_offset: int = 0,
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """Generic training loop for a single stage.

    Args:
        stage_name: Label for logging (e.g. ``"Stage1"``).
        model: The classifier model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimiser.
        scheduler: Learning-rate scheduler.
        device: Target device.
        num_epochs: Maximum epochs for this stage.
        early_stopper: ``EarlyStopping`` instance.
        checkpoint_path: Path to save the best model weights.
        writer: TensorBoard ``SummaryWriter``.
        global_epoch_offset: Epoch offset for consistent logging across stages.

    Returns:
        Tuple of (train_losses, val_losses, train_accs, val_accs, epochs_run).
    """
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        global_epoch = global_epoch_offset + epoch

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["accuracy"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{stage_name}] Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        writer.add_scalar(f"{stage_name}/train_loss", train_loss, global_epoch)
        writer.add_scalar(f"{stage_name}/val_loss", val_loss, global_epoch)
        writer.add_scalar(f"{stage_name}/train_acc", train_metrics["accuracy"], global_epoch)
        writer.add_scalar(f"{stage_name}/val_acc", val_metrics["accuracy"], global_epoch)
        writer.add_scalar(f"{stage_name}/lr", current_lr, global_epoch)
        writer.add_scalar(f"{stage_name}/val_f1", val_metrics["f1"], global_epoch)

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                f"{stage_name}/train_loss": train_loss,
                f"{stage_name}/val_loss": val_loss,
                f"{stage_name}/train_acc": train_metrics["accuracy"],
                f"{stage_name}/val_acc": val_metrics["accuracy"],
                f"{stage_name}/lr": current_lr,
                f"{stage_name}/val_f1": val_metrics["f1"],
                "epoch": global_epoch,
            })

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")

        # Early stopping
        if early_stopper(val_metrics["accuracy"]):
            print(f"[{stage_name}] Early stopping triggered at epoch {epoch}.")
            break

    return train_losses, val_losses, train_accs, val_accs, epoch


def main(cfg: Config) -> None:
    """Run the full two-stage training pipeline.

    Args:
        cfg: Global configuration object.
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # Data
    train_loader, val_loader, _test_loader, class_to_idx = build_dataloaders(cfg)

    # Model — Stage 1
    model = get_model(cfg, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = AdamW(
        get_trainable_params(model),
        lr=cfg.stage1_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=cfg.scheduler_patience,
        factor=cfg.scheduler_factor,
    )
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

    writer = SummaryWriter(log_dir=cfg.log_dir)

    if cfg.logger_backend == "wandb" and WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # Stage 1: Transfer Learning
    print("\n" + "=" * 60)
    print("STAGE 1 — Transfer Learning (frozen backbone)")
    print("=" * 60)
    s1_tl, s1_vl, s1_ta, s1_va, s1_epochs = _run_stage(
        stage_name="Stage1",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=cfg.stage1_epochs,
        early_stopper=early_stopper,
        checkpoint_path=str(cfg.checkpoint_path),
        writer=writer,
    )

    # Reload best weights before fine-tuning
    model.load_state_dict(torch.load(str(cfg.checkpoint_path), map_location=device, weights_only=True))

    # Stage 2: Fine-tuning
    print("\n" + "=" * 60)
    print("STAGE 2 — Fine-tuning (layer3 + layer4 + fc)")
    print("=" * 60)
    unfreeze_for_finetuning(model)

    optimizer = AdamW(
        get_trainable_params(model),
        lr=cfg.stage2_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=cfg.scheduler_patience,
        factor=cfg.scheduler_factor,
    )
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

    s2_tl, s2_vl, s2_ta, s2_va, _s2_epochs = _run_stage(
        stage_name="Stage2",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=cfg.stage2_epochs,
        early_stopper=early_stopper,
        checkpoint_path=str(cfg.checkpoint_path),
        writer=writer,
        global_epoch_offset=s1_epochs,
    )

    writer.close()

    # Combined curves
    all_tl = s1_tl + s2_tl
    all_vl = s1_vl + s2_vl
    all_ta = s1_ta + s2_ta
    all_va = s1_va + s2_va
    plot_training_curves(all_tl, all_vl, all_ta, all_va, save_path="training_curves.png")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    print("\n[TRAIN] Training complete. Best model saved to:", cfg.checkpoint_path)


def parse_args() -> Config:
    """Parse CLI arguments and return a ``Config`` instance.

    Returns:
        Populated ``Config`` object.
    """
    parser = argparse.ArgumentParser(description="Train WikiArt style classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=25)
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    args = parser.parse_args()

    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.stage1_lr = args.stage1_lr
    cfg.stage2_lr = args.stage2_lr
    cfg.device = args.device
    cfg.logger_backend = args.logger
    return cfg


if __name__ == "__main__":
    config = parse_args()
    main(config)
