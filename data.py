"""Data loading and preprocessing for the WikiArt style classifier.

Provides WikiArtDataset, dataset splitting, transforms, and DataLoader factories.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Config


class WikiArtDataset(Dataset):
    """PyTorch Dataset for WikiArt images organised in per-class subdirectories.

    Expected directory layout::

        data_dir/
        ├── Art_Nouveau/
        │   ├── img001.jpg
        │   └── ...
        ├── Baroque/
        └── ...

    Args:
        image_paths: List of absolute image file paths.
        labels: Corresponding integer labels.
        transform: Optional torchvision transform pipeline.
        class_to_idx: Mapping from class name to integer index.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx or {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """Return (image_tensor, label) for the given index."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def collect_samples(
    data_dir: str,
    class_names: List[str],
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Walk *data_dir* and collect (path, label) pairs for the given classes.

    Args:
        data_dir: Root directory containing one sub-folder per class.
        class_names: Ordered list of class folder names.

    Returns:
        Tuple of (image_paths, labels, class_to_idx).
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    image_paths: List[str] = []
    labels: List[int] = []

    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        if not class_dir.is_dir():
            print(f"[WARNING] Directory not found for class '{class_name}': {class_dir}")
            continue
        for entry in sorted(class_dir.iterdir()):
            if entry.suffix.lower() in valid_extensions:
                image_paths.append(str(entry))
                labels.append(class_to_idx[class_name])

    return image_paths, labels, class_to_idx


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """Split dataset into train / val / test with stratification per class.

    Args:
        image_paths: Full list of image paths.
        labels: Corresponding integer labels.
        split_ratios: (train, val, test) fractions — must sum to 1.
        seed: Random seed for reproducibility.

    Returns:
        Three tuples of (paths, labels) for train, val, and test splits.
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1."

    rng = random.Random(seed)

    # Group indices by class for stratified splitting.
    class_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for _cls, indices in sorted(class_indices.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        for i in train_idx:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])
        for i in val_idx:
            val_paths.append(image_paths[i])
            val_labels.append(labels[i])
        for i in test_idx:
            test_paths.append(image_paths[i])
            test_labels.append(labels[i])

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_train_transform(cfg: Config) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Args:
        cfg: Global configuration object.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])


def get_val_transform(cfg: Config) -> transforms.Compose:
    """Build the validation / test transform pipeline (no augmentation).

    Args:
        cfg: Global configuration object.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])


def build_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Convenience function: collect samples, split, and return DataLoaders.

    Args:
        cfg: Global configuration object.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx).
    """
    image_paths, labels, class_to_idx = collect_samples(cfg.data_dir, cfg.class_names)
    print(f"[DATA] Collected {len(image_paths)} images across {len(class_to_idx)} classes.")

    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(
        image_paths, labels, cfg.split_ratios, cfg.seed,
    )
    print(f"[DATA] Split sizes — train: {len(train_p)}, val: {len(val_p)}, test: {len(test_p)}")

    train_ds = WikiArtDataset(train_p, train_l, get_train_transform(cfg), class_to_idx)
    val_ds = WikiArtDataset(val_p, val_l, get_val_transform(cfg), class_to_idx)
    test_ds = WikiArtDataset(test_p, test_l, get_val_transform(cfg), class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_to_idx
