"""Download and organize WikiArt images from HuggingFace into per-class directories.

Downloads images for the 10 target styles used by the classifier and saves them
into ``data/<StyleName>/`` sub-folders.

Usage::

    python download_dataset.py                 # default: data/
    python download_dataset.py --output_dir my_data --max_per_class 500
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

TARGET_STYLES = {
    "Art Nouveau (Modern)": "Art_Nouveau",
    "Baroque": "Baroque",
    "Cubism": "Cubism",
    "Expressionism": "Expressionism",
    "Impressionism": "Impressionism",
    "Realism": "Realism",
    "Romanticism": "Romanticism",
    "Surrealism": "Surrealism",
    "Symbolism": "Symbolism",
    "Ukiyo-e": "Ukiyo_e",
}


def download_wikiart(
    output_dir: str = "data",
    max_per_class: int = 0,
) -> None:
    """Stream the Artificio/WikiArt dataset and save target-style images.

    Args:
        output_dir: Root directory to save images.
        max_per_class: Maximum images per class (0 = unlimited).
    """
    out = Path(output_dir)
    for folder in TARGET_STYLES.values():
        (out / folder).mkdir(parents=True, exist_ok=True)

    print("[DOWNLOAD] Loading Artificio/WikiArt (streaming) ...")
    ds = load_dataset("Artificio/WikiArt", split="train", streaming=True)

    counts: dict[str, int] = {v: 0 for v in TARGET_STYLES.values()}
    saved = 0

    for example in tqdm(ds, desc="Scanning dataset"):
        style_raw = example.get("style", "")
        if style_raw not in TARGET_STYLES:
            continue

        folder_name = TARGET_STYLES[style_raw]

        if max_per_class > 0 and counts[folder_name] >= max_per_class:
            if all(c >= max_per_class for c in counts.values()):
                break
            continue

        image = example["image"]
        idx = counts[folder_name]
        filename = f"{folder_name}_{idx:05d}.jpg"
        save_path = out / folder_name / filename

        try:
            image.convert("RGB").save(str(save_path), "JPEG", quality=95)
            counts[folder_name] += 1
            saved += 1
        except Exception as exc:
            print(f"[WARN] Failed to save {save_path}: {exc}")

    print(f"\n[DOWNLOAD] Done. Saved {saved} images total.")
    for name, count in sorted(counts.items()):
        print(f"  {name:20s}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WikiArt dataset")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Max images per class (0 = all available)",
    )
    args = parser.parse_args()

    download_wikiart(args.output_dir, args.max_per_class)
