import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def label_from_path(path: Path):
    lowered_parts = [part.lower() for part in path.parts]
    lowered_name = path.name.lower()

    if any("cat" in part for part in lowered_parts) or lowered_name.startswith("cat"):
        return "cat"

    if (
        any("dog" in part for part in lowered_parts)
        or any("noncat" in part for part in lowered_parts)
        or lowered_name.startswith("dog")
    ):
        return "noncat"

    return None


def copy_labeled_images(images, label, destination, max_count):
    labeled = [img for img in images if label_from_path(img) == label]
    random.shuffle(labeled)

    if max_count is not None:
        labeled = labeled[:max_count]

    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(labeled):
        target = destination / f"{label}_{index:05d}{source.suffix.lower()}"
        shutil.copy2(source, target)

    return len(labeled)


def prepare_from_h5(source_file: Path, output_root: Path, max_per_class, seed: int):
    import h5py

    random.seed(seed)

    with h5py.File(source_file, "r") as f:
        images = np.array(f["train_set_x"])
        labels = np.array(f["train_set_y"])

    cats_dir = output_root / "cats"
    noncats_dir = output_root / "noncats"

    if cats_dir.exists():
        shutil.rmtree(cats_dir)
    if noncats_dir.exists():
        shutil.rmtree(noncats_dir)

    cats_dir.mkdir(parents=True, exist_ok=True)
    noncats_dir.mkdir(parents=True, exist_ok=True)

    cat_indices = [i for i, y in enumerate(labels) if int(y) == 1]
    noncat_indices = [i for i, y in enumerate(labels) if int(y) == 0]

    random.shuffle(cat_indices)
    random.shuffle(noncat_indices)

    if max_per_class is not None:
        cat_indices = cat_indices[:max_per_class]
        noncat_indices = noncat_indices[:max_per_class]

    for idx, img_index in enumerate(cat_indices):
        image = Image.fromarray(images[img_index])
        image = image.convert("L").resize((64, 64))
        image.save(cats_dir / f"cat_{idx:05d}.png")

    for idx, img_index in enumerate(noncat_indices):
        image = Image.fromarray(images[img_index])
        image = image.convert("L").resize((64, 64))
        image.save(noncats_dir / f"noncat_{idx:05d}.png")

    print(f"Prepared dataset at {output_root}")
    print(f"Cats: {len(cat_indices)}")
    print(f"Noncats: {len(noncat_indices)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare cats/noncats folders from a Kaggle dataset")
    parser.add_argument("--source-root", type=str, default="data_raw", help="Root folder containing extracted Kaggle files")
    parser.add_argument("--output-root", type=str, default="data", help="Output root with cats/noncats folders")
    parser.add_argument("--max-per-class", type=int, default=300, help="Max images per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_root}")

    h5_files = sorted(source_root.glob("train*.h5")) or sorted(source_root.glob("*.h5"))
    if h5_files:
        prepare_from_h5(h5_files[0], output_root, args.max_per_class, args.seed)
        return

    images = find_images(source_root)
    if not images:
        raise ValueError(f"No images found under: {source_root}")

    cats_dir = output_root / "cats"
    noncats_dir = output_root / "noncats"

    if cats_dir.exists():
        shutil.rmtree(cats_dir)
    if noncats_dir.exists():
        shutil.rmtree(noncats_dir)

    cats_count = copy_labeled_images(images, "cat", cats_dir, args.max_per_class)
    noncats_count = copy_labeled_images(images, "noncat", noncats_dir, args.max_per_class)

    if cats_count == 0 or noncats_count == 0:
        raise ValueError(
            "Could not find both class types. Ensure dataset contains cat and dog/noncat images with identifiable names or folders."
        )

    print(f"Prepared dataset at {output_root}")
    print(f"Cats: {cats_count}")
    print(f"Noncats: {noncats_count}")


if __name__ == "__main__":
    main()
