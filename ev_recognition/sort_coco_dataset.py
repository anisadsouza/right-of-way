"""
sort_coco_dataset.py

Sorts the Roboflow "Indian emergency vehicles" dataset into
data/images/ambulance/, fire_truck/, police/, normal/.

Unlike prepare_dataset.py, this reads the _annotations.coco.json
file in each split (train/valid/test) to find out what's actually
in each image, since this dataset has no per-class folders.

HOW TO USE
----------
Run this from inside ev_recognition/, with your venv active:

    python sort_coco_dataset.py

It assumes the dataset is at:
    raw_datasets/Indian emergency vehicles/{train,valid,test}/

and writes into:
    data/images/{ambulance,fire_truck,police,normal}/

Edit SOURCE_DIR and DEST_DIR below if your paths differ.
This only COPIES files, your originals are untouched, safe to re-run.
"""

import json
import shutil
from pathlib import Path
from collections import Counter

SOURCE_DIR = Path("raw_datasets/Indian emergency vehicles")
DEST_DIR = Path("data/images")

KEYWORD_RULES = [
    ("ambulance", "ambulance"),
    ("fire", "fire_truck"),
    ("police", "police"),
    ("army", "normal"),
    ("car", "normal"),
    ("truck", "normal"),
    ("bus", "normal"),
    ("bike", "normal"),
    ("auto", "normal"),
    ("tempo", "normal"),
    ("vehicle", "normal"),
]

IGNORE_CATEGORIES = {"road_sign", "writing", "hose", "lamp", "symbol"}


def classify_category(name: str):
    name_lower = name.lower()
    if any(ignored in name_lower for ignored in IGNORE_CATEGORIES):
        return None
    for keyword, target in KEYWORD_RULES:
        if keyword in name_lower:
            return target
    return None


def copy_unique(src: Path, dest_dir: Path):
    dest = dest_dir / src.name
    i = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.copy2(src, dest)


def main():
    for folder in ["ambulance", "fire_truck", "police", "normal"]:
        (DEST_DIR / folder).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    unrecognised = Counter()
    splits_found = []

    for split in ["train", "valid", "test"]:
        split_dir = SOURCE_DIR / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"Skipping {split}: no _annotations.coco.json found at {ann_path}")
            continue

        splits_found.append(split)
        with open(ann_path) as f:
            coco = json.load(f)

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        image_categories = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            image_categories.setdefault(img_id, []).append(cat_name)

        for img_id, file_name in image_id_to_file.items():
            src_path = split_dir / file_name
            if not src_path.exists():
                continue

            cats_here = image_categories.get(img_id, [])
            if not cats_here:
                continue

            targets = [classify_category(c) for c in cats_here]
            targets = [t for t in targets if t is not None]

            if not targets:
                for c in cats_here:
                    unrecognised[c] += 1
                continue

            target_class = Counter(targets).most_common(1)[0][0]
            copy_unique(src_path, DEST_DIR / target_class)
            counts[target_class] += 1

    print("\n--- Sorting complete ---")
    print(f"Splits processed: {splits_found}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]} images")

    if unrecognised:
        print("\nCategory names seen but not matched to any class:")
        for cat, n in unrecognised.most_common():
            print(f"  {cat}: {n} images")
        print("If any of these should be ambulance/fire_truck/police, add a")
        print("matching keyword to KEYWORD_RULES near the top and re-run.")

    if not splits_found:
        print("\nNo splits found at all. Check SOURCE_DIR matches your actual folder layout,")
        print("run: ls \"raw_datasets/Indian emergency vehicles\" to see what's actually there.")


if __name__ == "__main__":
    main()