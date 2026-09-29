from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

CLASS_NAMES = ("ambulance", "fire_truck", "police", "normal")
SOURCE_LABELS = {
    "ambulance": "ambulance",
    "ambulance_108": "ambulance",
    "ambulance_sol": "ambulance",
    "fire_truck": "fire_truck",
    "police": "police",
    "army": "normal",
    "vehicle": "normal",
    "auto": "normal",
    "bike": "normal",
    "bus": "normal",
    "car": "normal",
    "tempo traveller": "normal",
    "truck": "normal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Roboflow Indian emergency-vehicle COCO export into classifier crops.")
    parser.add_argument("--source", type=Path, default=Path("raw_datasets/Indian emergency vehicles"))
    parser.add_argument("--out", type=Path, default=Path("data/images"))
    parser.add_argument("--max-per-class", type=int, default=1200)
    parser.add_argument("--padding", type=float, default=0.08)
    return parser.parse_args()


def expanded_box(box: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = box
    x_pad, y_pad = box_width * padding, box_height * padding
    left = max(0, int(x - x_pad))
    top = max(0, int(y - y_pad))
    right = min(width, int(x + box_width + x_pad))
    bottom = min(height, int(y + box_height + y_pad))
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Roboflow dataset folder not found: {args.source}")
    if args.max_per_class < 1:
        raise SystemExit("--max-per-class must be positive")

    for label in CLASS_NAMES:
        (args.out / label).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        annotation_path = args.source / split / "_annotations.coco.json"
        if not annotation_path.exists():
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_labels = {item["id"]: SOURCE_LABELS.get(item["name"].lower()) for item in data["categories"]}
        images = {item["id"]: item for item in data["images"]}
        for annotation in data["annotations"]:
            label = category_labels.get(annotation["category_id"])
            if label is None or counts[label] >= args.max_per_class:
                continue
            image_info = images[annotation["image_id"]]
            image_path = args.source / split / image_info["file_name"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                skipped["unreadable_image"] += 1
                continue
            left, top, right, bottom = expanded_box(annotation["bbox"], frame.shape[1], frame.shape[0], args.padding)
            crop = frame[top:bottom, left:right]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                skipped["small_crop"] += 1
                continue
            file_name = f"{split}_{annotation['image_id']}_{annotation['id']}.jpg"
            if cv2.imwrite(str(args.out / label / file_name), crop):
                counts[label] += 1
            else:
                skipped["write_failed"] += 1

    summary = {
        "source": str(args.source),
        "output": str(args.out),
        "max_per_class": args.max_per_class,
        "crops_created": dict(counts),
        "skipped": dict(skipped),
    }
    manifest_path = Path("data/roboflow_vision_manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    missing = [label for label in CLASS_NAMES if counts[label] == 0]
    if missing:
        raise SystemExit(f"No crops created for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
