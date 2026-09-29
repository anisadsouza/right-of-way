from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

IMAGE_CLASSES = ("ambulance", "fire_truck", "police", "normal")
AUDIO_CLASSES = ("siren", "ambient")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the emergency-vehicle training dataset.")
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--audio", type=Path, default=Path("data/audio"))
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--min-audio", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_summary.json"))
    return parser.parse_args()


def count_files(root: Path, classes: tuple[str, ...], extensions: set[str]) -> dict[str, int]:
    return {
        label: sum(1 for path in (root / label).rglob("*") if path.is_file() and path.suffix.lower() in extensions)
        for label in classes
    }


def main() -> None:
    args = parse_args()
    images = count_files(args.images, IMAGE_CLASSES, IMAGE_EXTENSIONS)
    audio = count_files(args.audio, AUDIO_CLASSES, AUDIO_EXTENSIONS)
    problems = []
    for label, count in images.items():
        if count < args.min_images:
            problems.append(f"images/{label}: {count}, need at least {args.min_images}")
    for label, count in audio.items():
        if count < args.min_audio:
            problems.append(f"audio/{label}: {count}, need at least {args.min_audio}")

    summary = {
        "image_counts": images,
        "audio_counts": audio,
        "image_total": sum(images.values()),
        "audio_total": sum(audio.values()),
        "minimums": {"images_per_class": args.min_images, "audio_files_per_class": args.min_audio},
        "ready_for_training": not problems,
        "problems": problems,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit("Dataset is not ready. Add the missing classes listed above.")


if __name__ == "__main__":
    main()
