from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from evr.validate_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS

VISION_ALIASES = {
    "ambulance": "ambulance", "ambulances": "ambulance",
    "fire": "fire_truck", "firetruck": "fire_truck", "fire_truck": "fire_truck", "firetrucks": "fire_truck",
    "police": "police", "police_car": "police", "policecar": "police",
    "normal": "normal", "car": "normal", "cars": "normal", "vehicle": "normal",
    "vehicles": "normal", "non_emergency": "normal", "nonemergency": "normal",
}
AUDIO_ALIASES = {
    "siren": "siren", "sirens": "siren", "ambulance": "siren", "police": "siren", "fire": "siren",
    "firetruck": "siren", "fire_truck": "siren",
    "ambient": "ambient", "traffic": "ambient", "noise": "ambient", "normal": "ambient", "background": "ambient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a downloaded dataset into this project's canonical class folders.")
    parser.add_argument("source", type=Path, help="Root folder of the downloaded, extracted dataset.")
    parser.add_argument("--kind", choices=("images", "audio"), required=True)
    parser.add_argument("--destination", type=Path, help="Default: data/images or data/audio")
    return parser.parse_args()


def normalized(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def label_for(path: Path, aliases: dict[str, str]) -> str | None:
    for parent in path.parents:
        label = aliases.get(normalized(parent.name))
        if label:
            return label
    return None


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Not a folder: {args.source}")
    destination = args.destination or Path("data") / args.kind
    aliases = VISION_ALIASES if args.kind == "images" else AUDIO_ALIASES
    extensions = IMAGE_EXTENSIONS if args.kind == "images" else AUDIO_EXTENSIONS
    copied: Counter[str] = Counter()
    skipped: list[str] = []

    for file in args.source.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in extensions:
            continue
        label = label_for(file, aliases)
        if label is None:
            skipped.append(str(file))
            continue
        target_dir = destination / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / file.name
        index = 1
        while target.exists():
            target = target_dir / f"{file.stem}_{index}{file.suffix.lower()}"
            index += 1
        shutil.copy2(file, target)
        copied[label] += 1

    manifest = {"source": str(args.source), "kind": args.kind, "copied": dict(copied), "skipped_unlabelled": skipped}
    manifest_path = Path("data/import_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not copied:
        raise SystemExit("No labelled files were copied. Rename class folders or add aliases in prepare_dataset.py.")


if __name__ == "__main__":
    main()
