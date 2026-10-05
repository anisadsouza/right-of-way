"""
rebalance_dataset.py

Caps oversized classes in data/images so fire_truck isn't drowned
out by a 20-to-1 imbalance. Moves excess images to a sidecar folder
(data/images_excluded) rather than deleting them, so nothing is lost,
you can always pull more back in later if needed.

Run from inside ev_recognition/, with your venv active:
    python rebalance_dataset.py
"""

import random
import shutil
from pathlib import Path

DATA_DIR = Path("data/images")
EXCLUDED_DIR = Path("data/images_excluded")
CAP_PER_CLASS = 2500   # adjust this number if you want a different ceiling
SEED = 42

valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    rng = random.Random(SEED)
    class_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])

    print(f"{'Class':<12} {'Before':>8} {'Kept':>8} {'Moved out':>10}")
    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext)
        before = len(files)

        if before <= CAP_PER_CLASS:
            print(f"{class_dir.name:<12} {before:>8} {before:>8} {0:>10}")
            continue

        rng.shuffle(files)
        keep = files[:CAP_PER_CLASS]
        move_out = files[CAP_PER_CLASS:]

        dest_dir = EXCLUDED_DIR / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in move_out:
            shutil.move(str(f), str(dest_dir / f.name))

        print(f"{class_dir.name:<12} {before:>8} {len(keep):>8} {len(move_out):>10}")

    print(f"\nExcess images moved to {EXCLUDED_DIR}/, not deleted. Safe to pull back in later if needed.")


if __name__ == "__main__":
    main()
