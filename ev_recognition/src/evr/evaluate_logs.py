from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize realtime detection logs.")
    parser.add_argument("log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.log.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit("Log is empty.")

    total = len(rows)
    emergency_frames = sum(1 for row in rows if row["state"] != "NORMAL")
    max_fused = max(float(row["fused"]) for row in rows)
    commands = sorted(set(row["command"] for row in rows))

    print(f"Frames logged: {total}")
    print(f"Emergency/yield frames: {emergency_frames}")
    print(f"Max fused confidence: {max_fused:.3f}")
    print(f"Commands used: {', '.join(commands)}")


if __name__ == "__main__":
    main()
