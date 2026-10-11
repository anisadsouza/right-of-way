"""Build 16 kHz mono test WAVs from ESC-50 for false-trigger and external siren checks.

Each output file joins every ESC-50 clip of one category, so one pipeline run
covers the whole category.

Usage:
    python prepare_esc50_tests.py --esc50 raw_datasets/ESC-50-master --out demo/esc50
"""
import argparse
import csv
import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16000
NEGATIVE_CATEGORIES = [
    "car_horn", "church_bells", "clock_alarm", "train", "helicopter",
    "crying_baby", "engine", "dog", "rain", "keyboard_typing",
]
POSITIVE_CATEGORIES = ["siren"]


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return rate, data.astype(np.float32)


def to_target_rate(rate, data):
    if rate == TARGET_RATE:
        return data
    divisor = gcd(rate, TARGET_RATE)
    return resample_poly(data, TARGET_RATE // divisor, rate // divisor)


def write_wav(path, data):
    clipped = np.clip(data, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_RATE)
        handle.writeframes(clipped.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50", required=True, help="Path to the extracted ESC-50 folder")
    parser.add_argument("--out", default="demo/esc50")
    args = parser.parse_args()

    root = Path(args.esc50)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(root / "meta" / "esc50.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for category in NEGATIVE_CATEGORIES + POSITIVE_CATEGORIES:
        files = sorted(
            r["filename"] for r in rows
            if r["category"] == category and (root / "audio" / r["filename"]).exists()
        )
        if not files:
            print(f"{category}: no clips found, skipped")
            continue
        pieces = []
        for name in files:
            rate, data = read_wav(root / "audio" / name)
            pieces.append(to_target_rate(rate, data))
        joined = np.concatenate(pieces)
        label = "siren_external" if category in POSITIVE_CATEGORIES else f"negative_{category}"
        write_wav(out / f"{label}.wav", joined)
        print(f"{label}.wav: {len(files)} clips, {len(joined) / TARGET_RATE:.0f} s")


if __name__ == "__main__":
    main()