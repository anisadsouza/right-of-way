"""Download ESC-50 clips and split them by the dataset's own folds.

Folds 1-4 (non-siren categories): individual 16 kHz mono WAVs written to the
training ambient folder, so the existing train_audio.py picks them up.
Fold 5 (all categories, including siren): joined per category into the test
folder and never used for training. ESC-50 keeps clips cut from the same source
recording in the same fold, so the test fold is independent of the training folds.

Run from ev_recognition/:
    python prepare_esc50_split.py
"""
import argparse
import csv
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

BASE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
RATE = 16000


def curl(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    temp = dest.with_suffix(".part")
    subprocess.run(
        ["curl", "-fsS", "--retry", "2", "--connect-timeout", "15", "--max-time", "120", "-o", str(temp), url],
        check=True,
    )
    temp.rename(dest)


def download_all(root, names):
    """Download with 8 workers, print progress, retry failures once."""
    pending = sorted(n for n in names if not (root / "audio" / n).exists())
    print(f"{len(names) - len(pending)} already downloaded, {len(pending)} to fetch", flush=True)
    for attempt in (1, 2):
        failed = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(curl, f"{BASE_URL}/audio/{n}", root / "audio" / n): n for n in pending}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    future.result()
                except Exception:
                    failed.append(futures[future])
                if done % 25 == 0 or done == len(pending):
                    print(f"  attempt {attempt}: {done}/{len(pending)} done, {len(failed)} failed", flush=True)
        if not failed:
            return
        pending = failed
    sys.exit(f"{len(pending)} files still failed. Check your connection and run the script again; finished files are kept.")


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        rate, channels = handle.getframerate(), handle.getnchannels()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return rate, data.astype(np.float32)


def convert(path):
    rate, data = read_wav(path)
    if rate != RATE:
        divisor = gcd(rate, RATE)
        data = resample_poly(data, RATE // divisor, rate // divisor)
    return data


def write_wav(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(np.clip(data, -32768, 32767).astype(np.int16).tobytes())


def spread_across_folds(rows, count):
    """Pick clips round-robin across folds so each category is varied."""
    by_fold = {}
    for row in sorted(rows, key=lambda r: r["filename"]):
        by_fold.setdefault(int(row["fold"]), []).append(row)
    picked, index = [], 0
    while len(picked) < count and any(index < len(v) for v in by_fold.values()):
        for fold in sorted(by_fold):
            if index < len(by_fold[fold]) and len(picked) < count:
                picked.append(by_fold[fold][index])
        index += 1
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50", default="raw_datasets/ESC-50-master")
    parser.add_argument("--train-out", default="data/audio/ambient")
    parser.add_argument("--test-out", default="demo/esc50_test")
    parser.add_argument("--train-per-category", type=int, default=5)
    parser.add_argument("--test-per-category", type=int, default=4)
    parser.add_argument("--test-fold", type=int, default=5)
    parser.add_argument("--add-train-sirens", action="store_true",
                        help="Also add ESC-50 siren clips from the training folds to the siren class")
    parser.add_argument("--siren-train-out", default="data/audio/siren")
    args = parser.parse_args()

    root, train_out, test_out = Path(args.esc50), Path(args.train_out), Path(args.test_out)
    meta = root / "meta" / "esc50.csv"
    curl(f"{BASE_URL}/meta/esc50.csv", meta)
    with open(meta, newline="") as handle:
        rows = list(csv.DictReader(handle))

    categories = sorted({r["category"] for r in rows})
    train_rows, test_rows = [], []
    for category in categories:
        in_category = [r for r in rows if r["category"] == category]
        train_pool = [r for r in in_category if int(r["fold"]) != args.test_fold]
        test_pool = [r for r in in_category if int(r["fold"]) == args.test_fold]
        if category != "siren":
            train_rows += spread_across_folds(train_pool, args.train_per_category)
        count = len(test_pool) if category == "siren" else args.test_per_category
        test_rows += [(category, r) for r in sorted(test_pool, key=lambda r: r["filename"])[:count]]

    siren_train_rows = []
    if args.add_train_sirens:
        siren_train_rows = sorted(
            (r for r in rows if r["category"] == "siren" and int(r["fold"]) != args.test_fold),
            key=lambda r: r["filename"],
        )

    needed = {r["filename"] for r in train_rows} | {r["filename"] for r in siren_train_rows}
    needed |= {r["filename"] for r in train_rows} | {r["filename"] for _, r in test_rows}
    download_all(root, needed)

    for row in train_rows:
        write_wav(train_out / f"esc50_{row['filename']}", convert(root / "audio" / row["filename"]))
    print(f"Training negatives: {len(train_rows)} clips from {len({r['category'] for r in train_rows})} categories -> {train_out}")

    if siren_train_rows:
        siren_out = Path(args.siren_train_out)
        for row in siren_train_rows:
            write_wav(siren_out / f"esc50_{row['filename']}", convert(root / "audio" / row["filename"]))
        print(f"Training sirens: {len(siren_train_rows)} clips -> {siren_out}")

    grouped = {}
    for category, row in test_rows:
        grouped.setdefault(category, []).append(convert(root / "audio" / row["filename"]))
    for category, pieces in grouped.items():
        name = "siren_external" if category == "siren" else f"negative_{category}"
        write_wav(test_out / f"{name}.wav", np.concatenate(pieces))
    print(f"Test files: {len(grouped)} categories, fold {args.test_fold} only -> {test_out}")


if __name__ == "__main__":
    main()