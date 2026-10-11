"""Evaluate the siren classifier on ESC-50 test WAVs built by prepare_esc50_tests.py.

Run from ev_recognition/:
    python -m evr.evaluate_esc50
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from evr.audio import AudioClassifier

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
MFCC_COUNT = 40
WINDOW_SECONDS = 1
DEFAULT_EMERGENCY_THRESHOLD = 0.80


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="demo/esc50")
    parser.add_argument("--model", default="models/audio_siren_classifier.joblib")
    parser.add_argument("--output", default="reports/esc50_results.csv")
    parser.add_argument("--threshold", type=float, default=DEFAULT_EMERGENCY_THRESHOLD)
    args = parser.parse_args()
    TEST_DIR = ROOT / args.test_dir
    OUTPUT = ROOT / args.output

    classifier = AudioClassifier(
        ROOT / args.model,
        ROOT / "models" / "audio_labels.txt",
        SAMPLE_RATE,
        MFCC_COUNT,
    )

    results = []
    for wav_path in sorted(TEST_DIR.glob("*.wav")):
        audio, rate = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            raise ValueError(f"{wav_path.name}: expected {SAMPLE_RATE} Hz, got {rate} Hz")

        chunk_size = SAMPLE_RATE * WINDOW_SECONDS
        predictions = []
        for start in range(0, len(audio) - chunk_size + 1, chunk_size):
            label, confidence, siren_score = classifier.predict(audio[start:start + chunk_size])
            predictions.append((label, confidence, siren_score))
        if not predictions:
            print(f"Skipped {wav_path.name}: no complete audio windows")
            continue

        total = len(predictions)
        siren_count = sum(label == "siren" for label, _, _ in predictions)
        strong_count = sum(score >= args.threshold for _, _, score in predictions)
        results.append({
            "category": wav_path.stem,
            "expected": "siren" if wav_path.stem == "siren_external" else "ambient",
            "windows_tested": total,
            "siren_predictions": siren_count,
            "siren_rate_pct": round(100 * siren_count / total, 2),
            f"windows_at_or_above_{args.threshold:.2f}": strong_count,
            "strong_rate_pct": round(100 * strong_count / total, 2),
            "mean_siren_score": round(float(np.mean([s for _, _, s in predictions])), 4),
        })

    if not results:
        raise SystemExit(f"No WAV files found in {TEST_DIR}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    strong_key = f"windows_at_or_above_{args.threshold:.2f}"
    print(f"ESC-50 results by category (siren-labelled / at or above {args.threshold:.2f})")
    for row in results:
        print(
            f"{row['category']}: {row['siren_predictions']}/{row['windows_tested']} "
            f"({row['siren_rate_pct']}%) | strong {row['strong_rate_pct']}%"
        )

    for name, expected in (("Negative-window false-trigger", "ambient"), ("External-siren detection", "siren")):
        group = [r for r in results if r["expected"] == expected]
        if not group:
            continue
        total = sum(r["windows_tested"] for r in group)
        labelled = sum(r["siren_predictions"] for r in group)
        strong = sum(r[strong_key] for r in group)
        print(
            f"{name} rate: {100 * labelled / total:.2f}% labelled siren, "
            f"{100 * strong / total:.2f}% at or above {args.threshold:.2f}"
        )
    print(f"Detailed results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
