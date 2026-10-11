"""Show how the siren score threshold trades false triggers against siren detection.

Run from ev_recognition/:
    python -m evr.sweep_audio_threshold --test-dir demo/esc50_test
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from evr.audio import AudioClassifier

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
MFCC_COUNT = 40
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="demo/esc50_test")
    parser.add_argument("--model", default="models/audio_siren_classifier.joblib")
    args = parser.parse_args()

    classifier = AudioClassifier(
        ROOT / args.model, ROOT / "models" / "audio_labels.txt", SAMPLE_RATE, MFCC_COUNT
    )
    negative_scores, siren_scores, per_category = [], [], {}
    for wav_path in sorted((ROOT / args.test_dir).glob("*.wav")):
        audio, rate = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            raise ValueError(f"{wav_path.name}: expected {SAMPLE_RATE} Hz, got {rate} Hz")
        scores = []
        for start in range(0, len(audio) - SAMPLE_RATE + 1, SAMPLE_RATE):
            _, _, score = classifier.predict(audio[start:start + SAMPLE_RATE])
            scores.append(score)
        if wav_path.stem == "siren_external":
            siren_scores += scores
        else:
            negative_scores += scores
            per_category[wav_path.stem] = scores

    negative, siren = np.array(negative_scores), np.array(siren_scores)
    print(f"Windows: {len(negative)} non-siren, {len(siren)} siren")
    print("threshold | non-siren windows at or above | siren windows at or above")
    for t in THRESHOLDS:
        print(f"   {t:.2f}   |           {100 * (negative >= t).mean():6.2f}%          |        {100 * (siren >= t).mean():6.2f}%")
    print("Highest non-siren categories at 0.50:")
    ranked = sorted(per_category.items(), key=lambda kv: -np.mean(np.array(kv[1]) >= 0.5))[:5]
    for name, scores in ranked:
        print(f"   {name}: {100 * np.mean(np.array(scores) >= 0.5):.1f}%")


if __name__ == "__main__":
    main()