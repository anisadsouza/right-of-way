from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evr.audio import extract_audio_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train siren audio classifier from WAV/MP3 files.")
    parser.add_argument("--data", type=Path, default=Path("data/audio"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--mfcc", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    labels = sorted([p.name for p in args.data.iterdir() if p.is_dir()])
    if not labels:
        raise SystemExit("Put audio files in data/audio/ambient, data/audio/siren, etc.")

    for label in labels:
        for audio_path in (args.data / label).glob("*"):
            if audio_path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                continue
            audio, _ = librosa.load(audio_path, sr=args.sample_rate, mono=True)
            for chunk in chunk_audio(audio, args.sample_rate):
                x_rows.append(extract_audio_features(chunk, args.sample_rate, args.mfcc))
                y_rows.append(label)

    x = np.vstack(x_rows)
    y = np.array(y_rows)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=250, max_depth=14, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test)))

    joblib.dump(model, args.out / "audio_siren_classifier.joblib")
    (args.out / "audio_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"Saved audio model in {args.out}")


def chunk_audio(audio: np.ndarray, sample_rate: int, seconds: float = 1.0, hop: float = 0.5):
    window = int(seconds * sample_rate)
    step = int(hop * sample_rate)
    if len(audio) < window:
        yield np.pad(audio, (0, window - len(audio)))
        return
    for start in range(0, len(audio) - window + 1, step):
        yield audio[start : start + window]


if __name__ == "__main__":
    main()
