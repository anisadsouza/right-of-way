from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evr.audio import extract_audio_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train siren audio classifier from WAV/MP3 files.")
    parser.add_argument("--data", type=Path, default=Path("data/audio"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--mfcc", type=int, default=40)
    parser.add_argument("--report", type=Path, default=Path("reports/audio_training_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    groups: list[str] = []
    labels = sorted([p.name for p in args.data.iterdir() if p.is_dir()])
    if not labels:
        raise SystemExit("Put audio files in data/audio/ambient, data/audio/siren, etc.")

    for label in labels:
        for audio_path in (args.data / label).rglob("*"):
            if audio_path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                continue
            audio, _ = librosa.load(audio_path, sr=args.sample_rate, mono=True)
            for chunk in chunk_audio(audio, args.sample_rate):
                x_rows.append(extract_audio_features(chunk, args.sample_rate, args.mfcc))
                y_rows.append(label)
                groups.append(str(audio_path))

    if not x_rows:
        raise SystemExit("No supported audio files found. Add files under data/audio/siren and data/audio/ambient.")

    x = np.vstack(x_rows)
    y = np.array(y_rows)
    group_array = np.array(groups)
    if len(set(group_array)) < 4 or len(set(y)) < 2:
        raise SystemExit("Need at least four recordings across both audio classes for a reliable held-out test set.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(x, y, groups=group_array))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise SystemExit("The recording-level split lost a class. Add more recordings to each audio class.")

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=250, max_depth=14, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "train_windows": len(x_train),
        "test_windows": len(x_test),
        "train_recordings": len(set(group_array[train_index])),
        "test_recordings": len(set(group_array[test_index])),
        "classification_report": report,
    }
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))

    joblib.dump(model, args.out / "audio_siren_classifier.joblib")
    (args.out / "audio_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
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
