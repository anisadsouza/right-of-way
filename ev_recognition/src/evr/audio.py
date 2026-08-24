from __future__ import annotations

from pathlib import Path

import joblib
import librosa
import numpy as np


class AudioClassifier:
    def __init__(self, model_path: Path, labels_path: Path, sample_rate: int, mfcc_count: int) -> None:
        self.model = joblib.load(model_path)
        self.labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
        self.sample_rate = sample_rate
        self.mfcc_count = mfcc_count

    def predict(self, audio: np.ndarray) -> tuple[str, float, float]:
        features = extract_audio_features(audio, self.sample_rate, self.mfcc_count).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
        else:
            label = self.model.predict(features)[0]
            probs = np.zeros(len(self.labels), dtype=np.float32)
            probs[self.labels.index(label)] = 1.0

        best_index = int(np.argmax(probs))
        label = self.labels[best_index]
        confidence = float(probs[best_index])
        siren_score = float(sum(probs[i] for i, name in enumerate(self.labels) if name != "ambient"))
        return label, confidence, siren_score


def extract_audio_features(audio: np.ndarray, sample_rate: int, mfcc_count: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=mfcc_count)
    delta = librosa.feature.delta(mfcc)
    stats = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
        ]
    )
    return stats.astype(np.float32)
