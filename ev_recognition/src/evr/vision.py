from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class VisionClassifier:
    def __init__(self, model_path: Path, labels_path: Path, input_size: int) -> None:
        self.labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
        self.input_size = input_size
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]["dtype"]

    def predict(self, frame_bgr: np.ndarray) -> tuple[str, float, float]:
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(np.float32) / 255.0
        batch = np.expand_dims(image, axis=0)

        if self.input_dtype == np.uint8:
            scale, zero_point = self.input_details[0]["quantization"]
            batch = batch / scale + zero_point
            batch = batch.astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]["index"], batch)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        if self.output_details[0]["dtype"] == np.uint8:
            scale, zero_point = self.output_details[0]["quantization"]
            output = scale * (output.astype(np.float32) - zero_point)

        probs = _softmax(output)
        best_index = int(np.argmax(probs))
        label = self.labels[best_index]
        confidence = float(probs[best_index])
        emergency_score = float(sum(probs[i] for i, name in enumerate(self.labels) if name != "normal"))
        return label, confidence, emergency_score


def estimate_approach_side(frame_bgr: np.ndarray) -> str:
    """Lightweight visual cue: bright/red-blue light clusters left/center/right."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    blue = cv2.inRange(hsv, (95, 80, 80), (135, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), blue)
    moments = cv2.moments(mask)
    if moments["m00"] < 50:
        return "unknown"
    cx = int(moments["m10"] / moments["m00"])
    width = frame_bgr.shape[1]
    if cx < width * 0.38:
        return "left"
    if cx > width * 0.62:
        return "right"
    return "center"


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values -= np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)
