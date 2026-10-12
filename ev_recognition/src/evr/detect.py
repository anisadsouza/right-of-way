"""Optional vehicle boxes for the demo view.

The project's own models answer one question per frame: is an emergency vehicle
present? They do not output boxes. This module adds a separate, general purpose
COCO detector (YOLOv8n) that draws boxes around cars, buses, trucks and
motorcycles so the demo is easier to follow. It is for the laptop demo only. The
Raspberry Pi runtime does not need it and does not load it.

Install on the demo computer only:  pip install ultralytics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# COCO class ids for road vehicles.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

NEUTRAL_COLOR = (230, 170, 40)   # BGR, blue-ish
EMERGENCY_COLOR = (40, 40, 235)  # BGR, red


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    ev_score: float | None = None  # only set when per-box scoring is on


class VehicleDetector:
    def __init__(self, weights: str | Path = "models/yolov8n.pt", confidence: float = 0.35, image_size: int = 480) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Vehicle boxes need ultralytics. Install it with: pip install ultralytics") from error
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame_bgr,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=list(VEHICLE_CLASSES),
            verbose=False,
        )[0]
        height, width = frame_bgr.shape[:2]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    max(x1, 0), max(y1, 0), min(x2, width - 1), min(y2, height - 1),
                    VEHICLE_CLASSES.get(int(box.cls[0]), "vehicle"),
                    float(box.conf[0]),
                )
            )
        detections.sort(key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
        return detections


def crop_for_classifier(frame_bgr: np.ndarray, det: Detection, pad: float = 0.1) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    pad_x = int((det.x2 - det.x1) * pad)
    pad_y = int((det.y2 - det.y1) * pad)
    x1, y1 = max(det.x1 - pad_x, 0), max(det.y1 - pad_y, 0)
    x2, y2 = min(det.x2 + pad_x, width), min(det.y2 + pad_y, height)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return frame_bgr[y1:y2, x1:x2]


def draw_boxes(view: np.ndarray, detections: list[Detection], scale: float, ev_threshold: float = 0.80) -> None:
    """Draw boxes on a (possibly downscaled) view. Coordinates are in original frame pixels."""
    for det in detections:
        x1, y1, x2, y2 = (int(v * scale) for v in (det.x1, det.y1, det.x2, det.y2))
        is_ev = det.ev_score is not None and det.ev_score >= ev_threshold
        color = EMERGENCY_COLOR if is_ev else NEUTRAL_COLOR
        text = f"{det.label} {det.confidence:.2f}"
        if det.ev_score is not None:
            text += f" | EV {det.ev_score:.2f}"
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 3 if is_ev else 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(view, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
        cv2.putText(view, text, (x1 + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)