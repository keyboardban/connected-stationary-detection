"""Ultralytics YOLO wrapper for person tracking."""

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """Segment and track people in individual video frames."""

    def __init__(self, model_name: str = "yolov8m-seg.pt") -> None:
        self.model = YOLO(model_name)

    def track_frame(
        self, frame: np.ndarray, tracker_type: str = "custom_botsort.yaml"
    ) -> list[Any]:
        """Run person-only segmentation tracking with Apple Silicon MPS."""
        return self.model.track(
            frame,
            persist=True,
            classes=[0],
            tracker=tracker_type,
            device="mps",
            verbose=False,
        )
