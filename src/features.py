"""Domain-specific appearance extraction for dominant torso colors."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


class AppearanceExtractor:
    """Extract the dominant shirt color from a person's torso region."""

    def __init__(self, noise_threshold: int = 20) -> None:
        if noise_threshold < 1:
            raise ValueError("noise_threshold must be positive.")
        self.noise_threshold = noise_threshold
        self.color_ranges: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {
            "Red": (
                (
                    np.array([0, 100, 70], dtype=np.uint8),
                    np.array([9, 255, 255], dtype=np.uint8),
                ),
                (
                    np.array([170, 100, 70], dtype=np.uint8),
                    np.array([180, 255, 255], dtype=np.uint8),
                ),
            ),
            "Orange": (
                (
                    np.array([10, 100, 80], dtype=np.uint8),
                    np.array([19, 255, 255], dtype=np.uint8),
                ),
            ),
            "Yellow": (
                (
                    np.array([20, 100, 80], dtype=np.uint8),
                    np.array([34, 255, 255], dtype=np.uint8),
                ),
            ),
            "Green": (
                (
                    np.array([35, 70, 60], dtype=np.uint8),
                    np.array([89, 255, 255], dtype=np.uint8),
                ),
            ),
            "Blue": (
                (
                    np.array([90, 80, 60], dtype=np.uint8),
                    np.array([129, 255, 255], dtype=np.uint8),
                ),
            ),
            "Purple": (
                (
                    np.array([130, 70, 60], dtype=np.uint8),
                    np.array([169, 255, 255], dtype=np.uint8),
                ),
            ),
        }

    def get_color(self, frame: np.ndarray, box: Sequence[float]) -> str:
        """Return the dominant primary color in the torso ROI."""
        if frame is None or frame.size == 0 or len(box) != 4:
            return "Unknown"

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return "Unknown"

        box_height = y2 - y1
        torso_y1 = y1 + int(round(box_height * 0.25))
        torso_y2 = y1 + int(round(box_height * 0.75))
        roi = frame[torso_y1:torso_y2, x1:x2]
        if roi.size == 0:
            return "Unknown"

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        scores: dict[str, int] = {}
        for color_name, ranges in self.color_ranges.items():
            color_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for lower_bound, upper_bound in ranges:
                color_mask = cv2.bitwise_or(
                    color_mask, cv2.inRange(hsv_roi, lower_bound, upper_bound)
                )
            scores[color_name] = cv2.countNonZero(color_mask)

        dominant_color, dominant_pixels = max(scores.items(), key=lambda item: item[1])
        if dominant_pixels < self.noise_threshold:
            return "Unknown"
        return dominant_color
