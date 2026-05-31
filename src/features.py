"""Domain-specific color extraction for clothing regions and lanyards."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Sequence

import cv2
import numpy as np


class AppearanceExtractor:
    """Extract dominant colors from clothing regions."""

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

    def _get_region_color(
        self,
        frame: np.ndarray,
        box: Sequence[float],
        start_ratio: float,
        end_ratio: float,
    ) -> str:
        """Return the dominant primary color in one vertical body region."""
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
        roi_y1 = y1 + int(round(box_height * start_ratio))
        roi_y2 = y1 + int(round(box_height * end_ratio))
        roi = frame[roi_y1:roi_y2, x1:x2]
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

    def get_color(self, frame: np.ndarray, box: Sequence[float]) -> str:
        """Return the dominant torso color as a low-weight fallback feature."""
        return self._get_region_color(frame, box, 0.25, 0.55)

    def get_pants_color(self, frame: np.ndarray, box: Sequence[float]) -> str:
        """Return the dominant pants color for uniform-resistant matching."""
        return self._get_region_color(frame, box, 0.55, 0.95)


class LanyardExtractor:
    """Detect thin colored lanyards without confusing blue shirts for straps."""

    def __init__(
        self,
        noise_threshold: int = 8,
        voting_window: int = 15,
    ) -> None:
        if noise_threshold < 1:
            raise ValueError("noise_threshold must be positive.")
        if voting_window < 1:
            raise ValueError("voting_window must be positive.")

        self.noise_threshold = noise_threshold
        self.voting_window = voting_window
        self.color_ranges = AppearanceExtractor().color_ranges
        self.color_history: defaultdict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.voting_window)
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))

    def _crop_chest_roi(
        self, frame: np.ndarray, box: Sequence[float]
    ) -> np.ndarray | None:
        """Crop the central upper chest where a lanyard is most likely visible."""
        if frame is None or frame.size == 0 or len(box) != 4:
            return None

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))
        if x2 <= x1 or y2 <= y1:
            return None

        width = x2 - x1
        height = y2 - y1
        roi_x1 = x1 + int(round(width * 0.30))
        roi_x2 = x1 + int(round(width * 0.70))
        roi_y1 = y1 + int(round(height * 0.18))
        roi_y2 = y1 + int(round(height * 0.52))
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi if roi.size else None

    def _score_color_components(
        self, hsv_roi: np.ndarray, color_name: str
    ) -> float:
        """Score thin central color components and reject shirt-sized regions."""
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower_bound, upper_bound in self.color_ranges[color_name]:
            mask = cv2.bitwise_or(
                mask, cv2.inRange(hsv_roi, lower_bound, upper_bound)
            )

        roi_area = float(mask.size)
        mask_pixels = cv2.countNonZero(mask)
        if mask_pixels < self.noise_threshold:
            return 0.0

        # A blue shirt can dominate this ROI. A real lanyard should remain a
        # narrow component rather than cover a large fraction of the chest.
        coverage = mask_pixels / roi_area
        max_coverage = 0.25 if color_name == "Blue" else 0.35
        if coverage > max_coverage:
            return 0.0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        roi_height, roi_width = mask.shape
        roi_center_x = roi_width / 2.0
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue

            area_ratio = area / roi_area
            x, _, width, height = cv2.boundingRect(contour)
            short_side = max(1, min(width, height))
            aspect_ratio = max(width, height) / short_side
            center_x = x + width / 2.0
            center_distance_ratio = abs(center_x - roi_center_x) / max(
                1.0, roi_width / 2.0
            )

            if not 0.002 <= area_ratio <= 0.18:
                continue
            if aspect_ratio < 1.8:
                continue
            if center_distance_ratio > 0.85:
                continue

            center_bonus = 1.0 - center_distance_ratio
            best_score = max(best_score, area * aspect_ratio * (1.0 + center_bonus))
        return best_score

    def get_color(
        self,
        frame: np.ndarray,
        box: Sequence[float],
        track_id: int,
    ) -> str:
        """Return a temporally smoothed shape-aware lanyard color."""
        roi = self._crop_chest_roi(frame, box)
        if roi is None:
            return self._get_voted_color(track_id)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv_roi)
        enhanced_value = self.clahe.apply(value)
        hsv_roi = cv2.merge((hue, saturation, enhanced_value))

        scores = {
            color_name: self._score_color_components(hsv_roi, color_name)
            for color_name in self.color_ranges
        }
        dominant_color, dominant_score = max(scores.items(), key=lambda item: item[1])
        if dominant_score > 0.0:
            self.color_history[int(track_id)].append(dominant_color)
        return self._get_voted_color(track_id)

    def _get_voted_color(self, track_id: int) -> str:
        """Return the most frequent recent valid observation for one Track ID."""
        history = self.color_history.get(int(track_id))
        if not history:
            return "Unknown"
        return Counter(history).most_common(1)[0][0]
