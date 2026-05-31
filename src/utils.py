"""Utility functions for drawing and bounding-box calculations."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def calculate_centroid(box: Sequence[float]) -> list[float]:
    """Return the center point of an ``[x1, y1, x2, y2]`` bounding box."""
    if len(box) != 4:
        raise ValueError("A bounding box must contain exactly four coordinates.")

    x1, y1, x2, y2 = (float(value) for value in box)
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def get_bottom_center(box: Sequence[float]) -> tuple[int, int]:
    """Return the bottom-center point of a person's bounding box."""
    if len(box) != 4:
        raise ValueError("A bounding box must contain exactly four coordinates.")

    x1, _, x2, y2 = (float(value) for value in box)
    return int((x1 + x2) / 2.0), int(y2)


def is_in_exclusion_zone(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """Return whether a point is strictly inside an exclusion polygon."""
    polygon_array = np.asarray(polygon, dtype=np.int32)
    if polygon_array.ndim != 2 or polygon_array.shape[1] != 2:
        raise ValueError("polygon must have shape (N, 2).")
    return cv2.pointPolygonTest(polygon_array, point, False) > 0


def draw_info(
    frame: np.ndarray,
    box: Sequence[float] | None = None,
    track_id: int | None = None,
    torso_color: str = "Unknown",
    max_stationary_duration: float = 0.0,
    is_stationary: bool = False,
    exclusion_polygon: np.ndarray | None = None,
    is_excluded: bool = False,
    pants_color: str = "Unknown",
    lanyard_color: str = "Unknown",
) -> np.ndarray:
    """Draw the exclusion zone and optional tracking information."""
    if exclusion_polygon is not None:
        polygon_array = np.asarray(exclusion_polygon, dtype=np.int32)
        cv2.polylines(frame, [polygon_array], True, (128, 128, 128), 2)

    if box is None:
        return frame

    x1, y1, x2, y2 = (int(round(value)) for value in box)
    if is_excluded:
        box_color = (128, 128, 128)
        label = f"ID: {track_id} | EXCLUDED"
    else:
        box_color = (0, 0, 255) if is_stationary else (0, 255, 0)
        label = (
            f"ID: {track_id} | Pants: {pants_color} | "
            f"Lanyard: {lanyard_color} | Torso: {torso_color} | "
            f"Max Stay: {max_stationary_duration:.2f}s"
        )

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    text_x = max(0, x1)
    text_y = max(text_height + baseline + 4, y1 - 8)
    background_top = max(0, text_y - text_height - baseline - 4)
    background_right = min(frame.shape[1] - 1, text_x + text_width + 6)

    cv2.rectangle(
        frame,
        (text_x, background_top),
        (background_right, text_y + baseline),
        box_color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 3, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame
