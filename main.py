"""Run longest-stationary-person detection on the entrance video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detector import PersonDetector
from src.features import AppearanceExtractor
from src.logic import StationaryTracker
from src.reid import CustomReID
from src.utils import (
    calculate_centroid,
    draw_info,
    get_bottom_center,
    is_in_exclusion_zone,
)


INPUT_VIDEO = Path("data/entrance.mov")
OUTPUT_VIDEO = Path("outputs/result.mp4")
TRACKER_TYPE = "custom_botsort.yaml"
HISTORY_SECONDS = 1.0
PIXEL_THRESHOLD = 25.0
EXCLUSION_POLYGON = np.array(
    [[750, 1600], [1585, 1440], [1660, 220], [820, 70]], dtype=np.int32
)


def main() -> None:
    """Process the full video and render tracking annotations."""
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid FPS or dimensions.")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    detector = PersonDetector("yolov8m-seg.pt")
    stationary_tracker = StationaryTracker(
        fps=fps,
        history_seconds=HISTORY_SECONDS,
        pixel_threshold=PIXEL_THRESHOLD,
    )
    appearance_extractor = AppearanceExtractor()
    reid_extractor = CustomReID()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            active_ids: set[int] = set()
            results = detector.track_frame(frame, tracker_type=TRACKER_TYPE)
            boxes = results[0].boxes if results else None
            draw_info(frame, exclusion_polygon=EXCLUSION_POLYGON)
            if boxes is not None and boxes.id is not None:
                xyxy_boxes = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                active_ids = {
                    int(track_id)
                    for box, track_id in zip(xyxy_boxes, track_ids)
                    if not is_in_exclusion_zone(
                        get_bottom_center(box), EXCLUSION_POLYGON
                    )
                }
                stationary_tracker.mark_lost_ids(active_ids)
                for box, track_id in zip(xyxy_boxes, track_ids):
                    track_id = int(track_id)
                    if is_in_exclusion_zone(
                        get_bottom_center(box), EXCLUSION_POLYGON
                    ):
                        draw_info(frame, box=box, track_id=track_id, is_excluded=True)
                        continue

                    centroid = calculate_centroid(box)
                    torso_color = appearance_extractor.get_color(frame, box)
                    current_embedding = reid_extractor.get_embedding(frame, box)
                    is_stationary, max_duration = stationary_tracker.update_track(
                        track_id, centroid, current_embedding, torso_color
                    )
                    draw_info(
                        frame,
                        box,
                        track_id,
                        torso_color,
                        max_duration,
                        is_stationary,
                    )
            else:
                stationary_tracker.mark_lost_ids(active_ids)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    longest_id, longest_duration = stationary_tracker.get_longest_stay()
    if longest_id is None:
        print("No tracked person was found in the video.")
    else:
        print(
            f"Longest stationary person: ID {longest_id} "
            f"with {longest_duration:.2f} seconds."
        )
    print(f"Rendered video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
