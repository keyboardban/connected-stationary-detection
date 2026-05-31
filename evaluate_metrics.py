"""Evaluate Longest Stay Detection pipeline variants on a short video segment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import cv2
import pandas as pd
import torch

from src.detector import PersonDetector
from src.features import AppearanceExtractor
from src.logic import StationaryTracker
from src.reid import CustomReID
from src.utils import calculate_centroid


VIDEO_PATH = Path("data/entrance.mov")
OUTPUT_CSV = Path("outputs/evaluation_metrics.csv")
DEFAULT_MAX_FRAMES = 300
HISTORY_SECONDS = 1.0
PIXEL_THRESHOLD = 25.0


class EvaluationConfig(TypedDict):
    """Configuration for one detector, tracker, and optional Re-ID pipeline."""

    experiment: str
    detection_model: str
    tracker: str
    reid_enabled: bool


EXPERIMENT_CONFIGS: list[EvaluationConfig] = [
    {
        "experiment": "Config A (Baseline)",
        "detection_model": "yolov8m.pt",
        "tracker": "bytetrack.yaml",
        "reid_enabled": False,
    },
    {
        "experiment": "Config B (Intermediate)",
        "detection_model": "yolov8m-pose.pt",
        "tracker": "botsort.yaml",
        "reid_enabled": False,
    },
    {
        "experiment": "Config C (Ours)",
        "detection_model": "yolov8m-seg.pt",
        "tracker": "custom_botsort.yaml",
        "reid_enabled": True,
    },
]

RESULT_COLUMNS = [
    "Experiment",
    "Detection Model",
    "Tracker",
    "ReID Enabled",
    "Total Unique IDs",
    "Average FPS",
]


def synchronize_mps() -> None:
    """Wait for queued MPS work so benchmark timing is accurate."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def format_markdown_table(dataframe: pd.DataFrame) -> str:
    """Return a Markdown table without requiring the optional tabulate package."""
    headers = [str(column) for column in dataframe.columns]
    rows = [
        [str(value) for value in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        cells = [
            value.ljust(widths[index]) for index, value in enumerate(row)
        ]
        return f"| {' | '.join(cells)} |"

    divider = f"| {' | '.join('-' * width for width in widths)} |"
    return "\n".join([format_row(headers), divider, *(format_row(row) for row in rows)])


def run_evaluation(
    config: EvaluationConfig,
    video_path: str | Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, object]:
    """Evaluate one pipeline variant over exactly ``max_frames`` frames."""
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid FPS metadata.")

    detector = PersonDetector(config["detection_model"])
    tracker_logic: StationaryTracker | None = None
    appearance_extractor: AppearanceExtractor | None = None
    reid_extractor: CustomReID | None = None
    if config["reid_enabled"]:
        tracker_logic = StationaryTracker(
            fps=fps,
            history_seconds=HISTORY_SECONDS,
            pixel_threshold=PIXEL_THRESHOLD,
        )
        appearance_extractor = AppearanceExtractor()
        reid_extractor = CustomReID(device="mps")

    unique_track_ids: set[int] = set()
    processed_frames = 0
    synchronize_mps()
    start_time = time.perf_counter()

    try:
        while processed_frames < max_frames:
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Video ended after {processed_frames} frames; "
                    f"{max_frames} frames were requested."
                )

            results = detector.track_frame(frame, tracker_type=config["tracker"])
            boxes = results[0].boxes if results else None
            active_ids: set[int] = set()
            if boxes is not None and boxes.id is not None:
                xyxy_boxes = boxes.xyxy.cpu().numpy()
                track_ids = [int(track_id) for track_id in boxes.id.int().cpu().tolist()]
                active_ids = set(track_ids)
                unique_track_ids.update(active_ids)

                if tracker_logic is not None:
                    tracker_logic.mark_lost_ids(active_ids)
                    if appearance_extractor is None or reid_extractor is None:
                        raise RuntimeError("Custom ReID dependencies were not initialized.")
                    for box, track_id in zip(xyxy_boxes, track_ids):
                        centroid = calculate_centroid(box)
                        torso_color = appearance_extractor.get_color(frame, box)
                        pants_color = appearance_extractor.get_pants_color(frame, box)
                        embeddings = reid_extractor.get_embedding(frame, box)
                        tracker_logic.update_track(
                            track_id, centroid, embeddings, torso_color, pants_color
                        )
            elif tracker_logic is not None:
                tracker_logic.mark_lost_ids(active_ids)

            processed_frames += 1
    finally:
        capture.release()

    synchronize_mps()
    elapsed_seconds = time.perf_counter() - start_time
    average_fps = processed_frames / elapsed_seconds if elapsed_seconds else 0.0
    return {
        "Experiment": config["experiment"],
        "Detection Model": config["detection_model"],
        "Tracker": config["tracker"],
        "ReID Enabled": config["reid_enabled"],
        "Total Unique IDs": len(unique_track_ids),
        "Average FPS": round(average_fps, 2),
    }


def main() -> None:
    """Evaluate every configured pipeline and save the comparison table."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results: list[dict[str, object]] = []

    for config in EXPERIMENT_CONFIGS:
        print(f"Evaluating {config['experiment']} ...")
        result = run_evaluation(config, VIDEO_PATH)
        evaluation_results.append(result)

    dataframe = pd.DataFrame(evaluation_results, columns=RESULT_COLUMNS)
    dataframe.to_csv(OUTPUT_CSV, index=False)

    print("\nEvaluation Metrics")
    print(format_markdown_table(dataframe))
    print(f"\nSaved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
