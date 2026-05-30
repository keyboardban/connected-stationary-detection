"""Benchmark detection and tracking combinations on the entrance video."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import pandas as pd

from src.detector import PersonDetector


INPUT_VIDEO = Path("data/entrance.mov")
OUTPUT_CSV = Path("outputs/experiments_result.csv")
MAX_FRAMES = 150
MODELS = ["yolov8n.pt", "yolov8m.pt", "yolo11m.pt"]
TRACKERS = ["bytetrack.yaml", "botsort.yaml"]


def benchmark_pipeline(model_name: str, tracker_type: str) -> dict[str, object]:
    """Benchmark one model/tracker combination over the first frames."""
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    detector = PersonDetector(model_name)
    unique_ids: set[int] = set()
    processed_frames = 0
    start_time = time.perf_counter()

    try:
        while processed_frames < MAX_FRAMES:
            success, frame = capture.read()
            if not success:
                break

            results = detector.track_frame(frame, tracker_type)
            boxes = results[0].boxes if results else None
            if boxes is not None and boxes.id is not None:
                unique_ids.update(
                    int(track_id) for track_id in boxes.id.int().cpu().tolist()
                )
            processed_frames += 1
    finally:
        capture.release()

    elapsed_seconds = time.perf_counter() - start_time
    processing_fps = processed_frames / elapsed_seconds if elapsed_seconds else 0.0
    return {
        "model": model_name,
        "tracker": tracker_type,
        "frames_processed": processed_frames,
        "processing_fps": round(processing_fps, 2),
        "total_unique_ids": len(unique_ids),
    }


def main() -> None:
    """Run every model/tracker combination and write a CSV report."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    benchmark_results: list[dict[str, object]] = []

    for model_name in MODELS:
        for tracker_type in TRACKERS:
            print(f"Benchmarking model={model_name}, tracker={tracker_type} ...")
            result = benchmark_pipeline(model_name, tracker_type)
            benchmark_results.append(result)
            print(result)

    dataframe = pd.DataFrame(benchmark_results)
    dataframe.to_csv(OUTPUT_CSV, index=False)
    print(f"Benchmark results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
