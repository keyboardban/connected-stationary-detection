# Longest Stay Detection

Computer-vision pipeline for detecting the person with the longest continuous
stationary duration in CCTV footage. The project is designed for Apple Silicon
and runs Ultralytics inference with `device="mps"`.

## Features

- YOLOv8 instance segmentation for overlapping people
- Occlusion-tuned BoT-SORT tracking
- Mask-aware, CLAHE-normalized MobileNetV2 embeddings with pants-first recovery
- Stable logical IDs on the rendered overlay after successful recovery
- Pants-color fallback matching with low-weight torso support
- Shape-aware lanyard detection with temporal voting
- Sliding-window stationary-duration analysis
- Polygon exclusion zone for ignoring people inside a configured area
- Annotated MP4 output with track IDs and stationary durations

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Input

Place the CCTV video at:

```text
data/entrance.mov
```

The input video is intentionally excluded from Git because CCTV footage may
contain personal data.

## Run

```bash
python main.py
```

The annotated result is written to:

```text
outputs/result.mp4
```

Latest full render result:

```text
Longest stationary person: ID 33 with 41.09 seconds.
```

## Benchmark

```bash
python experiments.py
```

Benchmark results are written to `outputs/experiments_result.csv`.

To compare the baseline, intermediate, and full pipelines on the first 300 frames:

```bash
python evaluate_metrics.py
```

Evaluation results are written to `outputs/evaluation_metrics.csv`. The report
separates raw tracker IDs from logical identities remaining after custom Re-ID
recovery.

For the full implementation history, architecture, output interpretation, and
measured pipeline comparison, read the
[detailed implementation report](docs/IMPLEMENTATION_REPORT.md).

## Notes

- Run on Apple Silicon with an MPS-enabled PyTorch installation.
- Model weights are downloaded automatically on first use.
- Adjust `EXCLUSION_POLYGON` in `main.py` for the target camera view.
