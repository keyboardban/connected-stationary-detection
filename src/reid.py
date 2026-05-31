"""Multi-region MobileNetV2 person re-identification embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


@dataclass
class RegionEmbeddings:
    """Normalized appearance vectors for complementary person regions."""

    head: torch.Tensor | None = None
    pants: torch.Tensor | None = None
    torso: torch.Tensor | None = None


class CustomReID:
    """Extract uniform-resistant multi-region embeddings on Apple Silicon MPS."""

    REGION_RANGES = {
        "head": (0.00, 0.25),
        "pants": (0.55, 0.95),
        "torso": (0.25, 0.55),
    }

    def __init__(self, device: str = "mps") -> None:
        if device != "mps":
            raise ValueError("CustomReID is configured for Apple Silicon MPS.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is not available. Run this pipeline on Apple Silicon with "
                "an MPS-enabled PyTorch installation."
            )

        self.device = torch.device(device)
        self.model = mobilenet_v2(weights="DEFAULT")
        self.model.classifier = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _prepare_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        start_ratio: float,
        end_ratio: float,
    ) -> torch.Tensor | None:
        """Crop and normalize one vertical region of a person bounding box."""
        height = y2 - y1
        roi_y1 = y1 + int(height * start_ratio)
        roi_y2 = y1 + int(height * end_ratio)
        crop = frame[
            max(0, roi_y1) : min(frame.shape[0], roi_y2),
            max(0, x1) : min(frame.shape[1], x2),
        ]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0
        return (tensor - self.mean) / self.std

    def get_embedding(
        self, frame: np.ndarray, box: Sequence[float]
    ) -> RegionEmbeddings:
        """Return a normalized CPU embedding for each useful body region.

        This is a real-time heuristic inspired by the Uniform Feature Separation
        concept described in the ACM 2024 PU-ReID study. It is intentionally not
        a reproduction of that paper's learned separation framework.

        The head crop reduces uniform bias, while the pants crop becomes the
        strongest cue for this camera because targets share shirts but wear
        different pants. The torso crop remains available at a low weight as a
        fallback. All valid crops are processed in one batched MPS forward pass
        to keep the richer representation practical for a hackathon pipeline.
        """
        empty_embeddings = RegionEmbeddings()
        if frame is None or frame.size == 0 or len(box) != 4:
            return empty_embeddings

        x1, y1, x2, y2 = (int(round(value)) for value in box)
        if x2 <= x1 or y2 <= y1:
            return empty_embeddings

        region_names: list[str] = []
        region_tensors: list[torch.Tensor] = []
        for region_name, (start_ratio, end_ratio) in self.REGION_RANGES.items():
            tensor = self._prepare_region(
                frame, x1, y1, x2, y2, start_ratio, end_ratio
            )
            if tensor is not None:
                region_names.append(region_name)
                region_tensors.append(tensor)

        if not region_tensors:
            return empty_embeddings

        batch = torch.stack(region_tensors).to(self.device)
        with torch.inference_mode():
            vectors = self.model(batch)
            vectors = F.normalize(vectors, p=2, dim=1).detach().cpu()

        embeddings = RegionEmbeddings()
        for region_name, vector in zip(region_names, vectors):
            setattr(embeddings, region_name, vector)
        return embeddings
