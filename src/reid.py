"""Custom MobileNetV2-based person re-identification embeddings."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class CustomReID:
    """Extract normalized person appearance embeddings on Apple Silicon MPS."""

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

    def get_embedding(
        self, frame: np.ndarray, box: Sequence[float]
    ) -> torch.Tensor | None:
        """Return a normalized CPU embedding for one cropped person."""
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

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0
        tensor = ((tensor - self.mean) / self.std).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embedding = self.model(tensor).flatten()
            embedding = F.normalize(embedding, p=2, dim=0)
        return embedding.detach().cpu()
