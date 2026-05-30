"""Sliding-window stationary analysis with custom appearance-based Re-ID."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrackState:
    """Mutable stationary-analysis state associated with one visible track."""

    track_id: int
    history: deque[np.ndarray]
    embedding: torch.Tensor | None = None
    color: str = "Unknown"
    current_frames: int = 0
    max_frames: int = 0
    is_stationary: bool = False
    lost_frames: int = 0
    last_centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )


class StationaryTracker:
    """Measure stationary duration and recover IDs after tracker switches."""

    def __init__(
        self,
        fps: float,
        history_seconds: float,
        pixel_threshold: float,
        similarity_threshold: float = 0.83,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if history_seconds <= 0:
            raise ValueError("history_seconds must be positive.")
        if pixel_threshold <= 0:
            raise ValueError("pixel_threshold must be positive.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")

        self.fps = float(fps)
        self.history_size = max(1, int(round(fps * history_seconds)))
        self.pixel_threshold = float(pixel_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.recovery_distance = self.pixel_threshold * 2.0
        self.max_orphan_frames = max(1, int(round(self.fps * 3.0)))
        self.tracks: dict[int, TrackState] = {}
        self.orphaned_tracks: dict[int, TrackState] = {}
        self.completed_tracks: list[TrackState] = []

    def _new_state(self, track_id: int) -> TrackState:
        return TrackState(track_id=track_id, history=deque(maxlen=self.history_size))

    def _recover_or_create(
        self,
        track_id: int,
        centroid: np.ndarray,
        current_embedding: torch.Tensor | None,
        color: str,
    ) -> TrackState:
        if track_id in self.orphaned_tracks:
            state = self.orphaned_tracks.pop(track_id)
            state.lost_frames = 0
            self.tracks[track_id] = state
            return state

        best_orphan_id: int | None = None
        best_similarity = -1.0
        if current_embedding is not None:
            for orphan_id, orphan in self.orphaned_tracks.items():
                if orphan.embedding is None:
                    continue
                similarity = float(
                    F.cosine_similarity(
                        current_embedding.unsqueeze(0),
                        orphan.embedding.unsqueeze(0),
                    ).item()
                )
                if similarity > best_similarity:
                    best_orphan_id = orphan_id
                    best_similarity = similarity

        if best_similarity <= self.similarity_threshold:
            best_orphan_id = self._find_color_fallback(centroid, color)

        if best_orphan_id is None:
            state = self._new_state(track_id)
        else:
            state = self.orphaned_tracks.pop(best_orphan_id)
            state.track_id = track_id
            state.lost_frames = 0

        self.tracks[track_id] = state
        return state

    def _find_color_fallback(
        self, centroid: np.ndarray, color: str
    ) -> int | None:
        """Use torso color and proximity only when embedding matching fails."""
        if color == "Unknown":
            return None

        best_orphan_id: int | None = None
        best_distance = float("inf")
        for orphan_id, orphan in self.orphaned_tracks.items():
            if orphan.color != color:
                continue
            distance = float(np.linalg.norm(centroid - orphan.last_centroid))
            if distance < self.recovery_distance and distance < best_distance:
                best_orphan_id = orphan_id
                best_distance = distance
        return best_orphan_id

    @staticmethod
    def _update_embedding(
        old_embedding: torch.Tensor | None,
        current_embedding: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if current_embedding is None:
            return old_embedding
        if old_embedding is None:
            return F.normalize(current_embedding, p=2, dim=0)
        return F.normalize(0.9 * old_embedding + 0.1 * current_embedding, p=2, dim=0)

    def update_track(
        self,
        track_id: int,
        centroid: Sequence[float],
        current_embedding: torch.Tensor | None,
        color: str,
    ) -> tuple[bool, float]:
        """Update one track and return stationary status and maximum seconds."""
        centroid_array = np.asarray(centroid, dtype=np.float64)
        if centroid_array.shape != (2,):
            raise ValueError("centroid must contain exactly two coordinates.")

        state = self.tracks.get(track_id)
        if state is None:
            state = self._recover_or_create(
                track_id, centroid_array, current_embedding, color
            )

        state.last_centroid = centroid_array
        state.embedding = self._update_embedding(state.embedding, current_embedding)
        if color != "Unknown":
            state.color = color
        state.history.append(centroid_array)

        if len(state.history) == self.history_size:
            history_mean = np.mean(np.asarray(state.history), axis=0)
            distance = float(np.linalg.norm(centroid_array - history_mean))
            if distance < self.pixel_threshold:
                state.is_stationary = True
                state.current_frames += 1
                state.max_frames = max(state.max_frames, state.current_frames)
            else:
                state.is_stationary = False
                state.current_frames = 0
        else:
            state.is_stationary = False

        return state.is_stationary, state.max_frames / self.fps

    def mark_lost_ids(self, active_ids: set[int]) -> None:
        """Move tracks missing from the analyzed area into the recovery pool."""
        expired_ids: list[int] = []
        for track_id, state in self.orphaned_tracks.items():
            state.lost_frames += 1
            if state.lost_frames > self.max_orphan_frames:
                expired_ids.append(track_id)
        for track_id in expired_ids:
            self.completed_tracks.append(self.orphaned_tracks.pop(track_id))

        lost_ids = set(self.tracks) - set(active_ids)
        for track_id in lost_ids:
            state = self.tracks.pop(track_id)
            state.lost_frames = 0
            self.orphaned_tracks[track_id] = state

    def get_longest_stay(self) -> tuple[int | None, float]:
        """Return the currently known ID and longest stationary duration."""
        all_states = (
            list(self.tracks.values())
            + list(self.orphaned_tracks.values())
            + self.completed_tracks
        )
        if not all_states:
            return None, 0.0

        longest_state = max(all_states, key=lambda state: state.max_frames)
        return longest_state.track_id, longest_state.max_frames / self.fps
