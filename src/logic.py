"""Sliding-window stationary analysis with multi-region appearance Re-ID."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.reid import RegionEmbeddings


@dataclass
class TrackState:
    """Mutable stationary-analysis state associated with one visible track."""

    track_id: int
    history: deque[np.ndarray]
    embeddings: RegionEmbeddings = field(default_factory=RegionEmbeddings)
    torso_color: str = "Unknown"
    pants_color: str = "Unknown"
    current_frames: int = 0
    max_frames: int = 0
    is_stationary: bool = False
    lost_frames: int = 0
    last_centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )


class StationaryTracker:
    """Measure stationary duration and recover IDs after tracker switches."""

    REGION_WEIGHTS = {
        "head": 0.25,
        "pants": 0.60,
        "torso": 0.15,
    }
    PANTS_COLOR_MISMATCH_PENALTY = 0.15

    def __init__(
        self,
        fps: float,
        history_seconds: float,
        pixel_threshold: float,
        similarity_threshold: float = 0.70,
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
        current_embeddings: RegionEmbeddings,
        torso_color: str,
        pants_color: str,
    ) -> TrackState:
        if track_id in self.orphaned_tracks:
            state = self.orphaned_tracks.pop(track_id)
            state.lost_frames = 0
            self.tracks[track_id] = state
            return state

        best_orphan_id = self._attempt_reid_recovery(
            current_embeddings, centroid, pants_color
        )
        if best_orphan_id is None:
            best_orphan_id = self._find_color_fallback(
                centroid, pants_color, torso_color
            )

        if best_orphan_id is None:
            state = self._new_state(track_id)
        else:
            state = self.orphaned_tracks.pop(best_orphan_id)
            state.track_id = track_id
            state.lost_frames = 0

        self.tracks[track_id] = state
        return state

    @staticmethod
    def _cosine_similarity(
        current_embedding: torch.Tensor | None,
        old_embedding: torch.Tensor | None,
    ) -> float | None:
        """Return cosine similarity when both region embeddings are available."""
        if current_embedding is None or old_embedding is None:
            return None
        return float(
            F.cosine_similarity(
                current_embedding.unsqueeze(0),
                old_embedding.unsqueeze(0),
            ).item()
        )

    def _multi_region_similarity(
        self,
        current_embeddings: RegionEmbeddings,
        old_embeddings: RegionEmbeddings,
    ) -> float | None:
        """Combine valid region scores, prioritizing pants over shared shirts."""
        weighted_score = 0.0
        available_weight = 0.0
        for region_name, weight in self.REGION_WEIGHTS.items():
            similarity = self._cosine_similarity(
                getattr(current_embeddings, region_name),
                getattr(old_embeddings, region_name),
            )
            if similarity is None:
                continue
            weighted_score += weight * similarity
            available_weight += weight

        if available_weight == 0.0:
            return None
        return weighted_score / available_weight

    def _attempt_reid_recovery(
        self,
        current_embeddings: RegionEmbeddings,
        current_centroid: np.ndarray,
        pants_color: str,
    ) -> int | None:
        """Recover the best nearby orphan using spatial-temporal constraints.

        Pants embeddings carry the most weight because shirts are identical in
        this scene. Head embeddings help break ties, while torso embeddings are
        deliberately weak. The distance gate rejects physically implausible
        teleports and the continuous penalty favors realistic nearby matches.
        """
        best_orphan_id: int | None = None
        highest_score = -1.0
        for orphan_id, orphan in self.orphaned_tracks.items():
            similarity = self._multi_region_similarity(
                current_embeddings, orphan.embeddings
            )
            if similarity is None:
                continue

            distance = float(
                np.linalg.norm(current_centroid - orphan.last_centroid)
            )
            if distance > 200.0:
                continue

            distance_penalty = distance / 1000.0
            color_penalty = 0.0
            if (
                pants_color != "Unknown"
                and orphan.pants_color != "Unknown"
                and pants_color != orphan.pants_color
            ):
                color_penalty = self.PANTS_COLOR_MISMATCH_PENALTY

            final_score = similarity - distance_penalty - color_penalty
            if final_score > highest_score:
                best_orphan_id = orphan_id
                highest_score = final_score

        if highest_score >= self.similarity_threshold:
            return best_orphan_id
        return None

    def _find_color_fallback(
        self,
        centroid: np.ndarray,
        pants_color: str,
        torso_color: str,
    ) -> int | None:
        """Use pants color first, then torso color, when embedding matching fails."""
        best_orphan_id = self._find_nearby_color_match(
            centroid, pants_color, color_attribute="pants_color"
        )
        if best_orphan_id is not None:
            return best_orphan_id
        return self._find_nearby_color_match(
            centroid, torso_color, color_attribute="torso_color"
        )

    def _find_nearby_color_match(
        self,
        centroid: np.ndarray,
        color: str,
        color_attribute: str,
    ) -> int | None:
        """Return the closest orphan with the same known regional color."""
        if color == "Unknown":
            return None

        best_orphan_id: int | None = None
        best_distance = float("inf")
        for orphan_id, orphan in self.orphaned_tracks.items():
            if getattr(orphan, color_attribute) != color:
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

    def _update_embeddings(
        self,
        old_embeddings: RegionEmbeddings,
        current_embeddings: RegionEmbeddings,
    ) -> RegionEmbeddings:
        """Apply EMA independently to every available regional embedding."""
        return RegionEmbeddings(
            head=self._update_embedding(old_embeddings.head, current_embeddings.head),
            pants=self._update_embedding(
                old_embeddings.pants, current_embeddings.pants
            ),
            torso=self._update_embedding(
                old_embeddings.torso, current_embeddings.torso
            ),
        )

    def update_track(
        self,
        track_id: int,
        centroid: Sequence[float],
        current_embeddings: RegionEmbeddings,
        torso_color: str,
        pants_color: str,
    ) -> tuple[bool, float]:
        """Update one track and return stationary status and maximum seconds."""
        centroid_array = np.asarray(centroid, dtype=np.float64)
        if centroid_array.shape != (2,):
            raise ValueError("centroid must contain exactly two coordinates.")

        state = self.tracks.get(track_id)
        if state is None:
            state = self._recover_or_create(
                track_id,
                centroid_array,
                current_embeddings,
                torso_color,
                pants_color,
            )

        state.last_centroid = centroid_array
        state.embeddings = self._update_embeddings(
            state.embeddings, current_embeddings
        )
        if torso_color != "Unknown":
            state.torso_color = torso_color
        if pants_color != "Unknown":
            state.pants_color = pants_color
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
