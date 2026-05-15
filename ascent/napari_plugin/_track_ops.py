"""Track-level edit operations with an undo/redo stack.

Operates on the napari Tracks-layer data model: an ``(N, 5)`` array with
columns ``(track_id, t, z, y, x)`` plus a parallel ``object_ids`` array
carried as a layer property.

Pure-Python so it can be unit-tested without Qt or napari. The widget in
``_widgets/_correction.py`` is a thin shell over this class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _Snapshot:
    """A single state of the editor — what we push onto the undo stack."""

    track_data: np.ndarray
    object_ids: np.ndarray
    label: str = ""


@dataclass
class TrackEditor:
    """Mutable container for a Tracks-layer's data with undo/redo."""

    track_data: np.ndarray
    object_ids: np.ndarray
    _undo: list[_Snapshot] = field(default_factory=list, init=False, repr=False)
    _redo: list[_Snapshot] = field(default_factory=list, init=False, repr=False)
    max_undo: int = 50

    def __post_init__(self) -> None:
        self._validate_shapes()
        self._sort()

    # --------------------------------------------------------------------- #
    # state inspection
    # --------------------------------------------------------------------- #

    def track_ids(self) -> np.ndarray:
        """Sorted unique track IDs currently in the editor."""
        if self.track_data.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(self.track_data[:, 0].astype(np.int64))

    def has_track(self, track_id: int) -> bool:
        return bool(np.any(self.track_data[:, 0].astype(np.int64) == int(track_id)))

    def frames_of(self, track_id: int) -> np.ndarray:
        mask = self.track_data[:, 0].astype(np.int64) == int(track_id)
        return self.track_data[mask, 1].astype(np.int64)

    # --------------------------------------------------------------------- #
    # mutations
    # --------------------------------------------------------------------- #

    def split(self, track_id: int, frame_t: int) -> int:
        """Split ``track_id`` so rows with ``t >= frame_t`` become a new track.

        Returns the new track ID assigned to the tail. Raises ``ValueError``
        if the split would create an empty new track.
        """
        track_id = int(track_id)
        frame_t = int(frame_t)
        ids = self.track_data[:, 0].astype(np.int64)
        ts = self.track_data[:, 1].astype(np.int64)
        mask_track = ids == track_id
        if not mask_track.any():
            raise KeyError(f"track {track_id} not present")

        tail_mask = mask_track & (ts >= frame_t)
        head_mask = mask_track & (ts < frame_t)
        if not tail_mask.any():
            raise ValueError(
                f"split({track_id}, t={frame_t}) leaves no rows in the new track"
            )
        if not head_mask.any():
            raise ValueError(
                f"split({track_id}, t={frame_t}) leaves no rows in the original track"
            )

        new_id = int(self._next_free_id())
        self._push_undo(f"split {track_id}@t={frame_t}")
        new_data = self.track_data.copy()
        new_data[tail_mask, 0] = new_id
        self.track_data = new_data
        self._sort()
        return new_id

    def merge(self, into: int, source: int) -> None:
        """Merge ``source`` into ``into``. Track IDs must not overlap in time."""
        into, source = int(into), int(source)
        if into == source:
            raise ValueError("cannot merge a track into itself")
        ids = self.track_data[:, 0].astype(np.int64)
        ts_into = self.track_data[ids == into, 1].astype(np.int64)
        ts_source = self.track_data[ids == source, 1].astype(np.int64)
        if ts_into.size == 0 or ts_source.size == 0:
            raise KeyError(f"missing track(s) for merge: into={into}, source={source}")
        overlap = np.intersect1d(ts_into, ts_source)
        if overlap.size > 0:
            raise ValueError(
                f"merge({into}, {source}) failed — overlapping frames {list(overlap)}"
            )
        self._push_undo(f"merge {source} → {into}")
        new_data = self.track_data.copy()
        new_data[ids == source, 0] = into
        self.track_data = new_data
        self._sort()

    def relabel(self, old_id: int, new_id: int) -> None:
        """Change a track's ID. ``new_id`` must not already exist."""
        old_id, new_id = int(old_id), int(new_id)
        if old_id == new_id:
            return
        if not self.has_track(old_id):
            raise KeyError(f"track {old_id} not present")
        if self.has_track(new_id):
            raise ValueError(
                f"relabel({old_id}, {new_id}) failed — {new_id} exists; use merge instead"
            )
        self._push_undo(f"relabel {old_id} → {new_id}")
        ids = self.track_data[:, 0].astype(np.int64)
        new_data = self.track_data.copy()
        new_data[ids == old_id, 0] = new_id
        self.track_data = new_data
        self._sort()

    def delete(self, track_id: int) -> None:
        track_id = int(track_id)
        if not self.has_track(track_id):
            raise KeyError(f"track {track_id} not present")
        self._push_undo(f"delete {track_id}")
        ids = self.track_data[:, 0].astype(np.int64)
        keep = ids != track_id
        self.track_data = self.track_data[keep]
        self.object_ids = self.object_ids[keep]
        # no sort needed — order preserved

    def remove_point(self, track_id: int, frame_t: int) -> None:
        track_id, frame_t = int(track_id), int(frame_t)
        ids = self.track_data[:, 0].astype(np.int64)
        ts = self.track_data[:, 1].astype(np.int64)
        match = (ids == track_id) & (ts == frame_t)
        if not match.any():
            raise KeyError(f"no point at track={track_id}, t={frame_t}")
        if int(match.sum()) > 1:
            raise ValueError(
                f"multiple points at track={track_id}, t={frame_t}; data corrupt"
            )
        self._push_undo(f"remove pt track={track_id} t={frame_t}")
        keep = ~match
        self.track_data = self.track_data[keep]
        self.object_ids = self.object_ids[keep]

    # --------------------------------------------------------------------- #
    # undo / redo
    # --------------------------------------------------------------------- #

    def undo(self) -> Optional[str]:
        if not self._undo:
            return None
        snap = self._undo.pop()
        self._redo.append(self._snapshot("redo"))
        # Copy on restore so a follow-up in-place op (e.g. delete) can't alias
        # back into the redo stack's snapshot buffer.
        self.track_data = snap.track_data.copy()
        self.object_ids = snap.object_ids.copy()
        return snap.label

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(self._snapshot("undo"))
        self.track_data = snap.track_data.copy()
        self.object_ids = snap.object_ids.copy()
        return snap.label

    # --------------------------------------------------------------------- #
    # internals
    # --------------------------------------------------------------------- #

    def _validate_shapes(self) -> None:
        if self.track_data.ndim != 2 or self.track_data.shape[1] != 5:
            raise ValueError(
                f"track_data must be (N, 5); got {self.track_data.shape}"
            )
        if self.object_ids.shape != (self.track_data.shape[0],):
            raise ValueError(
                "object_ids must have length N matching track_data; "
                f"got {self.object_ids.shape} vs N={self.track_data.shape[0]}"
            )

    def _sort(self) -> None:
        """Stable sort by (track_id, t) so napari renders edges correctly."""
        if self.track_data.shape[0] == 0:
            return
        order = np.lexsort((self.track_data[:, 1], self.track_data[:, 0]))
        self.track_data = self.track_data[order]
        self.object_ids = self.object_ids[order]

    def _next_free_id(self) -> int:
        ids = self.track_ids()
        if ids.size == 0:
            return 0
        return int(ids.max()) + 1

    def _snapshot(self, label: str) -> _Snapshot:
        return _Snapshot(
            track_data=self.track_data.copy(),
            object_ids=self.object_ids.copy(),
            label=label,
        )

    def _push_undo(self, label: str) -> None:
        self._undo.append(self._snapshot(label))
        if len(self._undo) > self.max_undo:
            self._undo.pop(0)
        self._redo.clear()
