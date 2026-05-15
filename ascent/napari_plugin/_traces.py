"""Per-track activity-trace extraction.

Given a 4-D image and a Tracks-layer data array, return one 1-D intensity
trace per track ID, sampled at the track's coordinates with an optional
ROI.

Pure-Python: no Qt / napari needed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

ReductionName = str

_REDUCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda a: float(a.mean()) if a.size else float("nan"),
    "max": lambda a: float(a.max()) if a.size else float("nan"),
    "sum": lambda a: float(a.sum()) if a.size else float("nan"),
}


def extract_traces(
    image: np.ndarray,
    track_data: np.ndarray,
    *,
    roi_radius: int = 2,
    reduction: ReductionName = "mean",
) -> dict[int, dict[str, np.ndarray]]:
    """Return one ``{frames, intensity}`` trace per track.

    Parameters
    ----------
    image
        4-D ``(T, Z, Y, X)`` array. Multi-channel layers should be passed as
        a single channel slice.
    track_data
        ``(N, 5)`` array with columns ``(track_id, t, z, y, x)``.
    roi_radius
        Box radius in voxels around each centroid. ``0`` samples a single
        voxel; ``r`` samples a ``(2r+1)`` cube clipped to image bounds.
    reduction
        ``"mean"``, ``"max"``, or ``"sum"`` over the ROI per timepoint.

    Returns
    -------
    dict
        ``{track_id: {"frames": (T_k,), "intensity": (T_k,)}}`` sorted by
        track ID and frame.

    Notes
    -----
    The inner sampling loop is per-row Python, but bounds-clipping and
    coordinate rounding are vectorized once up front; for typical workloads
    (≤ 200 tracks × ≤ 5000 frames = 1M rows) this runs in a few seconds. If
    that becomes a bottleneck, replace the per-row ``image[t, z0:z1, …]``
    slice with a single ``np.add.at`` over a precomputed sample grid.
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}; expected one of {list(_REDUCTIONS)}"
        )
    if roi_radius < 0:
        raise ValueError(f"roi_radius must be ≥ 0; got {roi_radius}")
    if image.ndim != 4:
        raise ValueError(f"image must be 4-D (T, Z, Y, X); got shape {image.shape}")
    if track_data.shape[0] == 0:
        return {}
    if track_data.ndim != 2 or track_data.shape[1] != 5:
        raise ValueError(
            f"track_data must be (N, 5); got shape {track_data.shape}"
        )

    T, Z, Y, X = image.shape
    reduce_fn = _REDUCTIONS[reduction]

    # Vectorized prep: cast columns once, clip frame indices, compute ROI
    # bounds in bulk. The per-row slicing loop afterwards only does the
    # actual array reduction, which is what dominates cost.
    ids = track_data[:, 0].astype(np.int64)
    ts = track_data[:, 1].astype(np.int64)
    zs = np.rint(track_data[:, 2]).astype(np.int64)
    ys = np.rint(track_data[:, 3]).astype(np.int64)
    xs = np.rint(track_data[:, 4]).astype(np.int64)

    valid = (ids >= 0) & (ts >= 0) & (ts < T)
    if not valid.any():
        return {}

    z0 = np.clip(zs - roi_radius, 0, Z)
    z1 = np.clip(zs + roi_radius + 1, 0, Z)
    y0 = np.clip(ys - roi_radius, 0, Y)
    y1 = np.clip(ys + roi_radius + 1, 0, Y)
    x0 = np.clip(xs - roi_radius, 0, X)
    x1 = np.clip(xs + roi_radius + 1, 0, X)

    order = np.lexsort((ts, ids))
    buckets: dict[int, dict[str, list]] = defaultdict(lambda: {"frames": [], "intensity": []})
    for i in order:
        if not valid[i]:
            continue
        tid = int(ids[i])
        t = int(ts[i])
        roi = image[t, z0[i]:z1[i], y0[i]:y1[i], x0[i]:x1[i]]
        b = buckets[tid]
        b["frames"].append(t)
        b["intensity"].append(reduce_fn(roi))

    return {
        tid: {
            "frames": np.asarray(b["frames"], dtype=np.int64),
            "intensity": np.asarray(b["intensity"], dtype=np.float64),
        }
        for tid, b in sorted(buckets.items())
    }
