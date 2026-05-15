"""Programmatic inference adapter for the ASCENT napari plugin.

Bridges in-memory napari layer arrays (Image + Points) to the file-based
``ascent.tools.run_ascent.run`` pipeline. The bridge is intentionally
file-based for M2 — the existing dataset class consumes paths, and writing
small temp files is far less invasive than refactoring the dataset to take
arrays. M4 may revisit this if the temp-file overhead matters in practice.

Pure-Python: no napari/Qt dependency. Safe to import without ``[gui]``.
"""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# Defaults match examples/configs/track_template.py at the time of writing.
# Anything the user does not override is filled in here.
_DEFAULTS: dict[str, Any] = {
    # dataset
    "dataset_image_channel": 0,
    "dataset_axis_order": "ZYX",
    "dataset_spacing": [1.0, 1.0, 1.0],
    "dataset_normalize": "percentile",
    "dataset_norm_p_low": 1.0,
    "dataset_norm_p_high": 99.99,
    "dataset_start_frame": None,
    "dataset_end_frame": None,
    # model
    "model_lf_patch_size_xy": 64,
    "model_lf_patch_size_z": 3,
    "model_lf_pretrained": "imagenet_channelvit_small_p16_DINO",
    "model_pe_num_mlp_layers": 3,
    "model_pe_norm": "batch",
    "model_pe_scaling": "relative",
    "model_tr_d_model": 256,
    "model_tr_nhead": 4,
    "model_tr_num_encoder_layers": 4,
    "model_tr_dim_feedforward": 512,
    "model_tr_dropout": 0.1,
    "model_tr_activation": "relu",
    "model_use_local_features": True,
    "model_use_positional_encoding": True,
    "model_use_transformer": True,
    # tracking
    "tracking_momentum": 0.5,
    "tracking_temperature": 0.05,
    "tracking_max_gap_frames": 9999,
    "tracking_w_within": 0,
    # runtime — output_* and device are set per-call below
    "runtime_batch_size_frame": 16,
    "runtime_loglevel": "INFO",
}


def build_config(model_ckpt: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> dict:
    """Merge plugin defaults with user overrides into a full run-config dict.

    The returned dict is missing only the per-call dataset paths and runtime
    output settings, which ``run_from_arrays`` fills in just before invocation.
    """
    cfg = dict(_DEFAULTS)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["model_ckpt"] = str(model_ckpt)
    return cfg


def write_image_h5(image: np.ndarray, path: Path, *, channel: int = 0) -> None:
    """Write a (T, Z, Y, X) array as ASCENT-format HDF5 (``t{n}/c{channel}``).

    Preserves the input dtype so downstream percentile normalization sees the
    same value range it would on disk.
    """
    if image.ndim != 4:
        raise ValueError(
            f"Image must be 4-D (T, Z, Y, X); got shape {image.shape}"
        )
    with h5py.File(path, "w") as f:
        for t in range(image.shape[0]):
            grp = f.create_group(f"t{t}")
            # Use np.asarray to ensure dask arrays are evaluated frame-by-frame
            grp.create_dataset(f"c{channel}", data=np.asarray(image[t]))


def write_detections_csv(
    coords: np.ndarray,
    path: Path,
    *,
    object_ids: Optional[np.ndarray] = None,
) -> None:
    """Write a (N, 4) array of (t, z, y, x) coords as ASCENT detections CSV.

    Auto-assigns sequential ``object_id`` if not provided. Frame indices are
    rounded to int (napari stores them as float).
    """
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(
            f"Points coords must be (N, 4) for (t, z, y, x); got shape {coords.shape}"
        )
    n = coords.shape[0]
    if object_ids is None:
        object_ids = np.arange(n, dtype=np.int64)
    elif object_ids.shape != (n,):
        raise ValueError(
            f"object_ids must have length {n}; got {object_ids.shape}"
        )
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "t", "z", "y", "x"])
        for i in range(n):
            t = int(round(coords[i, 0]))
            z, y, x = float(coords[i, 1]), float(coords[i, 2]), float(coords[i, 3])
            w.writerow([int(object_ids[i]), t, z, y, x])


def read_tracks_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a tracks CSV produced by ASCENT into napari's Tracks layer format.

    Returns ``(track_data, object_ids)`` where ``track_data`` has shape
    ``(N, 5)`` with columns ``(track_id, t, z, y, x)`` and ``object_ids`` has
    shape ``(N,)``.
    """
    with open(path, newline="") as f:
        next(f, None)  # drop header
        body = f.read()
    if not body.strip():
        return np.zeros((0, 5), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    raw = np.loadtxt(body.splitlines(), delimiter=",", dtype=np.float64, ndmin=2)
    track_data = np.column_stack(
        [raw[:, 0], raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]]
    )
    object_ids = raw[:, 1].astype(np.int64)
    return track_data, object_ids


def run_from_arrays(
    *,
    image: Optional[np.ndarray] = None,
    image_path: Optional[str | Path] = None,
    image_channel: int = 0,
    points: np.ndarray,
    model_ckpt: str | Path,
    object_ids: Optional[np.ndarray] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[Path] = None,
    output_prefix: str = "napari_run",
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end inference from napari layer arrays.

    When ``output_dir`` is provided and the run fails partway through, the
    intermediate ``{prefix}_pred_z.pt`` / ``{prefix}_pred_object_ids.pt``
    files are *not* cleaned up — they remain in the user's directory. With
    ``output_dir=None`` (the default), everything lives in a temp directory
    deleted on return regardless of success.

    Parameters
    ----------
    image
        Optional 4-D (T, Z, Y, X) array from a single-channel napari Image layer.
    image_path
        Optional path to an existing ASCENT HDF5 to avoid re-writing.
    image_channel
        Channel index if image_path is provided.
    points
        (N, 4) array of (t, z, y, x) coords from a Points layer.
    model_ckpt
        Path to a trained NETr ``.pth`` checkpoint.
    object_ids
        Optional per-row object IDs. Auto-assigned 0..N-1 if omitted.
    overrides
        Mapping of config keys (see ``_DEFAULTS``) to override values.
    output_dir
        Directory for the inference artefacts. If ``None``, a temp dir is
        used and the artefacts are deleted on return.
    output_prefix
        Filename prefix used for ``{prefix}_tracks.csv`` etc.

    Returns
    -------
    track_data, object_ids
        Same shapes as :func:`read_tracks_csv`.
    """
    # late import so this module stays importable without torch installed
    from ascent.tools.run_ascent import run

    cleanup = output_dir is None
    workdir_ctx = tempfile.TemporaryDirectory() if cleanup else None
    workdir = Path(workdir_ctx.name) if cleanup else Path(output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        coord_path = workdir / "input_dets.csv"
        write_detections_csv(points, coord_path, object_ids=object_ids)
        
        if image_path is not None:
            final_image_path = Path(image_path)
        else:
            final_image_path = workdir / "input.h5"
            write_image_h5(image, final_image_path, channel=0)

        cfg = build_config(model_ckpt, overrides)
        cfg["dataset_file_image"] = str(final_image_path)
        cfg["dataset_image_channel"] = image_channel
        cfg["dataset_file_coord"] = str(coord_path)
        cfg["runtime_output_dir"] = str(workdir)
        cfg["runtime_output_prefix"] = output_prefix
        cfg.setdefault("runtime_device", "cpu")

        if progress_callback:
            progress_callback("serialization_done", 0, 0)

        tracks_csv = run(cfg, configure_logging=False, progress_callback=progress_callback)
        return read_tracks_csv(tracks_csv)
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()


def resolve_device(name: str) -> str:
    """Translate ``"auto"`` into a concrete device string (cuda → mps → cpu)."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
