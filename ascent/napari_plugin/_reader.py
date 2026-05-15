"""napari readers for ASCENT's data formats.

Three formats are sniffed and dispatched from a single ``get_reader`` entry
point:

1. HDF5 movie with the layout ``t{frame}/c{channel}`` where each leaf dataset
   is a Z-Y-X volume — consumed by ``ascent run``.
2. Detection CSV with header ``object_id,t,z,y,x`` — produced by any 3-D
   segmentation tool (e.g. ``stardist-napari``, ``cellpose-napari``) and
   fed into ``ascent run``.
3. Tracks CSV with header ``TrackID,ObjectID,t,z,y,x`` — output of
   ``ascent run``.

Each reader returns the napari-standard ``(data, meta, layer_type)`` tuple list.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional, Sequence

import h5py
import numpy as np

LayerData = tuple[object, dict, str]
ReaderFunction = Callable[[str | Sequence[str]], list[LayerData]]


def get_reader(path: str | Sequence[str]) -> Optional[ReaderFunction]:
    """Return a reader callable if ``path`` is one ASCENT can open, else None.

    napari calls this function for every dropped file. We sniff the extension
    and (for CSVs and HDF5s) the contents to decide whether to claim the file.
    """
    if isinstance(path, (list, tuple)):
        if not path:
            return None
        # Only claim multi-file drops if every path is one of ours.
        if not all(_can_read(p) for p in path):
            return None
        return _read_many
    return _read_single if _can_read(path) else None


def _can_read(path: str) -> bool:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return _is_ascent_h5(p)
    if suffix == ".csv":
        return _classify_csv(p) is not None
    return False


def _read_many(paths: Sequence[str]) -> list[LayerData]:
    out: list[LayerData] = []
    for p in paths:
        out.extend(_read_single(p))
    return out


def _read_single(path: str) -> list[LayerData]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return [_read_ascent_h5(p)]
    if suffix == ".csv":
        kind = _classify_csv(p)
        if kind == "detections":
            return [_read_detections_csv(p)]
        if kind == "tracks":
            return [_read_tracks_csv(p)]
    raise ValueError(f"ASCENT reader cannot handle {path!r}")


# --------------------------------------------------------------------------- #
# HDF5
# --------------------------------------------------------------------------- #

_FRAME_PREFIX = "t"
_CHANNEL_PREFIX = "c"


def _is_ascent_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            frame_keys = _frame_keys(f)
            if not frame_keys:
                return False
            first = f[frame_keys[0]]
            return isinstance(first, h5py.Group) and any(
                k.startswith(_CHANNEL_PREFIX) and k[1:].isdigit() for k in first.keys()
            )
    except (OSError, KeyError):
        return False


def _frame_keys(f: h5py.File) -> list[str]:
    return sorted(
        (k for k in f.keys() if k.startswith(_FRAME_PREFIX) and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )


def _channel_keys(group: h5py.Group) -> list[str]:
    return sorted(
        (k for k in group.keys() if k.startswith(_CHANNEL_PREFIX) and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )


def _read_ascent_h5(path: Path) -> LayerData:
    """Lazily stack the ``t*/c*`` datasets into a (T, [C,] Z, Y, X) dask array.
    """
    import dask.array as da
    
    f = h5py.File(path, "r")
    frame_keys = _frame_keys(f)
    if not frame_keys:
        raise ValueError(f"No frame groups (t*) found in {path}")
    ch_keys = _channel_keys(f[frame_keys[0]])
    if not ch_keys:
        raise ValueError(f"No channel datasets (c*) in {frame_keys[0]} of {path}")

    if len(ch_keys) == 1:
        stack = da.stack([da.from_array(f[fk][ch_keys[0]]) for fk in frame_keys], axis=0)
        meta = {"name": path.stem, "metadata": {"source": str(path)}}
    else:
        shapes = {f[frame_keys[0]][ck].shape for ck in ch_keys}
        if len(shapes) > 1:
            raise ValueError(
                f"{path}: channel datasets in {frame_keys[0]} have mismatched "
                f"shapes {shapes}. ASCENT's reader requires all channels to "
                "share a common ZYX shape; load mismatched volumes as separate "
                "files instead."
            )
        stack = da.stack(
            [
                da.stack([da.from_array(f[fk][ck]) for ck in ch_keys], axis=0)
                for fk in frame_keys
            ],
            axis=0,
        )
        meta = {
            "name": [f"{path.stem} ch{ck[1:]}" for ck in ch_keys],
            "channel_axis": 1,
            "metadata": {"source": str(path), "channels": ch_keys},
        }

    return (stack, meta, "image")


# --------------------------------------------------------------------------- #
# CSVs
# --------------------------------------------------------------------------- #


def _loadtxt_skip_header(path: Path, *, expected_cols: int) -> np.ndarray:
    """Like ``np.loadtxt(skiprows=1)`` but returns an empty (0, expected_cols)
    array for header-only files instead of warning. Strips a UTF-8 BOM if present."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        next(f, None)  # drop header
        body = f.read()
    if not body.strip():
        return np.zeros((0, expected_cols), dtype=np.float64)
    return np.loadtxt(body.splitlines(), delimiter=",", dtype=np.float64, ndmin=2)


_DETECTION_HEADER = ("object_id", "t", "z", "y", "x")
_TRACKS_HEADER = ("trackid", "objectid", "t", "z", "y", "x")


def _classify_csv(path: Path) -> Optional[str]:
    """Return ``"detections"``, ``"tracks"``, or ``None`` based on header.

    Tolerant to:
    - UTF-8 BOM at the start of the file (Excel exports prepend it).
    - Case-insensitive column names (``TrackID`` vs ``trackid`` vs ``TRACKID``).
    """
    try:
        # ``utf-8-sig`` strips a BOM if present; otherwise behaves like utf-8.
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), None)
    except (OSError, StopIteration, UnicodeDecodeError):
        return None
    if not header:
        return None
    cols = tuple(c.strip().lower() for c in header)
    if cols[: len(_DETECTION_HEADER)] == _DETECTION_HEADER:
        return "detections"
    if cols[: len(_TRACKS_HEADER)] == _TRACKS_HEADER:
        return "tracks"
    return None


def _read_detections_csv(path: Path) -> LayerData:
    rows = _loadtxt_skip_header(path, expected_cols=5)
    if rows.shape[0] == 0:
        coords = np.zeros((0, 4), dtype=np.float64)
        object_ids = np.zeros((0,), dtype=np.int64)
    else:
        object_ids = rows[:, 0].astype(np.int64)
        coords = rows[:, 1:5]  # (t, z, y, x)
    meta = {
        "name": f"{path.stem} (detections)",
        "size": 4,
        "face_color": "transparent",
        "border_color": "yellow",
        "border_width": 0.15,
        "out_of_slice_display": False,
        "features": {"object_id": object_ids},
        "metadata": {"source": str(path), "ascent_layer_kind": "detections"},
    }
    return (coords, meta, "points")


def _read_tracks_csv(path: Path) -> LayerData:
    """Read TrackID,ObjectID,t,z,y,x → napari Tracks layer."""
    raw = _loadtxt_skip_header(path, expected_cols=6)
    if raw.shape[0] == 0:
        track_data = np.zeros((0, 5), dtype=np.float64)
        properties: dict[str, np.ndarray] = {"object_id": np.zeros((0,), dtype=np.int64)}
    else:
        # napari Tracks expects (track_id, t, [z,] y, x). Drop ObjectID into properties.
        track_data = np.column_stack(
            [raw[:, 0], raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]]
        )
        properties = {"object_id": raw[:, 1].astype(np.int64)}
    meta = {
        "name": f"{path.stem} (tracks)",
        "tail_length": 30,
        "head_length": 0,
        "properties": properties,
        "metadata": {"source": str(path), "ascent_layer_kind": "tracks"},
    }
    return (track_data, meta, "tracks")
