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
import dask.array as da
import dask

import re

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
        return _classify_csv(p)
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
        return _read_ascent_h5(p)
    if suffix == ".csv":
        return _read_tracks_csv(p)
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


# --- 1. Duck-Typed Class for hdf5 ---
class UnifiedHDF5Proxy:
    """
    An ND-agnostic virtual layout mapping a global coordinate system (T, C, ...spatial)
    directly to HDF5 paths while strictly adhering to NumPy slicing rules.
    """
    def __init__(self, filepath):
        self._file = h5py.File(filepath, 'r', rdcc_nbytes=536870912)
        
        # 1. Gather timepoint keys
        self.t_keys = sorted(
            [k for k in self._file.keys() if k.startswith('t') and k[1:].isdigit()],
            key=lambda k: int(k[1:])
        )
        if not self.t_keys:
            raise ValueError("No timepoint groups found in the HDF5 file.")
            
        # 2. Gather channel keys from the first frame
        first_t = self.t_keys[0]
        self.c_keys = sorted(
            [k for k in self._file[first_t].keys() if k.startswith('c') and k[1:].isdigit()],
            key=lambda k: int(k[1:])
        )
        if not self.c_keys:
            raise ValueError("No channel datasets found in the HDF5 file.")
            
        # 3. Read metadata template from one dataset
        template_ds = self._file[first_t][self.c_keys[0]]
        self.spatial_shape = template_ds.shape  # Can be (Y, X) or (Z, Y, X)
        self.dtype = template_ds.dtype
        self.chunks = template_ds.chunks or self.spatial_shape 
        
        # 4. Define dynamic global shape: (T, C, ...)
        self.shape = (len(self.t_keys), len(self.c_keys)) + self.spatial_shape
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        # Standardize key into a full self.ndim-tuple layout
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        
        t_dim, c_dim = key[0], key[1]
        spatial_slices = key[2:]  # Dynamically matches the spatial dims (2D or 3D)
        
        # Track whether the index type dictates dropping or keeping dimensions
        squeeze_t = isinstance(t_dim, int)
        squeeze_c = isinstance(c_dim, int)
        
        # Convert slices or integers into explicit lists of global coordinates
        t_indices = [t_dim] if squeeze_t else list(range(*t_dim.indices(self.shape[0])))
        c_indices = [c_dim] if squeeze_c else list(range(*c_dim.indices(self.shape[1])))
        
        # Build out the target array blocks sequentially
        t_blocks = []
        for t in t_indices:
            c_blocks = []
            for c in c_indices:
                data = self._file[self.t_keys[t]][self.c_keys[c]][spatial_slices]
                c_blocks.append(data)
            # Stack elements along the channel axis (axis 0 of this sub-block)
            t_blocks.append(np.stack(c_blocks, axis=0))
            
        # Stack elements along the time axis
        res = np.stack(t_blocks, axis=0)
        
        # Squeeze dimensions ONLY if they were requested as flat integers 
        if squeeze_c:
            res = np.squeeze(res, axis=1)
        if squeeze_t:
            res = np.squeeze(res, axis=0)
            
        return res


class LazyHDF5Volume:
    def __init__(self, filepath):
        proxy = UnifiedHDF5Proxy(filepath)
        
        # Chunking rules adapt automatically to 4D or 5D layouts
        dask_chunks = (1, 1) + proxy.chunks
        
        self.dask_array = da.from_array(
            proxy, 
            chunks=dask_chunks, 
        )
        
    @property
    def shape(self): return self.dask_array.shape
    @property
    def dtype(self): return self.dask_array.dtype
    @property
    def ndim(self): return self.dask_array.ndim
    def __getitem__(self, key): return self.dask_array[key]
    
# --- 2. The Worker Function --- #
def _read_ascent_h5(path):
    """
    Reads the filepath and returns a list of napari LayerData tuples.
    """
    print(f"Loading {path} via Custom Dask Reader...")
    
    # Instantiate your lazy volume
    lazy_volume = LazyHDF5Volume(path)
    
    # Define how napari should display this layer
    add_kwargs = {
        "name": "HDF5 Volume",
        "multiscale": False,
        "channel_axis": 1 if lazy_volume.ndim == 5 else None, 
        "cache": True
    }
    
    # Return exactly one LayerData tuple inside a list
    # Format: (data, meta_dict, layer_type)
    print(lazy_volume.shape)
    return [(lazy_volume, add_kwargs, 'image')]


# --------------------------------------------------------------------------- #
# CSVs
# --------------------------------------------------------------------------- #

_MINIMUM_HEADER = {"t", "y", "x"}

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
            print(f"header: {header}")
    except (OSError, StopIteration, UnicodeDecodeError):
        return False
    if not header:
        return False
    cols = {c.strip().lower() for c in header}
    if _MINIMUM_HEADER.issubset(cols):
        return True
    return False

def _read_tracks_csv(path):
    """
    Reader for CSV annotations. 
    Standardizes various column naming conventions into a unified Points layer.
    """
    import pandas as pd
    path = Path(path)
    
    # 1. Load the CSV (logic from get_annotation_csv)
    df = pd.read_csv(path, header=0)
    
    # Standardize column names
    name_map = {
        "ObjectID": "object_id",
        "TrackID": "track_id",
        "t_idx": "t"
    }
    df.rename(columns=name_map, inplace=True)
    
    # 2. Ensure essential columns exist
    # If no track/worldline ID exists, treat every point as its own object
    if "object_id" not in df.columns:
        df["object_id"] = np.arange(len(df))
    # makeTracks = True
    # if "track_id" not in df.columns:
    #     df["track_id"] = df["object_id"]
    #     makeTracks = False

    # Add provenance if missing (logic from get_annotation_csv)
    # if "provenance" not in df.columns:
    #     df["provenance"] = "csv"

    # 3. Extract coordinates and features (logic from getAscentPointsData)
    # Expected order: Time, Z, Y, X
    try:
        coords = df[["t", "z", "y", "x"]].to_numpy().astype(float)
    except KeyError:
        # Fallback for 2D data if 'z' is missing in some CSVs
        coords = df[["t", "y", "x"]].to_numpy().astype(float)

    features = {key:df[key].to_numpy() for key in df.keys() if key not in ["t","x","y","z"]}
    #     "track_id": df["track_id"].to_numpy(),
    #     "object_id": df["object_id"].to_numpy(),
    #     "provenance": df["provenance"].to_numpy(),
    # }

    # 4. Define Layer Metadata
    add_kwargs = {
        "name": path.stem,
        "features": features,
        # "face_color": "track_id", # Color points by track_id
        # "face_colormap": 'turbo',
        "size": 5,
        # "text": "track_id",
        "metadata": {"source": path,
                     "reader": "ascent"},
    }
    # if makeTracks:
    #     print([(coords, add_kwargs, "points"),make_tracks(coords,add_kwargs)])
    #     return [(coords, add_kwargs, "points"),make_tracks(coords,add_kwargs)]
    import napari
    try:
        viewer = napari.current_viewer()
        if viewer is not None: 
            from ._manager import get_or_create_manager
            get_or_create_manager(viewer)
    except Exception as e:
        print(f"Warning: Coult not auto-initialize LayerManager from reader: {e}")

    return [(coords, add_kwargs, "points")]

def make_tracks(coords,args) -> tuple[np.ndarray,dict,str]:
    data = np.concat([args["features"]['track_id'][:,None],coords],axis=1)
    add_kwargs = {
        "name": args["name"] + "_tracks",
        "features": args["features"],
        "color_by": "track_id",
        "colormap": "fire"
    }
    return (data,add_kwargs,"tracks")