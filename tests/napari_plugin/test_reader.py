"""Reader-layer tests for the ASCENT napari plugin.

These tests are pure-Python and do not require a Qt event loop, so they run
anywhere ``ascent`` is installed (no ``[gui]`` extras needed).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ascent.napari_plugin._reader import get_reader


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def ascent_h5(tmp_path: Path) -> Path:
    """A tiny 3-frame, single-channel HDF5 movie in the ASCENT layout."""
    p = tmp_path / "tiny.h5"
    rng = np.random.default_rng(0)
    with h5py.File(p, "w") as f:
        for t in range(3):
            g = f.create_group(f"t{t}")
            g.create_dataset("c0", data=rng.integers(0, 255, (4, 8, 8), dtype=np.uint8))
    return p


@pytest.fixture
def ascent_h5_2ch(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_2ch.h5"
    rng = np.random.default_rng(1)
    with h5py.File(p, "w") as f:
        for t in range(2):
            g = f.create_group(f"t{t}")
            g.create_dataset("c0", data=rng.integers(0, 255, (4, 8, 8), dtype=np.uint8))
            g.create_dataset("c1", data=rng.integers(0, 255, (4, 8, 8), dtype=np.uint8))
    return p


@pytest.fixture
def detections_csv(tmp_path: Path) -> Path:
    p = tmp_path / "dets.csv"
    p.write_text(
        "object_id,t,z,y,x\n"
        "0,0,1.5,2.5,3.5\n"
        "1,0,2.0,4.0,6.0\n"
        "2,1,1.5,2.5,3.5\n"
    )
    return p


@pytest.fixture
def tracks_csv(tmp_path: Path) -> Path:
    p = tmp_path / "tracks.csv"
    p.write_text(
        "TrackID,ObjectID,t,z,y,x\n"
        "0,0,0,1.5,2.5,3.5\n"
        "0,2,1,1.6,2.4,3.5\n"
        "1,1,0,2.0,4.0,6.0\n"
    )
    return p


@pytest.fixture
def unrelated_csv(tmp_path: Path) -> Path:
    p = tmp_path / "stuff.csv"
    p.write_text("a,b,c\n1,2,3\n")
    return p


# --------------------------------------------------------------------------- #
# get_reader dispatch
# --------------------------------------------------------------------------- #


def test_get_reader_claims_ascent_h5(ascent_h5):
    assert get_reader(str(ascent_h5)) is not None


def test_get_reader_rejects_unrelated_h5(tmp_path):
    p = tmp_path / "random.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("foo", data=np.zeros(3))
    assert get_reader(str(p)) is None


def test_get_reader_claims_detections_csv(detections_csv):
    assert get_reader(str(detections_csv)) is not None


def test_get_reader_claims_tracks_csv(tracks_csv):
    assert get_reader(str(tracks_csv)) is not None


def test_get_reader_rejects_unrelated_csv(unrelated_csv):
    assert get_reader(str(unrelated_csv)) is None


def test_get_reader_handles_missing_path(tmp_path):
    assert get_reader(str(tmp_path / "does_not_exist.h5")) is None


def test_get_reader_multi_file_all_ours(ascent_h5, detections_csv):
    assert get_reader([str(ascent_h5), str(detections_csv)]) is not None


def test_get_reader_multi_file_mixed_rejects(ascent_h5, unrelated_csv):
    assert get_reader([str(ascent_h5), str(unrelated_csv)]) is None


# --------------------------------------------------------------------------- #
# reader output shape
# --------------------------------------------------------------------------- #


def test_h5_single_channel_returns_4d_image(ascent_h5):
    reader = get_reader(str(ascent_h5))
    layers = reader(str(ascent_h5))
    assert len(layers) == 1
    data, meta, ltype = layers[0]
    assert ltype == "image"
    assert data.shape == (3, 4, 8, 8)  # T Z Y X
    assert meta["name"] == "tiny"
    assert "channel_axis" not in meta


def test_h5_multi_channel_sets_channel_axis(ascent_h5_2ch):
    reader = get_reader(str(ascent_h5_2ch))
    (data, meta, ltype), = reader(str(ascent_h5_2ch))
    assert ltype == "image"
    assert data.shape == (2, 2, 4, 8, 8)  # T C Z Y X
    assert meta["channel_axis"] == 1


def test_detections_csv_returns_points(detections_csv):
    reader = get_reader(str(detections_csv))
    (data, meta, ltype), = reader(str(detections_csv))
    assert ltype == "points"
    assert data.shape == (3, 4)  # N x (t,z,y,x)
    np.testing.assert_array_equal(meta["features"]["object_id"], [0, 1, 2])
    assert meta["metadata"]["ascent_layer_kind"] == "detections"


def test_tracks_csv_returns_tracks(tracks_csv):
    reader = get_reader(str(tracks_csv))
    (data, meta, ltype), = reader(str(tracks_csv))
    assert ltype == "tracks"
    assert data.shape == (3, 5)  # N x (track_id, t, z, y, x)
    np.testing.assert_array_equal(data[:, 0], [0, 0, 1])  # track ids
    np.testing.assert_array_equal(data[:, 1], [0, 1, 0])  # times
    np.testing.assert_array_equal(meta["properties"]["object_id"], [0, 2, 1])
    assert meta["metadata"]["ascent_layer_kind"] == "tracks"


def test_empty_detections_csv(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("object_id,t,z,y,x\n")
    reader = get_reader(str(p))
    (data, meta, ltype), = reader(str(p))
    assert ltype == "points"
    assert data.shape == (0, 4)


def test_empty_tracks_csv(tmp_path):
    p = tmp_path / "empty_tracks.csv"
    p.write_text("TrackID,ObjectID,t,z,y,x\n")
    reader = get_reader(str(p))
    (data, meta, ltype), = reader(str(p))
    assert ltype == "tracks"
    assert data.shape == (0, 5)


def test_csv_classifier_tolerates_utf8_bom(tmp_path):
    """Excel-exported CSVs commonly start with a BOM."""
    p = tmp_path / "with_bom.csv"
    p.write_bytes(b"\xef\xbb\xbfobject_id,t,z,y,x\n0,0,1,2,3\n")
    reader = get_reader(str(p))
    assert reader is not None
    (data, meta, ltype), = reader(str(p))
    assert ltype == "points"
    assert data.shape == (1, 4)


def test_csv_classifier_case_insensitive(tmp_path):
    """Header capitalization shouldn't matter."""
    p = tmp_path / "shouty.csv"
    p.write_text("TRACKID,OBJECTID,T,Z,Y,X\n0,0,0,1,2,3\n")
    reader = get_reader(str(p))
    assert reader is not None
    (data, meta, ltype), = reader(str(p))
    assert ltype == "tracks"
