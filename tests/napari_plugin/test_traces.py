"""Tests for activity-trace extraction (pure-Python)."""

from __future__ import annotations

import numpy as np
import pytest

from ascent.napari_plugin._traces import extract_traces


def _gradient_image() -> np.ndarray:
    """Returns a (T, Z, Y, X) image where the value at (t, z, y, x) == t*1000 + z*100 + y*10 + x."""
    T, Z, Y, X = 4, 3, 5, 5
    img = np.zeros((T, Z, Y, X), dtype=np.float64)
    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    img[t, z, y, x] = t * 1000 + z * 100 + y * 10 + x
    return img


def test_empty_tracks_returns_empty():
    img = np.zeros((2, 1, 1, 1))
    out = extract_traces(img, np.zeros((0, 5)))
    assert out == {}


def test_basic_extraction_radius_zero():
    img = _gradient_image()
    tracks = np.array(
        [
            [0, 0, 1, 2, 3],  # value 0 + 100 + 20 + 3 = 123
            [0, 1, 1, 2, 3],  # value 1000 + 100 + 20 + 3 = 1123
            [1, 0, 0, 0, 0],  # value 0
            [1, 2, 0, 0, 0],  # value 2000
        ],
        dtype=np.float64,
    )
    out = extract_traces(img, tracks, roi_radius=0)
    assert set(out) == {0, 1}
    np.testing.assert_array_equal(out[0]["frames"], [0, 1])
    np.testing.assert_array_equal(out[0]["intensity"], [123, 1123])
    np.testing.assert_array_equal(out[1]["intensity"], [0, 2000])


def test_radius_clips_to_bounds():
    img = np.ones((1, 2, 2, 2), dtype=np.float64)
    tracks = np.array([[0, 0, 0, 0, 0]], dtype=np.float64)
    out = extract_traces(img, tracks, roi_radius=10)
    assert out[0]["intensity"][0] == 1.0  # clipped to (1,2,2,2), all ones


def test_max_reduction():
    img = _gradient_image()
    tracks = np.array([[0, 0, 1, 2, 3]], dtype=np.float64)
    mean = extract_traces(img, tracks, roi_radius=1, reduction="mean")[0]["intensity"][0]
    mx = extract_traces(img, tracks, roi_radius=1, reduction="max")[0]["intensity"][0]
    assert mx >= mean


def test_invalid_reduction():
    with pytest.raises(ValueError, match="unknown reduction"):
        extract_traces(np.zeros((1, 1, 1, 1)), np.array([[0, 0, 0, 0, 0]]), reduction="median")


def test_invalid_radius():
    with pytest.raises(ValueError, match="roi_radius"):
        extract_traces(np.zeros((1, 1, 1, 1)), np.zeros((0, 5)), roi_radius=-1)


def test_invalid_image_shape():
    with pytest.raises(ValueError, match="4-D"):
        extract_traces(np.zeros((1, 1, 1)), np.zeros((0, 5)))


def test_out_of_bounds_frames_skipped():
    img = np.ones((2, 1, 1, 1))
    tracks = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0],  # past end
            [0, -1, 0, 0, 0],  # before start
        ],
        dtype=np.float64,
    )
    out = extract_traces(img, tracks)
    np.testing.assert_array_equal(out[0]["frames"], [0])
