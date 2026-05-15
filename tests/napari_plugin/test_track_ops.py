"""Unit tests for the TrackEditor (pure-Python, no Qt)."""

from __future__ import annotations

import numpy as np
import pytest

from ascent.napari_plugin._track_ops import TrackEditor


def _make_editor(rows):
    arr = np.asarray(rows, dtype=np.float64)
    oids = np.arange(len(rows), dtype=np.int64)
    return TrackEditor(track_data=arr, object_ids=oids)


def test_init_validates_shape():
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        TrackEditor(track_data=np.zeros((3, 4)), object_ids=np.zeros((3,), dtype=np.int64))


def test_init_validates_object_ids_length():
    with pytest.raises(ValueError, match="length N"):
        TrackEditor(
            track_data=np.zeros((3, 5)),
            object_ids=np.zeros((2,), dtype=np.int64),
        )


def test_track_ids_dedup_and_sort():
    ed = _make_editor(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [2, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(ed.track_ids(), [0, 1, 2])


def test_split_basic():
    ed = _make_editor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 3, 0, 0, 0],
        ]
    )
    new_id = ed.split(0, frame_t=2)
    assert new_id == 1
    assert sorted(ed.track_ids().tolist()) == [0, 1]
    np.testing.assert_array_equal(ed.frames_of(0), [0, 1])
    np.testing.assert_array_equal(ed.frames_of(1), [2, 3])


def test_split_at_first_frame_errors():
    ed = _make_editor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    with pytest.raises(ValueError, match="original"):
        ed.split(0, frame_t=0)


def test_split_at_one_past_last_errors():
    ed = _make_editor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    with pytest.raises(ValueError, match="new track"):
        ed.split(0, frame_t=2)  # no rows with t >= 2


def test_merge_basic():
    ed = _make_editor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 3, 0, 0, 0],
        ]
    )
    ed.merge(into=0, source=1)
    assert ed.track_ids().tolist() == [0]
    np.testing.assert_array_equal(ed.frames_of(0), [0, 1, 2, 3])


def test_merge_overlap_errors():
    ed = _make_editor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="overlap"):
        ed.merge(into=0, source=1)


def test_merge_self_errors():
    ed = _make_editor([[0, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="itself"):
        ed.merge(into=0, source=0)


def test_relabel_to_unused_id():
    ed = _make_editor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    ed.relabel(0, 7)
    assert ed.track_ids().tolist() == [7]


def test_relabel_to_existing_errors():
    ed = _make_editor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="merge instead"):
        ed.relabel(0, 1)


def test_relabel_self_is_noop():
    ed = _make_editor([[0, 0, 0, 0, 0]])
    ed.relabel(0, 0)
    assert ed.track_ids().tolist() == [0]


def test_delete_track():
    ed = _make_editor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    ed.delete(0)
    assert ed.track_ids().tolist() == [1]
    assert ed.track_data.shape[0] == 1


def test_remove_point():
    ed = _make_editor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 2, 0, 0, 0],
        ]
    )
    ed.remove_point(0, frame_t=1)
    np.testing.assert_array_equal(ed.frames_of(0), [0, 2])


def test_remove_point_missing_errors():
    ed = _make_editor([[0, 0, 0, 0, 0]])
    with pytest.raises(KeyError):
        ed.remove_point(0, frame_t=999)


def test_undo_redo_roundtrip():
    ed = _make_editor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 3, 0, 0, 0],
        ]
    )
    before = ed.track_data.copy()
    ed.split(0, frame_t=2)
    assert ed.track_ids().tolist() == [0, 1]
    ed.undo()
    np.testing.assert_array_equal(ed.track_data, before)
    ed.redo()
    assert ed.track_ids().tolist() == [0, 1]


def test_redo_cleared_on_new_mutation():
    """Standard editor behavior: any new mutation must clear the redo stack."""
    ed = _make_editor(
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
    )
    ed.delete(1)
    ed.undo()  # populates redo
    assert len(ed._redo) == 1
    ed.delete(0)  # any new mutation
    assert len(ed._redo) == 0
    # redo() now is a no-op
    assert ed.redo() is None


def test_undo_stack_capped():
    ed = _make_editor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    ed.max_undo = 3
    for _ in range(5):
        ed.relabel(0, 99)
        ed.relabel(99, 0)
    assert len(ed._undo) <= ed.max_undo
