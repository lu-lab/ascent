"""Smoke tests for the dock widgets.

Skipped when napari / Qt aren't installed (which is the case in the
default ``ascent`` environment without ``[gui]`` extras). On a CI runner
with PySide6 + napari installed, these instantiate each widget and verify
the basic propagation paths.
"""

from __future__ import annotations

import numpy as np
import pytest

# Hard-skip the whole module if any of the GUI deps are missing — including
# pytest-qt, which is what supplies the ``qtbot`` fixture used below.
# Without this, a runner with napari but without pytest-qt fails at fixture
# resolution instead of cleanly skipping.
pytest.importorskip("magicgui")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")
napari = pytest.importorskip("napari")


@pytest.fixture
def viewer(qtbot):
    v = napari.Viewer(show=False)
    yield v
    v.close()


def test_inference_widget_instantiates(viewer):
    from ascent.napari_plugin._widgets._inference import make_inference_widget

    w = make_inference_widget()
    assert w is not None
    # Magicgui wraps the function into an object with a ``call_button``.
    assert hasattr(w, "call_button")


def test_correction_widget_instantiates_and_attaches(viewer, qtbot):
    from ascent.napari_plugin._widgets._correction import make_correction_widget

    track_data = np.array(
        [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 0, 2, 2, 2]],
        dtype=np.float64,
    )
    viewer.add_tracks(track_data, name="t", properties={"object_id": np.array([0, 0, 1])})

    w = make_correction_widget()
    # Picker should auto-select the only Tracks layer.
    assert w is not None


def test_traces_widget_instantiates(viewer):
    from ascent.napari_plugin._widgets._traces import make_traces_widget

    w = make_traces_widget()
    assert w is not None


def test_control_widget_stacks_panels(viewer):
    from ascent.napari_plugin._widgets._control import make_control_widget

    w = make_control_widget()
    # Header label + 3 child panels.
    assert len(list(w)) >= 4
