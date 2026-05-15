"""Manual track-correction dock widget.

Wraps :class:`ascent.napari_plugin._track_ops.TrackEditor` in a magicgui
Container with split / merge / relabel / delete / undo / redo buttons and a
status line. Operates on a user-selected Tracks layer; mutations are flushed
back to the layer immediately so the viewer stays in sync.

magicgui and napari are imported lazily so the module is safe to import from
non-GUI environments.
"""


from typing import Any

import numpy as np


def make_correction_widget():
    """Build the correction dock widget."""
    from magicgui.widgets import (
        Container,
        Label,
        PushButton,
        SpinBox,
        create_widget,
    )
    from napari.utils.notifications import show_error, show_info

    from ascent.napari_plugin._track_ops import TrackEditor

    layer_picker = create_widget(annotation="napari.layers.Tracks", label="Tracks layer")
    track_a = SpinBox(label="Track A", value=0, min=0, max=10**9)
    track_b = SpinBox(label="Track B", value=1, min=0, max=10**9)
    frame_t = SpinBox(label="Frame t", value=0, min=0, max=10**9)
    btn_split = PushButton(label="Split A at frame t")
    btn_merge = PushButton(label="Merge B → A")
    btn_relabel = PushButton(label="Relabel A → B")
    btn_delete = PushButton(label="Delete A")
    btn_remove_pt = PushButton(label="Remove point (A, t)")
    btn_undo = PushButton(label="Undo")
    btn_redo = PushButton(label="Redo")
    status = Label(value="No tracks layer selected.")

    state: dict[str, Any] = {"editor": None, "layer": None}

    # ----------------------------------------------------------------- #
    # layer attach / sync helpers
    # ----------------------------------------------------------------- #

    def _object_ids_from_layer(layer) -> np.ndarray:
        n = layer.data.shape[0]
        props = getattr(layer, "properties", None) or {}
        oids = props.get("object_id")
        if oids is None or len(oids) != n:
            return np.full((n,), -1, dtype=np.int64)
        return np.asarray(oids, dtype=np.int64)

    def _attach(layer) -> None:
        if layer is None:
            state["editor"] = None
            state["layer"] = None
            status.value = "No tracks layer selected."
            return
        editor = TrackEditor(
            track_data=np.asarray(layer.data, dtype=np.float64).copy(),
            object_ids=_object_ids_from_layer(layer).copy(),
        )
        state["editor"] = editor
        state["layer"] = layer
        status.value = f"Attached '{layer.name}': {len(editor.track_ids())} tracks."

    def _flush() -> None:
        editor: TrackEditor = state["editor"]
        layer = state["layer"]
        if editor is None or layer is None:
            return
        # Set properties before data: napari's Tracks layer validates
        # property length against the *current* data, so updating data first
        # against the old, longer/shorter property arrays trips a brief
        # length-mismatch error.
        layer.properties = {"object_id": editor.object_ids}
        layer.data = editor.track_data

    def _guarded(action_label: str, fn) -> None:
        editor: TrackEditor = state["editor"]
        if editor is None:
            show_error("Pick a Tracks layer first.")
            return
        try:
            result = fn(editor)
        except (KeyError, ValueError) as e:
            show_error(f"{action_label}: {e}")
            return
        _flush()
        msg = f"{action_label} ok"
        if result is not None:
            msg = f"{msg} → {result}"
        status.value = msg
        show_info(msg)

    # ----------------------------------------------------------------- #
    # button slots
    # ----------------------------------------------------------------- #

    def _on_split() -> None:
        _guarded(
            f"split {track_a.value} @ t={frame_t.value}",
            lambda ed: f"new id {ed.split(track_a.value, frame_t.value)}",
        )

    def _on_merge() -> None:
        _guarded(
            f"merge {track_b.value} → {track_a.value}",
            lambda ed: ed.merge(into=track_a.value, source=track_b.value),
        )

    def _on_relabel() -> None:
        _guarded(
            f"relabel {track_a.value} → {track_b.value}",
            lambda ed: ed.relabel(track_a.value, track_b.value),
        )

    def _on_delete() -> None:
        _guarded(f"delete {track_a.value}", lambda ed: ed.delete(track_a.value))

    def _on_remove_pt() -> None:
        _guarded(
            f"remove pt ({track_a.value}, t={frame_t.value})",
            lambda ed: ed.remove_point(track_a.value, frame_t.value),
        )

    def _on_undo() -> None:
        _guarded("undo", lambda ed: ed.undo() or "(nothing to undo)")

    def _on_redo() -> None:
        _guarded("redo", lambda ed: ed.redo() or "(nothing to redo)")

    layer_picker.changed.connect(_attach)
    btn_split.clicked.connect(_on_split)
    btn_merge.clicked.connect(_on_merge)
    btn_relabel.clicked.connect(_on_relabel)
    btn_delete.clicked.connect(_on_delete)
    btn_remove_pt.clicked.connect(_on_remove_pt)
    btn_undo.clicked.connect(_on_undo)
    btn_redo.clicked.connect(_on_redo)

    # Initial attach if a Tracks layer is already selected when the widget
    # opens (magicgui's ``create_widget`` populates ``value`` from the viewer).
    if layer_picker.value is not None:
        _attach(layer_picker.value)

    return Container(
        widgets=[
            layer_picker,
            track_a,
            track_b,
            frame_t,
            btn_split,
            btn_merge,
            btn_relabel,
            btn_delete,
            btn_remove_pt,
            btn_undo,
            btn_redo,
            status,
        ],
        labels=True,
        name="ASCENT — Correct tracks",
    )
