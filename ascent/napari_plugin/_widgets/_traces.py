"""Activity-trace dock widget.

Plots one intensity-vs-frame line per track, sampled from a chosen Image
layer at the centroid of each Tracks-layer point with an optional ROI.

Uses ``pyqtgraph`` for the plot canvas, which is part of the ``[gui]``
extras. Lazy-imported so non-GUI environments can still load the package.
"""


from typing import Any

import numpy as np


def make_traces_widget():
    """Build the activity-trace dock widget."""
    from magicgui.widgets import ComboBox, Container, Label, PushButton, SpinBox, create_widget
    from napari.utils.notifications import show_error, show_info
    import pyqtgraph as pg

    from ascent.napari_plugin._traces import extract_traces

    image_picker = create_widget(annotation="napari.layers.Image", label="Image layer")
    tracks_picker = create_widget(annotation="napari.layers.Tracks", label="Tracks layer")
    roi_radius = SpinBox(label="ROI radius (vox)", value=2, min=0, max=64)
    reduction = ComboBox(label="Reduction", choices=["mean", "max", "sum"], value="mean")
    btn_refresh = PushButton(label="Compute traces")
    status = Label(value="Pick image + tracks layers, then Compute.")

    plot_widget = pg.PlotWidget()
    plot_widget.setBackground("w")
    plot_widget.setLabel("bottom", "Frame")
    plot_widget.setLabel("left", "Intensity")
    plot_widget.addLegend()

    state: dict[str, Any] = {"curves": []}
    palette = [
        (213, 94, 0),
        (0, 114, 178),
        (0, 158, 115),
        (240, 228, 66),
        (204, 121, 167),
        (86, 180, 233),
        (230, 159, 0),
    ]

    def _clear_plot() -> None:
        plot_widget.clear()
        # plot_widget.clear() leaves the previous legend orphaned in the
        # PlotItem; clearing it explicitly avoids a stack of duplicate legends.
        legend = plot_widget.plotItem.legend
        if legend is not None:
            legend.clear()
        state["curves"] = []

    def _on_compute() -> None:
        image = image_picker.value
        tracks = tracks_picker.value
        if image is None or tracks is None:
            show_error("Pick both an image layer and a tracks layer.")
            return
        img_arr = np.asarray(image.data)
        if img_arr.ndim != 4:
            show_error(
                f"Image must be 4-D (T, Z, Y, X); got {img_arr.shape}. "
                "Multi-channel layers should be split first."
            )
            return
        try:
            traces = extract_traces(
                img_arr,
                np.asarray(tracks.data),
                roi_radius=int(roi_radius.value),
                reduction=str(reduction.value),
            )
        except (ValueError, IndexError) as e:
            show_error(f"trace extraction failed: {e}")
            return

        _clear_plot()
        for i, (tid, trace) in enumerate(traces.items()):
            color = palette[i % len(palette)]
            curve = plot_widget.plot(
                trace["frames"],
                trace["intensity"],
                pen=pg.mkPen(color=color, width=1.5),
                name=f"track {tid}",
            )
            state["curves"].append(curve)

        status.value = f"{len(traces)} tracks plotted."
        show_info(status.value)

    btn_refresh.clicked.connect(_on_compute)

    container = Container(
        widgets=[image_picker, tracks_picker, roi_radius, reduction, btn_refresh, status],
        labels=True,
        name="ASCENT — Activity traces",
    )

    # Embed the pyqtgraph plot into the same dock as the controls.
    container.native.layout().addWidget(plot_widget)
    return container
