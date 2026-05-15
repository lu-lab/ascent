"""Top-level control dock widget for the ASCENT napari plugin.

A vertical stack of feature panels — header + inference + correction +
activity traces. Each panel is also reachable as a standalone widget via
the manifest, so users who want only one piece of the workflow can pick
just that command from the Plugins menu.

``magicgui`` and ``napari`` are imported lazily so this module stays
importable from non-GUI environments.
"""


_HEADER_MD = (
    "<b>ASCENT</b> &mdash; annotation-free 3-D neuron tracking.<br>"
    "Drop an HDF5 movie (<code>t{frame}/c{channel}</code>), a detections "
    "CSV (<code>object_id,t,z,y,x</code>), or a tracks CSV "
    "(<code>TrackID,ObjectID,t,z,y,x</code>) onto the viewer."
)


def make_control_widget():
    """Return the ASCENT control panel: header + inference + correction + traces."""
    from magicgui.widgets import Container, Label

    from ascent.napari_plugin._widgets._correction import make_correction_widget
    from ascent.napari_plugin._widgets._inference import make_inference_widget
    from ascent.napari_plugin._widgets._traces import make_traces_widget

    container = Container(
        widgets=[
            Label(value=_HEADER_MD),
            make_inference_widget(),
            make_correction_widget(),
            make_traces_widget(),
        ],
        labels=False,
        name="ASCENT",
    )
    # The HTML header label will stretch the dock to the width of the text
    # unless we explicitly tell the underlying Qt widget to word-wrap it.
    try:
        container[0].native.setWordWrap(True)
        container.native.setMaximumWidth(350)
    except Exception:
        pass

    return container
