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
from qtpy.QtWidgets import QWidget, QTabWidget, QVBoxLayout

class ControlWidget(QWidget):
    """Return the ASCENT control panel: header + inference + correction + traces."""
    from magicgui.widgets import Container, Label
    from napari.viewer import Viewer
    def __init__(self,viewer:Viewer,parent=None):
        super().__init__(parent)
        from ascent.napari_plugin._widgets._correction import CorrectionWidget
        from ascent.napari_plugin._widgets._inference import make_inference_widget
        from ascent.napari_plugin._widgets._traces import make_traces_widget
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.container = QTabWidget()
        self.container.addTab(make_inference_widget().native,"Inference")
        self.container.addTab(CorrectionWidget(viewer),"Correction")
        self.container.addTab(make_traces_widget().native,"Traces")
        layout.addWidget(self.container)
        return 
