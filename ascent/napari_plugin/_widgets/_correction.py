from __future__ import annotations
import numpy as np
import napari
from napari.layers import Image, Points, Tracks
from qtpy.QtWidgets import QSizePolicy, QWidget, QVBoxLayout, QSpinBox, QComboBox, QLabel, QPushButton, QCheckBox, QLineEdit
from typing import TYPE_CHECKING
from collections import deque
from vispy.color.colormap import Colormap
import warnings

if TYPE_CHECKING:
    import napari.viewer
_KEY_BINDINGS = (
"""
<p>The custom keybindings provided by this plugin are as follows:</p>

<ol>
    <li><b>w</b> → increment z slice</li>
    <li><b>d</b> → decrement z slice</li>
    <li><b>e</b> → increment time</li>
    <li><b>q</b> → decrement time</li>
    <li><b>y</b> → yank/copy label from selected point</li>
    <li><b>r</b> → reassign selected point</li>
    <li><b>t</b> → reassign track forward in time</li>
    <li><b>b</b> → reassign track backward in time</li>
    <li><b>c</b> → increment current label</li>
    <li><b>x</b> → decrement current label</li>
</ol>
""")
class CorrectionWidget(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        from napari.settings import get_settings
        settings = get_settings()
        settings.appearance.highlight = {"highlight_thickness":2,"highlight_color":[1,1,0,1]}

        self.viewer = viewer
        self._connected_layers = set()

        # history deques and restore flag for undo/redo functionality
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        self._is_restoring = False

        # Setup Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # UI Elements
        target_label = QLabel("Target Layer:")
        target_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)
        layout.addWidget(target_label)
        self.layer_combo = QComboBox()
        self._update_layer_choices()
        layout.addWidget(self.layer_combo)

        current_label = QLabel("Current Label:")
        current_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)
        layout.addWidget(current_label)
        self.label_spin = QSpinBox()
        self.label_spin.setRange(1, 9999)
        self.label_spin.setValue(1)

        self.label_spin.valueChanged.connect(self._on_label_changed)
        layout.addWidget(self.label_spin)

        self.savefile_line = QLineEdit(text="tracks_file.csv")
        layout.addWidget(self.savefile_line)
        self.save_btn = QPushButton(text="Save Tracks")
        self.save_btn.clicked.connect(self._save_tracks)
        layout.addWidget(self.save_btn)

        self.redo_btn = QPushButton(text="Redo Edit")
        layout.addWidget(self.redo_btn)
        self.redo_btn.clicked.connect(self._redo)
        self.undo_btn = QPushButton(text='Undo Edit')
        layout.addWidget(self.undo_btn)
        self.undo_btn.clicked.connect(self._undo)

        self.unique_chk = QCheckBox(text="Maintain 1 Object Per Track")
        self.unique_chk.setChecked(True)
        layout.addWidget(self.unique_chk)

        layout.addWidget(QLabel(_KEY_BINDINGS))

        # Layer Management Setup
        self.viewer.layers.events.inserted.connect(self._update_layer_choices)
        self.viewer.layers.events.removed.connect(self._update_layer_choices)
        self.layer_combo.currentIndexChanged.connect(self._on_active_layer_change)

        # Ensure a Tracking layer exists
        self._update_layer_choices()
        self._setup_keybindings()

        self._on_active_layer_change()

    # --- Properties --- #
    @property
    def active_layer(self) -> Points | None:
        """Returns the currently selected layer object from the combo box."""
        return self.layer_combo.currentData()

    # --- quick general helper functions --- #
    def layer_exists(self,layer_type,layer_name):
        "Helper to find if a layer with a specific type and name exists in current viewer"
        return any(isinstance(layer,layer_type) and layer.name == layer_name for layer in self.viewer.layers)

    def _get_points_layers(self) -> list[Points]:
        """Helper to find all Points layers."""
        return [l for l in self.viewer.layers if isinstance(l, Points) and "track_id" in l.features]
    
    def _save_state(self):
        "Helper to save state for undo functionality"
        if self.active_layer:
            current_data = np.copy(self.active_layer.data)
            current_features = self.active_layer.features.copy(deep=True)
            self.undo_stack.append((current_data,current_features))
        return

    # --- UI callbacks --- # 
    def _update_layer_choices(self, event=None):
        """Syncs the QComboBox with current Points layers."""
        self.layer_combo.blockSignals(True)

        current_selection = self.active_layer
        self.layer_combo.clear()

        layers = self._get_points_layers()
        for layer in layers:
            self.layer_combo.addItem(layer.name, layer)
        
        if current_selection in layers:
            self.layer_combo.setCurrentText(current_selection.name)
        
        self.layer_combo.blockSignals(False)
        return
    
    def _on_label_changed(self):
        """updates point layer defaults to match current layer in spinbox"""
        if self.active_layer is not None:
            self.active_layer.feature_defaults["track_id"] = int(self.label_spin.value())
            self.active_layer.feature_defaults["color"] = (int(self.label_spin.value())) / 300
        return
    
    def _save_tracks(self):
        """shortcut for saving tracks using the provided _writer"""
        saved_paths = self.viewer.layers.save(path=self.savefile_line.text(),selected=[self.active_layer],plugin="ascent")
        print(f"Tracks saved to {saved_paths}")
        return
    
    def _undo(self):
        if len(self.undo_stack) > 1:
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)

            previous_state = self.undo_stack[-1]

            self._is_restoring = True
            self.active_layer.data = np.copy(previous_state[0])
            self.active_layer.features = previous_state[1].copy(deep=True)
            self.active_layer.refresh_colors()
            self.active_layer.refresh_text()
            self._is_restoring = False
        return

    def _redo(self):
        if self.redo_stack:
            next_state = self.redo_stack.pop()
            self.undo_stack.append(next_state)

            self._is_restoring = True
            self.active_layer.data = np.copy(next_state[0])
            self.active_layer.features = next_state[1].copy(deep=True)
            self.active_layer.refresh_colors()
            self.active_layer.refresh_text()
            self._is_restoring = False
        return

    # --- Connecting to Active Layer --- # 
    def _on_active_layer_change(self):
        """Configure the layer for tracking when it is selected in the widget."""
        layer = self.active_layer
        if layer is None:
            return
        
        if layer not in self._connected_layers:
            layer.events.data.connect(self._on_point_changed)
            self._bind_layer_specific_keys(layer)
            self._connected_layers.add(layer)

        self._save_state()
        self._on_point_changed()
        self._on_label_changed()
        self.viewer.layers.selection.active = self.active_layer
        return
        
    def _bind_layer_specific_keys(self,layer:Points):
        """Adds keybindings that are Point Layer specific"""

        @layer.bind_key('d', overwrite=True)
        def dec_slice(_):
            self._move_dim(1, -1)

        @layer.bind_key('f',overwrite=True)
        def find_label(_):
            if self.active_layer is None:
                return
            t_idx = self.viewer.dims.current_step[0]
            t_mask = self.active_layer.data[:,0]==t_idx
            frame_pt_labels = self.active_layer.features['track_id'].values[t_mask]
            label = self.label_spin.value()
            if label in frame_pt_labels:
                self.active_layer.selected_data = set(np.where(t_mask * self.active_layer.features['track_id']==label)[0])
            return
        
        @layer.bind_key('y',overwrite=True)
        def yank_label(_):
            if self.active_layer is None:
                return
            active_pts = list(self.active_layer.selected_data)
            if len(active_pts) == 0:
                return
            new_label = self.active_layer.features["track_id"][active_pts[0]]
            self.label_spin.setValue(new_label)
            self._on_label_changed()
            return
        
        @layer.bind_key('r',overwrite=True)
        def reassign_pt(_):
            if self.active_layer is None:
                return
            selected_pts = self.active_layer.selected_data
            if self.unique_chk.isChecked():
                current_t = self.viewer.dims.current_step[0]
                next_label = self.active_layer.features["track_id"].max() + 1
                mask = (self.active_layer.data[:,0] == current_t) * (self.active_layer.features["track_id"] == self.label_spin.value())
                self.active_layer.features.loc[mask,'track_id'] = next_label
            self.active_layer.features.loc[selected_pts,'track_id'] = self.label_spin.value()

            self.active_layer.events.features()
            self.active_layer.refresh_colors()
            self.active_layer.refresh_text()
            self._save_state()
            return

        @layer.bind_key('t',overwrite=True)
        def reassign_track_future(_):
            selected_pts = self.active_layer.selected_data
            selected_labels = self.active_layer.features.loc[selected_pts,'track_id'].values
            label_mask = np.array([label in selected_labels for label in self.active_layer.features['track_id'].values])
            t_mask = self.active_layer.data[:,0] >= self.viewer.dims.current_step[0]
            if self.unique_chk.isChecked():
                next_label = self.active_layer.features["track_id"].max() + 1
                replace_label_mask = self.active_layer.features["track_id"] == self.label_spin.value()
                self.active_layer.features.loc[t_mask*replace_label_mask,'track_id'] = next_label
            self.active_layer.features.loc[t_mask*label_mask,'track_id'] = self.label_spin.value()

            self.active_layer.events.features()
            self.active_layer.refresh_colors()
            self.active_layer.refresh_text()
            self._save_state()
            return

        @layer.bind_key('b',overwrite=True)
        def reassign_track_past(_):
            selected_pts = self.active_layer.selected_data
            selected_labels = self.active_layer.features.loc[selected_pts,'track_id'].values
            label_mask = np.array([label in selected_labels for label in self.active_layer.features['track_id'].values])
            t_mask = self.active_layer.data[:,0] <= self.viewer.dims.current_step[0]
            if self.unique_chk.isChecked():
                next_label = self.active_layer.features["track_id"].max() + 1
                replace_label_mask = self.active_layer.features["track_id"] == self.label_spin.value()
                self.active_layer.features.loc[t_mask*replace_label_mask,'track_id'] = next_label
            self.active_layer.features.loc[t_mask*label_mask,'track_id'] = self.label_spin.value()

            self.active_layer.events.features()
            self.active_layer.refresh_colors()
            self.active_layer.refresh_text()
            self._save_state()
            return
    
    # --- update event callbacks --- # 
    def _on_point_changed(self):
        """callback for when user edits point layer. Ensure points and tracks stay in sync"""
        if self.active_layer is not None:
            if len(self.active_layer.features["object_id"]) > 0:
                self.active_layer.feature_defaults["object_id"]  = max(self.active_layer.features["object_id"])+1
            else:
                self.active_layer.feature_defaults["object_id"] = 1
            if  not self._is_restoring:
                if not np.array_equal(self.undo_stack[-1][0],self.active_layer.data):
                    self._save_state()
                    self.redo_stack.clear()
        return

    # --- general navigation/viewer-level keybindings --- #
    def _setup_keybindings(self):
        """Binds tracking navigation and labeling keys to the viewer."""
        @self.viewer.bind_key('q', overwrite=True)
        def dec_frame(_):
            self._move_dim(0, -1)

        @self.viewer.bind_key('e', overwrite=True)
        def inc_frame(_):
            self._move_dim(0, 1)

        @self.viewer.bind_key('w', overwrite=True)
        def inc_slice(_):
            self._move_dim(1, 1)

        @self.viewer.bind_key('d', overwrite=True)
        def dec_slice(_):
            self._move_dim(1, -1)

        @self.viewer.bind_key('c', overwrite=True)
        def inc_label(_):
            self.label_spin.setValue(self.label_spin.value() + 1)

        @self.viewer.bind_key('x', overwrite=True)
        def dec_label(_):
            self.label_spin.setValue(max(0, self.label_spin.value() - 1))

    def _move_dim(self, dimension: int, delta: int):
        """Helper to navigate Time (0) or Z (1) axes."""
        step = list(self.viewer.dims.current_step)
        if dimension < len(step):
            max_val = int(self.viewer.dims.range[dimension][1])
            step[dimension] = np.clip(step[dimension] + delta, 0, max_val)
            self.viewer.dims.current_step = tuple(step)