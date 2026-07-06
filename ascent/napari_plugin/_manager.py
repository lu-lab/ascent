import napari
from napari.layers import Points, Tracks
import numpy as np
_GLOBAL_MANAGER = None

class LayerManager:
    def __init__(self,viewer):
        self.viewer = viewer
        self.managed_layers = set()
        self.track_layers = {}

        self.viewer.layers.events.inserted.connect(self._auto_register) 
        self.viewer.layers.events.removed.connect(self._auto_clean)

        self.viewer.dims.events.current_step.connect(self._on_slice_changed)
        return
    
    def _auto_register(self,event):
        layer = event.value

        if isinstance(layer,Points) and layer.metadata.get("reader","") == "ascent":
            self.managed_layers.add(layer)
            self._apply_styling(layer)
            self._ensure_tracks(layer)
            self._on_slice_changed()
        return
    
    def _auto_clean(self,event):
        layer = event.value
        if layer in self.managed_layers:
            self.managed_layers.remove(layer)
            if layer.name in self.track_layers:
                self.track_layer.pop(layer.name,None)
        return
    
    def _ensure_tracks(self,layer):
        if "track_id" in layer.features and not self.layer_exists(Tracks,layer.name+"_tracks"):
            data,features = self.points2tracks(layer)
            tracks = self.viewer.add_tracks(data=data,features=features,name=layer.name+"_tracks")
            self.track_layers[layer.name] = tracks
            self.viewer.layers.move(self.viewer.layers.index(tracks.name),0)
            layer.events.data.connect(self._on_point_edit)
            layer.events.features.connect(self._on_point_edit)

    def _apply_styling(self,layer:Points):
        layer.text = {
            'string': '{track_id}' if "track_id" in layer.features else "{object_id}",
            'size': 10,
            'color': 'white',
        }        
        layer.face_color = "track_id" if "track_id" in layer.features else "object_id"
        layer.face_colormap = "turbo"
        layer.refresh_colors()
        return 
    
    def layer_exists(self,layer_type,layer_name):
        "Helper to find if a layer with a specific type and name exists in current viewer"
        return any(isinstance(layer,layer_type) and layer.name == layer_name for layer in self.viewer.layers)

    def _on_slice_changed(self,event=None):
        t = self.viewer.dims.current_step[0]
        for layer in self.managed_layers:
            layer.shown = (layer.data[:,0] == t)
        return 
    
    def _on_point_edit(self,event):
        self._sync_tracks(event._sources[0])
        return
    
    def _sync_tracks(self,layer):
        data,features = self.points2tracks(layer)
        self.track_layers[layer.name].data = data
        self.track_layers[layer.name].features = features
        self.track_layers[layer.name].refresh()
        return

    def points2tracks(self,layer) -> tuple[np.array,dict]:
        """Appends track_id column to point data for track layer"""
        data = layer.data
        data = np.concatenate([layer.features['track_id'].values[:,None],data],axis=1) 
        features = layer.features
        return (data,features)

def get_or_create_manager(viewer:napari.Viewer=None) -> LayerManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        if viewer is None:
            raise ValueError("viewer is a required argument to create new LayerManager")
        _GLOBAL_MANAGER = LayerManager(viewer)
    return _GLOBAL_MANAGER