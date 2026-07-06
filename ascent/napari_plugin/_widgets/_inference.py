"""Inference dock widget for the ASCENT napari plugin.

Consumes one Image and one Points layer from the open viewer, runs the NETr
embedding + Hungarian-tracking pipeline in a background thread, and adds a
new Tracks layer with the result. A CSV is also written to the user-chosen
output directory so that the run is reproducible outside napari.

This module imports ``magicgui`` and ``napari`` lazily inside the factory so
that it can sit in the manifest without forcing those imports at plugin
discovery time.
"""


from pathlib import Path
from typing import Optional

DEVICES = ["auto", "cpu", "cuda", "mps"]
NORMALIZE = ["percentile", "none"]
AXIS_ORDERS = ["ZYX", "ZXY", "YZX", "YXZ", "XZY", "XYZ"]


def make_inference_widget():
    """Build the inference dock widget.

    Returns a ``magicgui``-decorated callable which napari treats as a dock
    widget. The callable is the "Run" handler — pressing the button kicks off
    a background ``thread_worker`` that calls
    :func:`ascent.napari_plugin._inference.run_from_arrays`.
    """
    from magicgui import magicgui
    from magicgui.widgets import PushButton
    from napari.qt.threading import thread_worker
    from napari.utils.notifications import show_error, show_info

    from ascent.napari_plugin._inference import resolve_device, run_from_arrays

    @thread_worker
    def _runner(*, image, image_path, image_channel, points, object_ids, model_ckpt, overrides, output_dir, progress_callback):
        return run_from_arrays(
            image=image,
            image_path=image_path,
            image_channel=image_channel,
            points=points,
            object_ids=object_ids,
            model_ckpt=model_ckpt,
            overrides=overrides,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

    # State held in a closure so we can detect overlapping runs.
    state: dict = {"worker": None}

    @magicgui(
        call_button="Run inference",
        image_layer={"label": "Image layer"},
        points_layer={"label": "Detections layer"},
        model_ckpt={
            "label": "Model checkpoint (.pth)",
            "mode": "r",
            "filter": "*.pth *.pt",
        },
        model_patch_size_z = {"label": "Z patch size used to train model","min":1},
        device={"choices": DEVICES},
        normalize={"choices": NORMALIZE, "label": "Image normalization"},
        norm_p_low={"label": "Percentile low", "min": 0.0, "max": 100.0, "step": 0.1},
        norm_p_high={"label": "Percentile high", "min": 0.0, "max": 100.0, "step": 0.01},
        axis_order={
            "choices": AXIS_ORDERS,
            "label": "Volume axis order",
            "tooltip": (
                "Order of the spatial axes inside each (Z,Y,X) volume of the "
                "image layer. Picking the wrong order silently produces "
                "wrong tracks — match what your microscope writes."
            ),
        },
        spacing_z={"label": "Voxel spacing Z (µm)", "min": 0.0001, "step": 0.01},
        spacing_y={"label": "Voxel spacing Y (µm)", "min": 0.0001, "step": 0.01},
        spacing_x={"label": "Voxel spacing X (µm)", "min": 0.0001, "step": 0.01},
        batch_size={"label": "Batch size (frames)", "min": 1},
        momentum={"label": "Tracking momentum", "min": 0.0, "max": 1.0, "step": 0.05},
        temperature={"label": "Tracking temperature", "min": 0.001, "step": 0.001},
        max_gap_frames={"label": "Max gap frames", "min": 1},
        output_dir={
            "label": "Output dir (optional)",
            "mode": "d",
        },
    )
    def widget(
        viewer: "napari.viewer.Viewer",
        image_layer: "napari.layers.Image",
        points_layer: "napari.layers.Points",
        model_ckpt: Path = Path(""),
        model_patch_size_z: int = 3,
        device: str = "auto",
        normalize: str = "percentile",
        norm_p_low: float = 1.0,
        norm_p_high: float = 99.99,
        axis_order: str = "ZYX",
        spacing_z: float = 1.5,
        spacing_y: float = 0.36,
        spacing_x: float = 0.36,
        batch_size: int = 16,
        momentum: float = 0.5,
        temperature: float = 0.05,
        max_gap_frames: int = 9999,
        output_dir: Path = Path(""),
    ) -> None:
        if state["worker"] is not None:
            show_info("ASCENT inference is already running — wait for it to finish.")
            return
        if image_layer is None or points_layer is None:
            show_error("Pick both an image layer and a detections layer.")
            return
        if not model_ckpt or not Path(model_ckpt).is_file():
            show_error(f"Model checkpoint not found: {model_ckpt}")
            return

        image = image_layer.data
        if image.ndim > 4 or image.ndim < 3:
            show_error(
                f"Image layer must be 4-D (T, Z, Y, X); got {image.shape}. "
                "Multi-channel layers should be split into single-channel layers."
            )
            return

        points = points_layer.data
        if points.ndim != 2 or points.shape[1] != 4:
            show_error(
                f"Points layer must have shape (N, 4) for (t, z, y, x); "
                f"got {points.shape}."
            )
            return

        object_ids = None
        feats = getattr(points_layer, "features", None)
        if feats is not None and "object_id" in feats:
            object_ids = feats["object_id"].to_numpy()

        overrides = {
            "model_lf_patch_size_z": model_patch_size_z,
            "dataset_normalize": normalize,
            "dataset_norm_p_low": norm_p_low,
            "dataset_norm_p_high": norm_p_high,
            "dataset_axis_order": axis_order,
            "dataset_spacing": [spacing_z, spacing_y, spacing_x],
            "runtime_batch_size_frame": batch_size,
            "runtime_device": resolve_device(device),
            "tracking_momentum": momentum,
            "tracking_temperature": temperature,
            "tracking_max_gap_frames": max_gap_frames,
        }

        out_dir: Optional[Path] = Path(output_dir) if str(output_dir) else None

        image_data = image_layer.data
        image_path_to_use = None
        image_channel_to_use = 0
        
        source_path = getattr(image_layer.source, "path", None)
        if source_path and Path(source_path).suffix.lower() in (".h5", ".hdf5"):
            image_path_to_use = source_path
            import re
            m = re.search(r"ch(\d+)$", image_layer.name)
            if m:
                image_channel_to_use = int(m.group(1))
            image_data = None

        show_info("ASCENT inference started — this may take a while.")
        widget.call_button.enabled = False

        from qtpy.QtCore import QObject, Signal
        class ProgressEmitter(QObject):
            yielded = Signal(object)
        
        emitter = ProgressEmitter()

        def cb(kind, current, total):
            emitter.yielded.emit((kind, current, total))

        worker = _runner(
            image=image_data,
            image_path=image_path_to_use,
            image_channel=image_channel_to_use,
            points=points,
            object_ids=object_ids,
            model_ckpt=str(model_ckpt),
            overrides=overrides,
            output_dir=out_dir,
            progress_callback=cb,
        )
        worker.emitter = emitter  # prevent garbage collection
        state["worker"] = worker

        from napari.utils import progress
        pbar_phase = progress(total=3, desc="ASCENT: Serialization")
        pbar_batch = None

        def _on_yield(msg):
            nonlocal pbar_phase, pbar_batch
            kind, current, total = msg
            if kind == "serialization_done":
                if pbar_phase:
                    pbar_phase.update(1)
            elif kind == "extract_start":
                if pbar_phase:
                    pbar_phase.set_description("ASCENT: Feature Extraction")
                pbar_batch = progress(total=total, desc="Extracting batches")
            elif kind == "extract_step":
                if pbar_batch:
                    pbar_batch.update(1)
            elif kind == "extract_done":
                if pbar_batch:
                    pbar_batch.close()
                    pbar_batch = None
                if pbar_phase:
                    pbar_phase.update(1)
            elif kind == "tracking_start":
                if pbar_phase:
                    pbar_phase.set_description("ASCENT: Hungarian Tracking")
            elif kind == "tracking_done":
                if pbar_phase:
                    pbar_phase.update(1)
                    pbar_phase.close()
                    pbar_phase = None

        def _release() -> None:
            nonlocal pbar_phase, pbar_batch
            if pbar_batch is not None:
                pbar_batch.close()
            if pbar_phase is not None:
                pbar_phase.close()
            state["worker"] = None
            widget.call_button.enabled = True

        def _on_done(result):
            from ascent.napari_plugin._manager import get_or_create_manager
            track_data, oids = result
            
            # add track_id to features. data frame is copied then reassigned to trigger internal napari updates
            features  = points_layer.features.copy()
            features["track_id"] = track_data[:,0].astype(int)
            points_layer.features = features
            # points_layer.features["track_id"] = track_data[:,0].astype(int)
            # points_layer.properties["track_id"] = track_data[:,0].astype(int)

            manager = get_or_create_manager()
            manager._ensure_tracks(points_layer)
            manager._apply_styling(points_layer)

            # need to rescale face_contrast since napari doesn't automatically
            points_layer.face_contrast_limits = (track_data[:,0].min(),track_data[:,0].max())
            points_layer.refresh_colors()
            points_layer.refresh_text()

        def _on_error(err):
            show_error(f"ASCENT inference failed: {err!r}")

        emitter.yielded.connect(_on_yield)
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.finished.connect(_release)
        worker.start()

    def _update_scale() -> None:
        import napari
        sz = widget.spacing_z.value
        sy = widget.spacing_y.value
        sx = widget.spacing_x.value
        order = widget.axis_order.value

        scale = [1.0, 1.0, 1.0, 1.0]
        spacing_dict = {'Z': sz, 'Y': sy, 'X': sx}
        scale[1] = spacing_dict[order[0]]
        scale[2] = spacing_dict[order[1]]
        scale[3] = spacing_dict[order[2]]
        scale_tuple = tuple(scale)

        viewer = napari.current_viewer()
        if viewer is None:
            return
            
        for layer in viewer.layers:
            if layer.ndim == 4:
                layer.scale = scale_tuple

    widget.image_layer.changed.connect(_update_scale)
    widget.points_layer.changed.connect(_update_scale)
    widget.spacing_z.changed.connect(_update_scale)
    widget.spacing_y.changed.connect(_update_scale)
    widget.spacing_x.changed.connect(_update_scale)
    widget.axis_order.changed.connect(_update_scale)

    btn_apply = PushButton(text="Apply spacing to viewer")
    widget.insert(widget.index(widget.spacing_x) + 1, btn_apply)
    btn_apply.clicked.connect(_update_scale)

    return widget
