# ASCENT — napari plugin

**Annotation-free Self-supervised Contrastive Embeddings for 3-D Neuron Tracking.**

This plugin wraps the [ASCENT](https://github.com/lu-lab/ascent) tracking
pipeline in napari and adds an interactive correction + analysis layer on
top of it. It is designed for 3-D fluorescence-microscopy recordings of
behaving animals (originally *C. elegans*), but works on any 4-D
`(T, Z, Y, X)` movie with a per-frame detections layer.

## What it does

- **Reads** ASCENT-format HDF5 movies, detection CSVs, and tracks CSVs as
  napari Image, Points, and Tracks layers. Movies are **lazy-loaded via Dask** to open multi-gigabyte datasets instantly. Multi-channel data is automatically split.
- **Runs inference** with a trained NETr model — feature extraction +
  Hungarian linking — in a background thread, producing a Tracks layer. Features **multi-phase progress bars** and automatically **bypasses file serialization** if reading from a dropped `.h5` file.
- **Synchronizes Scale** via an explicit "Apply spacing to viewer" button that maps your physical voxel sizes directly to all 4D layers.
- **Lets you correct tracks** manually — split, merge, relabel, delete,
  remove individual points, with undo / redo.
- **Plots activity traces** — per-track intensity vs. frame, sampled with
  an ROI around each centroid (mean / max / sum reductions).

## What it does NOT do

- Detect / segment neurons. Bring detections from
  [`stardist-napari`](https://napari-hub.org/plugins/stardist-napari),
  [`cellpose-napari`](https://napari-hub.org/plugins/cellpose-napari), or
  any other detector — the plugin operates on a `Points` layer and is
  intentionally agnostic about where it came from.
- Train new NETr models. Use the `ascent train` CLI for that.

## Quick start

1. `pip install ascent[gui]`
2. Launch napari. ASCENT panels show up under **Plugins → ASCENT**.
3. Drop your 4-D HDF5 movie onto the viewer (multi-channel files automatically split into single-channel layers).
4. Drop a detections CSV (or run a detector plugin to make a Points layer).
5. Open **Plugins → ASCENT control panel**. This opens a unified interface for inference, correction, and traces.
6. Under the **Inference** tab, select your desired Image channel and Points layers.
7. Enter your physical voxel dimensions and click **Apply spacing to viewer** to scale the 4D view.
8. Point at a NETr `.pth` checkpoint and click **Run inference**.
9. Use the **Correct tracks** tab to clean up misassignments, and the **Activity traces** tab to inspect intensities.

Pre-trained NETr checkpoints (lightsheet, NeRVE, Opterra) are listed in the
[ASCENT README](https://github.com/lu-lab/ascent#available-pre-trained-netr-checkpoints).

## License

MIT.
