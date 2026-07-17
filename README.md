# ASCENT

*Annotation‑free Self‑supervised Contrastive Embeddings for 3‑D Neuron Tracking*

---

## 📑 Table of Contents

- [ASCENT](#ascent)
  - [📦 Installation](#-installation)
    - [Quick Install (from Source)](#quick-install-from-source)
  - [Docker (one image, no env juggling)](#docker-one-image-no-env-juggling)
  - [Detection (bring your own)](#detection-bring-your-own)
  - [🚀 Tracking Your Own Video with a Pre-trained NETr Model](#-tracking-your-own-video-with-a-pre-trained-netr-model)
    - [How ASCENT Runs Inference](#how-ascent-runs-inference)
    - [Available Pre-trained NETr Checkpoints](#available-pre-trained-netr-checkpoints)
    - [Example: Tracking Lightsheet Video](#example-tracking-lightsheet-video)
    - [Overriding Parameters from the Command Line](#overriding-parameters-from-the-command-line)
    - [Input File Formats](#input-file-formats)
    - [Output Files](#output-files)
    - [Recommended Parameters](#recommended-parameters)
    - [Minimal Workflow](#minimal-workflow)
  - [🧪 Training Your Own NETr Model](#-training-your-own-netr-model)
    - [Quick start](#quick-start)
    - [Configuration schema](#configuration-schema)
    - [Notes On Parameters](#notes-on-parameters)
    - [Tips](#tips)
  - [📜 License](#-license)

---

## 📦 Installation (ASCENT core)

ASCENT runs on Python **3.10–3.13** (Linux/macOS/Windows). It’s a standard PyTorch project; install either the CPU build or CUDA build depending on your machine.

---

### Quick Install (from Source)

```bash
# Create a fresh environment (recommended)
conda create -n ascent python=3.12 && conda activate ascent
# or: python -m venv .venv && source .venv/bin/activate

# Clone the repository
git clone https://github.com/lu-lab/ascent.git
cd ascent

# Install ASCENT
pip install -e .

# Sanity check
python -c "import ascent, torch; print('ASCENT', ascent.__version__, '| CUDA available:', torch.cuda.is_available())"
```

## Docker (one image, no env juggling)

A single Docker image ships the ASCENT inference / training pipeline. It is
PyTorch-only — detection (segmentation) is intentionally separated, since
the previous bundled-StarDist setup forced users into a TensorFlow + PyTorch
CUDA dance that was the source of most install pain. Run any segmentation
plugin you like (we recommend [`stardist-napari`](https://napari-hub.org/plugins/stardist-napari)
or [`cellpose-napari`](https://napari-hub.org/plugins/cellpose-napari)),
export a detections CSV, and feed it to ASCENT.

### Build

CPU image (multi-arch — works on linux/amd64 and linux/arm64, including
Docker Desktop on Apple Silicon):

```bash
docker build -t ascent:latest .
```

CUDA image (Linux + NVIDIA Container Toolkit on the host):

```bash
docker build     --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime     -t ascent:cuda .
```

### Run

Mount your data and configs at `/workspace` (the image's default WORKDIR):

```bash
# ASCENT inference
docker run --rm -v "$PWD:/workspace" ascent:latest ascent run \
    --config /workspace/track.py

# Download the three pretrained NETr checkpoints
docker run --rm -v "$PWD/checkpoints:/checkpoints" ascent:latest \
    fetch-models /checkpoints

# Interactive shell (nothing auto-activated; use `micromamba run -n ascent ...`)
docker run --rm -it -v "$PWD:/workspace" ascent:latest bash
```

GPU run:

```bash
docker run --gpus all --rm -v "$PWD:/workspace" ascent:cuda     ascent run --config /workspace/track.py
```

### What's inside

| Path | Contents |
|---|---|
| `/opt/ascent` | The source tree (editable install) |
| `/opt/conda/envs/ascent` | Python 3.12 + ASCENT + PyTorch |
| `/usr/local/bin/entrypoint` | Dispatcher — see `scripts/docker-entrypoint.sh` |

The image does **not** include the napari GUI plugin; for interactive use,
install ASCENT locally with `pip install ascent[gui]` (see the napari
plugin section below). It also does **not** include any segmentation /
detection model — see *Detection (bring your own)* below.


---

---

## Detection (bring your own)

ASCENT consumes a CSV of per-frame neuron centroids — it does **not** ship
its own segmenter. Pick whichever 3-D detector best fits your data and
have it emit a CSV with this header:

```
object_id,t,z,y,x
```

Recommended napari segmentation plugins (each lives in its own
environment, so there is no PyTorch / TensorFlow conflict to manage):

- [`stardist-napari`](https://napari-hub.org/plugins/stardist-napari) — the
  detector ASCENT was originally evaluated against. Pretrained models for
  *C. elegans* brain are available:
  [`celegans-free-NeRVE`](https://www.dropbox.com/scl/fo/dxcikcgwgi96yw5lefokq/AH8gG4qmRP86jTjiy4t-6GA?rlkey=8673p9td73sb1cnvmdu4phxlr&st=wfoj46mu&dl=0)
  (NeRVE imaging conditions, voxel 0.3226 × 0.3226 × 1.5 µm),
  [`celegans-device-Opterra`](https://www.dropbox.com/scl/fo/djsxhdxfco9xhijdxpk6w/ALbtald5Lnh2p1dOibEs4Q4?rlkey=t07vmppwwm04gtg9eww0wvgdl&st=qljcb71i&dl=0)
  (Opterra, voxel 0.243 × 0.243 × 1.5 µm).
- [`cellpose-napari`](https://napari-hub.org/plugins/cellpose-napari) — a
  PyTorch-native alternative; useful if you prefer to stay in one ML stack.
- Any other tool that emits per-frame instance masks. Convert masks →
  centroids with a few lines of `scipy.ndimage.center_of_mass` or
  `skimage.measure.regionprops` and dump as CSV.

If you used the previous `examples/scripts/stardist_segment.py` script,
its functionality is now provided by `stardist-napari`'s headless API
(`stardist-napari.run --input ... --modelpath ...`), or you can keep
running it from a prior commit of this repo against your existing
StarDist install.

Dataset details for the *C. elegans* StarDist models are in the bioRxiv
preprint, "Datasets and ground truth":
[https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1.full](https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1.full)

---

## 🚀 Tracking Your Own Video with a Pre-trained NETr Model

### How ASCENT Runs Inference

ASCENT’s `run` mode performs:

1. **Feature extraction** – NETr encodes each neuron candidate into a high‑dimensional **embedding vector** that captures appearance and context.
2. **Tracking** – **HungarianTracker** links candidates across frames using cosine similarity in embedding space, producing continuous neuron identities.

Both steps run sequentially in `ascent run` (see [`tools/run_ascent.py`](ascent/tools/run_ascent.py)).

---

### Available Pre-trained NETr Checkpoints

All checkpoints share the **same NETr architecture hyperparameters** as in `examples/configs/track_template.py`.

| Model ID          | Download |Trained on (dataset)                                   | Microscope / Modality          | Voxel size (µm)       | Sample |
| ----------------- | --- | ------------------------------------------------------ | ------------------------------ | -------------------------------- | --- |
| `NETr-lightsheet` | [link](https://www.dropbox.com/scl/fi/jju3gb44f8wtwi0mqewqe/NETr-lightsheet.pth?rlkey=lboxgt7a4iq7ez1x99t2ax22a&st=5vbuordg&dl=0) | Lightsheet microscopy recordings                       | Light-sheet microscope             | 0.36 x 0.36 x 1.26                                | Head-fixed *C. elegans* |
| `NETr-NeRVE`      | [link](https://www.dropbox.com/scl/fi/tj8oxcuvefz3fvjg18jhq/NETr-NeRVE.pth?rlkey=wee4aqpi04tny88f5r4e17ann&st=q3v3s1hn&dl=0) | **NeRVE** dataset        | Spinning-disk confocal         | 0.3226 × 0.3226 × 1.5     | Freely moving *C. elegans* |
| `NETr-Opterra`    | [link](https://www.dropbox.com/scl/fi/japv9qzzhxer63vdxg714/NETr-Opterra.pth?rlkey=nl1ryrgzfgets2i2ljha9ljiv&st=ut8zabuu&dl=0) | **In-house** dataset | Swept-field confocal (Opterra) | 0.243 × 0.243 × 1.5  | Immobilized *C. elegans* |

* Use these checkpoints directly with the workflow below in [Example: Tracking Lightsheet Video](#example-tracking-lightsheet-video).
* To point ASCENT to a checkpoint, either edit `model_ckpt` in the config or **override from the CLI** (see [Overriding Parameters from the Command Line](#overriding-parameters-from-the-command-line)).

**Usage**

```python
# examples/configs/track_template.py
model_ckpt = "/path/to/checkpoints/NETr-NeRVE.pth"  # or NETr-lightsheet / NETr-Opterra
```

or from the command line:

```bash
ascent run \
  --config examples/configs/track_template.py \
  --model_ckpt /path/to/checkpoints/NETr-Opterra.pth
```

---

### Example: Tracking Lightsheet Video

```bash
ascent run --config examples/configs/track_template.py
```

The config file defines:

* **Dataset paths** (`dataset_file_image`, `dataset_file_coord`)
* **Dataset parameters** (`dataset_image_channel`, `dataset_axis_order`, `dataset_normalize`, `dataset_norm_p_low`, `dataset_norm_p_high`, `dataset_spacing`)
* **Model checkpoint & architecture** (`model_ckpt`, NETr params, patch size)
* **Tracking parameters** (`tracking_momentum`, `tracking_temperature`, etc.)
* **Runtime settings** (`runtime_output_dir`, `runtime_output_prefix`, etc.)

---

### Overriding Parameters from the Command Line

Any config value can be overrideen at runtime:

```bash
ascent run \
  --config examples/configs/track_template.py \
  --dataset_file_image /path/to/raw.h5 \
  --dataset_file_coord /path/to/centroids.csv \
  --dataset_axis_order ZYX \
  --dataset_do_normalize true \
  --dataset_norm_p_low 1 \
  --dataset_norm_p_high 99.9 \
  --model_ckpt /path/to/checkpoints/NETr-lightsheet.pth \
  --runtime_output_dir ./outputs \
  --runtime_output_prefix sample
```

---

### Input File Formats

#### 1. Raw Video (HDF5)

```
root
└── t0
    └── c0    (Z × Y × X float32/uint16)
└── t1
    └── c0
...
```

* Axis order in each dataset is set via `dataset_axis_order` (`"ZYX"`, `"YXZ"`, etc.).
* `dataset_image_channel` selects the channel to load.

#### 2. Detections CSV

Comma-separated with at least:

```
object_id,t,z,y,x
0,0,15.2,64.1,33.8
1,0,28.7,50.2,70.1
...
```

Coordinates are in voxel units, matching the raw video.

---

### Output Files

When `runtime_output_dir="outdir"` and `runtime_output_prefix="sample"`:

* `outdir/sample_pred_z.pt` – NETr embeddings (PyTorch tensor)
* `outdir/sample_pred_object_ids.pt` – Object IDs corresponding to rows in the embedding tensor
* `outdir/sample_tracks.csv` – Napari-compatible track table:
  `TrackID,ObjectID,t,z,y,x`

---

### Recommended Parameters

* Start with the default `examples/configs/track_template.py`.

| Parameter                  | Recommended Value | Notes                                                                 |
|----------------------------|-------------------|----------------------------------------------------------------------|
| `dataset_normalize`        | `percentile`      | Chooses per-frame intensity normalization method (`none` or `percentile`). |
| `dataset_norm_p_low`       | `1.0`             | Lower percentile bound for normalization.                           |
| `dataset_norm_p_high`      | `99.99`           | Upper percentile bound for normalization.                           |
| `tracking_momentum`        | `0.5`             | Momentum factor for updating track embeddings over time.            |
| `tracking_temperature`     | `0.05`            | Softmax temperature for similarity scoring in the tracker.          |
| `tracking_max_gap_frames`  | `9999`            | Maximum allowed gap (in frames) when linking detections into tracks.|
| `tracking_w_within`        | `0`               | Weight for within-track distance when estimating cutoffs.           |
| `runtime_batch_size_frame` | `64`              | Number of frames processed per batch; increase if memory allows.    |
| `runtime_device`           | `"cuda"`          | Device for computation (`"cuda"` for GPU, `"cpu"` for CPU).          |

---

### Minimal Workflow

1. (Optional) Generate detections with StarDist 3‑D.
2. Point `model_ckpt` to one of the pre‑trained NETr checkpoints.
3. Run `ascent run --config examples/configs/track_template.py`.
4. Inspect `*_tracks.csv` in Napari.

---

## 🧪 Training Your Own NETr Model

### Quick Start

Train NETr from scratch or fine‑tune using a single **config file** and the training entry point:

```bash
ascent train \
  --config examples/configs/train_NETr_template.py
```

ASCENT supports multi‑GPU training via PyTorch DistributedDataParallel (DDP) and automatically uses **all visible GPUs**.

### Configuration Schema

A training config is a small Python (or YAML) file that declares the **model**, **dataset**, **transforms**, **dataloader**, **losses**, **optimizer**, and run settings.

See the [example](./examples/configs/train_NETr_template.py) for an example config.

### Notes On Parameters
* `device`: `"cpu"`, `"cuda"`, `"mps"`, or `"auto"`. `"auto"` picks an appropriate GPU device if available.
* `port`: Internal port number used by DDP. Not used for single‑GPU or CPU runs.
* `image_file`, `coord_file`: Paths to the image and detection data used for self‑supervised NETr training. See [Input File Formats](#input-file-formats) for details.
* `dataloader`: Key–value dictionary passed to PyTorch’s `DataLoader` (e.g., `num_workers`, `batch_size`). `batch_size` has the largest impact on GPU memory usage and training speed—pick the largest value that fits without OOM.
* `losses`: A list of losses. Each item specifies a `"class"` with `"params"`. The total loss is the **weighted sum** of items using each entry’s `"weight"`.
* `optimizer`, `scheduler`: Defaults to Adam with learning rate **1e‑3** and **no scheduler** if not specified in the config.
* `epochs`, `time_limit`: Maximum epochs and/or wall‑clock time (seconds). Training stops when either limit is reached. If unspecified, no limit is applied.
* `save_every_n_epochs`, `save_time_span`: Control checkpoint frequency by epoch count or elapsed time (seconds).
* `continue_training`: If `True`, resumes from the most recent checkpoint found alongside `model_save_path`.

### Tips

* Keep `axis_order`, `spacing`, and `image_channel` consistent with your data. Use percentile normalization to stabilize training across recordings.
* Start with `batch_size=4`; increase if memory allows. If you see too few objects per frame, reduce crop sizes or jitter ranges.
* You can list **multiple training datasets** (as a list) to mix sources; batches are interleaved across loaders each epoch.
* For custom learning‑rate schedules or per‑layer LRs, add a `scheduler` or `optimizer.layer_lrs` to the config.

---
---

## napari plugin (GUI)

ASCENT ships with a [napari](https://napari.org) plugin that wraps the same
inference pipeline in a graphical interface and adds manual track correction
plus an activity-trace viewer. It also reads the ASCENT HDF5 / detections /
tracks formats directly via drag-and-drop.

### Install

```bash
pip install ascent[gui]    # from PyPI: adds napari, PySide6, magicgui, pyqtgraph
# or: pip install -e .[gui]   # editable install from a local clone
napari                       # launch napari; the ASCENT plugin appears under Plugins
```

### What is in the plugin

| Panel | Purpose |
|---|---|
| Readers | Drop an HDF5 movie (`t{frame}/c{channel}`), detections CSV (`object_id,t,z,y,x`), or tracks CSV (`TrackID,ObjectID,t,z,y,x`) — they load as Image / Points / Tracks layers respectively. HDF5 movies are **lazy-loaded** via Dask, so multi-gigabyte datasets open instantly. Multi-channel datasets are automatically split into separate layers. |
| ASCENT inference | Pick an Image layer + a Points layer (your detections), point at a NETr `.pth` checkpoint, click Run inference. The pipeline runs in a background thread and adds a Tracks layer when done. **Multi-phase progress bars** keep you updated on extraction and tracking batches. |
| ASCENT correct tracks | Split / merge / relabel / delete tracks; remove individual points; undo / redo. Edits flush back to the Tracks layer immediately. |
| ASCENT activity traces | Pick an Image + a Tracks layer, set an ROI radius and reduction (mean/max/sum), click Compute traces. Plots one intensity-vs-frame line per track in a pyqtgraph canvas. |

### Detection is intentionally separate

The plugin does not ship a detector. Use any napari segmentation plugin to produce
a Points layer of centroids (or load your own CSV), then feed it to ASCENT.
Common picks:

- [stardist-napari](https://napari-hub.org/plugins/stardist-napari) — the detector
  ASCENT was originally evaluated against.
- [cellpose-napari](https://napari-hub.org/plugins/cellpose-napari) — a
  PyTorch-native alternative.

This decoupling means StarDist's TF dependency does not need to coexist with
ASCENT's PyTorch stack; the two plugins live in independent environments and
you get the best of each via napari layer interop.

### Demo Data
The data needed to test the plugin on the inhouse worm dataset and a portion of a hydra recording from Lagache, et al. can be found in the following dropbox folder: https://www.dropbox.com/scl/fo/7jrb3duabf8ks150gflzn/AFcsPNgepYjSbgN4IG37Qow?rlkey=frpxd9owg184t44oda9e9nrlz&st=vssjwwe8&dl=0

h5 files containing the recordings, csv files with sample segmentations, and pretrained model weights are provided for both datasets.

Lagache, T., Hanson, A., Fairhall, A. & Yuste, R. Robust single neuron tracking of
calcium imaging in behaving Hydra (2020). URL https://www.biorxiv.org/content/10.
1101/2020.06.22.165696v1. Pages: 2020.06.22.165696 Section: New Results.
### Tips

- The plugin requires a 4-D `(T, Z, Y, X)` Image layer. If your data is
  multi-channel, ASCENT's reader automatically splits them into single-channel layers (e.g., `[filename] ch0`); just pick the one you want to track.
- The "Volume axis order" picker must match how each frame is stored in the
  HDF5 — picking the wrong order produces wrong tracks silently.
- Use the **Apply spacing to viewer** button in the inference panel to explicitly synchronize the physical voxel spacing (Z, Y, X) to all 4D layers in the viewer.
- When running inference, if you use an Image layer that was loaded directly from an `.h5` file, the plugin bypasses the expensive file-dumping phase and points PyTorch directly to your original file, saving significant disk space and startup time.
- `pip install ascent` (without `[gui]`) installs only the CLI/training pieces,
  with no Qt or napari pulled in.

---

## 📜 License

ASCENT is released under the MIT license. © Haejun Han.
