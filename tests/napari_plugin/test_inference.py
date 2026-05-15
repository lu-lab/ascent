"""Tests for the pure-Python inference adapter (no Qt / napari needed).

The end-to-end pipeline (which calls into torch + the NETr model) is not
exercised here — that requires real model weights and substantial fixtures.
We test the round-trip glue that surrounds the model: layer-array → temp
files → tracks CSV → arrays.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ascent.napari_plugin._inference import (
    build_config,
    read_tracks_csv,
    resolve_device,
    write_detections_csv,
    write_image_h5,
)


# --------------------------------------------------------------------------- #
# write_image_h5
# --------------------------------------------------------------------------- #


def test_write_image_h5_round_trip(tmp_path: Path):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 1024, (3, 4, 6, 8), dtype=np.uint16)
    out = tmp_path / "movie.h5"
    write_image_h5(img, out)

    with h5py.File(out, "r") as f:
        assert sorted(f.keys()) == ["t0", "t1", "t2"]
        for t in range(3):
            np.testing.assert_array_equal(f[f"t{t}/c0"][()], img[t])
            assert f[f"t{t}/c0"].dtype == np.uint16


def test_write_image_h5_rejects_non_4d(tmp_path: Path):
    img = np.zeros((4, 6, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="4-D"):
        write_image_h5(img, tmp_path / "bad.h5")


def test_write_image_h5_custom_channel(tmp_path: Path):
    img = np.zeros((2, 1, 1, 1), dtype=np.float32)
    out = tmp_path / "ch.h5"
    write_image_h5(img, out, channel=2)
    with h5py.File(out, "r") as f:
        assert "c2" in f["t0"]


# --------------------------------------------------------------------------- #
# write_detections_csv
# --------------------------------------------------------------------------- #


def test_write_detections_csv_round_trip(tmp_path: Path):
    coords = np.array(
        [
            [0.0, 1.5, 2.5, 3.5],
            [0.7, 2.0, 4.0, 6.0],  # t=0.7 should round to 1
            [2.0, 1.5, 2.5, 3.5],
        ]
    )
    oids = np.array([10, 20, 30], dtype=np.int64)
    out = tmp_path / "dets.csv"
    write_detections_csv(coords, out, object_ids=oids)

    text = out.read_text()
    lines = text.strip().splitlines()
    assert lines[0] == "object_id,t,z,y,x"
    # frame index for row 1 should round 0.7 → 1
    assert lines[2].startswith("20,1,")


def test_write_detections_csv_auto_object_ids(tmp_path: Path):
    coords = np.zeros((4, 4))
    out = tmp_path / "dets.csv"
    write_detections_csv(coords, out)
    rows = [line.split(",") for line in out.read_text().strip().splitlines()[1:]]
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3]


def test_write_detections_csv_rejects_bad_shape(tmp_path: Path):
    with pytest.raises(ValueError, match="N, 4"):
        write_detections_csv(np.zeros((3, 5)), tmp_path / "x.csv")


def test_write_detections_csv_rejects_object_id_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match="length"):
        write_detections_csv(
            np.zeros((3, 4)),
            tmp_path / "x.csv",
            object_ids=np.array([0, 1]),
        )


# --------------------------------------------------------------------------- #
# read_tracks_csv
# --------------------------------------------------------------------------- #


def test_read_tracks_csv_round_trip(tmp_path: Path):
    p = tmp_path / "tracks.csv"
    p.write_text(
        "TrackID,ObjectID,t,z,y,x\n"
        "0,10,0,1.5,2.5,3.5\n"
        "0,30,1,1.6,2.4,3.5\n"
        "1,20,0,2.0,4.0,6.0\n"
    )
    track_data, oids = read_tracks_csv(p)
    assert track_data.shape == (3, 5)
    np.testing.assert_array_equal(track_data[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(track_data[:, 1], [0, 1, 0])
    np.testing.assert_array_equal(oids, [10, 30, 20])


def test_read_tracks_csv_empty(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("TrackID,ObjectID,t,z,y,x\n")
    track_data, oids = read_tracks_csv(p)
    assert track_data.shape == (0, 5)
    assert oids.shape == (0,)


# --------------------------------------------------------------------------- #
# build_config
# --------------------------------------------------------------------------- #


def test_build_config_has_required_keys():
    cfg = build_config("/tmp/fake.pth")
    # Keys consumed by run_ascent.run — required to avoid KeyError there.
    required = {
        "model_ckpt",
        "model_lf_patch_size_xy",
        "model_lf_patch_size_z",
        "model_lf_pretrained",
        "model_pe_num_mlp_layers",
        "model_pe_norm",
        "model_pe_scaling",
        "model_tr_d_model",
        "model_tr_nhead",
        "model_tr_num_encoder_layers",
        "model_tr_dim_feedforward",
        "model_tr_dropout",
        "model_tr_activation",
        "model_use_local_features",
        "model_use_positional_encoding",
        "model_use_transformer",
        "tracking_momentum",
        "tracking_temperature",
        "tracking_max_gap_frames",
        "tracking_w_within",
        "dataset_image_channel",
        "dataset_axis_order",
        "dataset_spacing",
        "dataset_normalize",
        "dataset_norm_p_low",
        "dataset_norm_p_high",
        "runtime_batch_size_frame",
    }
    missing = required - cfg.keys()
    assert not missing, f"build_config missing keys: {missing}"
    assert cfg["model_ckpt"] == "/tmp/fake.pth"


def test_build_config_overrides_win():
    cfg = build_config(
        "ckpt.pth",
        overrides={"tracking_momentum": 0.9, "dataset_spacing": [1, 1, 1]},
    )
    assert cfg["tracking_momentum"] == 0.9
    assert cfg["dataset_spacing"] == [1, 1, 1]


def test_build_config_ignores_none_overrides():
    cfg = build_config("ckpt.pth", overrides={"tracking_momentum": None})
    assert cfg["tracking_momentum"] == 0.5  # default preserved


# --------------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------------- #


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_known_value():
    out = resolve_device("auto")
    assert out in {"cpu", "cuda", "mps"}


# --------------------------------------------------------------------------- #
# run_from_arrays round-trip (mocked model)
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=False)
def _torch_or_skip():
    """The round-trip tests below import ``ascent.tools.run_ascent``, which
    eagerly imports torch. Skip cleanly if torch isn't on the test runner."""
    pytest.importorskip("torch")


def test_run_from_arrays_round_trip(_torch_or_skip, monkeypatch, tmp_path: Path):
    """Validate the file-bridge: layer arrays → temp files → parsed tracks.

    The model invocation is mocked — we replace ``ascent.tools.run_ascent.run``
    with a fake that reads the inputs we wrote, asserts they round-tripped,
    and emits a deterministic tracks CSV. This catches everything that
    ``run_from_arrays`` is responsible for *except* the actual NETr forward.
    """
    from ascent.napari_plugin import _inference

    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (3, 2, 4, 4), dtype=np.uint16)
    points = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.5, 1.5, 2.5],
            [1.0, 1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0, 3.0],
        ]
    )
    object_ids = np.array([10, 20, 10, 10], dtype=np.int64)

    def fake_run(cfg, *, configure_logging=True):
        # Inputs landed where we said they would.
        assert Path(cfg["dataset_file_image"]).is_file()
        assert Path(cfg["dataset_file_coord"]).is_file()
        assert configure_logging is False
        # Image written in t{n}/c0 layout.
        with h5py.File(cfg["dataset_file_image"], "r") as fh:
            assert sorted(fh.keys()) == ["t0", "t1", "t2"]
            np.testing.assert_array_equal(fh["t0/c0"][()], image[0])
        # Detections CSV header survived the round-trip.
        det_lines = Path(cfg["dataset_file_coord"]).read_text().strip().splitlines()
        assert det_lines[0] == "object_id,t,z,y,x"
        # object_id column matches what we passed in.
        det_oids = [int(line.split(",")[0]) for line in det_lines[1:]]
        assert det_oids == [10, 20, 10, 10]
        # Forward a deterministic tracks CSV — two tracks, four points.
        out_path = (
            Path(cfg["runtime_output_dir"])
            / f"{cfg['runtime_output_prefix']}_tracks.csv"
        )
        out_path.write_text(
            "TrackID,ObjectID,t,z,y,x\n"
            "0,10,0,1,2,3\n"
            "0,10,1,1,2,3\n"
            "0,10,2,1,2,3\n"
            "1,20,0,0.5,1.5,2.5\n"
        )
        return out_path

    monkeypatch.setattr("ascent.tools.run_ascent.run", fake_run)

    track_data, oids = _inference.run_from_arrays(
        image=image,
        points=points,
        object_ids=object_ids,
        model_ckpt="/tmp/fake.pth",
        output_dir=tmp_path,
    )
    assert track_data.shape == (4, 5)
    np.testing.assert_array_equal(track_data[:, 0], [0, 0, 0, 1])
    np.testing.assert_array_equal(oids, [10, 10, 10, 20])


def test_run_from_arrays_cleans_up_temp_dir(_torch_or_skip, monkeypatch):
    """When ``output_dir`` is None, the working dir must be deleted on return."""
    from ascent.napari_plugin import _inference

    captured: dict = {}

    def fake_run(cfg, *, configure_logging=True):
        captured["workdir"] = Path(cfg["runtime_output_dir"])
        out_path = captured["workdir"] / f"{cfg['runtime_output_prefix']}_tracks.csv"
        out_path.write_text("TrackID,ObjectID,t,z,y,x\n")
        return out_path

    monkeypatch.setattr("ascent.tools.run_ascent.run", fake_run)

    _inference.run_from_arrays(
        image=np.zeros((1, 1, 1, 1), dtype=np.uint8),
        points=np.zeros((0, 4)),
        model_ckpt="/tmp/fake.pth",
        output_dir=None,
    )
    assert "workdir" in captured
    assert not captured["workdir"].exists(), "temp dir should be cleaned up"
