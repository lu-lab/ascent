"""Manifest-shape and entry-point tests.

These do not require napari/Qt to be installed — they parse the YAML and
walk the package structure to confirm every command's ``python_name``
resolves to an importable module path.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")  # PyYAML is a napari dep, but skip cleanly if absent.


@pytest.fixture(scope="module")
def manifest() -> dict:
    pkg_files = importlib.resources.files("ascent.napari_plugin")
    manifest_path = pkg_files / "napari.yaml"
    with manifest_path.open() as f:
        return yaml.safe_load(f)


def test_manifest_has_required_top_level_keys(manifest):
    assert manifest["name"] == "ascent"
    assert "display_name" in manifest
    assert "contributions" in manifest


def test_manifest_commands_have_unique_ids(manifest):
    ids = [c["id"] for c in manifest["contributions"]["commands"]]
    assert len(ids) == len(set(ids)), f"duplicate command IDs: {ids}"


def test_manifest_widgets_reference_real_commands(manifest):
    cmd_ids = {c["id"] for c in manifest["contributions"]["commands"]}
    for w in manifest["contributions"]["widgets"]:
        assert w["command"] in cmd_ids, f"widget {w!r} references unknown command"


def test_manifest_readers_reference_real_commands(manifest):
    cmd_ids = {c["id"] for c in manifest["contributions"]["commands"]}
    for r in manifest["contributions"]["readers"]:
        assert r["command"] in cmd_ids, f"reader {r!r} references unknown command"


def test_command_python_names_resolve_module(manifest):
    """Each command's python_name parses as ``module:attr`` and the module imports.

    The attr itself may import GUI deps lazily, so we don't try to fetch it —
    just that the dotted module path exists. This catches typos like a wrong
    package name in the manifest.
    """
    for cmd in manifest["contributions"]["commands"]:
        pn = cmd["python_name"]
        module_path, _, attr = pn.partition(":")
        assert attr, f"command {cmd['id']} has malformed python_name {pn!r}"
        # Resolve to a file path without importing — avoids pulling in magicgui/Qt.
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"command {cmd['id']} module not found: {module_path}"


def test_reader_filename_patterns_cover_h5_and_csv(manifest):
    patterns: set[str] = set()
    for r in manifest["contributions"]["readers"]:
        patterns.update(p.lower() for p in r["filename_patterns"])
    for needed in ("*.h5", "*.hdf5", "*.csv"):
        assert needed in patterns, f"missing reader pattern {needed}"


def test_napari_manifest_entry_point_registered():
    """pyproject.toml exposes the manifest under the ``napari.manifest`` group."""
    eps = importlib.metadata.entry_points(group="napari.manifest")
    names = {ep.name for ep in eps}
    assert "ascent" in names, (
        f"napari.manifest entry point 'ascent' not registered (found: {names}). "
        "Re-run `pip install -e .`."
    )
    target = next(ep for ep in eps if ep.name == "ascent")
    assert target.value == "ascent.napari_plugin:napari.yaml"


def test_manifest_yaml_bundled_in_package():
    pkg_files = importlib.resources.files("ascent.napari_plugin")
    assert (pkg_files / "napari.yaml").is_file()
