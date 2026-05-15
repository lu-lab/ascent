"""napari plugin for ASCENT.

Provides a viewer integration for ASCENT — readers for the HDF5 movie /
detection-CSV / tracks-CSV formats, dock widgets to run inference, and (in
later milestones) manual track correction with an activity-trace viewer.

The plugin is registered via the npe2 manifest at
``ascent/napari_plugin/napari.yaml`` and exposed through the
``napari.manifest`` entry point in ``pyproject.toml``.

Submodules are deliberately *not* imported here so that the readers (which
depend only on numpy + h5py) can be used without the optional ``[gui]``
extras. napari resolves widget commands lazily via the manifest.
"""
