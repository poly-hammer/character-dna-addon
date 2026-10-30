"""Repeated DNA imports must not accumulate scale in shared bone templates."""

from copy import deepcopy

import bpy
import pytest

from character_dna.dna_io import importer
from character_dna.utilities import get_active_rig_instance, get_addon_ops_module
from constants import HEAD_DNA_FILE


def test_repeated_head_imports_keep_the_same_parent_bones(monkeypatch):
    # Isolate the shared template so even a failing regression cannot poison later tests.
    template = deepcopy(importer.EXTRA_BONES)
    monkeypatch.setattr(importer, "EXTRA_BONES", template)
    locations = {name: tuple(data["location"]) for name, data in template}
    poses = []
    try:
        for _ in range(2):
            bpy.ops.wm.read_homefile(use_empty=True)
            get_addon_ops_module().import_dna(
                filepath=str(HEAD_DNA_FILE),
                include_body=False,
                import_face_board=False,
                import_mesh=False,
                import_bones=True,
                import_shape_keys=False,
                import_materials=False,
            )
            instance = get_active_rig_instance()
            assert instance is not None
            rig = instance.head_rig
            assert rig is not None and isinstance(rig.data, bpy.types.Armature)
            poses.append(
                {name: tuple(value for row in rig.data.bones[name].matrix_local for value in row) for name in locations}
            )
        for name in locations:
            assert poses[1][name] == pytest.approx(poses[0][name], abs=1e-7), name
        assert {name: tuple(data["location"]) for name, data in template} == locations
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)
