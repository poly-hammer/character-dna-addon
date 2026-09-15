import bmesh
import bpy
import pytest

from constants import TEST_DNA_FOLDER

from character_dna.dna_io.calibrator import DNACalibrator
from character_dna.ui.callbacks import get_active_rig_instance


@pytest.fixture
def head_only_character(addon):
    """A head DNA imported on its own, reloaded per test since these mutate the topology."""
    from fixtures.scene import load_dna

    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=False,
        include_body=False,
    )
    yield get_active_rig_instance()
    bpy.ops.wm.read_homefile(use_empty=True)


def _calibrator(instance) -> DNACalibrator:
    from character_dna import utilities

    component = utilities.get_active_head()
    return DNACalibrator(
        instance=instance,
        linear_modifier=component.linear_modifier,
        file_name="head.dna",
        component_type="head",
        textures=False,
        normals=False,
    )


def test_calibrate_validation_passes_on_untouched_topology(head_only_character):
    valid, _, message, _ = _calibrator(head_only_character).validate_scene()

    assert valid, message


def test_calibrate_validation_rejects_an_added_vertex(head_only_character):
    mesh_object = head_only_character.head_mesh
    mesh = bmesh.new()
    mesh.from_mesh(mesh_object.data)
    mesh.verts.new((0.0, 0.0, 0.0))
    mesh.to_mesh(mesh_object.data)
    mesh.free()

    valid, title, message, _ = _calibrator(head_only_character).validate_scene()

    assert not valid
    assert "Vertex count mismatch" in title
    assert mesh_object.name in message
