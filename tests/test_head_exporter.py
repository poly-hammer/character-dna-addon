import pytest

from mathutils import Euler, Vector

from constants import (
    CI_EXPORT_MESH_NAMES,
    HEAD_DNA_FILE,
    IGNORED_BONE_ROTATIONS_ON_EXPORT,
    NORMAL_ROUND_TRIP_BOUNDS,
    TOLERANCE,
)
from utilities.assertions import assert_bone_definitions, assert_mesh_corner_normals, assert_mesh_geometry
from utilities.dna_data import get_mesh_names, get_test_bone_definitions_params, get_test_mesh_geometry_params


@pytest.mark.parametrize(
    ("bone_name", "attribute", "axis_name"), get_test_bone_definitions_params(dna_file_path=HEAD_DNA_FILE)
)
def test_bone_definitions(
    original_head_dna_json_data,
    exported_head_dna_json_data,
    bone_name: str,
    attribute: str,
    axis_name: str,
    changed_head_bone_name: str,
    changed_head_bone_rotation: tuple[Euler, Euler],
    changed_head_bone_location: tuple[Vector, Vector],
):
    assert_bone_definitions(
        expected_data=original_head_dna_json_data,
        current_data=exported_head_dna_json_data,
        bone_name=bone_name,
        attribute=attribute,
        axis_name=axis_name,
        changed_bone_name=changed_head_bone_name,
        changed_bone_rotation=changed_head_bone_rotation,
        changed_bone_location=changed_head_bone_location,
        tolerance=TOLERANCE[attribute],
        output_method="export",
        ignored_bones=IGNORED_BONE_ROTATIONS_ON_EXPORT,
    )


@pytest.mark.parametrize(
    ("mesh_name", "attribute", "axis_name"),
    get_test_mesh_geometry_params(
        lods=[0],
        vertex_positions=True,
        normals=False,
        uvs=True,
        dna_file_path=HEAD_DNA_FILE,
        ci_required=CI_EXPORT_MESH_NAMES,
    ),
)
def test_mesh_geometry(
    original_head_dna_json_data,
    exported_head_dna_json_data,
    mesh_name: str,
    attribute: str,
    axis_name: str,
    changed_head_mesh_name: int,
    changed_head_vertex_index: int,
    changed_head_vertex_location: tuple[Vector, Vector, Vector],
):
    assert_mesh_geometry(
        expected_data=original_head_dna_json_data,
        current_data=exported_head_dna_json_data,
        mesh_name=mesh_name,
        attribute=attribute,
        axis_name=axis_name,
        changed_mesh_name=changed_head_mesh_name,
        changed_vertex_index=changed_head_vertex_index,
        changed_vertex_location=changed_head_vertex_location,
        assert_mesh_indices=False,
        assert_index_order=False,
        tolerance=TOLERANCE[attribute],
        output_method="export",
    )


@pytest.mark.skip(reason="`import_normals` has no UI control, so no imported mesh carries the DNA's normals")
@pytest.mark.parametrize(
    "mesh_name", [name for name in get_mesh_names(HEAD_DNA_FILE) if name.endswith("_lod0_mesh")]
)
def test_mesh_normals(
    original_head_dna_json_data,
    exported_head_dna_json_data,
    mesh_name: str,
    changed_head_mesh_name: str,
    changed_head_normal_index: int,
    changed_head_normal_vector: tuple[Vector, Vector, Vector],
    changed_head_normal_neighbours: list[int],
):
    """The export rewrites the layouts, so normals are compared where they still mean the same
    thing: the normal each face corner resolves to."""
    assert_mesh_corner_normals(
        expected_data=original_head_dna_json_data,
        current_data=exported_head_dna_json_data,
        mesh_name=mesh_name,
        changed_mesh_name=changed_head_mesh_name,
        changed_normal_index=changed_head_normal_index,
        changed_normal_neighbours=changed_head_normal_neighbours,
        bounds=NORMAL_ROUND_TRIP_BOUNDS,
    )
