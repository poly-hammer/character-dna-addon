import pytest

from mathutils import Euler, Vector

from constants import (
    CI_CALIBRATE_MESH_NAMES,
    CI_SKIN_WEIGHT_MESH_NAMES,
    HEAD_DNA_FILE,
    IGNORED_BONE_ROTATIONS_ON_CALIBRATE,
    TOLERANCE,
)
from utilities.assertions import (
    assert_bone_behaviors,
    assert_bone_definitions,
    assert_mesh_geometry,
    assert_skin_weights,
)
from utilities.dna_data import (
    get_test_bone_behaviors_params,
    get_test_bone_definitions_params,
    get_test_mesh_geometry_params,
    get_test_skin_weights_params,
)


@pytest.mark.parametrize(
    ("bone_name", "attribute", "axis_name"), get_test_bone_definitions_params(dna_file_path=HEAD_DNA_FILE)
)
def test_bone_definitions(
    original_head_dna_json_data,
    calibrated_head_dna_json_data,
    bone_name: str,
    attribute: str,
    axis_name: str,
    changed_head_bone_name: str,
    changed_head_bone_rotation: tuple[Euler, Euler],
    changed_head_bone_location: tuple[Vector, Vector],
):
    assert_bone_definitions(
        expected_data=original_head_dna_json_data,
        current_data=calibrated_head_dna_json_data,
        bone_name=bone_name,
        attribute=attribute,
        axis_name=axis_name,
        changed_bone_name=changed_head_bone_name,
        changed_bone_rotation=changed_head_bone_rotation,
        changed_bone_location=changed_head_bone_location,
        output_method="calibrate",
        ignored_bones=IGNORED_BONE_ROTATIONS_ON_CALIBRATE,
    )


@pytest.mark.parametrize("bone_name", get_test_bone_behaviors_params(dna_file_path=HEAD_DNA_FILE))
def test_bone_behaviors(original_head_dna_json_data, calibrated_head_dna_json_data, bone_name: str):
    assert_bone_behaviors(
        expected_data=original_head_dna_json_data, current_data=calibrated_head_dna_json_data, bone_name=bone_name
    )


@pytest.mark.parametrize(
    ("mesh_name", "attribute", "axis_name"),
    get_test_mesh_geometry_params(
        vertex_positions=True,
        # Off while `import_normals` has no UI control, so no imported mesh carries the DNA's normals.
        normals=False,
        uvs=True,
        dna_file_path=HEAD_DNA_FILE,
        ci_required=CI_CALIBRATE_MESH_NAMES,
    ),
)
def test_mesh_geometry(
    original_head_dna_json_data,
    calibrated_head_dna_json_data,
    mesh_name: str,
    attribute: str,
    axis_name: str,
    changed_head_mesh_name: int,
    changed_head_vertex_index: int,
    changed_head_vertex_location: tuple[Vector, Vector, Vector],
    changed_head_normal_index: int,
    changed_head_normal_vector: tuple[Vector, Vector, Vector],
    changed_head_normal_neighbours: list[int],
    changed_head_lower_lod_vertices: list[dict],
):
    assert_mesh_geometry(
        expected_data=original_head_dna_json_data,
        current_data=calibrated_head_dna_json_data,
        mesh_name=mesh_name,
        attribute=attribute,
        axis_name=axis_name,
        changed_mesh_name=changed_head_mesh_name,
        changed_vertex_index=changed_head_vertex_index,
        changed_vertex_location=changed_head_vertex_location,
        changed_normal_index=changed_head_normal_index,
        changed_normal_vector=changed_head_normal_vector,
        changed_normal_neighbours=changed_head_normal_neighbours,
        lower_lod_vertices=changed_head_lower_lod_vertices,
        tolerance=TOLERANCE[attribute],
        assert_mesh_indices=True,
        output_method="calibrate",
    )


@pytest.mark.parametrize("head_lod_index", [0, 1, 2, 3, 4, 5, 6, 7])
def test_lower_lod_seam_aligns_to_body(
    calibrated_head_and_body_dna_json_data: dict,
    head_lod_index: int,
):
    """The neck edge-loop vertices on every head mesh must match the *exported*
    body DNA exactly (within DNA float precision) when ``align_head_and_body`` is
    enabled, so the head and body share a seam with no drift.

    Both components are exported through the calibrator with ``auto_update_lods``
    on, which regenerates every lower-LOD mesh via the UV-barycentric solver. The
    head and body lower LODs are propagated independently, so the exported seam only
    lines up when the head conforms onto the *exported* body (not the imported
    template). This test compares the two exported DNAs against each other so it
    fails if the head is snapped to the wrong (stale) body geometry."""
    from character_dna.constants import HEAD_TO_BODY_LOD_MAPPING
    from character_dna.utilities import get_head_to_body_edge_loop_mapping
    from constants import DNA_DEFINITION_VERSION, DNA_GEOMETRY_VERSION

    head_dna_json_data = calibrated_head_and_body_dna_json_data["head"]
    body_dna_json_data = calibrated_head_and_body_dna_json_data["body"]

    body_lod_index = HEAD_TO_BODY_LOD_MAPPING[head_lod_index]
    head_mesh_name = f"head_lod{head_lod_index}_mesh"
    body_mesh_name = f"body_lod{body_lod_index}_mesh"

    head_mesh_index = head_dna_json_data[DNA_DEFINITION_VERSION]["meshNames"].index(head_mesh_name)
    body_mesh_index = body_dna_json_data[DNA_DEFINITION_VERSION]["meshNames"].index(body_mesh_name)

    head_positions = head_dna_json_data[DNA_GEOMETRY_VERSION]["meshes"][head_mesh_index]["positions"]
    body_positions = body_dna_json_data[DNA_GEOMETRY_VERSION]["meshes"][body_mesh_index]["positions"]

    edge_loop_mapping = get_head_to_body_edge_loop_mapping().get(str(head_lod_index), {})
    assert edge_loop_mapping, f"No head-to-body edge-loop mapping defined for head LOD {head_lod_index}."

    for raw_head_vertex_index, raw_body_vertex_index in edge_loop_mapping.items():
        head_vertex_index = int(raw_head_vertex_index)
        body_vertex_index = int(raw_body_vertex_index)
        for axis in ("x", "y", "z"):
            head_value = head_positions[f"{axis}s"][head_vertex_index]
            body_value = body_positions[f"{axis}s"][body_vertex_index]
            assert head_value == pytest.approx(body_value, abs=1e-3), (
                f"Seam drift on {head_mesh_name} vertex {head_vertex_index} ({axis}): head {head_value} "
                f"!= body {body_mesh_name} vertex {body_vertex_index} {body_value}."
            )


@pytest.mark.parametrize(
    ("mesh_name", "attribute", "mesh_vertex_count"),
    get_test_skin_weights_params(dna_file_path=HEAD_DNA_FILE, ci_required=CI_SKIN_WEIGHT_MESH_NAMES),
)
def test_skin_weights(
    original_head_dna_json_data,
    calibrated_head_dna_json_data,
    mesh_name: str,
    attribute: str,
    mesh_vertex_count: int,
    changed_head_mesh_name: int,
    changed_head_vertex_group_name: str,
    changed_head_vertex_group_vertex_index: int,
    changed_head_vertex_group_weight: float,
):
    assert_skin_weights(
        expected_data=original_head_dna_json_data,
        current_data=calibrated_head_dna_json_data,
        mesh_name=mesh_name,
        mesh_vertex_count=mesh_vertex_count,
        attribute=attribute,
        changed_mesh_name=changed_head_mesh_name,
        changed_vertex_group_name=changed_head_vertex_group_name,
        changed_vertex_group_vertex_index=changed_head_vertex_group_vertex_index,
        changed_vertex_group_weight=changed_head_vertex_group_weight,
        tolerance=TOLERANCE[attribute],
    )
