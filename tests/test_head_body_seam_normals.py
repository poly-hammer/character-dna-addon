"""The DNA's own normals must reach the mesh, and must agree where the head meets the body.

The head and body DNAs store the same normal at every vertex the two meshes share. Normals
Blender derives from the faces instead disagree by several degrees there, because each mesh
averages only its own faces, and that shows as a shading crease along the neck.
"""

# standard library imports
import math

# third party imports
import bpy
import pytest

# local imports
from constants import BODY_DNA_FILE, HEAD_DNA_FILE

pytestmark = pytest.mark.skip(
    reason="`import_normals` has no UI control, so no imported mesh carries the DNA's normals"
)

SEAM_TOLERANCE_DEGREES = 0.5
"""Anything under this is invisible; the derived normals disagreed by a mean of 4.4 degrees."""


@pytest.fixture(scope="module")
def imported_skin(addon):
    bpy.ops.wm.read_homefile(use_empty=True)
    for file_path in (BODY_DNA_FILE, HEAD_DNA_FILE):
        bpy.ops.character_dna.import_dna(filepath=str(file_path), include_body=False)

    meshes = {mesh_object.name: mesh_object for mesh_object in bpy.data.objects if mesh_object.type == "MESH"}
    head = next(value for name, value in meshes.items() if "head_lod0" in name)
    body = next(value for name, value in meshes.items() if "body_lod0" in name)
    yield head, body


def corner_normal(mesh: bpy.types.Mesh, loop_index: int):
    return mesh.corner_normals[loop_index].vector.normalized()


def test_the_dna_normals_reach_the_mesh(imported_skin):
    head, body = imported_skin

    assert head.data.has_custom_normals, "The head lost the DNA's normals"
    assert body.data.has_custom_normals, "The body lost the DNA's normals"


def test_the_normals_are_unit_length(imported_skin):
    """The DNA stores directions, so the linear unit must not be applied to them."""
    head, _ = imported_skin

    lengths = [head.data.corner_normals[index].vector.length for index in range(0, len(head.data.loops), 997)]

    assert all(math.isclose(length, 1.0, abs_tol=1e-3) for length in lengths)


def test_the_head_and_body_agree_at_the_seam(imported_skin):
    head, body = imported_skin
    head_mesh, body_mesh = head.data, body.data

    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in head_mesh.polygons:
        for key in polygon.edge_keys:
            edge_counts[key] = edge_counts.get(key, 0) + 1
    border = {index for key, count in edge_counts.items() if count == 1 for index in key}
    assert border, "The head mesh has no open border, so there is no seam to check"

    body_loop_by_vertex: dict[int, int] = {}
    for polygon in body_mesh.polygons:
        for loop_index in polygon.loop_indices:
            body_loop_by_vertex.setdefault(body_mesh.loops[loop_index].vertex_index, loop_index)

    body_points = [body.matrix_world @ vertex.co for vertex in body_mesh.vertices]

    worst = 0.0
    compared = 0
    for polygon in head_mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = head_mesh.loops[loop_index].vertex_index
            if vertex_index not in border:
                continue
            point = head.matrix_world @ head_mesh.vertices[vertex_index].co
            nearest, distance = min(
                ((index, (candidate - point).length) for index, candidate in enumerate(body_points)),
                key=lambda entry: entry[1],
            )
            if distance > 1e-3 or nearest not in body_loop_by_vertex:
                continue
            angle = corner_normal(head_mesh, loop_index).angle(corner_normal(body_mesh, body_loop_by_vertex[nearest]))
            worst = max(worst, math.degrees(angle))
            compared += 1

    assert compared, "No coincident seam vertices were found"
    assert worst < SEAM_TOLERANCE_DEGREES, f"The seam normals disagree by up to {worst:.2f} degrees"
