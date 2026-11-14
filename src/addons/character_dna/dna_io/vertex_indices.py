"""Explicit correspondence between visible mesh vertices and DNA position slots."""

from typing import Any


def scene_to_dna(mesh: Any, position_count: int, faces: list[list[int]]) -> list[int]:
    """Validate canonical faces and resolve full or compact imported geometry.

    DNA may retain unreferenced positions anywhere in its arrays. Blender's
    importer omits them, but calibration must preserve their slots and values.
    """
    used = sorted({vertex for face in faces for vertex in face})
    count = len(mesh.vertices)
    if count == position_count:
        indices = list(range(count))
    elif count == len(used):
        indices = used
    else:
        raise ValueError(
            f"{mesh.name} has {count} vertices; DNA has {position_count} positions ({len(used)} referenced)"
        )
    if len(mesh.polygons) != len(faces):
        raise ValueError(f"{mesh.name} has changed DNA face topology")
    for polygon, expected in zip(mesh.polygons, faces, strict=True):
        actual = [indices[v] for v in polygon.vertices]
        if len(actual) != len(expected) or not any(
            actual == expected[offset:] + expected[:offset] for offset in range(len(expected))
        ):
            raise ValueError(f"{mesh.name} has changed DNA face topology at face {polygon.index}")
    return indices


def for_mesh(reader: Any, index: int, mesh: Any) -> list[int]:
    from .mesh_data import reader_faces

    return scene_to_dna(mesh, reader.getVertexPositionCount(index), reader_faces(reader, index))
