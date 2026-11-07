"""Array boundary for native DNA neckline topology, independent of Pro editors."""

from typing import Any

import numpy as np

from ..bindings import load_converter_core


def ordered_boundary(faces: list, vertices: set[int]) -> list[int]:
    """Capture polygon buffers and order the declared neckline in Rust."""
    corners = np.asarray([index for face in faces for index in face], dtype=np.int32)
    sizes = np.asarray([len(face) for face in faces], dtype=np.int32)
    count = max(max(vertices, default=-1), int(corners.max(initial=-1))) + 1
    return load_converter_core().topology_ordered_boundary(corners, sizes, count, vertices)


def match_boundaries(
    head: list[int], body: list[int], head_points: Any, body_points: Any, *, tolerance: float
) -> dict[int, int]:
    """Match captured component boundaries without retaining Blender objects."""
    return load_converter_core().topology_match_boundaries(
        head,
        body,
        np.ascontiguousarray(head_points, dtype=np.float64),
        np.ascontiguousarray(body_points, dtype=np.float64),
        tolerance,
    )
