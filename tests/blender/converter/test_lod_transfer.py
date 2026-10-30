"""Coverage for the shared LOD optimizations used during conversion."""

import numpy as np
import pytest

from character_dna.editors.shared.lod_propagation import _write_propagated_normals
from character_dna.editors.shared.uv_barycentric import (
    SourceUVMapping,
    barycentric_coords,
    deduplicate_texture_coordinates,
    solve_uv_barycentric_transfer,
)


def _dense_overlap_reference(mapping, point, boxes):
    """Independent original full scan, including its first-index tie rule."""
    if mapping.tri_count == 0 or not boxes:
        return -1
    from character_dna.editors.shared.uv_barycentric import CONTAINMENT_EPS

    mask = np.zeros(mapping.tri_count, dtype=bool)
    for lower, upper in boxes:
        mask |= (
            (mapping.bb_max[:, 0] >= lower[0])
            & (upper[0] >= mapping.bb_min[:, 0])
            & (mapping.bb_max[:, 1] >= lower[1])
            & (upper[1] >= mapping.bb_min[:, 1])
        )
    candidates = np.nonzero(mask)[0]
    if not candidates.size:
        return -1
    weights = barycentric_coords(mapping.tri_uv[candidates], np.broadcast_to(point, (len(candidates), 2)))
    contains = np.isfinite(weights).all(axis=1) & (weights >= -CONTAINMENT_EPS).all(axis=1)
    if contains.any():
        return int(candidates[contains].min())
    lower, upper = mapping.bb_min[candidates], mapping.bb_max[candidates]
    dx = np.maximum.reduce([lower[:, 0] - point[0], np.zeros(len(candidates)), point[0] - upper[:, 0]])
    dy = np.maximum.reduce([lower[:, 1] - point[1], np.zeros(len(candidates)), point[1] - upper[:, 1]])
    return int(candidates[np.argmin(np.sqrt(dx * dx + dy * dy))])


@pytest.mark.parametrize("layout", ["grid", "overlapping", "empty"])
def test_indexed_uv_fallback_matches_full_scan_at_boundaries_and_outside(layout):
    rng = np.random.default_rng(846)
    if layout == "grid":
        triangles = np.array([[(x, y), (x + 0.75, y), (x, y + 0.75)] for x in range(12) for y in range(12)])
        triangles = np.concatenate((triangles, triangles[:2]))  # Deliberate ties.
    elif layout == "overlapping":
        triangles = rng.uniform(-2, 12, size=(180, 3, 2))
    else:
        triangles = np.empty((0, 3, 2))
    uv = triangles.reshape((-1, 2))
    faces = list(np.arange(len(uv)).reshape((-1, 3)))
    mapping = SourceUVMapping(uv, np.arange(len(uv)), faces)
    samples = [np.array((0, 0)), np.array((0.75, 0)), np.array((-5, 20)), *rng.uniform(-3, 15, size=(80, 2))]
    if mapping.tri_count:
        samples.extend((mapping.bb_min[0], mapping.bb_max[0]))
    for point in samples:
        for boxes in (
            [],
            [(point - 0.1, point + 0.1)],
            [(point - 0.1, point + 0.1), (np.zeros(2), np.ones(2))],
            [(np.full(2, -20), np.full(2, 20))],
            [(np.full(2, -1e308), np.full(2, 1e308))],
            [(np.ones(2), np.zeros(2))],
            [(np.full(2, -np.inf), np.full(2, np.inf))],
            [(np.full(2, np.nan), np.ones(2))],
        ):
            assert mapping.nearest_overlapping_triangle(point, boxes) == _dense_overlap_reference(mapping, point, boxes)


def test_mirror_deduplication_matches_dense_reference():
    rng = np.random.default_rng(19)
    half = rng.uniform(size=(200, 2))
    other = half.copy()
    other[30:50] += 0.01
    uv = np.concatenate((half, other))
    expected = uv[:, 0].copy()
    match = (np.abs(half[:, None, :] - other[None, :, :]) < 0.0002).all(axis=2).any(axis=1)
    expected[:200][match] += 1
    np.testing.assert_array_equal(deduplicate_texture_coordinates(uv[:, 0], uv[:, 1]), expected)


def test_reused_uv_index_produces_identical_transfer():
    uv = np.array(((0, 0), (1, 0), (0, 1)), dtype=float)
    indices = np.arange(3)
    faces = [indices]
    source = np.column_stack((uv, np.ones(3)))
    target = uv * 0.5
    args = (uv, indices, faces, source, target, indices, faces, 3, np.zeros((3, 3)))
    mapping = SourceUVMapping(uv, indices, faces)
    np.testing.assert_array_equal(
        solve_uv_barycentric_transfer(*args), solve_uv_barycentric_transfer(*args, source_mapping=mapping)
    )


def test_lower_lod_normals_follow_new_geometry():
    class Reader:
        def getVertexNormalXs(self, _index):
            return [0.0] * 3

        def getVertexNormalYs(self, _index):
            return [0.0] * 3

        def getVertexNormalZs(self, _index):
            return [1.0] * 3

        def getVertexLayoutPositionIndices(self, _index):
            return [0, 1, 2]

        def getVertexLayoutNormalIndices(self, _index):
            return [0, 1, 2]

        def getFaceCount(self, _index):
            return 1

        def getFaceVertexLayoutIndices(self, _index, _face):
            return [0, 1, 2]

    class Writer:
        def setVertexNormals(self, **kwargs):
            self.normals = kwargs["normals"]

    writer = Writer()
    _write_propagated_normals(Reader(), writer, 0, np.array(((0, 0, 0), (1, 0, 0), (0, 0, 1))))
    np.testing.assert_array_equal(writer.normals, ((0, -1, 0),) * 3)
