"""Coverage for the shared LOD optimizations used during conversion."""

import numpy as np

from character_dna.editors.shared.lod_propagation import _write_propagated_normals
from character_dna.editors.shared.uv_barycentric import (
    SourceUVMapping,
    deduplicate_texture_coordinates,
    solve_uv_barycentric_transfer,
)


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
