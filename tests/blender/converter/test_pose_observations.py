"""Sparse head observations retain dense skinning and hierarchy validation."""

import numpy as np
import pytest

from character_dna.editors.converter.pose.observations import pool_weights


def test_sparse_head_pooling_matches_dense_weights_and_joint_order():
    names = ["root", "neck", "face", "tip", "unused"]
    parents = [-1, 0, 1, 2, -1]
    groups = {
        "root": [(0, 0.25), (1, 0.2)],
        "neck": [(0, 0.5)],
        "face": [(0, 0.1), (0, 0.125), (1, 0.8)],
        "tip": [(0, 0.125), (2, 1.0)],
    }
    dense = np.zeros((3, len(names)))
    for column, name in enumerate(names):
        for vertex, weight in groups.get(name, []):
            dense[vertex, column] = weight
    expected = [[0.75, 0.25], [0.8, 0.2], [1.0, 0.0]]
    for buffers in ({"groups": groups}, {"weights": dense}):
        actual = pool_weights(3, names, parents, ["neck", "root"], **buffers)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_sparse_head_pooling_rejects_invalid_influences_before_aggregation(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        pool_weights(1, ["root", "face"], [-1, 0], ["root"], groups={"root": [(0, 1)], "face": [(0, value)]})


def test_sparse_head_pooling_rejects_cycles_and_unowned_weights():
    with pytest.raises(ValueError, match="Cyclic"):
        pool_weights(1, ["a", "b"], [1, 0], ["root"], groups={"a": [(0, 1)]})
    with pytest.raises(ValueError, match="no anatomical owner"):
        pool_weights(1, ["face"], [-1], ["root"], groups={"face": [(0, 1)]})
