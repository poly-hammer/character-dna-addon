"""Tests for the UV-barycentric orphan fallback used by the Raw Control
Editor mirror / flip operators.

The mirror / flip math pairs each vertex with its exact UV mirror partner
(:func:`core.build_uv_vertex_pairs`); vertices with no exact partner
(orphans) are resampled from the source-side delta field via
:func:`character_dna.editors.shared.uv_symmetry.barycentric_mirror_deltas`.
These tests verify:

* the pure barycentric resampler interpolates and reflects an orphan's
  mirrored delta correctly, and is deterministic (idempotent inputs ->
  identical output);
* :func:`core.compute_mirrored_vertex_positions` keeps the exact-pair
  path bit-exact while applying a supplied ``orphan_deltas`` entry;
* :func:`core.compute_flipped_vertex_positions` does the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from character_dna.editors.raw_control_editor import core
from character_dna.editors.shared.uv_symmetry import barycentric_mirror_deltas


VertexSide = core.VertexSide
MirrorDirection = core.MirrorDirection


# ---------------------------------------------------------------------------
# Pure barycentric resampler
# ---------------------------------------------------------------------------


def test_barycentric_mirror_deltas_interpolates_and_reflects():
    """An orphan whose reflected UV is the centroid of a source triangle
    gets the mean of the triangle's deltas with X negated."""
    # Source-side triangle (u > 0.5), three loops -> three vertices.
    layout_uv = np.array([[0.7, 0.0], [0.9, 0.0], [0.8, 0.4]], dtype=np.float64)
    layout_position_index = np.array([0, 1, 2], dtype=np.int64)
    faces = [np.array([0, 1, 2], dtype=np.int64)]

    # Per-vertex delta field (vertex 3 is the orphan, delta unused).
    deltas = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Orphan UV reflects to the triangle centroid (0.8, 0.4/3).
    centroid_v = 0.4 / 3.0
    orphan_uv = np.array([[1.0 - 0.8, centroid_v]], dtype=np.float64)

    result = barycentric_mirror_deltas(layout_uv, layout_position_index, faces, deltas, [3], orphan_uv)

    assert set(result.keys()) == {3}
    # Mean of the three deltas = (2, 20, 200); X negated by the mirror.
    assert result[3] == pytest.approx([-2.0, 20.0, 200.0])


def test_barycentric_mirror_deltas_skips_unmapped():
    """A query whose reflected UV lands outside every source triangle is
    omitted from the result (caller leaves it untouched)."""
    layout_uv = np.array([[0.7, 0.0], [0.9, 0.0], [0.8, 0.4]], dtype=np.float64)
    layout_position_index = np.array([0, 1, 2], dtype=np.int64)
    faces = [np.array([0, 1, 2], dtype=np.int64)]
    deltas = np.zeros((4, 3), dtype=np.float64)

    # Reflected UV (0.95, 0.95) is far outside the source triangle.
    far_uv = np.array([[1.0 - 0.95, 0.95]], dtype=np.float64)
    result = barycentric_mirror_deltas(layout_uv, layout_position_index, faces, deltas, [3], far_uv)
    assert result == {}


def test_barycentric_mirror_deltas_is_deterministic():
    """Identical inputs produce identical output (idempotent resample)."""
    layout_uv = np.array([[0.7, 0.0], [0.9, 0.0], [0.8, 0.4]], dtype=np.float64)
    layout_position_index = np.array([0, 1, 2], dtype=np.int64)
    faces = [np.array([0, 1, 2], dtype=np.int64)]
    deltas = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    orphan_uv = np.array([[1.0 - 0.8, 0.4 / 3.0]], dtype=np.float64)

    first = barycentric_mirror_deltas(layout_uv, layout_position_index, faces, deltas, [3], orphan_uv)
    second = barycentric_mirror_deltas(layout_uv, layout_position_index, faces, deltas, [3], orphan_uv)
    assert first.keys() == second.keys()
    assert first[3] == pytest.approx(list(second[3]))


# ---------------------------------------------------------------------------
# Core mirror with orphan fallback
# ---------------------------------------------------------------------------

# Vertex roles:
#   0 = LEFT  (source)      paired with 1
#   1 = RIGHT (destination) paired with 0
#   2 = RIGHT (destination) ORPHAN (partner == self)
#   3 = CENTER
_NEUTRAL = [
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (-2.0, 5.0, 0.0),
    (0.0, 9.0, 0.0),
]
_PARTNER = [1, 0, 2, 3]
_SIDE = [VertexSide.LEFT, VertexSide.RIGHT, VertexSide.RIGHT, VertexSide.CENTER]


def test_mirror_applies_orphan_delta_and_keeps_pairs_exact():
    """Paired destination vertex mirrors its partner's delta bit-exactly;
    the orphan receives the supplied ``orphan_deltas`` entry."""
    # Sculpt only the source vertex 0 by (0, 1, 2).
    current = [
        (1.0, 1.0, 2.0),  # 0: source moved
        (-1.0, 0.0, 0.0),  # 1: untouched (will be overwritten)
        (-2.0, 5.0, 0.0),  # 2: untouched orphan
        (0.0, 9.0, 0.0),  # 3: center
    ]
    orphan_deltas = {2: (0.5, 0.6, 0.7)}

    out = core.compute_mirrored_vertex_positions(
        _NEUTRAL,
        current,
        _PARTNER,
        _SIDE,
        MirrorDirection.LEFT_TO_RIGHT,
        orphan_deltas=orphan_deltas,
    )

    # Paired vertex 1: neutral + mirror(delta of 0). delta = (0,1,2); X
    # negated -> (0,1,2); new = (-1, 1, 2).
    assert out[1] == pytest.approx([-1.0, 1.0, 2.0])
    # Orphan vertex 2: neutral + orphan_delta.
    assert out[2] == pytest.approx([-1.5, 5.6, 0.7])
    # Source vertex 0 and center vertex 3 untouched.
    assert out[0] == pytest.approx([1.0, 1.0, 2.0])
    assert out[3] == pytest.approx([0.0, 9.0, 0.0])


def test_mirror_without_orphan_deltas_leaves_orphan_untouched():
    """The legacy behavior (no orphan_deltas) is preserved: the orphan
    keeps its current position."""
    current = [
        (1.0, 1.0, 2.0),
        (-1.0, 0.0, 0.0),
        (-2.0, 5.0, 0.0),
        (0.0, 9.0, 0.0),
    ]
    out = core.compute_mirrored_vertex_positions(
        _NEUTRAL,
        current,
        _PARTNER,
        _SIDE,
        MirrorDirection.LEFT_TO_RIGHT,
    )
    assert out[2] == pytest.approx([-2.0, 5.0, 0.0])


def test_mirror_orphan_respects_selection():
    """An orphan outside the selection is not mirrored even when an
    ``orphan_deltas`` entry exists."""
    current = [
        (1.0, 1.0, 2.0),
        (-1.0, 0.0, 0.0),
        (-2.0, 5.0, 0.0),
        (0.0, 9.0, 0.0),
    ]
    out = core.compute_mirrored_vertex_positions(
        _NEUTRAL,
        current,
        _PARTNER,
        _SIDE,
        MirrorDirection.LEFT_TO_RIGHT,
        selected_indices={1},  # vertex 2 (orphan) not selected
        orphan_deltas={2: (0.5, 0.6, 0.7)},
    )
    assert out[2] == pytest.approx([-2.0, 5.0, 0.0])
    assert out[1] == pytest.approx([-1.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Core flip with orphan fallback
# ---------------------------------------------------------------------------


def test_flip_applies_orphan_delta():
    """A flip orphan (no exact partner) receives the supplied resampled
    delta while paired vertices swap their deltas as before."""
    # Sculpt source 0 by (0, 1, 2) and destination 1 by (0, 3, 4).
    current = [
        (1.0, 1.0, 2.0),
        (-1.0, 3.0, 4.0),
        (-2.0, 5.0, 0.0),
        (0.0, 9.0, 0.0),
    ]
    orphan_deltas = {2: (0.5, 0.6, 0.7)}

    out = core.compute_flipped_vertex_positions(
        _NEUTRAL,
        current,
        _PARTNER,
        _SIDE,
        orphan_deltas=orphan_deltas,
    )

    # Pair (0, 1) swaps deltas with X negated.
    #   delta_0 = (0,1,2) -> X negated (0,1,2) -> applied to vertex 1
    #   delta_1 = (0,3,4) -> X negated (0,3,4) -> applied to vertex 0
    assert out[0] == pytest.approx([1.0, 3.0, 4.0])
    assert out[1] == pytest.approx([-1.0, 1.0, 2.0])
    # Orphan vertex 2: neutral + orphan_delta.
    assert out[2] == pytest.approx([-1.5, 5.6, 0.7])
    # Center vertex 3 untouched (no delta sculpted).
    assert out[3] == pytest.approx([0.0, 9.0, 0.0])


def test_flip_without_orphan_deltas_leaves_orphan_untouched():
    """Legacy flip behavior preserved when no orphan_deltas supplied."""
    current = [
        (1.0, 1.0, 2.0),
        (-1.0, 3.0, 4.0),
        (-2.0, 5.0, 0.0),
        (0.0, 9.0, 0.0),
    ]
    out = core.compute_flipped_vertex_positions(_NEUTRAL, current, _PARTNER, _SIDE)
    assert out[2] == pytest.approx([-2.0, 5.0, 0.0])
