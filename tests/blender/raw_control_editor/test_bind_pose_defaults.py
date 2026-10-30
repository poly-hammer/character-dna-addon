"""Tests for the bind-pose ("default" sentinel) constants and predicate.

These tests exercise only the headless parts of the bind-pose plumbing
(constants and :func:`is_default_row`). End-to-end commit tests
require a live Blender session and the Ada DNA -- they are covered
by manual probes in ``scratches/raw-control-editor/``."""

from __future__ import annotations

import pytest

from character_dna.editors.raw_control_editor.constants import (
    DEFAULT_RAW_CONTROL_INDEX,
    DEFAULT_RAW_CONTROL_NAME,
    DEFAULT_RAW_CONTROL_VALUE,
    NEUTRAL_VERTEX_DELTA_THRESHOLD,
)


def test_default_constants_are_distinct() -> None:
    """The sentinel index must be distinct from every valid DNA raw
    control index (which start at 0) and from the IntProperty
    default of -1 (so a stale row isn't mistaken for the sentinel)."""
    assert DEFAULT_RAW_CONTROL_INDEX < -1
    assert DEFAULT_RAW_CONTROL_NAME == "default"
    assert DEFAULT_RAW_CONTROL_VALUE == 0.0
    assert NEUTRAL_VERTEX_DELTA_THRESHOLD > 0.0
    assert NEUTRAL_VERTEX_DELTA_THRESHOLD < 1e-3


@pytest.mark.parametrize(
    ("index", "name", "expected"),
    [
        (DEFAULT_RAW_CONTROL_INDEX, DEFAULT_RAW_CONTROL_NAME, True),
        (0, "CTRL_expressions.browDownL", False),
        (DEFAULT_RAW_CONTROL_INDEX, "CTRL_expressions.browDownL", False),
        (0, DEFAULT_RAW_CONTROL_NAME, False),
        (-1, "", False),
    ],
)
def test_is_default_row(index: int, name: str, expected: bool) -> None:
    """``is_default_row`` requires BOTH the sentinel index and the
    sentinel name -- either alone is rejected so a corrupted row
    can't accidentally trigger the bind-pose commit path."""
    from character_dna.editors.raw_control_editor.utilities import is_default_row

    class _Item:
        def __init__(self, raw_control_index: int, name: str) -> None:
            self.raw_control_index = raw_control_index
            self.name = name

    assert is_default_row(_Item(index, name)) is expected


def test_is_default_row_handles_none() -> None:
    from character_dna.editors.raw_control_editor.utilities import is_default_row

    assert is_default_row(None) is False


class _FakeReader:
    """Minimal RigLogic reader stub for :func:`build_writeback_mask`.

    Models a 4-joint chain ``root(0) -> mid(1) -> leafA(2), leafB(3)``
    with a single LOD0 mesh (index 0) of two vertices, each skinned to
    one leaf. Joint group 0 is driven by raw control 5 and outputs the
    two leaves' cells, so ``joints_driven_by_raw_control(5) == {2, 3}``
    while every other (including the bind-pose sentinel) is undriven.
    """

    def __init__(self) -> None:
        # vertex 0 -> leafA (joint 2), vertex 1 -> leafB (joint 3)
        self._vertex_joint = [2, 3]

    def getJointCount(self) -> int:
        return 4

    def getJointName(self, i: int) -> str:
        return ["FACIAL_C_FacialRoot", "mid", "leafA", "leafB"][i]

    def getJointParentIndex(self, i: int) -> int:
        # root is its own parent (RigLogic convention)
        return [0, 0, 1, 1][i]

    def getVertexPositionXs(self, mesh_index: int) -> list[float]:
        return [0.0, 0.0]

    def getSkinWeightsJointIndices(self, mesh_index: int, v_index: int) -> list[int]:
        return [self._vertex_joint[v_index]]

    def getSkinWeightsValues(self, mesh_index: int, v_index: int) -> list[float]:
        return [1.0]

    def getJointGroupCount(self) -> int:
        return 1

    def getJointGroupInputIndices(self, jg: int) -> list[int]:
        return [5]

    def getJointGroupOutputIndices(self, jg: int) -> list[int]:
        # joints 2 and 3, 9 channels each
        return [2 * 9, 3 * 9]


def _both_leaves_displaced() -> dict[int, list[tuple[float, float, float]]]:
    # 1 cm move on both verts -> both leaves cross the displacement gate.
    return {0: [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]}


def test_build_writeback_mask_bind_pose_skips_driven_clip() -> None:
    """The bind-pose sentinel (negative ``raw_control_index``) is not a
    real raw control, so no joint group drives it. The matcher mask must
    fall back to direct vertex evidence instead of being zeroed by the
    driven-clip -- otherwise the operator reports "updated 0 bones"."""
    from character_dna.editors.raw_control_editor.constants import DEFAULT_RAW_CONTROL_INDEX
    from character_dna.editors.raw_control_editor.core import build_writeback_mask

    reader = _FakeReader()
    matcher_mask, writeback_mask, stats = build_writeback_mask(
        reader, _both_leaves_displaced(), DEFAULT_RAW_CONTROL_INDEX
    )

    # Both displaced leaves are unlocked; their shared ancestors stay locked.
    assert matcher_mask == [False, False, True, True]
    assert writeback_mask == matcher_mask
    assert stats["matcher_unlocked"] == 2
    assert stats["driven_by_raw_control"] == 0


def test_build_writeback_mask_real_control_applies_driven_clip() -> None:
    """A real raw control still gets the driven clip: only joints whose
    cells the control actually writes survive into the matcher mask."""
    from character_dna.editors.raw_control_editor.core import build_writeback_mask

    reader = _FakeReader()
    # Raw control 5 drives leaves 2 and 3 (see stub joint group).
    matcher_mask, _writeback_mask, stats = build_writeback_mask(reader, _both_leaves_displaced(), 5)

    assert matcher_mask == [False, False, True, True]
    assert stats["driven_by_raw_control"] == 2
    assert stats["matcher_unlocked"] == 2


def test_build_writeback_mask_real_control_drops_undriven() -> None:
    """When the displaced leaves are NOT driven by the active control,
    the driven clip zeroes the matcher mask (the original safety
    behaviour the bind-pose path must bypass)."""
    from character_dna.editors.raw_control_editor.core import build_writeback_mask

    reader = _FakeReader()
    # Raw control 99 drives nothing -> driven clip empties the matcher mask.
    matcher_mask, _writeback_mask, stats = build_writeback_mask(reader, _both_leaves_displaced(), 99)

    assert matcher_mask == [False, False, False, False]
    assert stats["driven_by_raw_control"] == 0
    assert stats["matcher_unlocked"] == 0
