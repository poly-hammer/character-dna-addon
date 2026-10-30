"""Tests for the pure-DNA RBF backward solve (``editors/shared/rbf_solve.py``).

``resolve_blend_shape_rbf_targets`` maps a blend-shape channel name to the RBF
solver pose(s) whose stored neck/head quaternion drives that channel to ~1.0,
using ONLY the DNA RBF behavior (solver raw-control values + pose output weights)
and the PSD matrix. It handles two cases:

* a directly RBF-driven channel (``head_turnUp_U``) whose driving control IS an
  RBF output control, and
* a PSD combination corrective (``HturnUp_NKstretch_UL``) whose PSD row has an RBF
  output control among its product columns.

The fast unit tests use a hand-built fake reader that reproduces the head solver's
multi-driver layout (3 drivers = 12 values/pose, pose-major then driver-major). A
guarded real-rig test loads the shipped Ada head DNA and pins the head-turn solve
when the DNA actually ships the RBF solver.
"""

from __future__ import annotations

import pytest

from character_dna.editors.shared.rbf_solve import (
    get_rbf_pose_target,
    resolve_blend_shape_rbf_targets,
    resolve_corrective_rbf_targets,
)


# Three driver bones, four quaternion components each -> raw controls 251..262.
_NECK_RAW = {
    251: "neck_01.qx",
    252: "neck_01.qy",
    253: "neck_01.qz",
    254: "neck_01.qw",
    255: "neck_02.qx",
    256: "neck_02.qy",
    257: "neck_02.qz",
    258: "neck_02.qw",
    259: "head.qx",
    260: "head.qy",
    261: "head.qz",
    262: "head.qw",
}

# Per-pose driver quaternions in DNA (qx, qy, qz, qw) order, pose-major/driver-major.
_POSE_VALUES = {
    0: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # default / identity
    1: [0, 0, 0.0877, 0.9961, 0, 0, 0.1743, 0.9847, 0, 0, 0.2152, 0.9766],  # turnUp
    2: [0, 0, -0.0877, 0.9961, 0, 0, -0.1743, 0.9847, 0, 0, -0.2152, 0.9766],  # turnDown
    7: [0, 0, 0.04, 0.999, 0, 0, 0.08, 0.997, 0, 0, 0.10, 0.995],  # combination
}
_POSE_ORDER = [0, 1, 2, 7]

# Each pose's RBF output controls -> weights. Output control 808 (head_turnUp_U)
# peaks at the clean pose 1 (weight 1.0); the combination pose 7 only fires it 0.5.
_POSE_OUTPUTS = {
    0: ([], []),
    1: ([808], [1.0]),
    2: ([811], [1.0]),
    7: ([808], [0.5]),
}

# Blend-shape channel -> driving control (raw control or PSD).
_CHANNEL_NAMES = {
    675: "head_turnUp_U",
    693: "HturnUp_NKstretch_UL",
    0: "jaw_open",
}
_CHANNEL_OUTPUT_INDICES = [675, 693, 0]
_CHANNEL_INPUT_INDICES = [808, 300, 50]  # 808 = RBF out, 300 = PSD, 50 = plain raw

# PSD row 300 = head_turnUp_U (RBF out 808) x neckStretchL (raw 205).
_PSD_ROWS = [300, 300]
_PSD_COLUMNS = [808, 205]


class _FakeRBFReader:
    """Minimal stand-in for the riglogic reader surface used by ``rbf_solve``."""

    # -- RBF solver / pose ---------------------------------------------------
    def getRBFSolverCount(self) -> int:
        return 1

    def getRBFSolverRawControlIndices(self, _solver_index: int) -> list[int]:
        return list(_NECK_RAW.keys())

    def getRBFSolverRawControlValues(self, _solver_index: int) -> list[float]:
        return [v for pose in _POSE_ORDER for v in _POSE_VALUES[pose]]

    def getRBFSolverPoseIndices(self, _solver_index: int) -> list[int]:
        return list(_POSE_ORDER)

    def getRBFSolverDistanceMethod(self, _solver_index: int) -> int:
        return 1  # Quaternion

    def getRBFSolverTwistAxis(self, _solver_index: int) -> int:
        return 0  # X

    def getRBFPoseName(self, pose_index: int) -> str:
        # Authored pose names (direction + angle), as the real head solver stores.
        return {0: "default", 1: "turnUp_55", 2: "turnDown_50", 7: "fwd"}.get(pose_index, f"pose_{pose_index}")

    def getRBFPoseScale(self, _pose_index: int) -> float:
        return 1.0

    def getRBFPoseOutputControlIndices(self, pose_index: int) -> list[int]:
        return list(_POSE_OUTPUTS[pose_index][0])

    def getRBFPoseOutputControlWeights(self, pose_index: int) -> list[float]:
        return list(_POSE_OUTPUTS[pose_index][1])

    # -- raw controls --------------------------------------------------------
    def getRawControlName(self, index: int) -> str:
        return _NECK_RAW.get(index, f"CTRL_expressions.ctrl_{index}")

    # -- blend-shape channels ------------------------------------------------
    def getBlendShapeChannelCount(self) -> int:
        return 700

    def getBlendShapeChannelName(self, index: int) -> str:
        return _CHANNEL_NAMES.get(index, "")

    def getBlendShapeChannelOutputIndices(self) -> list[int]:
        return list(_CHANNEL_OUTPUT_INDICES)

    def getBlendShapeChannelInputIndices(self) -> list[int]:
        return list(_CHANNEL_INPUT_INDICES)

    # -- PSD -----------------------------------------------------------------
    def getPSDRowIndices(self) -> list[int]:
        return list(_PSD_ROWS)

    def getPSDColumnIndices(self) -> list[int]:
        return list(_PSD_COLUMNS)


# ===========================================================================
# get_rbf_pose_target -- multi-driver layout decode
# ===========================================================================
def test_get_rbf_pose_target_reads_three_drivers() -> None:
    target = get_rbf_pose_target(_FakeRBFReader(), 0, 1)
    assert target is not None
    assert target.pose_index == 1
    assert target.distance_method == "Quaternion"
    assert target.twist_axis == "X"
    assert [d.bone_name for d in target.drivers] == ["neck_01", "neck_02", "head"]
    assert target.drivers[0].raw_control_indices == (251, 252, 253, 254)
    assert target.drivers[2].raw_control_indices == (259, 260, 261, 262)
    # DNA (qx, qy, qz, qw) -> Blender (w, x, y, z).
    assert target.drivers[0].quaternion_wxyz == pytest.approx((0.9961, 0.0, 0.0, 0.0877))
    assert target.drivers[2].quaternion_wxyz == pytest.approx((0.9766, 0.0, 0.0, 0.2152))


def test_get_rbf_pose_target_default_pose_is_identity() -> None:
    target = get_rbf_pose_target(_FakeRBFReader(), 0, 0)
    assert target is not None
    for driver in target.drivers:
        assert driver.quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_get_rbf_pose_target_unknown_pose_returns_none() -> None:
    assert get_rbf_pose_target(_FakeRBFReader(), 0, 99) is None


# ===========================================================================
# resolve_blend_shape_rbf_targets -- channel -> RBF pose target
# ===========================================================================
def test_direct_rbf_channel_resolves_to_max_weight_pose() -> None:
    targets = resolve_blend_shape_rbf_targets(_FakeRBFReader(), "head_turnUp_U")
    assert len(targets) == 1
    # Clean single-direction pose 1 (weight 1.0), not the combination pose 7 (0.5).
    assert targets[0].pose_index == 1
    assert targets[0].drivers[0].quaternion_wxyz == pytest.approx((0.9961, 0.0, 0.0, 0.0877))


def test_psd_corrective_resolves_via_rbf_output_column() -> None:
    targets = resolve_blend_shape_rbf_targets(_FakeRBFReader(), "HturnUp_NKstretch_UL")
    assert len(targets) == 1
    assert targets[0].pose_index == 1


def test_non_rbf_channel_returns_empty() -> None:
    assert resolve_blend_shape_rbf_targets(_FakeRBFReader(), "jaw_open") == []


def test_unknown_channel_returns_empty() -> None:
    assert resolve_blend_shape_rbf_targets(_FakeRBFReader(), "does_not_exist") == []


def test_none_reader_returns_empty() -> None:
    assert resolve_blend_shape_rbf_targets(None, "head_turnUp_U") == []


# ===========================================================================
# resolve_corrective_rbf_targets -- PSD corrective `_tgt` target -> RBF pose
# ===========================================================================
def test_corrective_target_resolves_via_pose_name() -> None:
    # ``head_turnUp_tgt`` is not a channel; its stem matches the RBF pose
    # ``turnUp_55`` (base ``turnUp``), which stores the neck/head quaternion.
    targets = resolve_corrective_rbf_targets(_FakeRBFReader(), "head_turnUp_tgt")
    assert len(targets) == 1
    assert targets[0].pose_index == 1


def test_translation_corrective_resolves_via_pose_name() -> None:
    # ``head_fwd_tgt`` has no blend-shape channel of its own, but the RBF solver
    # owns a pose literally named ``fwd`` -- the deterministic, name-based link.
    targets = resolve_corrective_rbf_targets(_FakeRBFReader(), "head_fwd_tgt")
    assert len(targets) == 1
    assert targets[0].pose_index == 7


def test_corrective_target_exact_channel_name_still_resolves() -> None:
    targets = resolve_corrective_rbf_targets(_FakeRBFReader(), "head_turnUp_U")
    assert len(targets) == 1
    assert targets[0].pose_index == 1


def test_corrective_with_no_matching_pose_returns_empty() -> None:
    # No RBF pose named ``back`` in the fake solver -> no target.
    assert resolve_corrective_rbf_targets(_FakeRBFReader(), "head_back_tgt") == []


def test_corrective_none_reader_returns_empty() -> None:
    assert resolve_corrective_rbf_targets(None, "head_turnUp_tgt") == []


# ===========================================================================
# Real-rig pin -- the shipped Ada head DNA (skipped if it ships no RBF solver)
# ===========================================================================
def test_real_ada_dna_head_turn_solve() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    if int(reader.getRBFSolverCount()) == 0:
        pytest.skip("shipped Ada head DNA has no RBF solver")

    # Find any head-turn channel the DNA actually exposes.
    channel_count = int(reader.getBlendShapeChannelCount())
    names = {str(reader.getBlendShapeChannelName(i)) for i in range(channel_count)}
    head_turn = next((n for n in ("head_turnUp_M", "head_turnUp_U", "head_tiltLeft_M") if n in names), None)
    if head_turn is None:
        pytest.skip("shipped Ada head DNA exposes no head-turn channel")

    targets = resolve_blend_shape_rbf_targets(reader, head_turn)
    assert targets, f"{head_turn!r} should resolve to an RBF pose target"
    target = targets[0]
    assert target.distance_method == "Quaternion"
    # Drivers are the neck/head quaternion bones, full quaternion (4 components each).
    bones = {d.bone_name for d in target.drivers}
    assert bones <= {"neck_01", "neck_02", "head"}
    assert bones, "expected at least one neck/head driver"
    # The pose stores a non-identity rotation (it drives the channel to ~1.0).
    assert any(d.quaternion_wxyz[1:] != pytest.approx((0.0, 0.0, 0.0)) for d in target.drivers)


def test_real_ada_dna_corrective_target_solve() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    if int(reader.getRBFSolverCount()) == 0:
        pytest.skip("shipped Ada head DNA has no RBF solver")

    channel_count = int(reader.getBlendShapeChannelCount())
    names = {str(reader.getBlendShapeChannelName(i)) for i in range(channel_count)}
    if "head_turnUp_U" not in names:
        pytest.skip("shipped Ada head DNA exposes no head_turnUp channel")

    # The Raw Control Editor names this corrective by its rig-definition `_tgt`
    # target, which is not itself a channel -- the stem resolves to the RBF pose
    # whose authored name shares the direction.
    targets = resolve_corrective_rbf_targets(reader, "head_turnUp_tgt")
    assert len(targets) == 1, "head_turnUp_tgt should collapse to a single RBF pose"
    assert targets[0].distance_method == "Quaternion"

    # A translation/combination head corrective (``head_back_tgt``) is authored for
    # the RBF pose literally named ``back`` -- it resolves deterministically by
    # pose name even though it owns no blend-shape channel of its own.
    back = resolve_corrective_rbf_targets(reader, "head_back_tgt")
    assert len(back) == 1, "head_back_tgt should resolve to the 'back' RBF pose"
    assert back[0].pose_name == "back"
    assert any(d.quaternion_wxyz[1:] != pytest.approx((0.0, 0.0, 0.0)) for d in back[0].drivers)
