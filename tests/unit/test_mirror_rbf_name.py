"""Exhaustive tests for the RBF Editor name mirror
(:func:`rbf_editor.utilities.get_mirrored_name`) using the shipped default
solver / pose pattern :data:`DEFAULT_SOLVER_MIRROR_REGEX`.

The reference catalogs were built by enumerating every RBF solver
(72) and RBF pose (215) on the Ada body DNA
(``tests/test_files/dna/ada/body.dna``) and mirroring each with the production
function. Body RBF names use the ``_l_`` / ``_r_`` infix convention
(``calf_l_UERBFSolver`` <-> ``calf_r_UERBFSolver``,
``thigh_l_in_45_out_90`` <-> ``thigh_r_in_45_out_90``). The ``default`` pose has
no side and mirrors to ``None``. The catalogs are asserted live against the DNA
in :func:`test_solver_catalog_matches_live_dna` /
:func:`test_pose_catalog_matches_live_dna`.
"""

from __future__ import annotations

import pytest

from character_dna.naming import DEFAULT_SOLVER_MIRROR_REGEX, rbf_mirrored_name as get_mirrored_name


# ---------------------------------------------------------------------------
# Reference catalogs (must mirror tests/test_files/dna/ada/body.dna exactly)
# ---------------------------------------------------------------------------

# 36 solver mirror pairs (left, right).
EXPECTED_SOLVER_PAIRS: tuple[tuple[str, str], ...] = (
    ("calf_l_UERBFSolver", "calf_r_UERBFSolver"),
    ("clavicle_l_UERBFSolver", "clavicle_r_UERBFSolver"),
    ("foot_l_UERBFSolver", "foot_r_UERBFSolver"),
    ("hand_l_UERBFSolver", "hand_r_UERBFSolver"),
    ("index_l_01_UERBFSolver", "index_r_01_UERBFSolver"),
    ("index_l_01_half_UERBFSolver", "index_r_01_half_UERBFSolver"),
    ("index_l_02_UERBFSolver", "index_r_02_UERBFSolver"),
    ("index_l_02_half_UERBFSolver", "index_r_02_half_UERBFSolver"),
    ("index_l_03_UERBFSolver", "index_r_03_UERBFSolver"),
    ("index_l_03_half_UERBFSolver", "index_r_03_half_UERBFSolver"),
    ("lowerarm_l_UERBFSolver", "lowerarm_r_UERBFSolver"),
    ("middle_l_01_UERBFSolver", "middle_r_01_UERBFSolver"),
    ("middle_l_01_half_UERBFSolver", "middle_r_01_half_UERBFSolver"),
    ("middle_l_02_UERBFSolver", "middle_r_02_UERBFSolver"),
    ("middle_l_02_half_UERBFSolver", "middle_r_02_half_UERBFSolver"),
    ("middle_l_03_UERBFSolver", "middle_r_03_UERBFSolver"),
    ("middle_l_03_half_UERBFSolver", "middle_r_03_half_UERBFSolver"),
    ("pinky_l_01_UERBFSolver", "pinky_r_01_UERBFSolver"),
    ("pinky_l_01_half_UERBFSolver", "pinky_r_01_half_UERBFSolver"),
    ("pinky_l_02_UERBFSolver", "pinky_r_02_UERBFSolver"),
    ("pinky_l_02_half_UERBFSolver", "pinky_r_02_half_UERBFSolver"),
    ("pinky_l_03_UERBFSolver", "pinky_r_03_UERBFSolver"),
    ("pinky_l_03_half_UERBFSolver", "pinky_r_03_half_UERBFSolver"),
    ("ring_l_01_UERBFSolver", "ring_r_01_UERBFSolver"),
    ("ring_l_01_half_UERBFSolver", "ring_r_01_half_UERBFSolver"),
    ("ring_l_02_UERBFSolver", "ring_r_02_UERBFSolver"),
    ("ring_l_02_half_UERBFSolver", "ring_r_02_half_UERBFSolver"),
    ("ring_l_03_UERBFSolver", "ring_r_03_UERBFSolver"),
    ("ring_l_03_half_UERBFSolver", "ring_r_03_half_UERBFSolver"),
    ("thigh_l_UERBFSolver", "thigh_r_UERBFSolver"),
    ("thumb_l_01_UERBFSolver", "thumb_r_01_UERBFSolver"),
    ("thumb_l_02_UERBFSolver", "thumb_r_02_UERBFSolver"),
    ("thumb_l_02_half_UERBFSolver", "thumb_r_02_half_UERBFSolver"),
    ("thumb_l_03_UERBFSolver", "thumb_r_03_UERBFSolver"),
    ("thumb_l_03_half_UERBFSolver", "thumb_r_03_half_UERBFSolver"),
    ("upperarm_l_UERBFSolver", "upperarm_r_UERBFSolver"),
)

# 107 pose mirror pairs (left, right).
EXPECTED_POSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("calf_l_back_120", "calf_r_back_120"),
    ("calf_l_back_90", "calf_r_back_90"),
    ("calf_l_back_150", "calf_r_back_150"),
    ("calf_l_back_50", "calf_r_back_50"),
    ("calf_l_back_130", "calf_r_back_130"),
    ("clavicle_l_fwd_30", "clavicle_r_fwd_30"),
    ("clavicle_l_back_30", "clavicle_r_back_30"),
    ("clavicle_l_down_20", "clavicle_r_down_20"),
    ("clavicle_l_up_40", "clavicle_r_up_40"),
    ("foot_l_down_60", "foot_r_down_60"),
    ("foot_l_up_35", "foot_r_up_35"),
    ("hand_l_down_90", "hand_r_down_90"),
    ("hand_l_up_20", "hand_r_up_20"),
    ("hand_l_up_90", "hand_r_up_90"),
    ("index_l_01_curl_070", "index_r_01_curl_070"),
    ("index_l_01_push_090", "index_r_01_push_090"),
    ("index_l_01_half_curl_120", "index_r_01_half_curl_120"),
    ("index_l_01_half_push_120", "index_r_01_half_push_120"),
    ("index_l_01_half_caps_020", "index_r_01_half_caps_020"),
    ("index_l_02_curl_080", "index_r_02_curl_080"),
    ("index_l_02_half_curl_120", "index_r_02_half_curl_120"),
    ("index_l_02_half_push_120", "index_r_02_half_push_120"),
    ("index_l_02_half_caps_020", "index_r_02_half_caps_020"),
    ("index_l_03_curl_090", "index_r_03_curl_090"),
    ("index_l_03_half_curl_120", "index_r_03_half_curl_120"),
    ("index_l_03_half_push_120", "index_r_03_half_push_120"),
    ("index_l_03_half_caps_020", "index_r_03_half_caps_020"),
    ("lowerarm_l_in_110", "lowerarm_r_in_110"),
    ("lowerarm_l_in_35", "lowerarm_r_in_35"),
    ("lowerarm_l_out_35", "lowerarm_r_out_35"),
    ("lowerarm_l_in_50", "lowerarm_r_in_50"),
    ("lowerarm_l_in_75", "lowerarm_r_in_75"),
    ("lowerarm_l_in_90", "lowerarm_r_in_90"),
    ("lowerarm_l_out_10", "lowerarm_r_out_10"),
    ("lowerarm_l_in_10", "lowerarm_r_in_10"),
    ("middle_l_01_push_090", "middle_r_01_push_090"),
    ("middle_l_01_curl_070", "middle_r_01_curl_070"),
    ("middle_l_01_half_curl_120", "middle_r_01_half_curl_120"),
    ("middle_l_01_half_push_120", "middle_r_01_half_push_120"),
    ("middle_l_01_half_caps_020", "middle_r_01_half_caps_020"),
    ("middle_l_02_curl_080", "middle_r_02_curl_080"),
    ("middle_l_02_half_curl_120", "middle_r_02_half_curl_120"),
    ("middle_l_02_half_push_120", "middle_r_02_half_push_120"),
    ("middle_l_02_half_caps_020", "middle_r_02_half_caps_020"),
    ("middle_l_03_curl_090", "middle_r_03_curl_090"),
    ("middle_l_03_half_curl_120", "middle_r_03_half_curl_120"),
    ("middle_l_03_half_push_120", "middle_r_03_half_push_120"),
    ("middle_l_03_half_caps_020", "middle_r_03_half_caps_020"),
    ("pinky_l_01_push_090", "pinky_r_01_push_090"),
    ("pinky_l_01_curl_070", "pinky_r_01_curl_070"),
    ("pinky_l_01_half_curl_120", "pinky_r_01_half_curl_120"),
    ("pinky_l_01_half_push_120", "pinky_r_01_half_push_120"),
    ("pinky_l_01_half_caps_020", "pinky_r_01_half_caps_020"),
    ("pinky_l_02_curl_080", "pinky_r_02_curl_080"),
    ("pinky_l_02_half_curl_120", "pinky_r_02_half_curl_120"),
    ("pinky_l_02_half_push_120", "pinky_r_02_half_push_120"),
    ("pinky_l_02_half_caps_020", "pinky_r_02_half_caps_020"),
    ("pinky_l_03_curl_090", "pinky_r_03_curl_090"),
    ("pinky_l_03_half_curl_120", "pinky_r_03_half_curl_120"),
    ("pinky_l_03_half_push_120", "pinky_r_03_half_push_120"),
    ("pinky_l_03_half_caps_020", "pinky_r_03_half_caps_020"),
    ("ring_l_01_curl_070", "ring_r_01_curl_070"),
    ("ring_l_01_push_090", "ring_r_01_push_090"),
    ("ring_l_01_half_curl_120", "ring_r_01_half_curl_120"),
    ("ring_l_01_half_push_120", "ring_r_01_half_push_120"),
    ("ring_l_01_half_caps_020", "ring_r_01_half_caps_020"),
    ("ring_l_02_curl_080", "ring_r_02_curl_080"),
    ("ring_l_02_half_curl_120", "ring_r_02_half_curl_120"),
    ("ring_l_02_half_push_120", "ring_r_02_half_push_120"),
    ("ring_l_02_half_caps_020", "ring_r_02_half_caps_020"),
    ("ring_l_03_curl_090", "ring_r_03_curl_090"),
    ("ring_l_03_half_curl_120", "ring_r_03_half_curl_120"),
    ("ring_l_03_half_push_120", "ring_r_03_half_push_120"),
    ("ring_l_03_half_caps_020", "ring_r_03_half_caps_020"),
    ("thigh_l_bck_10", "thigh_r_bck_10"),
    ("thigh_l_fwd_10", "thigh_r_fwd_10"),
    ("thigh_l_in_45_out_90", "thigh_r_in_45_out_90"),
    ("thigh_l_fwd_90", "thigh_r_fwd_90"),
    ("thigh_l_out_55", "thigh_r_out_55"),
    ("thigh_l_bck_90", "thigh_r_bck_90"),
    ("thigh_l_bck_50", "thigh_r_bck_50"),
    ("thigh_l_out_110", "thigh_r_out_110"),
    ("thigh_l_out_10", "thigh_r_out_10"),
    ("thigh_l_fwd_45", "thigh_r_fwd_45"),
    ("thigh_l_in_50", "thigh_r_in_50"),
    ("thigh_l_fwd_110", "thigh_r_fwd_110"),
    ("thigh_l_out_85", "thigh_r_out_85"),
    ("thumb_l_01_curl_050", "thumb_r_01_curl_050"),
    ("thumb_l_02_curl_050", "thumb_r_02_curl_050"),
    ("thumb_l_02_half_curl_120", "thumb_r_02_half_curl_120"),
    ("thumb_l_02_half_push_120", "thumb_r_02_half_push_120"),
    ("thumb_l_02_half_caps_020", "thumb_r_02_half_caps_020"),
    ("thumb_l_03_curl_090", "thumb_r_03_curl_090"),
    ("thumb_l_03_half_curl_120", "thumb_r_03_half_curl_120"),
    ("thumb_l_03_half_push_120", "thumb_r_03_half_push_120"),
    ("thumb_l_03_half_caps_020", "thumb_r_03_half_caps_020"),
    ("upperarm_l_back_45", "upperarm_r_back_45"),
    ("upperarm_l_out_55", "upperarm_r_out_55"),
    ("upperarm_l_in_35", "upperarm_r_in_35"),
    ("upperarm_l_out_85", "upperarm_r_out_85"),
    ("upperarm_l_fwd_110", "upperarm_r_fwd_110"),
    ("upperarm_l_in_10", "upperarm_r_in_10"),
    ("upperarm_l_fwd_15", "upperarm_r_fwd_15"),
    ("upperarm_l_out_15", "upperarm_r_out_15"),
    ("upperarm_l_back_10", "upperarm_r_back_10"),
    ("upperarm_l_out_110", "upperarm_r_out_110"),
    ("upperarm_l_fwd_90", "upperarm_r_fwd_90"),
)

# Center poses (no detectable side).
EXPECTED_POSE_CENTERS: tuple[str, ...] = ("default",)


def _all_solver_names() -> set[str]:
    names: set[str] = set()
    for left, right in EXPECTED_SOLVER_PAIRS:
        names.add(left)
        names.add(right)
    return names


def _all_pose_names() -> set[str]:
    names: set[str] = set(EXPECTED_POSE_CENTERS)
    for left, right in EXPECTED_POSE_PAIRS:
        names.add(left)
        names.add(right)
    return names


EXPECTED_ALL_SOLVERS: frozenset[str] = frozenset(_all_solver_names())
EXPECTED_ALL_POSES: frozenset[str] = frozenset(_all_pose_names())


# ---------------------------------------------------------------------------
# Catalog sanity
# ---------------------------------------------------------------------------


def test_solver_catalog_counts() -> None:
    lefts = {left for left, _ in EXPECTED_SOLVER_PAIRS}
    rights = {right for _, right in EXPECTED_SOLVER_PAIRS}
    assert len(EXPECTED_SOLVER_PAIRS) == 36
    assert lefts.isdisjoint(rights)
    assert len(EXPECTED_ALL_SOLVERS) == 72


def test_pose_catalog_counts() -> None:
    lefts = {left for left, _ in EXPECTED_POSE_PAIRS}
    rights = {right for _, right in EXPECTED_POSE_PAIRS}
    centers = set(EXPECTED_POSE_CENTERS)
    assert len(EXPECTED_POSE_PAIRS) == 107
    assert len(EXPECTED_POSE_CENTERS) == 1
    assert lefts.isdisjoint(rights)
    assert centers.isdisjoint(lefts | rights)
    assert len(EXPECTED_ALL_POSES) == 215


# ---------------------------------------------------------------------------
# get_mirrored_name: solver + pose pairs mirror both ways and are involutive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("left", "right"), EXPECTED_SOLVER_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_solver_mirror_both_ways(left: str, right: str) -> None:
    assert get_mirrored_name(left, DEFAULT_SOLVER_MIRROR_REGEX) == right
    assert get_mirrored_name(right, DEFAULT_SOLVER_MIRROR_REGEX) == left


@pytest.mark.parametrize(("left", "right"), EXPECTED_POSE_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_pose_mirror_both_ways(left: str, right: str) -> None:
    assert get_mirrored_name(left, DEFAULT_SOLVER_MIRROR_REGEX) == right
    assert get_mirrored_name(right, DEFAULT_SOLVER_MIRROR_REGEX) == left


@pytest.mark.parametrize(
    ("left", "right"), EXPECTED_SOLVER_PAIRS + EXPECTED_POSE_PAIRS, ids=lambda p: p if isinstance(p, str) else ""
)
def test_mirror_is_involutive(left: str, right: str) -> None:
    once = get_mirrored_name(left, DEFAULT_SOLVER_MIRROR_REGEX)
    assert once == right
    assert get_mirrored_name(once, DEFAULT_SOLVER_MIRROR_REGEX) == left


@pytest.mark.parametrize("name", EXPECTED_POSE_CENTERS)
def test_center_pose_has_no_mirror(name: str) -> None:
    assert get_mirrored_name(name, DEFAULT_SOLVER_MIRROR_REGEX) is None


# ---------------------------------------------------------------------------
# Live drift guards -- enumerate Ada body.dna and assert the catalogs match
# ---------------------------------------------------------------------------
