import pytest

from character_dna.constants import EXCLUDED_FACE_BOARD_CONTROLS, EYE_AIM_BONES, FACE_BOARD_SWITCHES
from character_dna.validators import (
    Severity,
    validate_face_board_animation,
    validate_skeleton_animation,
)


BODY_BONES = [
    "root",
    "pelvis",
    "spine_01",
    "spine_02",
    "spine_03",
    "clavicle_l",
    "clavicle_r",
    "upperarm_l",
    "upperarm_r",
    "thigh_l",
    "thigh_r",
    "hand_l",
    "hand_r",
    "neck_01",
    "head",
]

FACE_BOARD_CONTROLS = [
    "CTRL_C_jaw",
    "CTRL_C_jaw_fwdBack",
    "CTRL_C_jaw_openExtreme",
    "CTRL_L_jaw_clench",
    "CTRL_R_jaw_clench",
    "CTRL_L_neck_stretch",
    "CTRL_R_neck_stretch",
    "CTRL_neck_digastricUpDown",
    "CTRL_L_mouth_cornerPull",
    "CTRL_R_mouth_cornerPull",
    "CTRL_L_brow_down",
    "CTRL_R_brow_down",
]

EXCLUDED_CONTROLS = frozenset(EXCLUDED_FACE_BOARD_CONTROLS) | frozenset(EYE_AIM_BONES) | frozenset(FACE_BOARD_SWITCHES)


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_matching_skeleton_animation_is_valid():
    report = validate_skeleton_animation(BODY_BONES, BODY_BONES, component="body")

    assert report.is_valid
    assert not report.issues


def test_matching_face_board_animation_is_valid():
    report = validate_face_board_animation(FACE_BOARD_CONTROLS, FACE_BOARD_CONTROLS)

    assert report.is_valid


@pytest.mark.parametrize(
    "name",
    sorted(
        set(EXCLUDED_FACE_BOARD_CONTROLS)
        | set(EYE_AIM_BONES)
        | set(FACE_BOARD_SWITCHES)
        | {"FRM_C_tongue_move", "GRP_faceGUI", "headGui_grp", "headRig_grp", "CTRL_helper_grp", "camera"}
    ),
)
def test_face_board_helpers_are_not_validated(name: str):
    report = validate_face_board_animation(
        [*FACE_BOARD_CONTROLS, name], FACE_BOARD_CONTROLS, exclude_controls=EXCLUDED_CONTROLS
    )

    assert report.is_valid
    assert not report.issues


def test_face_board_unknown_control_still_warns():
    report = validate_face_board_animation(
        [*FACE_BOARD_CONTROLS, "CTRL_missing", "CTRL_faceGUI", "FRM_C_tongue_move"],
        FACE_BOARD_CONTROLS,
        exclude_controls=EXCLUDED_CONTROLS,
    )

    assert report.is_valid
    assert codes(report) == {"node_unexpected"}
    assert report.warnings[0].actual == 1
    assert "CTRL_missing" in report.warnings[0].message


def test_face_board_helpers_cannot_satisfy_coverage():
    helpers = [*FACE_BOARD_SWITCHES, *EXCLUDED_FACE_BOARD_CONTROLS, *EYE_AIM_BONES]
    report = validate_face_board_animation(helpers, helpers, exclude_controls=EXCLUDED_CONTROLS)

    assert not report.is_valid
    assert "node_unexpected" not in codes(report)


def test_extra_fbx_nodes_only_warn():
    source = [*BODY_BONES, "extra_prop_bone", "camera_helper"]

    report = validate_skeleton_animation(source, BODY_BONES, component="body")

    assert report.is_valid
    assert codes(report) == {"node_unexpected"}
    assert report.warnings[0].severity is Severity.WARNING


def test_extra_rig_bones_do_not_fail_validation():
    target = [*BODY_BONES, *[f"twist_{index}" for index in range(200)]]

    report = validate_skeleton_animation(BODY_BONES, target, component="body")

    assert report.is_valid


def test_face_board_animation_onto_body_is_rejected():
    report = validate_skeleton_animation(FACE_BOARD_CONTROLS, BODY_BONES, component="body")

    assert not report.is_valid
    assert "wrong_animation_type" in codes(report)
    assert "face board" in report.summary()


def test_body_animation_onto_face_board_is_rejected():
    report = validate_face_board_animation(BODY_BONES, FACE_BOARD_CONTROLS)

    assert not report.is_valid
    assert "wrong_animation_type" in codes(report)
    assert "skeleton animation" in report.summary()


def test_empty_source_is_rejected():
    report = validate_skeleton_animation([], BODY_BONES, component="body")

    assert not report.is_valid
    assert codes(report) == {"no_animation_nodes"}


def test_empty_target_is_rejected():
    report = validate_skeleton_animation(BODY_BONES, [], component="body")

    assert not report.is_valid
    assert codes(report) == {"no_target_bones"}


@pytest.mark.parametrize(
    ("matched_count", "expected_valid"),
    [
        (len(BODY_BONES), True),
        (10, True),  # 10/15 == 67%, above the default 60% threshold
        (8, False),  # 8/15 == 53%, below it
        (2, False),  # too few matches regardless of ratio
    ],
)
def test_coverage_threshold(matched_count: int, expected_valid: bool):
    source = [*BODY_BONES[:matched_count], *[f"unrelated_{index}" for index in range(len(BODY_BONES) - matched_count)]]

    report = validate_skeleton_animation(source, BODY_BONES, component="body")

    assert report.is_valid is expected_valid
    if not expected_valid:
        assert "bone_coverage_too_low" in codes(report)


def test_a_tiny_perfectly_matching_file_is_still_rejected():
    # A handful of matching names would otherwise score 100% coverage.
    source = BODY_BONES[:3]

    report = validate_skeleton_animation(source, BODY_BONES, component="body")

    assert not report.is_valid
    assert "bone_coverage_too_low" in codes(report)


def test_missing_bones_are_listed_in_the_message():
    # Pad with unrelated names so the target is the smaller set and coverage is
    # measured against it.
    matched = [name for name in BODY_BONES if name != "pelvis"][:8]
    source = [*matched, *[f"unrelated_{index}" for index in range(20)]]

    report = validate_skeleton_animation(source, BODY_BONES, component="body")

    assert not report.is_valid
    assert "pelvis" in report.summary()


def test_summary_filters_by_severity():
    source = [*BODY_BONES, "extra_bone"]

    report = validate_skeleton_animation(source, BODY_BONES, component="body")

    assert report.summary() == ""
    assert "extra_bone" in report.summary(Severity.WARNING)
    assert "extra_bone" in report.summary(None)
