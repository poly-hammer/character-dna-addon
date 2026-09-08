"""Validate that an FBX animation actually belongs to the rig it is being imported onto.

The checks are deliberately name based and Blender free: they compare the node
names in an FBX clip against the bone names of the target rig. Extra nodes in
the FBX are expected and never block an import; a clip that clearly belongs to a
different rig does.
"""

# standard library imports
import logging

from collections.abc import Iterable

# local imports
from .base import Severity, ValidationReport


logger = logging.getLogger(__name__)

# Fraction of the smaller of the two name sets that must match. Comparing against
# the smaller set means neither a rig with extra bones nor an FBX with extra nodes
# is penalized, which is the whole point of the check.
MINIMUM_BONE_COVERAGE = 0.6

# Guards the degenerate case where a tiny FBX matches a handful of bones and would
# otherwise score perfect coverage.
MINIMUM_MATCHED_BONES = 8

# Bones that only ever appear on a MetaHuman body/head skeleton.
SKELETON_SIGNATURE_BONES = frozenset(
    {
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
    }
)

# Face board controls are uniformly prefixed.
FACE_BOARD_CONTROL_PREFIX = "CTRL_"

# Fraction of names that must be controls before a clip is considered a face board clip.
FACE_BOARD_CONTROL_RATIO = 0.5

# Number of names listed in a message before it is truncated.
MAX_REPORTED_NAMES = 10


def is_face_board_animation_control(name: str, exclude_controls: frozenset[str] = frozenset()) -> bool:
    """Return whether a node is an importable face-board control rather than a helper."""
    return name.startswith(FACE_BOARD_CONTROL_PREFIX) and not name.endswith("_grp") and name not in exclude_controls


def _format_names(names: Iterable[str]) -> str:
    """Return a short, sorted, comma separated preview of ``names``."""
    ordered = sorted(names)
    preview = ", ".join(ordered[:MAX_REPORTED_NAMES])
    if len(ordered) > MAX_REPORTED_NAMES:
        preview += f", ... (+{len(ordered) - MAX_REPORTED_NAMES} more)"
    return preview


def looks_like_face_board(names: Iterable[str]) -> bool:
    """Return ``True`` when a set of node names looks like a face board clip.

    Args:
        names: Node or bone names to inspect.

    Returns:
        ``True`` when most of the names are face board controls.
    """
    names = list(names)
    if not names:
        return False
    controls = sum(1 for name in names if name.startswith(FACE_BOARD_CONTROL_PREFIX))
    return controls / len(names) >= FACE_BOARD_CONTROL_RATIO


def looks_like_skeleton(names: Iterable[str]) -> bool:
    """Return ``True`` when a set of node names looks like a MetaHuman skeleton clip.

    Args:
        names: Node or bone names to inspect.

    Returns:
        ``True`` when enough signature body bones are present.
    """
    matched = SKELETON_SIGNATURE_BONES.intersection(names)
    return len(matched) >= 4


class AnimationValidator:
    """Compare the node names in an FBX clip against a target rig's bone names."""

    #: Human readable name of the kind of animation this validator accepts.
    expected_kind = "animation"

    def __init__(
        self,
        source_names: Iterable[str],
        target_names: Iterable[str],
        subject: str,
        minimum_coverage: float = MINIMUM_BONE_COVERAGE,
    ) -> None:
        """Initialize the validator.

        Args:
            source_names: Node names found in the FBX file.
            target_names: Bone names the import would write to.
            subject: Label for the target used in messages, such as ``"body"``.
            minimum_coverage: Fraction of the smaller name set that must match.
        """
        self._source_names = frozenset(source_names)
        self._target_names = frozenset(target_names)
        self._subject = subject
        self._minimum_coverage = minimum_coverage

    @property
    def matched_names(self) -> frozenset[str]:
        """Names present in both the FBX and the target rig."""
        return self._source_names & self._target_names

    @property
    def coverage(self) -> float:
        """Fraction of the smaller name set that matched, in ``[0, 1]``."""
        smaller = min(len(self._source_names), len(self._target_names))
        if smaller == 0:
            return 0.0
        return len(self.matched_names) / smaller

    def validate(self) -> ValidationReport:
        """Run every check.

        Returns:
            The collected report. Errors mean the file is very likely wrong.
        """
        report = ValidationReport(db_name=self._subject)
        if not self._source_names:
            report.add(
                code="no_animation_nodes",
                severity=Severity.ERROR,
                message="The FBX file does not contain any animated nodes.",
                expected=self.expected_kind,
                actual=None,
            )
            return report

        if not self._target_names:
            report.add(
                code="no_target_bones",
                severity=Severity.ERROR,
                message=f"The {self._subject} rig has no bones to import animation onto.",
                expected=self._subject,
                actual=None,
            )
            return report

        self._check_animation_type(report)
        self._check_coverage(report)
        self._check_unexpected_nodes(report)
        return report

    def _check_animation_type(self, report: ValidationReport) -> None:
        """Add an issue when the clip clearly belongs to a different kind of rig."""

    def _check_coverage(self, report: ValidationReport) -> None:
        matched = self.matched_names
        coverage = self.coverage
        if coverage >= self._minimum_coverage and len(matched) >= MINIMUM_MATCHED_BONES:
            return

        missing = self._target_names - self._source_names
        report.add(
            code="bone_coverage_too_low",
            severity=Severity.ERROR,
            message=(
                f"This file does not look like {self._subject} animation. Only "
                f"{len(matched)} of the {len(self._target_names)} {self._subject} bones were found "
                f"in it ({coverage:.0%} match).\n"
                f"Missing bones: {_format_names(missing)}"
            ),
            expected=self._minimum_coverage,
            actual=coverage,
        )

    def _check_unexpected_nodes(self, report: ValidationReport) -> None:
        unexpected = self._source_names - self._target_names
        if not unexpected:
            return
        report.add(
            code="node_unexpected",
            severity=Severity.WARNING,
            message=(
                f"The FBX file contains {len(unexpected)} node(s) with no matching "
                f"{self._subject} bone; they were ignored: {_format_names(unexpected)}"
            ),
            expected=None,
            actual=len(unexpected),
        )


class SkeletonAnimationValidator(AnimationValidator):
    """Validate a clip that is being imported onto a body or head skeleton."""

    expected_kind = "skeletal animation"

    def _check_animation_type(self, report: ValidationReport) -> None:
        if not looks_like_face_board(self._source_names):
            return
        report.add(
            code="wrong_animation_type",
            severity=Severity.ERROR,
            message=(
                f"This looks like face board animation, not {self._subject} animation. "
                f"Import it with the Face Board importer instead."
            ),
            expected=self.expected_kind,
            actual="face board animation",
        )


class FaceBoardAnimationValidator(AnimationValidator):
    """Validate a clip that is being imported onto the face board."""

    expected_kind = "face board animation"

    def __init__(
        self,
        source_names: Iterable[str],
        target_names: Iterable[str],
        subject: str,
        minimum_coverage: float = MINIMUM_BONE_COVERAGE,
        exclude_controls: frozenset[str] = frozenset(),
    ) -> None:
        """Limit matching to importable controls while retaining original clip-type evidence."""
        self._all_source_names = frozenset(source_names)
        super().__init__(
            (name for name in self._all_source_names if is_face_board_animation_control(name, exclude_controls)),
            (name for name in target_names if is_face_board_animation_control(name, exclude_controls)),
            subject,
            minimum_coverage,
        )

    def validate(self) -> ValidationReport:
        """Report a wrong-kind clip even when filtering leaves no importable controls."""
        report = super().validate()
        if not self._source_names or not self._target_names:
            self._check_animation_type(report)
        return report

    def _check_animation_type(self, report: ValidationReport) -> None:
        if not looks_like_skeleton(self._all_source_names):
            return
        report.add(
            code="wrong_animation_type",
            severity=Severity.ERROR,
            message=(
                "This looks like body or head skeleton animation, not face board animation. "
                "Import it with the Body or Head importer instead."
            ),
            expected=self.expected_kind,
            actual="skeletal animation",
        )


def validate_skeleton_animation(
    source_names: Iterable[str],
    target_names: Iterable[str],
    component: str = "body",
    minimum_coverage: float = MINIMUM_BONE_COVERAGE,
) -> ValidationReport:
    """Validate an FBX clip against a body or head skeleton.

    Args:
        source_names: Node names found in the FBX file.
        target_names: Bone names on the target rig.
        component: ``"body"`` or ``"head"``, used in messages.
        minimum_coverage: Fraction of the smaller name set that must match.

    Returns:
        The validation report.
    """
    return SkeletonAnimationValidator(source_names, target_names, component, minimum_coverage).validate()


def validate_face_board_animation(
    source_names: Iterable[str],
    target_names: Iterable[str],
    minimum_coverage: float = MINIMUM_BONE_COVERAGE,
    exclude_controls: frozenset[str] = frozenset(),
) -> ValidationReport:
    """Validate an FBX clip against the face board.

    Args:
        source_names: Node names found in the FBX file.
        target_names: Control bone names on the face board.
        minimum_coverage: Fraction of the smaller name set that must match.
        exclude_controls: Control names intentionally ignored by the importer.

    Returns:
        The validation report.
    """
    return FaceBoardAnimationValidator(
        source_names, target_names, "face board", minimum_coverage, exclude_controls
    ).validate()
