"""Exhaustive tests for :func:`core.mirror_raw_control_name` and
:func:`core.classify_raw_control_side`.

The reference catalog (EXPECTED_PAIRS + EXPECTED_CENTERS) was built by
enumerating every ``CTRL_expressions.*`` raw control on the Ada head
DNA (``tests/test_files/dna/ada/head.dna`` -- 251 names total). The
catalog is asserted live against the DNA in
:func:`test_catalog_matches_live_dna` so any future schema drift
(addition / rename of a raw control) immediately fails the suite.

Three mirror conventions are exercised:

* Trailing single uppercase ``L`` / ``R`` (e.g. ``browDownL``,
  ``mouthPressUL``, ``eyeLookLeftL`` -- the trailing token wins over
  the ``Left`` infix here).
* ``L`` / ``R`` immediately before a ``Ph<digit>`` suffix
  (``mouthLipsStickyLPh1`` -- 3 pairs).
* Whole-word ``Left`` / ``Right`` optionally followed by a single
  uppercase letter (``mouthLeft``, ``jawLeft``, ``teethLeftU``,
  ``mouthUpperLipShiftLeft``, ``tongueTwistLeft``, ``tongueTipLeft``).
"""

from __future__ import annotations

import pytest

from character_dna import naming as core
from character_dna.naming import RAW_CONTROL_PREFIX  # noqa: F401 - re-exported to the live-DNA catalog check


# ---------------------------------------------------------------------------
# Reference catalog (must mirror tests/test_files/dna/ada/head.dna exactly)
# ---------------------------------------------------------------------------

# 105 mirror pairs (left short name, right short name). Sides ``L``/``R``
# are assigned by Epic's naming convention -- ``L`` is the character's
# left, ``R`` is the character's right (mirrored from camera view).
EXPECTED_PAIRS: tuple[tuple[str, str], ...] = (
    # ----- trailing uppercase L / R (the dominant convention) -----
    ("browDownL", "browDownR"),
    ("browLateralL", "browLateralR"),
    ("browRaiseInL", "browRaiseInR"),
    ("browRaiseOuterL", "browRaiseOuterR"),
    ("earUpL", "earUpR"),
    ("eyeBlinkL", "eyeBlinkR"),
    ("eyeLidPressL", "eyeLidPressR"),
    ("eyeWidenL", "eyeWidenR"),
    ("eyeSquintInnerL", "eyeSquintInnerR"),
    ("eyeCheekRaiseL", "eyeCheekRaiseR"),
    ("eyeFaceScrunchL", "eyeFaceScrunchR"),
    ("eyeUpperLidUpL", "eyeUpperLidUpR"),
    ("eyeRelaxL", "eyeRelaxR"),
    ("eyeLowerLidUpL", "eyeLowerLidUpR"),
    ("eyeLowerLidDownL", "eyeLowerLidDownR"),
    ("eyeLookUpL", "eyeLookUpR"),
    ("eyeLookDownL", "eyeLookDownR"),
    # ``eyeLookLeftL`` / ``eyeLookLeftR`` -- trailing letter beats the
    # ``Left`` infix (which is the *look direction*, not the side).
    ("eyeLookLeftL", "eyeLookLeftR"),
    ("eyeLookRightL", "eyeLookRightR"),
    ("eyePupilWideL", "eyePupilWideR"),
    ("eyePupilNarrowL", "eyePupilNarrowR"),
    ("eyelashesUpINL", "eyelashesUpINR"),
    ("eyelashesUpOUTL", "eyelashesUpOUTR"),
    ("eyelashesDownINL", "eyelashesDownINR"),
    ("eyelashesDownOUTL", "eyelashesDownOUTR"),
    ("noseWrinkleL", "noseWrinkleR"),
    ("noseWrinkleUpperL", "noseWrinkleUpperR"),
    ("noseNostrilDepressL", "noseNostrilDepressR"),
    ("noseNostrilDilateL", "noseNostrilDilateR"),
    ("noseNostrilCompressL", "noseNostrilCompressR"),
    ("noseNasolabialDeepenL", "noseNasolabialDeepenR"),
    ("mouthCheekSuckL", "mouthCheekSuckR"),
    ("mouthCheekBlowL", "mouthCheekBlowR"),
    ("mouthLipsBlowL", "mouthLipsBlowR"),
    ("mouthUpperLipRaiseL", "mouthUpperLipRaiseR"),
    ("mouthLowerLipDepressL", "mouthLowerLipDepressR"),
    ("mouthCornerPullL", "mouthCornerPullR"),
    ("mouthStretchL", "mouthStretchR"),
    ("mouthStretchLipsCloseL", "mouthStretchLipsCloseR"),
    ("mouthDimpleL", "mouthDimpleR"),
    ("mouthCornerDepressL", "mouthCornerDepressR"),
    ("mouthPressUL", "mouthPressUR"),
    ("mouthPressDL", "mouthPressDR"),
    ("mouthLipsPurseUL", "mouthLipsPurseUR"),
    ("mouthLipsPurseDL", "mouthLipsPurseDR"),
    ("mouthLipsTowardsUL", "mouthLipsTowardsUR"),
    ("mouthLipsTowardsDL", "mouthLipsTowardsDR"),
    ("mouthFunnelUL", "mouthFunnelUR"),
    ("mouthFunnelDL", "mouthFunnelDR"),
    ("mouthLipsTogetherUL", "mouthLipsTogetherUR"),
    ("mouthLipsTogetherDL", "mouthLipsTogetherDR"),
    ("mouthUpperLipBiteL", "mouthUpperLipBiteR"),
    ("mouthLowerLipBiteL", "mouthLowerLipBiteR"),
    ("mouthLipsTightenUL", "mouthLipsTightenUR"),
    ("mouthLipsTightenDL", "mouthLipsTightenDR"),
    ("mouthLipsPressL", "mouthLipsPressR"),
    ("mouthSharpCornerPullL", "mouthSharpCornerPullR"),
    ("mouthStickyUINL", "mouthStickyUINR"),
    ("mouthStickyUOUTL", "mouthStickyUOUTR"),
    ("mouthStickyDINL", "mouthStickyDINR"),
    ("mouthStickyDOUTL", "mouthStickyDOUTR"),
    ("mouthLipsPushUL", "mouthLipsPushUR"),
    ("mouthLipsPushDL", "mouthLipsPushDR"),
    ("mouthLipsPullUL", "mouthLipsPullUR"),
    ("mouthLipsPullDL", "mouthLipsPullDR"),
    ("mouthLipsThinUL", "mouthLipsThinUR"),
    ("mouthLipsThinDL", "mouthLipsThinDR"),
    ("mouthLipsThickUL", "mouthLipsThickUR"),
    ("mouthLipsThickDL", "mouthLipsThickDR"),
    ("mouthLipsThinInwardUL", "mouthLipsThinInwardUR"),
    ("mouthLipsThinInwardDL", "mouthLipsThinInwardDR"),
    ("mouthLipsThickInwardUL", "mouthLipsThickInwardUR"),
    ("mouthLipsThickInwardDL", "mouthLipsThickInwardDR"),
    ("mouthCornerSharpenUL", "mouthCornerSharpenUR"),
    ("mouthCornerSharpenDL", "mouthCornerSharpenDR"),
    ("mouthCornerRounderUL", "mouthCornerRounderUR"),
    ("mouthCornerRounderDL", "mouthCornerRounderDR"),
    ("mouthUpperLipTowardsTeethL", "mouthUpperLipTowardsTeethR"),
    ("mouthLowerLipTowardsTeethL", "mouthLowerLipTowardsTeethR"),
    ("mouthUpperLipRollInL", "mouthUpperLipRollInR"),
    ("mouthUpperLipRollOutL", "mouthUpperLipRollOutR"),
    ("mouthLowerLipRollInL", "mouthLowerLipRollInR"),
    ("mouthLowerLipRollOutL", "mouthLowerLipRollOutR"),
    ("mouthCornerUpL", "mouthCornerUpR"),
    ("mouthCornerDownL", "mouthCornerDownR"),
    ("mouthCornerWideL", "mouthCornerWideR"),
    ("mouthCornerNarrowL", "mouthCornerNarrowR"),
    ("jawClenchL", "jawClenchR"),
    ("jawChinRaiseDL", "jawChinRaiseDR"),
    ("jawChinRaiseUL", "jawChinRaiseUR"),
    ("jawChinCompressL", "jawChinCompressR"),
    ("neckStretchL", "neckStretchR"),
    ("neckMastoidContractL", "neckMastoidContractR"),
    # ----- L / R before a ``Ph<digit>`` suffix -----
    ("mouthLipsStickyLPh1", "mouthLipsStickyRPh1"),
    ("mouthLipsStickyLPh2", "mouthLipsStickyRPh2"),
    ("mouthLipsStickyLPh3", "mouthLipsStickyRPh3"),
    # ----- whole-word ``Left`` / ``Right`` -----
    ("mouthLeft", "mouthRight"),
    ("mouthUpperLipShiftLeft", "mouthUpperLipShiftRight"),
    ("mouthLowerLipShiftLeft", "mouthLowerLipShiftRight"),
    ("jawLeft", "jawRight"),
    ("teethLeftU", "teethRightU"),
    ("teethLeftD", "teethRightD"),
    ("tongueLeft", "tongueRight"),
    ("tongueTwistLeft", "tongueTwistRight"),
    ("tongueTipLeft", "tongueTipRight"),
)


# 41 center names (no L/R counterpart). Includes the U/D axis pairs
# (``mouthUp`` / ``mouthDown``, ``teethUpU`` / ``teethDownU``, etc.)
# which are mid-line controls -- ``U``/``D`` here denote up/down, not
# a left/right side -- and the ``Ph`` phoneme series on the neck.
EXPECTED_CENTERS: tuple[str, ...] = (
    "eyeParallelLookDirection",
    "mouthUp",
    "mouthDown",
    "mouthStickyUC",
    "mouthStickyDC",
    "jawOpen",
    "jawFwd",
    "jawBack",
    "jawOpenExtreme",
    "neckSwallowPh1",
    "neckSwallowPh2",
    "neckSwallowPh3",
    "neckSwallowPh4",
    "neckThroatDown",
    "neckThroatUp",
    "neckDigastricDown",
    "neckDigastricUp",
    "neckThroatExhale",
    "neckThroatInhale",
    "teethUpU",
    "teethUpD",
    "teethDownU",
    "teethDownD",
    "teethFwdU",
    "teethFwdD",
    "teethBackU",
    "teethBackD",
    "tongueUp",
    "tongueDown",
    "tongueOut",
    "tongueIn",
    "tongueBendUp",
    "tongueBendDown",
    "tongueTipUp",
    "tongueTipDown",
    "tongueWide",
    "tongueNarrow",
    "tonguePress",
    # lowercase trailing ``l`` -- intentionally NOT mirrored by the
    # case-sensitive trailing-letter rule.
    "tongueRoll",
    "tongueThick",
    "tongueThin",
)


def _expected_short_name_set() -> set[str]:
    """Catalog as a flat set of short names (no ``CTRL_expressions.``
    prefix). Used as the ``known_short_names`` argument to the
    twin-gated classifier and as the drift guard against the live DNA."""
    names: set[str] = set(EXPECTED_CENTERS)
    for left, right in EXPECTED_PAIRS:
        names.add(left)
        names.add(right)
    return names


EXPECTED_ALL_SHORT_NAMES: frozenset[str] = frozenset(_expected_short_name_set())


# ---------------------------------------------------------------------------
# Catalog sanity (catches mistakes in the hardcoded reference itself)
# ---------------------------------------------------------------------------


def test_catalog_pair_and_center_counts() -> None:
    """The catalog must contain exactly 105 pairs + 41 centers = 251
    short names, with no overlap between the two sets."""
    pair_lefts = {left for left, _ in EXPECTED_PAIRS}
    pair_rights = {right for _, right in EXPECTED_PAIRS}
    centers = set(EXPECTED_CENTERS)

    assert len(EXPECTED_PAIRS) == 105
    assert len(EXPECTED_CENTERS) == 41
    assert len(pair_lefts) == len(EXPECTED_PAIRS), "duplicate left name in EXPECTED_PAIRS"
    assert len(pair_rights) == len(EXPECTED_PAIRS), "duplicate right name in EXPECTED_PAIRS"
    assert pair_lefts.isdisjoint(pair_rights), "name appears as both left and right"
    assert centers.isdisjoint(pair_lefts | pair_rights), "center name also listed as a pair"
    assert len(EXPECTED_ALL_SHORT_NAMES) == 251


# ---------------------------------------------------------------------------
# mirror_raw_control_name: pairs round-trip cleanly in both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_mirror_left_to_right(left: str, right: str) -> None:
    mirrored, side = core.mirror_raw_control_name(left)
    assert side == "L", f"expected 'L' side for {left!r}, got {side!r}"
    assert mirrored == right, f"{left!r} -> {mirrored!r}, expected {right!r}"


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_mirror_right_to_left(left: str, right: str) -> None:
    mirrored, side = core.mirror_raw_control_name(right)
    assert side == "R", f"expected 'R' side for {right!r}, got {side!r}"
    assert mirrored == left, f"{right!r} -> {mirrored!r}, expected {left!r}"


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_mirror_is_involutive(left: str, right: str) -> None:
    """Mirroring twice returns the original name."""
    once_l, _ = core.mirror_raw_control_name(left)
    twice_l, _ = core.mirror_raw_control_name(once_l) if once_l else (None, None)
    once_r, _ = core.mirror_raw_control_name(right)
    twice_r, _ = core.mirror_raw_control_name(once_r) if once_r else (None, None)
    assert twice_l == left
    assert twice_r == right


# ---------------------------------------------------------------------------
# mirror_raw_control_name: centers return (None, None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_CENTERS)
def test_mirror_center_returns_none(name: str) -> None:
    mirrored, side = core.mirror_raw_control_name(name)
    assert mirrored is None and side is None, f"{name!r} unexpectedly returned ({mirrored!r}, {side!r})"


# ---------------------------------------------------------------------------
# classify_raw_control_side: twin-gated classification matches catalog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_classify_pair(left: str, right: str) -> None:
    known = EXPECTED_ALL_SHORT_NAMES
    assert core.classify_raw_control_side(left, set(known)) == "L"
    assert core.classify_raw_control_side(right, set(known)) == "R"


@pytest.mark.parametrize("name", EXPECTED_CENTERS)
def test_classify_center(name: str) -> None:
    known = set(EXPECTED_ALL_SHORT_NAMES)
    assert core.classify_raw_control_side(name, known) == "C"


def test_classify_orphan_falls_through_to_center() -> None:
    """A name whose mirror partner is absent from ``known_short_names``
    must classify as center (``C``) -- the twin-exists gate. Drop
    ``browDownR`` from the known set: ``browDownL`` should no longer
    classify as ``L``."""
    known = set(EXPECTED_ALL_SHORT_NAMES) - {"browDownR"}
    assert core.classify_raw_control_side("browDownL", known) == "C"


# ---------------------------------------------------------------------------
# Live drift guard -- enumerate Ada head.dna and assert the catalog matches
# ---------------------------------------------------------------------------
