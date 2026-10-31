"""Exhaustive tests for the Shape Key Editor side classifier
(:func:`shared.utilities.classify_side_by_regex` /
:func:`shared.utilities.regex_side_mirror`) using the shipped default pattern
:data:`DEFAULT_SIDE_MIRROR_REGEX`.

The reference catalog (EXPECTED_PAIRS + EXPECTED_CENTERS) was built by
enumerating every blend shape channel on the Ada head DNA
(``tests/test_files/dna/ada/head.dna`` -- 782 channels total) and classifying
each with the production twin-gated regex classifier. The catalog is asserted
live against the DNA in :func:`test_catalog_matches_live_dna`, so any future
schema drift (added / renamed channel) immediately fails the suite.

The default pattern handles the four MetaHuman channel conventions:

* trailing ``_L`` / ``_R`` suffix (``brow_down_L``),
* trailing capital ``L`` / ``R`` after an upper/lower or in/out token
  (``mouth_press_UL``, ``Braise_Ewiden_INL``, ``HturnUp_NKstretch_ML``),
* whole-word ``_left`` / ``_right`` suffix (``brow_raiseOuter_left``),
* ``_L_`` / ``_R_`` infix.

Classification is twin-gated: a channel is only sided when its mirrored twin
also exists, so center channels that incidentally end in a side letter
(none here) would stay center, and correctives (``*_cor``), ``mouth_up`` /
``jaw_open``, head turn/tilt and tongue ``*_IN`` channels are all center.
"""

from __future__ import annotations

import pytest

from character_dna.naming import DEFAULT_SIDE_MIRROR_REGEX, classify_side_by_regex, regex_side_mirror


# ---------------------------------------------------------------------------
# Reference catalog (must mirror tests/test_files/dna/ada/head.dna exactly)
# ---------------------------------------------------------------------------

# 341 mirror pairs (left channel, right channel).
EXPECTED_PAIRS: tuple[tuple[str, str], ...] = (
    ("brow_down_L", "brow_down_R"),
    ("brow_lateral_L", "brow_lateral_R"),
    ("brow_raiseIn_L", "brow_raiseIn_R"),
    ("brow_raiseOuter_left", "brow_raiseOuter_right"),
    ("Bdown_Blateral__browLower_L", "Bdown_Blateral__browLower_R"),
    ("BraiseIn_Bdown__browInnerRaise_L", "BraiseIn_Bdown__browInnerRaise_R"),
    ("BraiseIn_Bdown_Blateral_L", "BraiseIn_Bdown_Blateral_R"),
    ("Braise_Ewiden_INL", "Braise_Ewiden_INR"),
    ("Braise_Ewiden_OUTL", "Braise_Ewiden_OUTR"),
    ("Braise_Eblink_INL", "Braise_Eblink_INR"),
    ("Braise_Eblink_OUTL", "Braise_Eblink_OUTR"),
    ("Braise_ElookDown_INL", "Braise_ElookDown_INR"),
    ("Braise_ElookDown_OUTL", "Braise_ElookDown_OUTR"),
    ("ear_up_L", "ear_up_R"),
    ("eye_blink_L", "eye_blink_R"),
    ("brow_raise_L", "brow_raise_R"),
    ("ElidPress_Eblink_L", "ElidPress_Eblink_R"),
    ("eye_widen_L", "eye_widen_R"),
    ("eye_squintInner_L", "eye_squintInner_R"),
    ("eye_cheekRaise_L", "eye_cheekRaise_R"),
    ("EcheekRaise_EsquintInner_L", "EcheekRaise_EsquintInner_R"),
    ("EsquintInner_Eblink_L", "EsquintInner_Eblink_R"),
    ("EcheekRaise_Eblink_L", "EcheekRaise_Eblink_R"),
    ("EcheekRaise_EsquintInner_Eblink_L", "EcheekRaise_EsquintInner_Eblink_R"),
    ("eye_faceScrunch_L", "eye_faceScrunch_R"),
    ("EfaceScrunch_Eblink_L", "EfaceScrunch_Eblink_R"),
    ("eye_upperLidUp_L", "eye_upperLidUp_R"),
    ("eye_relax_L", "eye_relax_R"),
    ("eye_lowerLidUp_L", "eye_lowerLidUp_R"),
    ("eye_lowerLidDown_L", "eye_lowerLidDown_R"),
    ("eye_lookUp_L", "eye_lookUp_R"),
    ("eye_lookDown_L", "eye_lookDown_R"),
    ("eye_lookLeft_L", "eye_lookLeft_R"),
    ("eye_lookRight_L", "eye_lookRight_R"),
    ("ElookUp_ElookLeft_L", "ElookUp_ElookLeft_R"),
    ("ElookUp_ElookRight_L", "ElookUp_ElookRight_R"),
    ("ElookDown_ElookLeft_L", "ElookDown_ElookLeft_R"),
    ("ElookDown_ElookRight_L", "ElookDown_ElookRight_R"),
    ("ElookUp_Eblink_L", "ElookUp_Eblink_R"),
    ("ElookDown_Eblink_L", "ElookDown_Eblink_R"),
    ("ElookLeft_Eblink_L", "ElookLeft_Eblink_R"),
    ("ElookRight_Eblink_L", "ElookRight_Eblink_R"),
    ("ElookUp_ElookLeft_Eblink_L", "ElookUp_ElookLeft_Eblink_R"),
    ("ElookUp_ElookRight_Eblink_L", "ElookUp_ElookRight_Eblink_R"),
    ("ElookDown_ElookLeft_Eblink_L", "ElookDown_ElookLeft_Eblink_R"),
    ("ElookDown_ElookRight_Eblink_L", "ElookDown_ElookRight_Eblink_R"),
    ("ElookDown_Ewiden_L", "ElookDown_Ewiden_R"),
    ("EcheekRaise_NSwrinkle_L", "EcheekRaise_NSwrinkle_R"),
    ("eye_pupilWide_L", "eye_pupilWide_R"),
    ("eye_pupilNarrow_L", "eye_pupilNarrow_R"),
    ("nose_wrinkle_left", "nose_wrinkle_right"),
    ("nose_wrinkleFull_L", "nose_wrinkleFull_R"),
    ("NSwrinkle_Jopen_L", "NSwrinkle_Jopen_R"),
    ("nose_nostrilsDepress_L", "nose_nostrilsDepress_R"),
    ("nose_nostrilsDilate_L", "nose_nostrilsDilate_R"),
    ("nose_nostrilsCompress_L", "nose_nostrilsCompress_R"),
    ("nose_nasolabialDeepener_L", "nose_nasolabialDeepener_R"),
    ("cheek_suck_L", "cheek_suck_R"),
    ("cheek_blow_left", "cheek_blow_right"),
    ("Cblow_MlipsBlow_L", "Cblow_MlipsBlow_R"),
    ("mouth_lipsBlow_L", "mouth_lipsBlow_R"),
    ("mouth_left", "mouth_right"),
    ("mouth_upperLipRaise_left", "mouth_upperLipRaise_right"),
    ("MupperLipRaise_Jopen_L", "MupperLipRaise_Jopen_R"),
    ("mouth_lowerLipDepress_left", "mouth_lowerLipDepress_right"),
    ("MlowerLipDepress_Jopen_L", "MlowerLipDepress_Jopen_R"),
    ("MupperLipRaise_MlowerLipDepress_L", "MupperLipRaise_MlowerLipDepress_R"),
    ("MupperLipRaise_MlowerLipDepress_Jopen_L", "MupperLipRaise_MlowerLipDepress_Jopen_R"),
    ("mouth_cornerPull_left", "mouth_cornerPull_right"),
    ("mouth_stretch_left", "mouth_stretch_right"),
    ("Mstretch_MstretchLipsClose_L", "Mstretch_MstretchLipsClose_R"),
    ("McornerPull_Mstretch_L", "McornerPull_Mstretch_R"),
    ("McornerPull_Mstretch_Jopen_L", "McornerPull_Mstretch_Jopen_R"),
    ("mouth_dimple_left", "mouth_dimple_right"),
    ("mouth_cornerDepress_L", "mouth_cornerDepress_R"),
    ("mouth_press_UL", "mouth_press_UR"),
    ("mouth_press_DL", "mouth_press_DR"),
    ("mouth_lipsPurse_UL", "mouth_lipsPurse_UR"),
    ("mouth_lipsPurse_DL", "mouth_lipsPurse_DR"),
    ("mouth_lipsTowards_UL", "mouth_lipsTowards_UR"),
    ("mouth_lipsTowards_DL", "mouth_lipsTowards_DR"),
    ("mouth_funnel_UL", "mouth_funnel_UR"),
    ("mouth_funnel_DL", "mouth_funnel_DR"),
    ("Mpurse_Mtowards__pucker_UL", "Mpurse_Mtowards__pucker_UR"),
    ("Mpurse_Mtowards__pucker_DL", "Mpurse_Mtowards__pucker_DR"),
    ("Mpurse_Mfunnel_UL", "Mpurse_Mfunnel_UR"),
    ("Mpurse_Mfunnel_DL", "Mpurse_Mfunnel_DR"),
    ("Mfunnel_Mtowards_UL", "Mfunnel_Mtowards_UR"),
    ("Mfunnel_Mtowards_DL", "Mfunnel_Mtowards_DR"),
    ("Mpurse_Mtowards_Mfunnel__oh_UL", "Mpurse_Mtowards_Mfunnel__oh_UR"),
    ("Mpurse_Mtowards_Mfunnel__oh_DL", "Mpurse_Mtowards_Mfunnel__oh_DR"),
    ("Mpurse_Jopen_UL", "Mpurse_Jopen_UR"),
    ("Mpurse_Jopen_DL", "Mpurse_Jopen_DR"),
    ("Mtowards_Jopen_UL", "Mtowards_Jopen_UR"),
    ("Mtowards_Jopen_DL", "Mtowards_Jopen_DR"),
    ("Mfunnel_Jopen_UL", "Mfunnel_Jopen_UR"),
    ("Mfunnel_Jopen_DL", "Mfunnel_Jopen_DR"),
    ("Mpurse_Mtowards_Jopen__puckerJawOpen_UL", "Mpurse_Mtowards_Jopen__puckerJawOpen_UR"),
    ("Mpurse_Mtowards_Jopen__puckerJawOpen_DL", "Mpurse_Mtowards_Jopen__puckerJawOpen_DR"),
    ("Mpurse_Mfunnel_Jopen_UL", "Mpurse_Mfunnel_Jopen_UR"),
    ("Mpurse_Mfunnel_Jopen_DL", "Mpurse_Mfunnel_Jopen_DR"),
    ("Mfunnel_Mtowards_Jopen_UL", "Mfunnel_Mtowards_Jopen_UR"),
    ("Mfunnel_Mtowards_Jopen_DL", "Mfunnel_Mtowards_Jopen_DR"),
    ("Mpurse_Mtowards_Mfunnel_Jopen__ohJawOpen_UL", "Mpurse_Mtowards_Mfunnel_Jopen__ohJawOpen_UR"),
    ("Mpurse_Mtowards_Mfunnel_Jopen__ohJawOpen_DL", "Mpurse_Mtowards_Mfunnel_Jopen__ohJawOpen_DR"),
    ("McornerPull_Jopen_L", "McornerPull_Jopen_R"),
    ("McornerPull_MlowerLipDepress_L", "McornerPull_MlowerLipDepress_R"),
    ("McornerPull_MlowerLipDepress_Jopen_L", "McornerPull_MlowerLipDepress_Jopen_R"),
    ("Mdimple_Jopen_L", "Mdimple_Jopen_R"),
    ("Mstretch_Jopen_L", "Mstretch_Jopen_R"),
    ("Mfunnel_MupperLipRaise_UL", "Mfunnel_MupperLipRaise_UR"),
    ("Mfunnel_MupperLipRaise_DL", "Mfunnel_MupperLipRaise_DR"),
    ("Mfunnel_MlowerLipDepress_UL", "Mfunnel_MlowerLipDepress_UR"),
    ("Mfunnel_MlowerLipDepress_DL", "Mfunnel_MlowerLipDepress_DR"),
    (
        "Mfunnel_MupperLipRaise_MlowerLipDepress__funnelWide_UL",
        "Mfunnel_MupperLipRaise_MlowerLipDepress__funnelWide_UR",
    ),
    (
        "Mfunnel_MupperLipRaise_MlowerLipDepress__funnelWide_DL",
        "Mfunnel_MupperLipRaise_MlowerLipDepress__funnelWide_DR",
    ),
    ("Mpress_Jopen_UL", "Mpress_Jopen_UR"),
    ("Mpress_Jopen_DL", "Mpress_Jopen_DR"),
    ("MlipsTogether_Jopen_UL", "MlipsTogether_Jopen_UR"),
    ("MlipsTogether_Jopen_DL", "MlipsTogether_Jopen_DR"),
    ("MlipsTogether_Mpress_Jopen__mouthSuck_UL", "MlipsTogether_Mpress_Jopen__mouthSuck_UR"),
    ("MlipsTogether_Mpress_Jopen__mouthSuck_DL", "MlipsTogether_Mpress_Jopen__mouthSuck_DR"),
    ("MupperLipRaise_NSwrinkle_L", "MupperLipRaise_NSwrinkle_R"),
    ("MupperLipRaise_McornerDepress_L", "MupperLipRaise_McornerDepress_R"),
    ("McornerDepress_NSwrinkle_L", "McornerDepress_NSwrinkle_R"),
    ("McornerDepress_Jopen_L", "McornerDepress_Jopen_R"),
    ("MupperLipRaise_NSwrinkle_McornerDepress_L", "MupperLipRaise_NSwrinkle_McornerDepress_R"),
    ("MupperLipRaise_NSwrinkle_Jopen_L", "MupperLipRaise_NSwrinkle_Jopen_R"),
    ("MupperLipRaise_McornerDepress_Jopen_L", "MupperLipRaise_McornerDepress_Jopen_R"),
    ("McornerDepress_NSwrinkle_Jopen_L", "McornerDepress_NSwrinkle_Jopen_R"),
    ("MupperLipRaise_NSwrinkle_McornerDepress_Jopen_L", "MupperLipRaise_NSwrinkle_McornerDepress_Jopen_R"),
    ("Mstretch_MupperLipRaise_L", "Mstretch_MupperLipRaise_R"),
    ("Mstretch_MupperLipRaise_Jopen_L", "Mstretch_MupperLipRaise_Jopen_R"),
    ("mouth_upperLipBite_L", "mouth_upperLipBite_R"),
    ("mouth_lowerLipBite_L", "mouth_lowerLipBite_R"),
    ("mouth_lipsBite_L", "mouth_lipsBite_R"),
    ("MupperLipBite_Jopen_L", "MupperLipBite_Jopen_R"),
    ("MlowerLipBite_Jopen_L", "MlowerLipBite_Jopen_R"),
    ("MlowerLipBite_MupperLipBite_Jopen_L", "MlowerLipBite_MupperLipBite_Jopen_R"),
    ("mouth_lipsTighten_UL", "mouth_lipsTighten_UR"),
    ("mouth_lipsTighten_DL", "mouth_lipsTighten_DR"),
    ("mouth_lipsPress_L", "mouth_lipsPress_R"),
    ("McornerPull_Mpurse_UL", "McornerPull_Mpurse_UR"),
    ("McornerPull_Mpurse_DL", "McornerPull_Mpurse_DR"),
    ("McornerPull_Mtowards_UL", "McornerPull_Mtowards_UR"),
    ("McornerPull_Mtowards_DL", "McornerPull_Mtowards_DR"),
    ("McornerPull_Mfunnel_UL", "McornerPull_Mfunnel_UR"),
    ("McornerPull_Mfunnel_DL", "McornerPull_Mfunnel_DR"),
    ("McornerPull_Mpurse_Mtowards__cornerPullPucker_UL", "McornerPull_Mpurse_Mtowards__cornerPullPucker_UR"),
    ("McornerPull_Mpurse_Mtowards__cornerPullPucker_DL", "McornerPull_Mpurse_Mtowards__cornerPullPucker_DR"),
    ("McornerPull_Mpurse_Mfunnel_UL", "McornerPull_Mpurse_Mfunnel_UR"),
    ("McornerPull_Mpurse_Mfunnel_DL", "McornerPull_Mpurse_Mfunnel_DR"),
    ("McornerPull_Mfunnel_Mtowards_UL", "McornerPull_Mfunnel_Mtowards_UR"),
    ("McornerPull_Mfunnel_Mtowards_DL", "McornerPull_Mfunnel_Mtowards_DR"),
    ("McornerPull_Mpurse_Mtowards_Mfunnel__cornerPullOh_UL", "McornerPull_Mpurse_Mtowards_Mfunnel__cornerPullOh_UR"),
    ("McornerPull_Mpurse_Mtowards_Mfunnel__cornerPullOh_DL", "McornerPull_Mpurse_Mtowards_Mfunnel__cornerPullOh_DR"),
    ("MupperLipRaise_Mtighten_UL", "MupperLipRaise_Mtighten_UR"),
    ("MupperLipRaise_Mtighten_DL", "MupperLipRaise_Mtighten_DR"),
    ("MlowerLipDepress_Mtighten_UL", "MlowerLipDepress_Mtighten_UR"),
    ("MlowerLipDepress_Mtighten_DL", "MlowerLipDepress_Mtighten_DR"),
    ("MupperLipRaise_MlowerLipDepress_Mtighten_UL", "MupperLipRaise_MlowerLipDepress_Mtighten_UR"),
    ("MupperLipRaise_MlowerLipDepress_Mtighten_DL", "MupperLipRaise_MlowerLipDepress_Mtighten_DR"),
    ("mouth_sharpCornerPull_L", "mouth_sharpCornerPull_R"),
    ("MsharpCornerPull_Jopen_L", "MsharpCornerPull_Jopen_R"),
    ("mouth_sticky_UINL", "mouth_sticky_UINR"),
    ("mouth_sticky_UOUTL", "mouth_sticky_UOUTR"),
    ("mouth_sticky_DINL", "mouth_sticky_DINR"),
    ("mouth_sticky_DOUTL", "mouth_sticky_DOUTR"),
    ("mouth_lipsSticky_L_ph1", "mouth_lipsSticky_R_ph1"),
    ("mouth_lipsSticky_L_ph2", "mouth_lipsSticky_R_ph2"),
    ("mouth_lipsSticky_L_ph3", "mouth_lipsSticky_R_ph3"),
    ("mouth_lipsPush_UL", "mouth_lipsPush_UR"),
    ("mouth_lipsPush_DL", "mouth_lipsPush_DR"),
    ("mouth_lipsPull_UL", "mouth_lipsPull_UR"),
    ("mouth_lipsPull_DL", "mouth_lipsPull_DR"),
    ("mouth_lipsThin_UL", "mouth_lipsThin_UR"),
    ("mouth_lipsThin_DL", "mouth_lipsThin_DR"),
    ("mouth_lipsThick_UL", "mouth_lipsThick_UR"),
    ("mouth_lipsThick_DL", "mouth_lipsThick_DR"),
    ("mouth_lipsThinInward_UL", "mouth_lipsThinInward_UR"),
    ("mouth_lipsThinInward_DL", "mouth_lipsThinInward_DR"),
    ("mouth_lipsThickInward_UL", "mouth_lipsThickInward_UR"),
    ("mouth_lipsThickInward_DL", "mouth_lipsThickInward_DR"),
    ("mouth_cornerSharpen_UL", "mouth_cornerSharpen_UR"),
    ("mouth_cornerSharpen_DL", "mouth_cornerSharpen_DR"),
    ("mouth_cornerRounder_UL", "mouth_cornerRounder_UR"),
    ("mouth_cornerRounder_DL", "mouth_cornerRounder_DR"),
    ("McornerPull_EcheekRaise_L", "McornerPull_EcheekRaise_R"),
    ("Mstretch_JlowerChinRaise_L", "Mstretch_JlowerChinRaise_R"),
    ("Mdimple_MupperLipRaise_L", "Mdimple_MupperLipRaise_R"),
    ("Mdimple_MlowerLipDepress_L", "Mdimple_MlowerLipDepress_R"),
    ("Mdimple_MupperLipRaise_MlowerLipDepress__ee_L", "Mdimple_MupperLipRaise_MlowerLipDepress__ee_R"),
    ("McornerPull_MsharpCornerPull_L", "McornerPull_MsharpCornerPull_R"),
    ("McornerPull_Mdimple_L", "McornerPull_Mdimple_R"),
    ("Mstretch_MlowerLipDepress_L", "Mstretch_MlowerLipDepress_R"),
    ("McornerPull_MupperLipRaise_L", "McornerPull_MupperLipRaise_R"),
    ("McornerPull_NSwrinkle_L", "McornerPull_NSwrinkle_R"),
    ("McornerPull_MupperLipRaise_NSwrinkle_L", "McornerPull_MupperLipRaise_NSwrinkle_R"),
    ("McornerPull_MupperLipRaise_Jopen_L", "McornerPull_MupperLipRaise_Jopen_R"),
    ("McornerPull_NSwrinkle_Jopen_L", "McornerPull_NSwrinkle_Jopen_R"),
    ("McornerPull_MupperLipRaise_NSwrinkle_Jopen_L", "McornerPull_MupperLipRaise_NSwrinkle_Jopen_R"),
    ("Mstretch_NSdepress_L", "Mstretch_NSdepress_R"),
    ("Mpurse_Mtighten_UL", "Mpurse_Mtighten_UR"),
    ("Mpurse_Mtighten_DL", "Mpurse_Mtighten_DR"),
    ("Mtowards_Mtighten_UL", "Mtowards_Mtighten_UR"),
    ("Mtowards_Mtighten_DL", "Mtowards_Mtighten_DR"),
    ("Mfunnel_Mtighten_UL", "Mfunnel_Mtighten_UR"),
    ("Mfunnel_Mtighten_DL", "Mfunnel_Mtighten_DR"),
    ("Mpurse_Mtowards_Mtighten__puckerTighten_UL", "Mpurse_Mtowards_Mtighten__puckerTighten_UR"),
    ("Mpurse_Mtowards_Mtighten__puckerTighten_DL", "Mpurse_Mtowards_Mtighten__puckerTighten_DR"),
    ("Mpurse_Mfunnel_Mtighten_UL", "Mpurse_Mfunnel_Mtighten_UR"),
    ("Mpurse_Mfunnel_Mtighten_DL", "Mpurse_Mfunnel_Mtighten_DR"),
    ("Mfunnel_Mtowards_Mtighten_UL", "Mfunnel_Mtowards_Mtighten_UR"),
    ("Mfunnel_Mtowards_Mtighten_DL", "Mfunnel_Mtowards_Mtighten_DR"),
    ("Mpurse_Mtowards_Mfunnel_Mtighten__ohTighten_UL", "Mpurse_Mtowards_Mfunnel_Mtighten__ohTighten_UR"),
    ("Mpurse_Mtowards_Mfunnel_Mtighten__ohTighten_DL", "Mpurse_Mtowards_Mfunnel_Mtighten__ohTighten_DR"),
    ("Mstretch_Mpurse_UL", "Mstretch_Mpurse_UR"),
    ("Mstretch_Mpurse_DL", "Mstretch_Mpurse_DR"),
    ("Mstretch_Mtowards_UL", "Mstretch_Mtowards_UR"),
    ("Mstretch_Mtowards_DL", "Mstretch_Mtowards_DR"),
    ("Mstretch_Mfunnel_UL", "Mstretch_Mfunnel_UR"),
    ("Mstretch_Mfunnel_DL", "Mstretch_Mfunnel_DR"),
    ("Mstretch_Mpurse_Mtowards__mouthStretchPucker_UL", "Mstretch_Mpurse_Mtowards__mouthStretchPucker_UR"),
    ("Mstretch_Mpurse_Mtowards__mouthStretchPucker_DL", "Mstretch_Mpurse_Mtowards__mouthStretchPucker_DR"),
    ("Mstretch_Mpurse_Mfunnel_UL", "Mstretch_Mpurse_Mfunnel_UR"),
    ("Mstretch_Mpurse_Mfunnel_DL", "Mstretch_Mpurse_Mfunnel_DR"),
    ("Mstretch_Mfunnel_Mtowards_UL", "Mstretch_Mfunnel_Mtowards_UR"),
    ("Mstretch_Mfunnel_Mtowards_DL", "Mstretch_Mfunnel_Mtowards_DR"),
    ("Mstretch_Mpurse_Mtowards_Mfunnel__mouthStretchOh_UL", "Mstretch_Mpurse_Mtowards_Mfunnel__mouthStretchOh_UR"),
    ("Mstretch_Mpurse_Mtowards_Mfunnel__mouthStretchOh_DL", "Mstretch_Mpurse_Mtowards_Mfunnel__mouthStretchOh_DR"),
    ("Mdimple_Mpurse_UL", "Mdimple_Mpurse_UR"),
    ("Mdimple_Mpurse_DL", "Mdimple_Mpurse_DR"),
    ("Mdimple_Mtowards_UL", "Mdimple_Mtowards_UR"),
    ("Mdimple_Mtowards_DL", "Mdimple_Mtowards_DR"),
    ("Mdimple_Mfunnel_UL", "Mdimple_Mfunnel_UR"),
    ("Mdimple_Mfunnel_DL", "Mdimple_Mfunnel_DR"),
    ("Mdimple_Mpurse_Mtowards__dimplePucker_UL", "Mdimple_Mpurse_Mtowards__dimplePucker_UR"),
    ("Mdimple_Mpurse_Mtowards__dimplePucker_DL", "Mdimple_Mpurse_Mtowards__dimplePucker_DR"),
    ("Mdimple_Mpurse_Mfunnel_UL", "Mdimple_Mpurse_Mfunnel_UR"),
    ("Mdimple_Mpurse_Mfunnel_DL", "Mdimple_Mpurse_Mfunnel_DR"),
    ("Mdimple_Mfunnel_Mtowards_UL", "Mdimple_Mfunnel_Mtowards_UR"),
    ("Mdimple_Mfunnel_Mtowards_DL", "Mdimple_Mfunnel_Mtowards_DR"),
    ("Mdimple_Mpurse_Mtowards_Mfunnel__dimpleOh_UL", "Mdimple_Mpurse_Mtowards_Mfunnel__dimpleOh_UR"),
    ("Mdimple_Mpurse_Mtowards_Mfunnel__dimpleOh_DL", "Mdimple_Mpurse_Mtowards_Mfunnel__dimpleOh_DR"),
    ("Mstretch_Mdimple_L", "Mstretch_Mdimple_R"),
    ("McornerPull_Mstretch_Mdimple_L", "McornerPull_Mstretch_Mdimple_R"),
    ("Mstretch_MlowerLipDepress_Jopen_L", "Mstretch_MlowerLipDepress_Jopen_R"),
    ("McornerPull_Mstretch_MupperLipRaise_L", "McornerPull_Mstretch_MupperLipRaise_R"),
    ("McornerPull_Mstretch_MlowerLipDepress_L", "McornerPull_Mstretch_MlowerLipDepress_R"),
    ("McornerPull_MlowerLipDepress_MupperLipRaise_L", "McornerPull_MlowerLipDepress_MupperLipRaise_R"),
    ("Mstretch_MupperLipRaise_MlowerLipDepress_L", "Mstretch_MupperLipRaise_MlowerLipDepress_R"),
    (
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_L",
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_R",
    ),
    ("McornerPull_Mstretch_MupperLipRaise_Jopen_L", "McornerPull_Mstretch_MupperLipRaise_Jopen_R"),
    ("McornerPull_Mstretch_MlowerLipDepress_Jopen_L", "McornerPull_Mstretch_MlowerLipDepress_Jopen_R"),
    ("McornerPull_MlowerLipDepress_MupperLipRaise_Jopen_L", "McornerPull_MlowerLipDepress_MupperLipRaise_Jopen_R"),
    ("Mstretch_MlowerLipDepress_MupperLipRaise_Jopen_L", "Mstretch_MlowerLipDepress_MupperLipRaise_Jopen_R"),
    (
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_Jopen_L",
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_Jopen_R",
    ),
    ("Mstretch_NSwrinkle_L", "Mstretch_NSwrinkle_R"),
    ("Mstretch_MupperLipRaise_NSwrinkle_L", "Mstretch_MupperLipRaise_NSwrinkle_R"),
    ("McornerPull_MupperLipRaise_Mdimple_L", "McornerPull_MupperLipRaise_Mdimple_R"),
    ("Mstretch_MupperLipRaise_Mdimple_L", "Mstretch_MupperLipRaise_Mdimple_R"),
    ("McornerPull_MlowerLipDepress_Mdimple_L", "McornerPull_MlowerLipDepress_Mdimple_R"),
    ("Mstretch_MlowerLipDepress_Mdimple_L", "Mstretch_MlowerLipDepress_Mdimple_R"),
    ("McornerPull_Mstretch_MupperLipRaise_Mdimple_L", "McornerPull_Mstretch_MupperLipRaise_Mdimple_R"),
    ("McornerPull_Mstretch_MlowerLipDepress_Mdimple_L", "McornerPull_Mstretch_MlowerLipDepress_Mdimple_R"),
    ("McornerPull_MupperLipRaise_MlowerLipDepress_Mdimple_L", "McornerPull_MupperLipRaise_MlowerLipDepress_Mdimple_R"),
    ("Mstretch_MupperLipRaise_MlowerLipDepress_Mdimple_L", "Mstretch_MupperLipRaise_MlowerLipDepress_Mdimple_R"),
    (
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_Mdimple_L",
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_Mdimple_R",
    ),
    ("McornerPull_Mdimple_Jopen_L", "McornerPull_Mdimple_Jopen_R"),
    ("Mstretch_Mdimple_Jopen_L", "Mstretch_Mdimple_Jopen_R"),
    ("McornerPull_Mstretch_Mdimple_Jopen_L", "McornerPull_Mstretch_Mdimple_Jopen_R"),
    ("McornerPull_Mstretch_NSwrinkle_L", "McornerPull_Mstretch_NSwrinkle_R"),
    ("McornerPull_Mstretch_MupperLipRaise_NSwrinkle_L", "McornerPull_Mstretch_MupperLipRaise_NSwrinkle_R"),
    ("Mstretch_NSwrinkle_Jopen_L", "Mstretch_NSwrinkle_Jopen_R"),
    ("Mstretch_MupperLipRaise_NSwrinkle_Jopen_L", "Mstretch_MupperLipRaise_NSwrinkle_Jopen_R"),
    ("McornerPull_Mstretch_NSwrinkle_Jopen_L", "McornerPull_Mstretch_NSwrinkle_Jopen_R"),
    ("McornerPull_Mstretch_MupperLipRaise_NSwrinkle_Jopen_L", "McornerPull_Mstretch_MupperLipRaise_NSwrinkle_Jopen_R"),
    ("McornerPull_JopenExtreme_L", "McornerPull_JopenExtreme_R"),
    ("Mstretch_JopenExtreme_L", "Mstretch_JopenExtreme_R"),
    ("MupperLipRaise_JopenExtreme_L", "MupperLipRaise_JopenExtreme_R"),
    ("MlowerLipDepress_JopenExtreme_L", "MlowerLipDepress_JopenExtreme_R"),
    ("McornerPull_Mstretch_JopenExtreme_L", "McornerPull_Mstretch_JopenExtreme_R"),
    ("McornerPull_MupperLipRaise_JopenExtreme_L", "McornerPull_MupperLipRaise_JopenExtreme_R"),
    ("Mstretch_MupperLipRaise_JopenExtreme_L", "Mstretch_MupperLipRaise_JopenExtreme_R"),
    ("McornerPull_MlowerLipDepress_JopenExtreme_L", "McornerPull_MlowerLipDepress_JopenExtreme_R"),
    ("Mstretch_MlowerLipDepress_JopenExtreme_L", "Mstretch_MlowerLipDepress_JopenExtreme_R"),
    ("MupperLipRaise_MlowerLipDepress_JopenExtreme_L", "MupperLipRaise_MlowerLipDepress_JopenExtreme_R"),
    ("McornerPull_Mstretch_MupperLipRaise_JopenExtreme_L", "McornerPull_Mstretch_MupperLipRaise_JopenExtreme_R"),
    ("McornerPull_Mstretch_MlowerLipDepress_JopenExtreme_L", "McornerPull_Mstretch_MlowerLipDepress_JopenExtreme_R"),
    (
        "McornerPull_MlowerLipDepress_MupperLipRaise_JopenExtreme_L",
        "McornerPull_MlowerLipDepress_MupperLipRaise_JopenExtreme_R",
    ),
    (
        "Mstretch_MlowerLipDepress_MupperLipRaise_JopenExtreme_L",
        "Mstretch_MlowerLipDepress_MupperLipRaise_JopenExtreme_R",
    ),
    (
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtreme_L",
        "McornerPull_Mstretch_MupperLipRaise_MlowerLipDepress_JopenExtreme_R",
    ),
    ("mouth_upperLipTowardsTeeth_L", "mouth_upperLipTowardsTeeth_R"),
    ("mouth_lowerLipTowardsTeeth_L", "mouth_lowerLipTowardsTeeth_R"),
    ("mouth_upperLipRollIn_L", "mouth_upperLipRollIn_R"),
    ("mouth_upperLipRollOut_L", "mouth_upperLipRollOut_R"),
    ("mouth_lowerLipRollIn_L", "mouth_lowerLipRollIn_R"),
    ("mouth_lowerLipRollOut_L", "mouth_lowerLipRollOut_R"),
    ("mouth_cornersUp_L", "mouth_cornersUp_R"),
    ("mouth_cornersDown_L", "mouth_cornersDown_R"),
    ("mouth_cornersWide_L", "mouth_cornersWide_R"),
    ("mouth_cornersNarrow_L", "mouth_cornersNarrow_R"),
    ("jaw_left", "jaw_right"),
    ("Jleft_MlipsTogether_UL", "Jleft_MlipsTogether_UR"),
    ("Jleft_MlipsTogether_DL", "Jleft_MlipsTogether_DR"),
    ("Jright_MlipsTogether_UL", "Jright_MlipsTogether_UR"),
    ("Jright_MlipsTogether_DL", "Jright_MlipsTogether_DR"),
    ("Jleft_MlipsTogether_Jopen_UL", "Jleft_MlipsTogether_Jopen_UR"),
    ("Jleft_MlipsTogether_Jopen_DL", "Jleft_MlipsTogether_Jopen_DR"),
    ("Jright_MlipsTogether_Jopen_UL", "Jright_MlipsTogether_Jopen_UR"),
    ("Jright_MlipsTogether_Jopen_DL", "Jright_MlipsTogether_Jopen_DR"),
    ("jaw_clench_L", "jaw_clench_R"),
    ("jaw_lowerChinRaise_L", "jaw_lowerChinRaise_R"),
    ("JlowerChinRaise_JupperChinRaise_L", "JlowerChinRaise_JupperChinRaise_R"),
    ("JlowerChinRaise_MupperLipRaise_L", "JlowerChinRaise_MupperLipRaise_R"),
    ("JlowerChinRaise_Jopen_L", "JlowerChinRaise_Jopen_R"),
    ("jaw_chinCompress_L", "jaw_chinCompress_R"),
    ("neck_stretch_L", "neck_stretch_R"),
    ("NKstretch_Mstretch_L", "NKstretch_Mstretch_R"),
    ("NKstretch_Jopen_L", "NKstretch_Jopen_R"),
    ("NKstretch_Mstretch_Jopen_L", "NKstretch_Mstretch_Jopen_R"),
    ("neck_mastoidContract_L", "neck_mastoidContract_R"),
    ("HturnUp_NKstretch_UL", "HturnUp_NKstretch_UR"),
    ("HturnUp_NKstretch_ML", "HturnUp_NKstretch_MR"),
    ("HturnUp_NKstretch_DL", "HturnUp_NKstretch_DR"),
    ("HturnDown_NKstretch_UL", "HturnDown_NKstretch_UR"),
    ("HturnDown_NKstretch_ML", "HturnDown_NKstretch_MR"),
    ("HturnDown_NKstretch_DL", "HturnDown_NKstretch_DR"),
    ("HturnLeft_NKstretch_UL", "HturnLeft_NKstretch_UR"),
    ("HturnLeft_NKstretch_ML", "HturnLeft_NKstretch_MR"),
    ("HturnLeft_NKstretch_DL", "HturnLeft_NKstretch_DR"),
    ("HturnRight_NKstretch_UL", "HturnRight_NKstretch_UR"),
    ("HturnRight_NKstretch_ML", "HturnRight_NKstretch_MR"),
    ("HturnRight_NKstretch_DL", "HturnRight_NKstretch_DR"),
    ("HtiltLeft_NKstretch_UL", "HtiltLeft_NKstretch_UR"),
    ("HtiltLeft_NKstretch_ML", "HtiltLeft_NKstretch_MR"),
    ("HtiltLeft_NKstretch_DL", "HtiltLeft_NKstretch_DR"),
    ("HtiltRight_NKstretch_UL", "HtiltRight_NKstretch_UR"),
    ("HtiltRight_NKstretch_ML", "HtiltRight_NKstretch_MR"),
    ("HtiltRight_NKstretch_DL", "HtiltRight_NKstretch_DR"),
    ("tongue_left_IN", "tongue_right_IN"),
)

# 100 center channels (no detectable side, or no mirror twin).
EXPECTED_CENTERS: tuple[str, ...] = (
    "nose_wrinkle_cor",
    "cheek_blow_cor",
    "mouth_up",
    "mouth_down",
    "mouth_upperLipRaise_cor",
    "mouth_lowerLipDepress_cor",
    "mouth_cornerPull_cor",
    "mouth_stretch_cor",
    "mouth_dimple_cor",
    "mouth_sticky_UC",
    "mouth_sticky_DC",
    "mouth_upperLipShiftLeft",
    "mouth_upperLipShiftRight",
    "mouth_lowerLipShiftLeft",
    "mouth_lowerLipShiftRight",
    "jaw_open",
    "jaw_fwd",
    "jaw_back",
    "Jleft_Jopen_cor",
    "Jright_Jopen_cor",
    "jaw_openExtreme_cor",
    "neck_swallow_ph1",
    "neck_swallow_ph2",
    "neck_swallow_ph3",
    "neck_swallow_ph4",
    "neck_throatDown",
    "neck_throatUp",
    "neck_digastricDown",
    "neck_digastricUp",
    "neck_throatExhale",
    "neck_throatInhale",
    "head_turnUp_U",
    "head_turnUp_M",
    "head_turnUp_D",
    "head_turnDown_U",
    "head_turnDown_M",
    "head_turnDown_D",
    "head_turnLeft_U",
    "head_turnLeft_M",
    "head_turnLeft_D",
    "head_turnRight_U",
    "head_turnRight_M",
    "head_turnRight_D",
    "head_tiltLeft_U",
    "head_tiltLeft_M",
    "head_tiltLeft_D",
    "head_tiltRight_U",
    "head_tiltRight_M",
    "head_tiltRight_D",
    "HturnUp_NKthroatExhale_U",
    "HturnUp_NKthroatExhale_M",
    "HturnUp_NKthroatExhale_D",
    "HturnUp_NKthroatInhale_U",
    "HturnUp_NKthroatInhale_M",
    "HturnUp_NKthroatInhale_D",
    "HturnDown_NKthroatExhale_U",
    "HturnDown_NKthroatExhale_M",
    "HturnDown_NKthroatExhale_D",
    "HturnDown_NKthroatInhale_U",
    "HturnDown_NKthroatInhale_M",
    "HturnDown_NKthroatInhale_D",
    "tongue_up_IN",
    "tongue_down_IN",
    "tongue_out_IN",
    "tongue_in_IN",
    "tongue_bendUp_IN",
    "tongue_bendDown_IN",
    "tongue_twistLeft_IN",
    "tongue_twistRight_IN",
    "tongue_tipUp_IN",
    "tongue_tipDown_IN",
    "tongue_tipLeft_IN",
    "tongue_tipRight_IN",
    "tongue_wide_IN",
    "tongue_narrow_IN",
    "tongue_press_IN",
    "tongue_roll_IN",
    "tongue_thick_IN",
    "tongue_thin_IN",
    "tongue_retraction_IN",
    "tongue_dartingOut_IN",
    "tongue_protrusion_IN",
    "tongue_outLeft_IN",
    "tongue_outRight_IN",
    "tongue_narrowLeft_IN",
    "tongue_narrowRight_IN",
    "tongue_outBendUp_IN",
    "tongue_outBendDown_IN",
    "tongue_outUp_IN",
    "tongue_upBendUp_IN",
    "tongue_upBendDown_IN",
    "tongue_bendUpTwistLeft_IN",
    "tongue_bendUpTwistRight_IN",
    "tongue_bendDownTwistLeft_IN",
    "tongue_bendDownTwistRight_IN",
    "tongue_outRoll_IN",
    "tongue_lateralLeft_IN",
    "tongue_lateralRight_IN",
    "tongue_elevation_IN",
    "tongue_depression_IN",
)


def _all_channel_names() -> set[str]:
    names: set[str] = set(EXPECTED_CENTERS)
    for left, right in EXPECTED_PAIRS:
        names.add(left)
        names.add(right)
    return names


EXPECTED_ALL_NAMES: frozenset[str] = frozenset(_all_channel_names())


# ---------------------------------------------------------------------------
# Catalog sanity (catches mistakes in the hardcoded reference itself)
# ---------------------------------------------------------------------------


def test_catalog_pair_and_center_counts() -> None:
    pair_lefts = {left for left, _ in EXPECTED_PAIRS}
    pair_rights = {right for _, right in EXPECTED_PAIRS}
    centers = set(EXPECTED_CENTERS)

    assert len(EXPECTED_PAIRS) == 341
    assert len(EXPECTED_CENTERS) == 100
    assert len(pair_lefts) == len(EXPECTED_PAIRS), "duplicate left name in EXPECTED_PAIRS"
    assert len(pair_rights) == len(EXPECTED_PAIRS), "duplicate right name in EXPECTED_PAIRS"
    assert pair_lefts.isdisjoint(pair_rights), "name appears as both left and right"
    assert centers.isdisjoint(pair_lefts | pair_rights), "center name also listed as a pair"
    assert len(EXPECTED_ALL_NAMES) == 782


# ---------------------------------------------------------------------------
# regex_side_mirror: pairs mirror in both directions and are involutive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_mirror_left_to_right(left: str, right: str) -> None:
    assert regex_side_mirror(left, DEFAULT_SIDE_MIRROR_REGEX) == right


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_mirror_right_to_left(left: str, right: str) -> None:
    assert regex_side_mirror(right, DEFAULT_SIDE_MIRROR_REGEX) == left


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_mirror_is_involutive(left: str, right: str) -> None:
    once = regex_side_mirror(left, DEFAULT_SIDE_MIRROR_REGEX)
    assert once == right
    assert regex_side_mirror(once, DEFAULT_SIDE_MIRROR_REGEX) == left


# ---------------------------------------------------------------------------
# classify_side_by_regex (twin-gated): pairs are L/R, centers are C
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("left", "right"), EXPECTED_PAIRS, ids=lambda p: p if isinstance(p, str) else "")
def test_classify_pair(left: str, right: str) -> None:
    known = set(EXPECTED_ALL_NAMES)
    assert classify_side_by_regex(left, DEFAULT_SIDE_MIRROR_REGEX, known) == "L"
    assert classify_side_by_regex(right, DEFAULT_SIDE_MIRROR_REGEX, known) == "R"


@pytest.mark.parametrize("name", EXPECTED_CENTERS)
def test_classify_center(name: str) -> None:
    known = set(EXPECTED_ALL_NAMES)
    assert classify_side_by_regex(name, DEFAULT_SIDE_MIRROR_REGEX, known) == "C"


def test_classify_orphan_falls_through_to_center() -> None:
    """Twin-exists gate: drop a right twin from the known set and its left
    partner must fall back to center."""
    left, right = EXPECTED_PAIRS[0]
    known = set(EXPECTED_ALL_NAMES) - {right}
    assert classify_side_by_regex(left, DEFAULT_SIDE_MIRROR_REGEX, known) == "C"


# ---------------------------------------------------------------------------
# Live drift guard -- enumerate Ada head.dna and assert the catalog matches
# ---------------------------------------------------------------------------
