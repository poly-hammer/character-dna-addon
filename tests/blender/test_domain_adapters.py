"""Editor APIs and preference defaults use the tested Python domain functions."""

from character_dna import naming
from character_dna.editors.raw_control_editor import core
from character_dna.editors.rbf_editor import properties as rbf_properties, utilities as rbf_utilities
from character_dna.editors.shape_key_editor import properties as shape_properties
from character_dna.editors.shared import utilities


def test_editor_naming_adapters_share_domain_implementations():
    for name in (
        "classify_raw_control_side",
        "mirror_raw_control_name",
        "get_mirrored_name",
        "get_mirror_side_replacement",
    ):
        assert getattr(core, name) is getattr(naming, name)
    for name in ("classify_side_by_regex", "regex_side_mirror"):
        assert getattr(utilities, name) is getattr(naming, name)
    assert rbf_utilities.get_mirrored_name is naming.rbf_mirrored_name
    assert rbf_utilities.can_mirror_name is naming.can_mirror_rbf_name
    assert rbf_utilities.get_mirror_side_replacement is naming.rbf_mirror_side_replacement
    assert rbf_properties.DEFAULT_SOLVER_MIRROR_REGEX == naming.DEFAULT_SOLVER_MIRROR_REGEX
    assert rbf_properties.DEFAULT_POSE_MIRROR_REGEX == naming.DEFAULT_POSE_MIRROR_REGEX
    assert rbf_properties.DEFAULT_BONE_MIRROR_REGEX == naming.DEFAULT_BONE_MIRROR_REGEX
    assert shape_properties.DEFAULT_SIDE_MIRROR_REGEX == naming.DEFAULT_SIDE_MIRROR_REGEX
