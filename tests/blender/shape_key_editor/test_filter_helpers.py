"""Tests for the shared UIList filter/sort helpers used by the Shape Key Editor
and the Raw Control Editor.

``classify_side_by_regex`` classifies a channel/control name as left/right/center
from its name alone (no DNA lookup), so combination correctives such as
``Bdown_Blateral__browLower_L`` are still sided by their ``_L`` / ``_R`` suffix.

``value_sorted_neworder`` builds the Blender ``filter_items`` reorder list
(``neworder[original_index] = display_slot``) and is always applied AFTER every
filter so ordering never interferes with the visible-item flags. Pinned indices
(the bind-pose sentinel row) are forced to the front.
"""

from __future__ import annotations

from character_dna.editors.shared.utilities import classify_side_by_regex, value_sorted_neworder


_DEFAULT_PATTERN = r"(?P<prefix>.+?)(?P<side>_[LlRr]_|_[LlRr]$)"


# ---------------------------------------------------------------------------
# classify_side_by_regex
# ---------------------------------------------------------------------------
def test_trailing_suffix_classifies_left_and_right():
    assert classify_side_by_regex("Bdown_Blateral__browLower_L", _DEFAULT_PATTERN) == "L"
    assert classify_side_by_regex("Bdown_Blateral__browLower_R", _DEFAULT_PATTERN) == "R"


def test_infix_token_classifies_side():
    assert classify_side_by_regex("eye_L_blink", _DEFAULT_PATTERN) == "L"
    assert classify_side_by_regex("eye_R_blink", _DEFAULT_PATTERN) == "R"


def test_unmatched_name_is_center():
    assert classify_side_by_regex("jawOpen", _DEFAULT_PATTERN) == "C"
    assert classify_side_by_regex("mouthStretch", _DEFAULT_PATTERN) == "C"


def test_lowercase_side_token_is_classified():
    assert classify_side_by_regex("brow_l", _DEFAULT_PATTERN) == "L"
    assert classify_side_by_regex("brow_r", _DEFAULT_PATTERN) == "R"


def test_invalid_pattern_is_center_not_error():
    # Unbalanced parenthesis -> re.error -> treated as center.
    assert classify_side_by_regex("brow_L", r"(?P<side>_[LR]") == "C"


def test_pattern_without_side_group_is_center():
    assert classify_side_by_regex("brow_L", r"(?P<prefix>.+)") == "C"


# ---------------------------------------------------------------------------
# value_sorted_neworder
# ---------------------------------------------------------------------------
def _display_order(neworder: list[int]) -> list[int]:
    """Invert a ``neworder`` (original -> slot) into ``slot -> original`` so a
    test can read the visible order directly."""
    order = [0] * len(neworder)
    for original_index, slot in enumerate(neworder):
        order[slot] = original_index
    return order


def test_descending_value_order():
    neworder = value_sorted_neworder([0.1, 0.9, 0.5])
    # Highest value (index 1) first, then index 2, then index 0.
    assert _display_order(neworder) == [1, 2, 0]


def test_ascending_value_order():
    neworder = value_sorted_neworder([0.1, 0.9, 0.5], reverse=False)
    assert _display_order(neworder) == [0, 2, 1]


def test_empty_values():
    assert value_sorted_neworder([]) == []


def test_pinned_index_forced_to_front():
    # Index 0 is the sentinel with the lowest value but must stay first.
    neworder = value_sorted_neworder([0.0, 0.9, 0.5], pinned_indices=[0])
    assert _display_order(neworder) == [0, 1, 2]


def test_out_of_range_pinned_index_ignored():
    neworder = value_sorted_neworder([0.1, 0.9], pinned_indices=[5])
    assert _display_order(neworder) == [1, 0]


def test_neworder_is_a_valid_permutation():
    neworder = value_sorted_neworder([0.3, 0.3, 0.7, 0.1], pinned_indices=[3])
    assert sorted(neworder) == [0, 1, 2, 3]
