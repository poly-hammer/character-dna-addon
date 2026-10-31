"""Unit coverage for the Raw Control Editor paste-from-raw-control helpers.

The source enumeration and default-twin resolution are pure/lightweight (no live
Blender rig), so they are exercised directly with small fakes. The pose-bone
paste itself (``paste_raw_control_pose`` / ``compute_raw_control_pose_bases``)
needs a live rig + DNA reader and is covered by manual probes.
"""

import pytest

from character_dna.editors.raw_control_editor.constants import DEFAULT_RAW_CONTROL_INDEX, DEFAULT_RAW_CONTROL_NAME


utilities = pytest.importorskip("character_dna.editors.raw_control_editor.utilities")


class _Row:
    def __init__(self, name: str, raw_control_index: int) -> None:
        self.name = name
        self.raw_control_index = raw_control_index


class _Editor:
    def __init__(self, rows: "list[_Row]", active_index: int) -> None:
        self.raw_controls = rows
        self.raw_controls_active_index = active_index


def _sample_editor(active_index: int) -> _Editor:
    rows = [
        _Row(DEFAULT_RAW_CONTROL_NAME, DEFAULT_RAW_CONTROL_INDEX),
        _Row("CTRL_expressions.browDownL", 0),
        _Row("CTRL_expressions.browDownR", 1),
        _Row("CTRL_expressions.jawOpen", 2),
    ]
    return _Editor(rows, active_index)


def test_paste_source_items_excludes_active_and_default():
    editor = _sample_editor(active_index=1)  # active = browDownL
    items = utilities.raw_control_paste_source_items(None, editor)
    full_names = {full for full, _short, _desc in items}
    short_names = {short for _full, short, _desc in items}
    assert full_names == {"CTRL_expressions.browDownR", "CTRL_expressions.jawOpen"}
    assert short_names == {"browDownR", "jawOpen"}
    assert DEFAULT_RAW_CONTROL_NAME not in full_names


def test_active_raw_control_short_name_strips_prefix():
    editor = _sample_editor(active_index=1)
    assert utilities.active_raw_control_short_name(editor) == "browDownL"


def test_default_paste_raw_control_name_returns_twin():
    known = {"CTRL_expressions.browDownR", "CTRL_expressions.jawOpen"}
    assert utilities.default_paste_raw_control_name("browDownL", known) == "CTRL_expressions.browDownR"


def test_default_paste_raw_control_name_none_when_twin_absent():
    assert utilities.default_paste_raw_control_name("browDownL", {"CTRL_expressions.jawOpen"}) is None


def test_default_paste_raw_control_name_none_for_center_control():
    known = {"CTRL_expressions.browDownR", "CTRL_expressions.jawOpen"}
    assert utilities.default_paste_raw_control_name("jawOpen", known) is None
