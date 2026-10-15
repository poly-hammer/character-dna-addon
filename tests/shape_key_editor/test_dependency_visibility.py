"""Tests for the in-edit dependency-visibility model.

The Shape Key Editor builds a read-only dependency list on edit-mode entry
(:func:`build_dependency_items`) and exposes two pure helpers used by the UI and
the live-preview callback:

* :func:`any_dependency_hidden` -- disables the edit tools + Commit while any
  dependency's eye is off, since those operators require the full chain.
* :func:`active_raw_dependency_indices` -- the raw-control indices driven to 1.0
  for the currently-visible raw dependencies.

The dependency rows are recovered purely from the DNA behavior (no rig
definition, no name conventions, no ``bpy``), so these tests drive them with a
minimal fake reader / head instance modeled on the real RigLogic surface.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from character_dna.editors.shape_key_editor.utilities import (
    DependencyRow,
    active_raw_dependency_indices,
    any_dependency_hidden,
    build_dependency_items,
)


_PREFIX = "CTRL_expressions."


class _FakeBehaviorReader:
    """Minimal behavior-reader surface used by the dependency backsolve."""

    def __init__(
        self,
        *,
        raw_control_names: list[str],
        channel_names: list[str],
        bsc_input_indices: list[int],
        bsc_output_indices: list[int],
        psd_rows: list[int],
        psd_columns: list[int],
    ) -> None:
        self._raw = raw_control_names
        self._channels = channel_names
        self._bsc_in = bsc_input_indices
        self._bsc_out = bsc_output_indices
        self._psd_rows = psd_rows
        self._psd_cols = psd_columns

    def getRawControlCount(self) -> int:
        return len(self._raw)

    def getRawControlName(self, index: int) -> str:
        return self._raw[index]

    def getBlendShapeChannelCount(self) -> int:
        return len(self._channels)

    def getBlendShapeChannelName(self, index: int) -> str:
        return self._channels[index]

    def getBlendShapeChannelInputIndices(self) -> list[int]:
        return self._bsc_in

    def getBlendShapeChannelOutputIndices(self) -> list[int]:
        return self._bsc_out

    def getPSDRowIndices(self) -> list[int]:
        return self._psd_rows

    def getPSDColumnIndices(self) -> list[int]:
        return self._psd_cols


def _build_reader() -> _FakeBehaviorReader:
    """A rig with a center channel (jaw_open <- raw jawOpen) and a combination
    corrective (jaw_openExtreme_cor <- psd of jawOpen * jawOpenExtreme).

    control space: raw [0,4) | psd [4,5)
      raw 0 = browDownL, 1 = browDownR, 2 = jawOpen, 3 = jawOpenExtreme
      psd 4 = jawOpen * jawOpenExtreme (columns 2, 3)
    channels:
      0 brow_down_L          <- raw 0
      1 brow_down_R          <- raw 1
      2 jaw_open             <- raw 2
      3 jaw_openExtreme_cor  <- psd 4
    """
    return _FakeBehaviorReader(
        raw_control_names=[
            _PREFIX + "browDownL",
            _PREFIX + "browDownR",
            _PREFIX + "jawOpen",
            _PREFIX + "jawOpenExtreme",
        ],
        channel_names=["brow_down_L", "brow_down_R", "jaw_open", "jaw_openExtreme_cor"],
        bsc_input_indices=[0, 1, 2, 4],
        bsc_output_indices=[0, 1, 2, 3],
        psd_rows=[4, 4],
        psd_columns=[2, 3],
    )


def _instance(reader: _FakeBehaviorReader | None, outputs: list[float]) -> SimpleNamespace:
    """A fake rig instance exposing only what ``build_dependency_items`` reads."""
    head_instance = SimpleNamespace(getBlendShapeOutputs=lambda: list(outputs))
    return SimpleNamespace(edit_target=SimpleNamespace(dna_reader=reader, state=head_instance))


def _editor(rows: list[SimpleNamespace]) -> SimpleNamespace:
    """A fake editor exposing only ``dependency_items``."""
    return SimpleNamespace(dependency_items=rows)


def _row(*, kind: str, show: bool, raw_index: int = -1, channel_index: int = -1) -> SimpleNamespace:
    """A fake dependency row with the attributes the helpers read."""
    return SimpleNamespace(
        kind=kind,
        show_dependency=show,
        raw_control_index=raw_index,
        channel_index=channel_index,
    )


# ---------------------------------------------------------------------------
# build_dependency_items
# ---------------------------------------------------------------------------
def test_build_dependency_items_combination_corrective() -> None:
    reader = _build_reader()
    # jaw_open active in the driven pose; the edited channel (idx 3) is skipped
    # as a dependency and appended last.
    instance = _instance(reader, outputs=[0.0, 0.0, 1.0, 0.6])

    rows = build_dependency_items(instance, "jaw_openExtreme_cor")

    assert rows == [
        DependencyRow("jawOpen", "RAW", False, raw_control_index=2),
        DependencyRow("jawOpenExtreme", "RAW", False, raw_control_index=3),
        DependencyRow("jaw_open", "SHAPE", False, channel_index=2, active_value=1.0),
        DependencyRow("jaw_openExtreme_cor", "SHAPE", True, channel_index=3, active_value=1.0),
    ]


def test_build_dependency_items_snapshots_active_value() -> None:
    reader = _build_reader()
    instance = _instance(reader, outputs=[0.0, 0.0, 0.42, 0.6])

    rows = build_dependency_items(instance, "jaw_openExtreme_cor")

    shape_dep = next(r for r in rows if r.name == "jaw_open")
    assert shape_dep.kind == "SHAPE"
    assert shape_dep.active_value == pytest.approx(0.42)


def test_build_dependency_items_skips_inactive_channels() -> None:
    reader = _build_reader()
    # Only the edited channel is active -- no other shape dependencies.
    instance = _instance(reader, outputs=[0.0, 0.0, 0.0, 0.6])

    rows = build_dependency_items(instance, "jaw_openExtreme_cor")

    shape_names = [r.name for r in rows if r.kind == "SHAPE"]
    assert shape_names == ["jaw_openExtreme_cor"]
    assert rows[-1].is_edited is True


def test_build_dependency_items_simple_channel_single_raw() -> None:
    reader = _build_reader()
    instance = _instance(reader, outputs=[0.0, 0.0, 1.0, 0.0])

    rows = build_dependency_items(instance, "jaw_open")

    assert rows == [
        DependencyRow("jawOpen", "RAW", False, raw_control_index=2),
        DependencyRow("jaw_open", "SHAPE", True, channel_index=2, active_value=1.0),
    ]


def test_build_dependency_items_without_reader_returns_single_edited_row() -> None:
    instance = _instance(None, outputs=[])

    rows = build_dependency_items(instance, "jaw_open")

    assert rows == [DependencyRow("jaw_open", "SHAPE", True)]


# ---------------------------------------------------------------------------
# any_dependency_hidden
# ---------------------------------------------------------------------------
def test_any_dependency_hidden_false_when_all_shown() -> None:
    editor = _editor(
        [
            _row(kind="RAW", show=True, raw_index=2),
            _row(kind="SHAPE", show=True, channel_index=2),
        ]
    )
    assert any_dependency_hidden(editor) is False


def test_any_dependency_hidden_true_when_one_hidden() -> None:
    editor = _editor(
        [
            _row(kind="RAW", show=True, raw_index=2),
            _row(kind="SHAPE", show=False, channel_index=2),
        ]
    )
    assert any_dependency_hidden(editor) is True


def test_any_dependency_hidden_false_for_empty_list() -> None:
    assert any_dependency_hidden(_editor([])) is False


# ---------------------------------------------------------------------------
# active_raw_dependency_indices
# ---------------------------------------------------------------------------
def test_active_raw_dependency_indices_only_visible_raw_rows() -> None:
    editor = _editor(
        [
            _row(kind="RAW", show=True, raw_index=2),
            _row(kind="RAW", show=False, raw_index=3),  # hidden -> dropped
            _row(kind="SHAPE", show=True, channel_index=2),  # not a raw row
        ]
    )
    assert active_raw_dependency_indices(editor) == [2]


def test_active_raw_dependency_indices_dedupes_and_sorts() -> None:
    editor = _editor(
        [
            _row(kind="RAW", show=True, raw_index=5),
            _row(kind="RAW", show=True, raw_index=2),
            _row(kind="RAW", show=True, raw_index=2),  # duplicate
        ]
    )
    assert active_raw_dependency_indices(editor) == [2, 5]


def test_active_raw_dependency_indices_drops_negative_indices() -> None:
    editor = _editor(
        [
            _row(kind="RAW", show=True, raw_index=-1),  # unset -> dropped
            _row(kind="RAW", show=True, raw_index=4),
        ]
    )
    assert active_raw_dependency_indices(editor) == [4]
