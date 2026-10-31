"""Tests for the pure-DNA blend-shape backward solve (no rig definition).

``resolve_blend_shape_raw_controls`` maps a blend shape channel name to the raw
controls that must be 1.0 for it to appear, using ONLY the DNA behavior: the
blend-shape-channel mapping (input/output indices) and the PSD matrix. A simple
shape resolves to its one driving raw control; a combination corrective resolves
to every raw control of its PSD.

The fast unit tests use a hand-built fake behavior reader; the real-rig test
loads the shipped Ada head DNA and pins the ``jaw_open`` / ``jaw_openExtreme_cor``
cases, so the resolver stays correct against the actual data RigLogic evaluates.
"""

from __future__ import annotations

import pytest

from character_dna.editors.shared.dependency_chain import resolve_blend_shape_raw_controls


_PREFIX = "CTRL_expressions."


class _FakeBehaviorReader:
    """Minimal stand-in for the riglogic behavior reader surface used by
    ``resolve_blend_shape_raw_controls``."""

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
    """A 3-raw-control rig with one simple channel and one PSD corrective.

    control space: raw [0,3) | psd [3,4)
      raw 0 = jawOpen, raw 1 = jawOpenExtreme, raw 2 = browDown
      psd 3 = jawOpen * jawOpenExtreme  (columns 0 and 1)

    channels:
      0 'jaw_open'              <- control 0 (raw jawOpen)
      1 'jaw_openExtreme_cor'   <- control 3 (the PSD)
      2 'brow_down'             <- control 2 (raw browDown)
      3 'undriven_map'          (never appears in output indices)
    """
    return _FakeBehaviorReader(
        raw_control_names=[_PREFIX + "jawOpen", _PREFIX + "jawOpenExtreme", _PREFIX + "browDown"],
        channel_names=["jaw_open", "jaw_openExtreme_cor", "brow_down", "undriven_map"],
        bsc_input_indices=[0, 3, 2],
        bsc_output_indices=[0, 1, 2],
        psd_rows=[3, 3],
        psd_columns=[0, 1],
    )


def test_simple_channel_resolves_to_single_raw_control() -> None:
    reader = _build_reader()
    assert resolve_blend_shape_raw_controls(reader, "jaw_open") == [0]


def test_corrective_channel_resolves_to_all_psd_raw_controls() -> None:
    reader = _build_reader()
    # The PSD's columns are raw controls 0 (jawOpen) and 1 (jawOpenExtreme).
    assert resolve_blend_shape_raw_controls(reader, "jaw_openExtreme_cor") == [0, 1]


def test_other_simple_channel_is_independent() -> None:
    reader = _build_reader()
    assert resolve_blend_shape_raw_controls(reader, "brow_down") == [2]


def test_unknown_channel_returns_empty() -> None:
    reader = _build_reader()
    assert resolve_blend_shape_raw_controls(reader, "does_not_exist") == []


def test_undriven_channel_returns_empty() -> None:
    reader = _build_reader()
    # 'undriven_map' is a named channel but is never an output index (e.g. a
    # head-movement map driven outside the blend-shape behavior).
    assert resolve_blend_shape_raw_controls(reader, "undriven_map") == []


def test_none_reader_returns_empty() -> None:
    assert resolve_blend_shape_raw_controls(None, "jaw_open") == []


# ---------------------------------------------------------------------------
# Real-rig pin -- the shipped Ada head DNA
# ---------------------------------------------------------------------------
def _raw_index(reader, short_name: str) -> int:
    target = _PREFIX + short_name
    for i in range(int(reader.getRawControlCount())):
        if str(reader.getRawControlName(i)) == target:
            return i
    raise AssertionError(f"raw control {target!r} not found in DNA")


@pytest.mark.parametrize(
    ("channel", "expected_shorts"),
    [
        ("jaw_open", ["jawOpen"]),
        ("jaw_openExtreme_cor", ["jawOpen", "jawOpenExtreme"]),
    ],
)
def test_real_ada_dna_backsolve(channel: str, expected_shorts: list[str]) -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    resolved = resolve_blend_shape_raw_controls(reader, channel)
    expected = sorted(_raw_index(reader, short) for short in expected_shorts)
    assert resolved == expected, (
        f"{channel!r} -> {[str(reader.getRawControlName(i)) for i in resolved]}, "
        f"expected {[_PREFIX + s for s in expected_shorts]}"
    )
