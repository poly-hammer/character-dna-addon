"""Tests for the forward blend-shape solve used by the "Transfer Shape to Shape
Key" operator's poll gate.

``resolve_raw_control_active_shape_key_channels`` returns the blend shape
channels RigLogic activates when ONLY a single raw control is driven to 1.0:
its direct channels plus channels driven by a PSD whose only column is that
control. A base control (``jawOpen``) returns its one channel; a joints-only
control (``jawOpenExtreme``) returns none -- proving the raw-control -> shape-key
mapping is NOT strictly 1-to-1 and the poll gate is required.
"""

from __future__ import annotations

from character_dna.editors.shared.dependency_chain import (
    resolve_raw_control_active_shape_key_channels,
    resolve_raw_control_shape_key_layers,
)


_PREFIX = "CTRL_expressions."


class _FakeBehaviorReader:
    """Minimal behavior reader for the forward solve."""

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
    """control space: raw [0,3) | psd [3,5)
      raw 0 = jawOpen, raw 1 = jawOpenExtreme, raw 2 = tongueUp
      psd 3 = jawOpen * jawOpenExtreme        (columns 0, 1 -- multi-input)
      psd 4 = tongueUp                        (column 2 only -- single-input)

    channels:
      0 'jaw_open'            <- control 0 (raw jawOpen, direct)
      1 'jaw_openExtreme_cor' <- control 3 (multi-input PSD)
      2 'tongue_up_cor'       <- control 4 (single-input PSD on tongueUp)
    """
    return _FakeBehaviorReader(
        raw_control_names=[_PREFIX + "jawOpen", _PREFIX + "jawOpenExtreme", _PREFIX + "tongueUp"],
        channel_names=["jaw_open", "jaw_openExtreme_cor", "tongue_up_cor"],
        bsc_input_indices=[0, 3, 4],
        bsc_output_indices=[0, 1, 2],
        psd_rows=[3, 3, 4],
        psd_columns=[0, 1, 2],
    )


def test_base_control_activates_its_single_direct_channel() -> None:
    reader = _build_reader()
    # jawOpen (raw 0) directly drives channel 0; the multi-input PSD it also
    # feeds stays at 0 (needs jawOpenExtreme too).
    assert resolve_raw_control_active_shape_key_channels(reader, 0) == [0]


def test_joints_only_control_activates_no_channel() -> None:
    reader = _build_reader()
    # jawOpenExtreme (raw 1) drives no direct channel and its only PSD is
    # multi-input, so alone at 1.0 it activates nothing -> no shape key.
    assert resolve_raw_control_active_shape_key_channels(reader, 1) == []


def test_single_input_psd_channel_is_activated() -> None:
    reader = _build_reader()
    # tongueUp (raw 2) drives the single-input PSD 4 -> channel 2 fires at 1.0.
    assert resolve_raw_control_active_shape_key_channels(reader, 2) == [2]


def test_out_of_range_control_returns_empty() -> None:
    reader = _build_reader()
    assert resolve_raw_control_active_shape_key_channels(reader, 99) == []
    assert resolve_raw_control_active_shape_key_channels(reader, -1) == []


def test_none_reader_returns_empty() -> None:
    assert resolve_raw_control_active_shape_key_channels(None, 0) == []


# ---------------------------------------------------------------------------
# Real-rig pin -- the shipped Ada head DNA
# ---------------------------------------------------------------------------
def _raw_index(reader, short_name: str) -> int:
    target = _PREFIX + short_name
    for i in range(int(reader.getRawControlCount())):
        if str(reader.getRawControlName(i)) == target:
            return i
    raise AssertionError(f"raw control {target!r} not found in DNA")


def test_real_ada_dna_jaw_open_activates_exactly_one_channel() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    channels = resolve_raw_control_active_shape_key_channels(reader, _raw_index(reader, "jawOpen"))
    assert len(channels) == 1
    assert str(reader.getBlendShapeChannelName(channels[0])) == "jaw_open"


def test_real_ada_dna_jaw_open_extreme_is_joints_only() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    # jawOpenExtreme is a joints-only additive control -> no shape key, so the
    # transfer operator's poll is False for it.
    assert resolve_raw_control_active_shape_key_channels(reader, _raw_index(reader, "jawOpenExtreme")) == []


# ---------------------------------------------------------------------------
# Layered resolution (top channel + underneath) used by the transfer op
# ---------------------------------------------------------------------------
def test_layers_base_control_is_one_deep() -> None:
    reader = _build_reader()
    # jawOpen authors jaw_open (channel 0) with nothing underneath.
    assert resolve_raw_control_shape_key_layers(reader, 0) == (0, [])


def test_layers_additive_control_sits_on_its_base() -> None:
    reader = _build_reader()
    # jawOpenExtreme authors the corrective (channel 1) on top of jaw_open (0).
    assert resolve_raw_control_shape_key_layers(reader, 1) == (1, [0])


def test_layers_single_input_corrective_is_one_deep() -> None:
    reader = _build_reader()
    # tongueUp's single-input PSD corrective (channel 2) has no lower base.
    assert resolve_raw_control_shape_key_layers(reader, 2) == (2, [])


def test_layers_out_of_range_returns_none() -> None:
    reader = _build_reader()
    assert resolve_raw_control_shape_key_layers(reader, 99) is None
    assert resolve_raw_control_shape_key_layers(None, 0) is None


def test_layers_ambiguous_top_returns_none() -> None:
    # A base control (raw 0) that drives BOTH a direct channel and TWO
    # single-input PSD correctives -> two PSD-driven tops -> ambiguous.
    reader = _FakeBehaviorReader(
        raw_control_names=[_PREFIX + "browDown"],
        channel_names=["brow_down", "brow_down_corA", "brow_down_corB"],
        bsc_input_indices=[0, 1, 2],  # channel 0 <- raw 0; channels 1,2 <- PSDs 1,2
        bsc_output_indices=[0, 1, 2],
        psd_rows=[1, 2],
        psd_columns=[0, 0],  # both PSDs driven solely by raw 0
    )
    assert resolve_raw_control_shape_key_layers(reader, 0) is None


def test_real_ada_dna_layers_jaw_open() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    result = resolve_raw_control_shape_key_layers(reader, _raw_index(reader, "jawOpen"))
    assert result is not None
    top, lower = result
    assert str(reader.getBlendShapeChannelName(top)) == "jaw_open"
    assert lower == []


def test_real_ada_dna_layers_jaw_open_extreme_sits_on_jaw_open() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    result = resolve_raw_control_shape_key_layers(reader, _raw_index(reader, "jawOpenExtreme"))
    assert result is not None
    top, lower = result
    assert str(reader.getBlendShapeChannelName(top)) == "jaw_openExtreme_cor"
    assert [str(reader.getBlendShapeChannelName(c)) for c in lower] == ["jaw_open"]
