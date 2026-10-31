"""Tests for the dependency_chain module after the pure-DNA refactor.

Two clearly separated responsibilities:

* :func:`build_psd_corrective_rows` -- reads the rig definition to enumerate the
  named PSD combination correctives surfaced for the editor's read-only "key
  poses" list. This is the ONLY rig-definition dependency that remains, and no
  operator logic relies on it.
* :func:`resolve_raw_control_activation_indices` -- the pure-DNA solve that every
  preview / commit / match-bones path actually uses: given a raw control, return
  the raw controls to drive to 1.0 (itself plus any prerequisite base controls),
  recovered straight from the DNA behavior with NO rig definition and NO name
  parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from character_dna.editors.shared.dependency_chain import (
    build_psd_corrective_rows,
    clear_graph_cache,
    resolve_raw_control_activation_indices,
)


_PREFIX = "CTRL_expressions."


@pytest.fixture(autouse=True)
def _isolate_graph_cache():
    """``build_psd_corrective_rows`` caches one catalog per ``db_name``; clear it
    around every test so fixtures with reused names never see a stale catalog."""
    clear_graph_cache()
    yield
    clear_graph_cache()


# ===========================================================================
# PSD corrective listing (rig definition) -- the editor's "key poses" list.
# ===========================================================================
class _NameReader:
    """Minimal reader exposing only the raw-control name table (all that
    ``build_psd_corrective_rows`` needs to resolve base-control indices)."""

    def __init__(self, raw_control_names: list[str]) -> None:
        self._names = raw_control_names

    def getRawControlCount(self) -> int:
        return len(self._names)

    def getRawControlName(self, index: int) -> str:
        return self._names[index]


@dataclass
class _Deformer:
    deformation_type: str


@dataclass
class _Expr:
    name: str
    control: str | None = None
    mesh_deformers: list[_Deformer] = field(default_factory=list)


@dataclass
class _PsdDef:
    name: str
    target: str
    layer: int = 1
    corrective: str = ""


@dataclass
class _PsdInput:
    expression: str


@dataclass
class _PsdNet:
    output_expression: str
    psd_definition: str
    inputs: list[_PsdInput] = field(default_factory=list)


@dataclass
class _FakeRigDefinition:
    expressions: list[_Expr]
    psd_definitions: list[_PsdDef]
    psd_nets: list[_PsdNet]
    db_name: str = "test"


def _names(*shorts: str) -> list[str]:
    return [_PREFIX + s for s in shorts]


def _ctrl(short: str) -> str:
    return _PREFIX + short


def _blend(name: str, ctrl: str | None = None) -> _Expr:
    """A base raw-control / corrective expression that drives blend shapes."""
    return _Expr(name, ctrl, [_Deformer("JointsAndBlendShapes")])


def _joints(name: str, ctrl: str | None = None) -> _Expr:
    """A combination-only raw-control expression that drives joints only."""
    return _Expr(name, ctrl, [_Deformer("JointsOnly")])


def _additive_rig() -> tuple[_NameReader, _FakeRigDefinition]:
    """``jawOpenExtreme`` owns a dedicated 1:1 corrective (psd_def named after the
    control) -- it must be excluded from the computed corrective listing."""
    reader = _NameReader(_names("jawOpen", "jawOpenExtreme", "browDownL"))
    rig = _FakeRigDefinition(
        db_name="additive",
        expressions=[
            _blend("jaw_open", _ctrl("jawOpen")),
            _joints("jaw_openExtreme", _ctrl("jawOpenExtreme")),
            _blend("brow_down_L", _ctrl("browDownL")),
        ],
        psd_definitions=[
            _PsdDef("jaw_openExtreme", target="jawOpenExtreme_tgt", layer=1, corrective="jaw_openExtreme_cor"),
        ],
        psd_nets=[
            _PsdNet(
                "jaw_openExtreme_cor",
                "jaw_openExtreme",
                [_PsdInput("jaw_open"), _PsdInput("jaw_openExtreme")],
            ),
        ],
    )
    return reader, rig


def _combo_rig() -> tuple[_NameReader, _FakeRigDefinition]:
    """A single combination corrective driven by three base raw controls."""
    reader = _NameReader(_names("jawOpen", "mouthStretchL", "mouthStretchR"))
    rig = _FakeRigDefinition(
        db_name="combo",
        expressions=[
            _blend("jaw_open", _ctrl("jawOpen")),
            _blend("mouth_stretch_L", _ctrl("mouthStretchL")),
            _blend("mouth_stretch_R", _ctrl("mouthStretchR")),
            _Expr("Mstretch_Jopen_tgt", None, [_Deformer("BlendShapesOnly")]),
        ],
        psd_definitions=[
            _PsdDef("Mstretch_Jopen", target="Mstretch_Jopen_tgt", layer=2, corrective="Mstretch_Jopen_cor"),
        ],
        psd_nets=[
            _PsdNet(
                "Mstretch_Jopen_cor",
                "Mstretch_Jopen",
                [_PsdInput("jaw_open"), _PsdInput("mouth_stretch_L"), _PsdInput("mouth_stretch_R")],
            ),
        ],
    )
    return reader, rig


def _layered_rig() -> tuple[_NameReader, _FakeRigDefinition]:
    """Two stacked correctives at layers 1 and 2 -- exercises the (layer, name)
    sort of the corrective listing."""
    reader = _NameReader(_names("jawOpen", "mouthStretchL", "mouthStretchR"))
    rig = _FakeRigDefinition(
        db_name="layered",
        expressions=[
            _blend("jaw_open", _ctrl("jawOpen")),
            _blend("mouth_stretch_L", _ctrl("mouthStretchL")),
            _blend("mouth_stretch_R", _ctrl("mouthStretchR")),
            _Expr("Mstretch_tgt", None, [_Deformer("BlendShapesOnly")]),
            _Expr("Mstretch_Jopen_tgt", None, [_Deformer("JointsAndBlendShapes")]),
        ],
        psd_definitions=[
            _PsdDef("Mstretch", target="Mstretch_tgt", layer=1, corrective="Mstretch_cor"),
            _PsdDef("Mstretch_Jopen", target="Mstretch_Jopen_tgt", layer=2, corrective="Mstretch_Jopen_cor"),
        ],
        psd_nets=[
            _PsdNet("Mstretch_cor", "Mstretch", [_PsdInput("mouth_stretch_L"), _PsdInput("mouth_stretch_R")]),
            _PsdNet(
                "Mstretch_Jopen_cor",
                "Mstretch_Jopen",
                [_PsdInput("jaw_open"), _PsdInput("mouth_stretch_L"), _PsdInput("mouth_stretch_R")],
            ),
        ],
    )
    return reader, rig


def _icon_rig() -> tuple[_NameReader, _FakeRigDefinition]:
    """Two sibling correctives over the same base controls: one drives a blend
    shape, the other is joints-only -- to assert ``deforms_blends``."""
    reader = _NameReader(_names("a", "b"))
    rig = _FakeRigDefinition(
        db_name="icon",
        expressions=[
            _blend("ea", _ctrl("a")),
            _blend("eb", _ctrl("b")),
            _Expr("blend_tgt", None, [_Deformer("BlendShapesOnly")]),
            _Expr("bone_tgt", None, [_Deformer("JointsOnly")]),
        ],
        psd_definitions=[
            _PsdDef("blend", target="blend_tgt", layer=1, corrective="blend_cor"),
            _PsdDef("bone", target="bone_tgt", layer=1, corrective="bone_cor"),
        ],
        psd_nets=[
            _PsdNet("blend_cor", "blend", [_PsdInput("ea"), _PsdInput("eb")]),
            _PsdNet("bone_cor", "bone", [_PsdInput("ea"), _PsdInput("eb")]),
        ],
    )
    return reader, rig


def test_build_psd_corrective_rows_names_by_target() -> None:
    reader, rig = _combo_rig()
    rows = build_psd_corrective_rows(reader, rig)
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "Mstretch_Jopen_tgt"
    assert row.layer == 2
    assert row.base_control_indices == (0, 1, 2)
    assert row.deforms_blends is True


def test_build_psd_corrective_rows_excludes_additive_layer() -> None:
    """The combination-only raw control jawOpenExtreme owns a dedicated 1:1
    corrective (psd_def named after it); that is represented by the raw-control
    row, so it must not appear in the computed corrective listing."""
    reader, rig = _additive_rig()
    assert build_psd_corrective_rows(reader, rig) == []


def test_build_psd_corrective_rows_sorted_by_layer_then_name() -> None:
    reader, rig = _layered_rig()
    rows = build_psd_corrective_rows(reader, rig)
    assert [(row.name, row.layer) for row in rows] == [
        ("Mstretch_tgt", 1),
        ("Mstretch_Jopen_tgt", 2),
    ]


def test_build_psd_corrective_rows_without_rig_is_empty() -> None:
    reader, _ = _combo_rig()
    assert build_psd_corrective_rows(reader, None) == []


def test_deforms_blends_distinguishes_blend_and_joint_correctives() -> None:
    reader, rig = _icon_rig()
    rows = {row.name: row for row in build_psd_corrective_rows(reader, rig)}
    assert rows["blend_tgt"].deforms_blends is True
    assert rows["bone_tgt"].deforms_blends is False


class _ChannelReader:
    """Reader double exposing raw-control names plus the blend-shape-channel
    behavior surface, so the channel fallback for net-less layer-1 correctives
    can backward-solve base controls from the DNA."""

    def __init__(
        self,
        *,
        raw_control_names: list[str],
        channel_names: list[str],
        bsc_input_indices: list[int],
        bsc_output_indices: list[int],
    ) -> None:
        self._raw = raw_control_names
        self._channels = channel_names
        self._bsc_in = bsc_input_indices
        self._bsc_out = bsc_output_indices

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
        return []

    def getPSDColumnIndices(self) -> list[int]:
        return []


def _netless_layer1_rig() -> tuple[_ChannelReader, _FakeRigDefinition]:
    """A layer-1 corrective driven directly by a single raw control (``tongue_up``)
    has no ``psd_net``; its base control must be recovered from the DNA blend-shape
    channel it drives (``tongue_up_IN`` <- ``tongueUp``)."""
    reader = _ChannelReader(
        raw_control_names=_names("tongueUp"),
        channel_names=["tongue_up_IN", "tongue_upBendUp_IN"],
        bsc_input_indices=[0, 0],
        bsc_output_indices=[0, 1],
    )
    rig = _FakeRigDefinition(
        db_name="netless",
        expressions=[
            _Expr("tongue_up", None, []),
            _Expr("tongue_up_tgt", None, [_Deformer("JointsAndBlendShapes")]),
        ],
        psd_definitions=[
            _PsdDef("tongue_up", target="tongue_up_tgt", layer=1, corrective="tongue_up_cor"),
        ],
        psd_nets=[],
    )
    return reader, rig


def test_build_psd_corrective_rows_recovers_netless_layer1_from_channels() -> None:
    """A layer-1 corrective with no psd_nets recovers its single base control from
    the blend-shape channel it drives (matched by the corrective stem)."""
    reader, rig = _netless_layer1_rig()
    rows = build_psd_corrective_rows(reader, rig)
    assert len(rows) == 1
    assert rows[0].name == "tongue_up_tgt"
    # Only ``tongue_up_IN`` matches the stem (``tongue_upBendUp_IN`` does not start
    # with ``tongue_up_``), so just its single driver is recovered.
    assert rows[0].base_control_indices == (0,)


# ===========================================================================
# Pure-DNA raw-control activation solve (no rig definition).
# ===========================================================================
class _BehaviorReader:
    """Minimal stand-in for the riglogic behavior reader surface used by
    ``resolve_raw_control_activation_indices``.

    ``bsc_input_indices`` lists the control index driving each blend-shape
    channel; any raw-control index that appears there is a *base* control. The
    PSD matrix is the paired ``psd_rows`` (PSD control index) / ``psd_columns``
    (raw-control index) arrays."""

    def __init__(
        self,
        *,
        raw_control_names: list[str],
        bsc_input_indices: list[int],
        psd_rows: list[int],
        psd_columns: list[int],
    ) -> None:
        self._raw = raw_control_names
        self._bsc_in = bsc_input_indices
        self._psd_rows = psd_rows
        self._psd_cols = psd_columns

    def getRawControlCount(self) -> int:
        return len(self._raw)

    def getRawControlName(self, index: int) -> str:
        return self._raw[index]

    def getBlendShapeChannelInputIndices(self) -> list[int]:
        return self._bsc_in

    def getPSDRowIndices(self) -> list[int]:
        return self._psd_rows

    def getPSDColumnIndices(self) -> list[int]:
        return self._psd_cols


def _behavior_reader() -> _BehaviorReader:
    """A 6-raw-control rig exercising every branch of the resolver.

      raw 0 = jawOpen                (base; drives blends)
      raw 1 = jawOpenExtreme         (additive; joints only)
      raw 2 = browDown               (base)
      raw 3 = mouthStretchL          (base)
      raw 4 = mouthStretchLipsCloseL (additive)
      raw 5 = teethUp                (joints only; in no PSD -> solo)

    PSD controls (index >= rawControlCount = 6):
      psd 6 = {jawOpen, jawOpenExtreme}
      psd 7 = {jawOpen, mouthStretchL, jawOpenExtreme}   (jawOpen co-occurs in
              every PSD containing jawOpenExtreme; mouthStretchL does not)
      psd 8 = {mouthStretchL, mouthStretchLipsCloseL}
    """
    return _BehaviorReader(
        raw_control_names=_names(
            "jawOpen", "jawOpenExtreme", "browDown", "mouthStretchL", "mouthStretchLipsCloseL", "teethUp"
        ),
        bsc_input_indices=[0, 2, 3],  # base controls drive blend shapes
        psd_rows=[6, 6, 7, 7, 7, 8, 8],
        psd_columns=[0, 1, 0, 3, 1, 3, 4],
    )


@pytest.mark.parametrize("base_index", [0, 2, 3])
def test_base_control_activates_solo(base_index: int) -> None:
    """A control that directly drives a blend shape is authored on its own."""
    reader = _behavior_reader()
    assert resolve_raw_control_activation_indices(reader, base_index) == [base_index]


def test_additive_control_adds_universal_base() -> None:
    """jawOpenExtreme (raw 1) co-occurs with jawOpen (raw 0) in BOTH PSDs that
    contain it, but with mouthStretchL (raw 3) in only one -- so jawOpen is the
    sole prerequisite."""
    reader = _behavior_reader()
    assert resolve_raw_control_activation_indices(reader, 1) == [0, 1]


def test_second_additive_control_adds_its_own_base() -> None:
    reader = _behavior_reader()
    # mouthStretchLipsCloseL (raw 4) -> mouthStretchL (raw 3).
    assert resolve_raw_control_activation_indices(reader, 4) == [3, 4]


def test_joints_only_control_without_psd_is_solo() -> None:
    reader = _behavior_reader()
    # teethUp (raw 5) participates in no PSD -> nothing to co-activate but itself.
    assert resolve_raw_control_activation_indices(reader, 5) == [5]


def test_out_of_range_index_is_empty() -> None:
    reader = _behavior_reader()
    assert resolve_raw_control_activation_indices(reader, 99) == []
    assert resolve_raw_control_activation_indices(reader, -1) == []


def test_none_reader_is_empty() -> None:
    assert resolve_raw_control_activation_indices(None, 0) == []


# ---------------------------------------------------------------------------
# Real-rig pin -- the shipped Ada head DNA.
# ---------------------------------------------------------------------------
# (combination-only control, sorted prerequisite controls). The COMPLETE set of
# raw-control -> raw-control dependencies in MH.6: a control that drives joints
# only and universally co-occurs with a blend-driving base in the DNA PSD matrix.
_REAL_RAW_DEPENDENCIES = [
    ("eyeLidPressL", ["eyeBlinkL"]),
    ("eyeLidPressR", ["eyeBlinkR"]),
    ("jawOpenExtreme", ["jawOpen"]),
    ("mouthStretchLipsCloseL", ["mouthStretchL"]),
    ("mouthStretchLipsCloseR", ["mouthStretchR"]),
    ("noseWrinkleUpperL", ["noseWrinkleL"]),
    ("noseWrinkleUpperR", ["noseWrinkleR"]),
]

# Independent BASE controls (drive blend shapes) -- must resolve solo.
_REAL_BASE_CONTROLS = [
    "jawOpen",
    "eyeBlinkL",
    "eyeBlinkR",
    "mouthStretchL",
    "noseWrinkleL",
    "mouthCornerPullL",
    "browDownL",
]


@pytest.fixture(scope="module")
def real_reader():
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    return get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")


def _raw_index(reader, short_name: str) -> int:
    target = _PREFIX + short_name
    for i in range(int(reader.getRawControlCount())):
        if str(reader.getRawControlName(i)) == target:
            return i
    raise AssertionError(f"raw control {target!r} not found in DNA")


@pytest.mark.parametrize(("control", "expected_parents"), _REAL_RAW_DEPENDENCIES)
def test_real_combination_only_dependencies(real_reader, control, expected_parents) -> None:
    index = _raw_index(real_reader, control)
    resolved = resolve_raw_control_activation_indices(real_reader, index)
    expected = sorted(_raw_index(real_reader, short) for short in [*expected_parents, control])
    assert resolved == expected, f"{control!r} -> {[str(real_reader.getRawControlName(i)) for i in resolved]}"


@pytest.mark.parametrize("control", _REAL_BASE_CONTROLS)
def test_real_base_controls_resolve_solo(real_reader, control) -> None:
    index = _raw_index(real_reader, control)
    assert resolve_raw_control_activation_indices(real_reader, index) == [index]


def test_real_dependency_set_is_exactly_the_seven(real_reader) -> None:
    """Guard against over-triggering: scanning EVERY raw control on the real MH.6
    rig yields exactly the seven verified dependencies -- no more, no less."""
    found: dict[str, list[str]] = {}
    for index in range(int(real_reader.getRawControlCount())):
        name = str(real_reader.getRawControlName(index))
        if not name.startswith(_PREFIX):
            continue
        indices = resolve_raw_control_activation_indices(real_reader, index)
        parents = [j for j in indices if j != index]
        if parents:
            found[name.replace(_PREFIX, "")] = sorted(
                str(real_reader.getRawControlName(j)).replace(_PREFIX, "") for j in parents
            )
    assert found == {control: sorted(parents) for control, parents in _REAL_RAW_DEPENDENCIES}
