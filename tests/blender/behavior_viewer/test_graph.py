"""Tests for the Behavior Viewer live behavior-graph builder.

Two layers:

* A synthetic DNA reader (no Blender, no real DNA) driven by hand-supplied
  ``raw_values`` / ``blend_outputs`` arrays -- exercises the Active Pose root,
  the active-only filtering, the value labels, the Raw Control -> Bone Pose +
  Shape Key wiring and the subset-covering shape-key layering (with arbitrary,
  non-1.0 values).
* Real-Ada ``head.dna`` pins that prove the builder reads the actual DNA joint
  behavior (cell-accurate bone poses) and blend-shape behavior (the channel a
  raw control drives) for a controlled active state.
"""

from __future__ import annotations

import pytest

from character_dna.editors.behavior_viewer.graph import (
    KIND_ACTIVE_POSE,
    KIND_ANIMATED_MAP,
    KIND_BONE,
    KIND_RAW,
    KIND_RBF_SOLVER,
    KIND_SHAPE,
    build_behavior_graph,
)


# ---------------------------------------------------------------------------
# Synthetic DNA: 3 raw controls, 5 channels, 2 PSD combinations, 1 joint group
# where only jawOpen moves a joint.
# ---------------------------------------------------------------------------
class _Reader:
    _raw = ["CTRL_expressions.jawOpen", "CTRL_expressions.mouthStretchL", "CTRL_expressions.mouthStretchR"]
    #  channel:      0          1                2               3           4               5
    _channels = ["jaw_open", "mouth_stretch_L", "mouth_stretch_R", "Mstretch", "Jopen_Mstretch", "head_turnLeft_U"]
    _joints = ["FACIAL_C_Jaw", "FACIAL_C_Neck"]

    def getRawControlCount(self) -> int:
        return len(self._raw)

    def getRawControlName(self, index: int) -> str:
        return self._raw[index]

    def getBlendShapeChannelCount(self) -> int:
        return len(self._channels)

    def getBlendShapeChannelName(self, index: int) -> str:
        return self._channels[index]

    def getBlendShapeChannelOutputIndices(self) -> list[int]:
        return [0, 1, 2, 3, 4]

    def getBlendShapeChannelInputIndices(self) -> list[int]:
        # channels 0..2 <- raw controls 0..2; channel 3 <- PSD 3; channel 4 <- PSD 4
        return [0, 1, 2, 3, 4]

    def getPSDRowIndices(self) -> list[int]:
        return [3, 3, 4, 4, 4]

    def getPSDColumnIndices(self) -> list[int]:
        return [1, 2, 0, 1, 2]

    def getPSDValues(self) -> list[float]:
        return [1.0, 1.0, 0.9, 1.0, 1.0]

    def getPSDCount(self) -> int:
        return 2

    def getJointCount(self) -> int:
        return len(self._joints)

    def getJointName(self, index: int) -> str:
        return self._joints[index]

    def getJointGroupCount(self) -> int:
        return 2

    def getJointGroupInputIndices(self, group: int) -> list[int]:
        # group 0 <- raw controls 0,1,2; group 1 <- the RBF output control 5.
        return [0, 1, 2] if group == 0 else [5]

    def getJointGroupOutputIndices(self, group: int) -> list[int]:
        # output 0 -> joint 0 (jaw); output 9 -> joint 1 (neck). group 1 drives neck.
        return [0, 9] if group == 0 else [9]

    def getJointGroupValues(self, _group: int) -> list[float]:
        # row-major (n_inputs=3): row0 = [jawOpen, mStretchL, mStretchR]; row1 zero.
        return [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]

    # animated maps: one wrinkle mask driven by raw control 0 (jawOpen), ramp [0,1].
    def getAnimatedMapCount(self) -> int:
        return 1

    def getAnimatedMapName(self, index: int) -> str:
        return ["head_wm1_jaw_msk"][index]

    def getAnimatedMapInputIndices(self) -> list[int]:
        return [0]

    def getAnimatedMapOutputIndices(self) -> list[int]:
        return [0]

    def getAnimatedMapFromValues(self) -> list[float]:
        return [0.0]

    def getAnimatedMapToValues(self) -> list[float]:
        return [1.0]

    def getAnimatedMapSlopeValues(self) -> list[float]:
        return [1.0]

    def getAnimatedMapCutValues(self) -> list[float]:
        return [0.0]

    # RBF: one solver (neck raws 1,2) with one pose driving channel 5 + joint 1.
    def getRBFSolverCount(self) -> int:
        return 1

    def getRBFSolverRawControlIndices(self, _solver: int) -> list[int]:
        return [1, 2]

    def getRBFSolverName(self, _solver: int) -> str:
        return "neck"

    def getRBFSolverPoseIndices(self, _solver: int) -> list[int]:
        return [0]

    def getRBFPoseName(self, _pose: int) -> str:
        return "head_turnLeft"

    def getRBFPoseBlendShapeChannelOutputIndices(self, _pose: int) -> list[int]:
        return [5]

    def getRBFPoseJointOutputIndices(self, _pose: int) -> list[int]:
        return [9]  # joint 1 (FACIAL_C_Neck)

    def getRBFPoseAnimatedMapOutputIndices(self, _pose: int) -> list[int]:
        return []

    def getRBFPoseOutputControlIndices(self, _pose: int) -> list[int]:
        return [5]  # offset = raw(3) + psd(2) + ml(0) = 5 -> rbf_control_values[0]


def _node(graph, node_id: str) -> dict:
    return next(n["data"] for n in graph.nodes if n["data"]["id"] == node_id)


def _has_node(graph, node_id: str) -> bool:
    return any(n["data"]["id"] == node_id for n in graph.nodes)


def _edge_pairs(graph) -> set[tuple[str, str]]:
    return {(e["data"]["source"], e["data"]["target"]) for e in graph.edges}


# ---------------------------------------------------------------------------
# Full active pose
# ---------------------------------------------------------------------------
def _full_graph():
    raw_values = [0.75, 0.5, 0.5]
    blend_outputs = [0.75, 0.5, 0.5, 0.25, 0.34]
    return build_behavior_graph(_Reader(), raw_values, blend_outputs, pose_label="mouth_lowerLipDepress")


def test_active_pose_root_and_activated_raw_controls() -> None:
    graph = _full_graph()
    root = _node(graph, "active_pose")
    assert root["kind"] == KIND_ACTIVE_POSE
    assert root["label"] == "mouth_lowerLipDepress"
    assert root["count"] == 3
    assert root["layer"] == 0

    assert _node(graph, "raw:0")["kind"] == KIND_RAW
    assert _node(graph, "raw:0")["label"] == "jawOpen"
    assert _node(graph, "raw:0")["value"] == 0.75
    assert _node(graph, "raw:1")["value"] == 0.5
    # active pose drives every activated raw control.
    assert ("active_pose", "raw:0") in _edge_pairs(graph)
    assert ("active_pose", "raw:1") in _edge_pairs(graph)


def test_raw_control_activates_bone_pose_and_shape_key() -> None:
    graph = _full_graph()
    # jawOpen poses a bone (cell-accurate) AND drives its primary shape key.
    bone = _node(graph, "bone:raw:0")
    assert bone["kind"] == KIND_BONE
    assert bone["bones"] == ["FACIAL_C_Jaw"]
    assert ("raw:0", "bone:raw:0") in _edge_pairs(graph)

    shape = _node(graph, "shape:0")
    assert shape["kind"] == KIND_SHAPE
    assert shape["label"] == "jaw_open"
    assert shape["value"] == 0.75
    assert ("raw:0", "shape:0") in _edge_pairs(graph)

    # mouthStretch controls have a primary shape but pose no bone (zero cells).
    assert not _has_node(graph, "bone:raw:1")
    assert _has_node(graph, "shape:1")


def test_shape_key_layering_ordered_by_activation() -> None:
    graph = _full_graph()
    # Primary shapes at column 2; Mstretch (depth 1) at 3; Jopen_Mstretch (2) at 4.
    assert _node(graph, "shape:0")["layer"] == 2
    assert _node(graph, "shape:3")["layer"] == 3
    assert _node(graph, "shape:4")["layer"] == 4

    pairs = _edge_pairs(graph)
    # Mstretch is used after mouth_stretch_L/R primary shapes.
    assert ("shape:1", "shape:3") in pairs
    assert ("shape:2", "shape:3") in pairs
    # Jopen_Mstretch is used after the Mstretch corrective and the jaw_open shape.
    assert ("shape:3", "shape:4") in pairs
    assert ("shape:0", "shape:4") in pairs
    # Its covered atoms are NOT wired directly from mouth_stretch shapes again.
    assert ("shape:1", "shape:4") not in pairs


def test_shape_key_definition_carries_live_product() -> None:
    graph = _full_graph()
    definition = _node(graph, "shape:4")["definition"]
    assert definition["output"] == "Jopen_Mstretch"
    # Read-only op/operand rows: weight, then the raw-control product.
    assert definition["operations"] == [
        {"op": "=", "operand": "0.9"},
        {"op": "\u00d7", "operand": "jawOpen"},
        {"op": "\u00d7", "operand": "mouthStretchL"},
        {"op": "\u00d7", "operand": "mouthStretchR"},
    ]


def test_primary_shape_and_raw_expression() -> None:
    graph = _full_graph()
    # Primary blend shape: a single assignment from its raw control.
    primary = _node(graph, "shape:0")["definition"]
    assert primary["output"] == "jaw_open"
    assert primary["operations"] == [{"op": "=", "operand": "jawOpen"}]
    # Raw control + joint group: base inputs -> no op rows, just a note.
    raw_def = _node(graph, "raw:0")["definition"]
    assert raw_def["output"] == "jawOpen"
    assert raw_def["operations"] == []
    assert "Raw control" in raw_def["note"]
    bone_def = _node(graph, "bone:raw:0")["definition"]
    assert bone_def["operations"] == []
    assert "Joint group" in bone_def["note"]


def test_animated_map_node_ramps_off_its_driver() -> None:
    graph = build_behavior_graph(_Reader(), [0.75, 0.5, 0.5], [0.75, 0.5, 0.5, 0.25, 0.34], animated_map_outputs=[0.6])
    node = _node(graph, "anim:0")
    assert node["kind"] == KIND_ANIMATED_MAP
    assert node["label"] == "head_wm1_jaw_msk"
    assert node["value"] == 0.6
    # Wired (ramp) from jawOpen's primary blend shape.
    assert ("shape:0", "anim:0") in _edge_pairs(graph)
    assert node["definition"]["operations"] == [{"op": "ramp", "operand": "jawOpen"}]
    assert "slope" in node["definition"]["note"]


def test_animated_map_absent_when_inactive() -> None:
    graph = build_behavior_graph(_Reader(), [0.75, 0.5, 0.5], [0.75, 0.5, 0.5, 0.25, 0.34], animated_map_outputs=[0.0])
    assert not _has_node(graph, "anim:0")


def test_raw_controls_carry_rbf_flag() -> None:
    graph = _full_graph()
    assert _node(graph, "raw:0")["rbf"] is False  # jawOpen is not an RBF input
    assert _node(graph, "raw:1")["rbf"] is True  # in the RBF solver's raw controls


def test_rbf_solver_surfaces_driven_blend_shape_and_joints() -> None:
    # Head turned: channel 5 (head_turnLeft_U) is RBF-driven -- not visible through
    # the blend-shape behavior, so it only appears via the RBF solver.
    blend_outputs = [0.75, 0.5, 0.5, 0.25, 0.34, 0.9]
    graph = build_behavior_graph(_Reader(), [0.75, 0.5, 0.5], blend_outputs, rbf_control_values=[0.66])

    solver = _node(graph, "rbf:0")
    assert solver["kind"] == KIND_RBF_SOLVER
    assert solver["label"] == "neck"
    assert "value" not in solver  # solvers have no per-pose live weight
    # Shared neck raw controls drive the solver.
    assert ("raw:1", "rbf:0") in _edge_pairs(graph)
    assert ("raw:2", "rbf:0") in _edge_pairs(graph)
    # The RBF-driven blend shape appears under the solver with its live value.
    shape = _node(graph, "shape:5")
    assert shape["label"] == "head_turnLeft_U"
    assert shape["value"] == 0.9
    assert ("rbf:0", "shape:5") in _edge_pairs(graph)
    # And the joint group it drives (resolved via the owning joint group).
    assert _node(graph, "bone:rbf:0")["bones"] == ["FACIAL_C_Neck"]
    assert ("rbf:0", "bone:rbf:0") in _edge_pairs(graph)


def test_no_rbf_solver_when_inactive() -> None:
    # No RBF-driven channel active and no RBF output-control value -> no RBF solver.
    assert not _has_node(_full_graph(), "rbf:0")


# ---------------------------------------------------------------------------
# Arbitrary / partial poses (values need not be 1.0)
# ---------------------------------------------------------------------------
def test_arbitrary_partial_pose_shows_only_active_nodes() -> None:
    # Only jawOpen active, at 0.6; correctives inactive.
    graph = build_behavior_graph(_Reader(), [0.6, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.0])
    ids = {n["data"]["id"] for n in graph.nodes}
    assert ids == {"active_pose", "raw:0", "bone:raw:0", "shape:0"}
    assert _node(graph, "raw:0")["value"] == 0.6
    assert _node(graph, "shape:0")["value"] == 0.6
    assert _node(graph, "active_pose")["label"] == "Active Pose"


def test_empty_when_nothing_active() -> None:
    assert build_behavior_graph(_Reader(), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]).nodes == []
    assert build_behavior_graph(None, [1.0], [1.0]).nodes == []


def test_threshold_filters_negligible_activation() -> None:
    graph = build_behavior_graph(_Reader(), [1e-6, 0.0, 0.0], [1e-6, 0.0, 0.0, 0.0, 0.0])
    assert graph.nodes == []


# ---------------------------------------------------------------------------
# Real Ada head.dna pins
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_reader():
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    return get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")


def _raw_index(reader, short_name: str) -> int:
    target = f"CTRL_expressions.{short_name}"
    for i in range(int(reader.getRawControlCount())):
        if str(reader.getRawControlName(i)) == target:
            return i
    raise AssertionError(f"raw control {target!r} not found in DNA")


def _channel_index(reader, name: str) -> int:
    for i in range(int(reader.getBlendShapeChannelCount())):
        if str(reader.getBlendShapeChannelName(i)) == name:
            return i
    raise AssertionError(f"channel {name!r} not found in DNA")


def test_real_jaw_open_active_state(real_reader) -> None:
    reader = real_reader
    raw_count = int(reader.getRawControlCount())
    channel_count = int(reader.getBlendShapeChannelCount())
    jaw = _raw_index(reader, "jawOpen")
    jaw_open_channel = _channel_index(reader, "jaw_open")

    # Controlled active state: only jawOpen driven, its primary channel active.
    raw_values = [0.0] * raw_count
    raw_values[jaw] = 1.0
    blend_outputs = [0.0] * channel_count
    blend_outputs[jaw_open_channel] = 1.0

    graph = build_behavior_graph(reader, raw_values, blend_outputs)

    assert _node(graph, "active_pose")["count"] == 1
    assert _node(graph, f"raw:{jaw}")["value"] == 1.0
    # jawOpen's primary shape key resolves to the real jaw_open channel.
    shape = _node(graph, f"shape:{jaw_open_channel}")
    assert shape["label"] == "jaw_open"
    assert (f"raw:{jaw}", f"shape:{jaw_open_channel}") in _edge_pairs(graph)
    # jawOpen poses real bones (cell-accurate, non-empty joint names).
    bone = _node(graph, f"bone:raw:{jaw}")
    assert bone["kind"] == KIND_BONE
    assert len(bone["bones"]) > 0
    assert all(isinstance(name, str) and name for name in bone["bones"])


def test_real_rbf_raw_control_flagged(real_reader) -> None:
    reader = real_reader
    rbf_raws: set[int] = set()
    for solver in range(int(reader.getRBFSolverCount())):
        rbf_raws.update(int(i) for i in reader.getRBFSolverRawControlIndices(solver))
    assert rbf_raws, "Ada has RBF (neck-rotation) raw controls"

    rbf_index = min(rbf_raws)
    raw_count = int(reader.getRawControlCount())
    raw_values = [0.0] * raw_count
    raw_values[rbf_index] = 1.0
    graph = build_behavior_graph(reader, raw_values, [0.0] * int(reader.getBlendShapeChannelCount()))
    assert _node(graph, f"raw:{rbf_index}")["rbf"] is True
