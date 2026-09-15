"""Bind-time numeric plans and bulk RNA capture for the native frame evaluator."""

from __future__ import annotations

from array import array
from typing import Any


def prepare(record: dict[str, Any], native: Any) -> Any:
    """Resolve topology and control indices once; numerical evaluation lives in C++."""
    bones = record["rig"].pose.bones
    indices = {bone.name: index for index, bone in enumerate(bones)}
    face = record["face"]
    face_indices = {bone.name: index for index, bone in enumerate(face.pose.bones)} if face else {}
    rest = array("f", [0.0]) * (len(bones) * 16)
    record["rig"].data.bones.foreach_get("matrix_local", rest)
    parents = array("i", [indices[bone.parent.name] if bone.parent else -1 for bone in bones])
    gui = array("i")
    for index, name, axis in record["gui"]:
        center = face_indices.get("CTRL_C_eye", -1) if name in {"CTRL_L_eye", "CTRL_R_eye"} else -1
        gui.extend((index, face_indices.get(name, -1), "xyz".index(axis), center))
    raw = array(
        "i", [value for index, name, axis in record["raw"] for value in (index, indices[name], "wxyz".index(axis))]
    )
    chain = array("i")
    nodes = {}

    def add_node(name: str) -> int:
        if name in nodes:
            return nodes[name]
        bone = bones[name]
        parent = add_node(bone.parent.name) if bone.parent else -1
        external = name in record["input_bones"]
        slot = record["bone_slots"].get(name, -1)
        scale_mode = ("FULL", "FIX_SHEAR", "ALIGNED", "AVERAGE", "NONE", "NONE_LEGACY").index(bone.bone.inherit_scale)
        flags = (
            int(external)
            | (int(not bone.bone.use_local_location) << 1)
            | (int(not bone.bone.use_inherit_rotation) << 2)
            | (scale_mode << 3)
        )
        node = len(chain) // 4
        chain.extend((indices[name], parent, slot, flags))
        nodes[name] = node
        return node

    eyes = array("i")
    targets = []
    controls = {(name, axis): index for index, name, axis in record["gui"]}
    if record["component"] == "head" and face:
        for target_index, side in enumerate(("L", "R")):
            eye, target, control = f"FACIAL_{side}_Eye", f"CTRL_{side}_eyeAim", f"CTRL_{side}_eye"
            targets.append(target)
            if eye in indices and target in face_indices:
                eyes.extend(
                    (add_node(eye), target_index, controls.get((control, "x"), -1), controls.get((control, "y"), -1))
                )
    record["capture_sizes"] = (len(bones), len(face_indices))
    record["eye_targets"] = targets
    return native.create_frame_plan(
        native.create_session(record["model"]),
        record["plan"],
        rest,
        parents,
        gui,
        raw,
        chain,
        eyes,
        len(face_indices),
        record["component"] == "body",
    )


def buffers(record: dict[str, Any]) -> dict[str, Any]:
    """Allocate stable caller-owned capture storage once per evaluation context."""
    bones, face = record["capture_sizes"]
    return {
        "poses": array("f", [0.0]) * (bones * 16),
        "bases": array("f", [0.0]) * (bones * 16),
        "face": array("f", [0.0]) * (face * 3),
        "spatial": array("f", [0.0]) * 38,
    }


def capture(record: dict[str, Any], context: dict[str, Any], graph: Any) -> bool:
    """Copy public RNA buffers."""
    rig = record["rig"].evaluated_get(graph)
    rig.pose.bones.foreach_get("matrix", context["poses"])
    spatial = context["spatial"]
    spatial[:16] = array("f", (value for column in rig.matrix_world.col for value in column))
    aim = False
    if record["face"]:
        face = record["face"].evaluated_get(graph)
        face.pose.bones.foreach_get("location", context["face"])
        spatial[16:32] = array("f", (value for column in face.matrix_world.col for value in column))
        switch = face.pose.bones.get("CTRL_lookAtSwitch")
        aim = bool(switch and switch.location.y >= 0.99)
        if aim:
            for index, name in enumerate(record["eye_targets"]):
                target = face.pose.bones.get(name)
                if target:
                    spatial[32 + index * 3 : 35 + index * 3] = array("f", target.head)
    if aim:
        record["rig"].pose.bones.foreach_get("matrix_basis", context["bases"])
    return aim
