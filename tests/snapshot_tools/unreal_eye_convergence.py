"""Run capture(output_path) in Unreal Editor to snapshot its eye convergence graph.

Only a transient Control Rig is evaluated; no assets or level actors are changed.
The reference fixture is tests/test_files/json/eye_convergence_unreal.json.
"""

import json

from pathlib import Path

import unreal


def _key(name):
    return unreal.RigElementKey(
        name=name, type=unreal.RigElementType.NULL if name.endswith("_space") else unreal.RigElementType.CONTROL
    )


def _transform(value):
    return {
        "location": [value.translation.x, value.translation.y, value.translation.z],
        "rotation": [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
        "scale": [value.scale3d.x, value.scale3d.y, value.scale3d.z],
    }


def capture(output_path):
    """Capture the graph's output for neutral, moved and independently offset eyes."""
    blueprint = unreal.load_asset("/Game/MetaHumans/Common/Face/Face_ControlBoard_CtrlRig")
    rig = blueprint.create_control_rig()
    rig.request_init()
    rig.execute("PrepareForExecution")
    hierarchy = rig.get_hierarchy()
    names = ["CTRL_C_eyesAim", "CTRL_L_eyeAim_space", "CTRL_R_eyeAim_space", "CTRL_L_eyeAim", "CTRL_R_eyeAim"]
    result = {
        "engine": unreal.SystemLibrary.get_engine_version(),
        "asset": blueprint.get_path_name(),
        "initial": {n: _transform(hierarchy.get_global_transform(_key(n), initial=True)) for n in names},
        "samples": [],
    }
    for case in ("neutral", "moved_center", "eye_offsets"):
        hierarchy.reset_pose_to_initial(unreal.RigElementType.ALL)
        hierarchy.set_control_value(_key("CTRL_lookAtSwitch"), hierarchy.make_control_value_from_float(1.0))
        if case != "neutral":
            center = hierarchy.get_global_transform(_key("CTRL_C_eyesAim"))
            center.translation += unreal.Vector(4.0, -9.0, 2.0)
            center.rotation = unreal.Rotator(pitch=15.0, yaw=20.0, roll=-10.0).quaternion() * center.rotation
            center.scale3d = unreal.Vector(1.2, 1.2, 1.2)
            hierarchy.set_global_transform(_key("CTRL_C_eyesAim"), center)
        if case == "eye_offsets":
            for side, delta in (("L", unreal.Vector(1.0, 2.0, -0.5)), ("R", unreal.Vector(-0.5, 1.0, 0.3))):
                local = hierarchy.get_local_transform(_key(f"CTRL_{side}_eyeAim"))
                local.translation += delta
                hierarchy.set_local_transform(_key(f"CTRL_{side}_eyeAim"), local)
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            hierarchy.set_control_value(_key("CTRL_convergenceSwitch"), hierarchy.make_control_value_from_float(value))
            if not rig.execute("Forwards Solve"):
                raise RuntimeError("Unreal did not execute the facial rig's forward solve")
            result["samples"].append(
                {
                    "case": case,
                    "value": value,
                    "transforms": {n: _transform(hierarchy.get_global_transform(_key(n))) for n in names},
                }
            )
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
