"""Native-only settings, LOD, and evaluated graph isolation checks."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import bpy
import numpy as np

from numpy.typing import ArrayLike


def matrix_values(rig: bpy.types.Object) -> np.ndarray:
    return np.asarray(
        [[value for row in bone.matrix_basis for value in row] for bone in rig.pose.bones], dtype=np.float32
    )


def assert_arrays(actual: ArrayLike, expected: ArrayLike, label: str) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"Invalid arrays: {label}")
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    if error > 1e-5:
        raise AssertionError(f"{label}: error {error}")
    return error


def run(fixture: Path, output: Path) -> None:  # noqa: PLR0915
    bpy.ops.wm.open_mainfile(filepath=str(fixture))
    if "character_dna" in bpy.context.preferences.addons:
        raise AssertionError("Native-only test loaded the add-on")
    head = next(obj for obj in bpy.data.objects if "_riglogic_head" in obj)
    body = next(obj for obj in bpy.data.objects if "_riglogic_body" in obj)
    settings = head["_riglogic_head"]
    body_settings = body["_riglogic_body"]
    scene = bpy.context.scene
    scene.frame_set(37)
    graph = bpy.context.evaluated_depsgraph_get()
    baseline_head = matrix_values(head.evaluated_get(graph))
    baseline_body = matrix_values(body.evaluated_get(graph))
    results = []

    for name in ("evaluate_bones", "enabled"):
        settings[name] = False
        body_settings[name] = False
        bpy.app.riglogic.rebuild_bindings()
        graph = bpy.context.evaluated_depsgraph_get()
        assert_arrays(matrix_values(head.evaluated_get(graph)), matrix_values(head), f"head {name}")
        assert_arrays(matrix_values(body.evaluated_get(graph)), matrix_values(body), f"body {name}")
        settings[name] = True
        body_settings[name] = True
        bpy.app.riglogic.rebuild_bindings()
        graph = bpy.context.evaluated_depsgraph_get()
        assert_arrays(matrix_values(head.evaluated_get(graph)), baseline_head, f"head resume {name}")
        assert_arrays(matrix_values(body.evaluated_get(graph)), baseline_body, f"body resume {name}")
        results.append(name)

    for name in ("evaluate_shape_keys", "evaluate_texture_masks"):
        settings[name] = False
        bpy.app.riglogic.rebuild_bindings()
        graph = bpy.context.evaluated_depsgraph_get()
        if name == "evaluate_shape_keys":
            for target in settings["keys"]:
                key = target["key"]
                assert_arrays(
                    [block.value for block in key.evaluated_get(graph).key_blocks],
                    [block.value for block in key.key_blocks],
                    "disabled shape keys",
                )
        else:
            material = settings["material"]
            node = material.node_tree.nodes[settings["texture_node"]]
            evaluated_node = material.evaluated_get(graph).node_tree.nodes[node.name]
            assert_arrays(
                [evaluated_node.inputs[item["socket"]].default_value for item in settings["maps"]],
                [node.inputs[item["socket"]].default_value for item in settings["maps"]],
                "disabled maps",
            )
        settings[name] = True
        bpy.app.riglogic.rebuild_bindings()
        results.append(name)

    body_session = bpy.app.riglogic.load(body_settings["path"], True)
    body_lods = bpy.app.riglogic.info(body_session)["lod_count"]
    for lod in (2, 4, 0):
        settings["lod"] = lod
        body_settings["lod"] = min(lod, body_lods - 1)
        bpy.app.riglogic.rebuild_bindings()
        graph = bpy.context.evaluated_depsgraph_get()
        if not np.isfinite(matrix_values(head.evaluated_get(graph))).all():
            raise AssertionError("Non-finite LOD output")
    assert_arrays(matrix_values(head.evaluated_get(graph)), baseline_head, "LOD0 restored")
    results.append("LOD transitions")

    body_settings["evaluate_rbfs"] = False
    bpy.app.riglogic.rebuild_bindings()
    graph = bpy.context.evaluated_depsgraph_get()
    driven = [item["bone"] for item in body_settings["joints"]]
    neutral = np.asarray(
        [[value for row in body.evaluated_get(graph).pose.bones[name].matrix_basis for value in row] for name in driven]
    )
    scene.frame_set(81)
    graph = bpy.context.evaluated_depsgraph_get()
    after = np.asarray(
        [[value for row in body.evaluated_get(graph).pose.bones[name].matrix_basis for value in row] for name in driven]
    )
    assert_arrays(after, neutral, "RBF-off driver independence")
    body_settings["evaluate_rbfs"] = True
    bpy.app.riglogic.rebuild_bindings()
    results.append("RBF-off neutral outputs")

    face = settings["face_board"]
    for enabled in (1.0, 0.0, 1.0):
        face.pose.bones["CTRL_lookAtSwitch"].location.y = enabled
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated_face = face.evaluated_get(graph)
        if evaluated_face.pose.bones["CTRL_C_eyesAim"].hide != (enabled < 0.99):
            raise AssertionError("Eye-aim control visibility did not follow the switch")
    face.pose.bones["CTRL_lookAtSwitch"].location.y = 0.0
    results.append("eye-aim control visibility")

    second = scene.view_layers.new("NativeIsolation")
    original_layer = bpy.context.view_layer
    with bpy.context.temp_override(view_layer=second):
        scene.frame_set(83)
        second_graph = bpy.context.evaluated_depsgraph_get()
        second_copy = head.evaluated_get(second_graph)
        second_values = matrix_values(second_copy)
    scene.frame_set(83)
    first_graph = bpy.context.evaluated_depsgraph_get()
    assert_arrays(matrix_values(head.evaluated_get(first_graph)), second_values, "view layer output")
    if head.evaluated_get(first_graph).as_pointer() == second_copy.as_pointer():
        raise AssertionError("View layers unexpectedly share evaluated object")
    with bpy.context.temp_override(view_layer=original_layer):
        scene.frame_set(17)
    results.append("independent view layers")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"passed": True, "checks": results, "binary": bpy.app.binary_path}, stream, indent=2)
    print(f"NATIVE SETTINGS PASS: {results}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.fixture, args.output)
