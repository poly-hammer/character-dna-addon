"""Native character duplication, head-only, and file-handle lifecycle tests."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile

from pathlib import Path

import bpy
import numpy as np


def append_character(fixture: Path) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Collection]:
    """Append a whole scene so Blender remaps every internal native ID reference."""
    active_scene = bpy.context.scene
    with tempfile.TemporaryDirectory(prefix="native-append-") as temporary:
        source = Path(temporary) / fixture.name
        shutil.copyfile(fixture, source)
        with bpy.data.libraries.load(str(source), link=False) as (available, requested):
            requested.scenes = [available.scenes[0]]
    imported = requested.scenes[0]
    objects = list(imported.objects)
    root = bpy.data.collections.new(f"NativeCharacter_{len(bpy.data.scenes)}")
    active_scene.collection.children.link(root)
    for collection in list(imported.collection.children):
        root.children.link(collection)
    for obj in list(imported.collection.objects):
        root.objects.link(obj)
    bpy.data.scenes.remove(imported)
    head = next(obj for obj in objects if "_riglogic_head" in obj)
    body = next(obj for obj in objects if "_riglogic_body" in obj)
    return head, body, root


def pose(rig: bpy.types.Object, graph: bpy.types.Depsgraph) -> np.ndarray:
    return np.asarray(
        [[value for row in bone.matrix for value in row] for bone in rig.evaluated_get(graph).pose.bones],
        dtype=np.float32,
    )


def run(fixture: Path, output: Path) -> None:  # noqa: PLR0915
    bpy.ops.wm.open_mainfile(filepath=str(fixture))
    head = next(obj for obj in bpy.data.objects if "_riglogic_head" in obj)
    body = next(obj for obj in bpy.data.objects if "_riglogic_body" in obj)
    bpy.context.scene.frame_set(47)
    graph = bpy.context.evaluated_depsgraph_get()
    original_head = pose(head, graph)
    original_body = pose(body, graph)
    second_head, second_body, _root = append_character(fixture)
    if second_head["_riglogic_head"]["material"] == head["_riglogic_head"]["material"]:
        raise AssertionError("Appended character shares the native material writer")
    if second_head["_riglogic_head"]["face_board"] == head["_riglogic_head"]["face_board"]:
        raise AssertionError("Appended character shares GUI inputs")
    bpy.app.riglogic.rebuild_bindings()
    bpy.context.scene.frame_set(47)
    graph = bpy.context.evaluated_depsgraph_get()
    for actual, expected in ((pose(second_head, graph), original_head), (pose(second_body, graph), original_body)):
        if not np.allclose(actual, expected, rtol=0, atol=1e-5):
            raise AssertionError("Appended native character differs at the same frame")
    face = second_head["_riglogic_head"]["face_board"]
    face.animation_data_clear()
    face.pose.bones["CTRL_C_jaw"].location.y = 0.8
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    if not np.array_equal(pose(head, graph), original_head) or not np.array_equal(pose(body, graph), original_body):
        raise AssertionError("Editing the second character changed the first")
    if np.allclose(pose(second_head, graph), original_head, rtol=0, atol=1e-5):
        raise AssertionError("Independent GUI edit did not drive the second head")
    checks = ["independent appended character", "isolated GUI edit"]

    with tempfile.TemporaryDirectory(prefix="native-riglogic-dna-") as temporary:
        path = Path(temporary) / "head.dna"
        shutil.copyfile(head["_riglogic_head"]["path"], path)
        replacement = Path(temporary) / "replacement.dna"
        shutil.copyfile(path, replacement)
        head["_riglogic_head"]["path"] = str(path)
        bpy.app.riglogic.rebuild_bindings()
        bpy.context.view_layer.update()
        replacement.replace(path)
        bpy.app.riglogic.rebuild_bindings()
        bpy.context.view_layer.update()
        if not np.allclose(pose(head, bpy.context.evaluated_depsgraph_get()), original_head, rtol=0, atol=1e-5):
            raise AssertionError("DNA reload changed matching output")
        checks.append("Windows DNA replacement and reload")
        head["_riglogic_head"]["path"] = second_head["_riglogic_head"]["path"]
        bpy.app.riglogic.rebuild_bindings()
        bpy.context.view_layer.update()

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            for bone in obj.pose.bones:
                for constraint in list(bone.constraints):
                    if getattr(constraint, "target", None) == second_body:
                        bone.constraints.remove(constraint)
    bpy.data.objects.remove(second_body, do_unlink=True)
    second_head["_riglogic_head"]["face_board"] = None
    bpy.app.riglogic.rebuild_bindings()
    bpy.context.scene.frame_set(23)
    graph = bpy.context.evaluated_depsgraph_get()
    if not np.isfinite(pose(second_head, graph)).all():
        raise AssertionError("Head-only evaluation produced non-finite output")
    checks.append("head without body or face board")
    bpy.data.objects.remove(second_head, do_unlink=True)
    bpy.app.riglogic.rebuild_bindings()
    bpy.context.scene.frame_set(47)
    graph = bpy.context.evaluated_depsgraph_get()
    if not np.array_equal(pose(head, graph), original_head):
        raise AssertionError("Deleting the duplicate changed the remaining character")
    checks.append("delete native character and rebuild")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"passed": True, "checks": checks}, stream, indent=2)
    print(f"NATIVE ISOLATION PASS: {checks}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.fixture, args.output)
