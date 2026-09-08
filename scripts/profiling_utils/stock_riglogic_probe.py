"""Probe supported bulk RNA operations in a disposable stock Blender process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from array import array
from pathlib import Path
from typing import Any

import bpy


def runtime_manifest() -> dict[str, Any]:
    """Identify the runtime and refuse the previous fork's native API."""
    if hasattr(bpy.app, "riglogic"):
        raise RuntimeError("Run this probe in stock Blender, not the RigLogic fork")
    if not bpy.app.background or bpy.data.filepath:
        raise RuntimeError("Use --background --factory-startup in a disposable process")
    with Path(bpy.app.binary_path).open("rb") as stream:
        binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "binary": bpy.app.binary_path,
        "sha256": binary_hash,
        "version": bpy.app.version_string,
        "build_hash": bpy.app.build_hash.decode(),
        "python": sys.version,
    }


def make_rig() -> tuple[Any, Any]:
    """Create mixed-mode bones and independently weighted mesh vertices."""
    bpy.ops.object.select_all(action="DESELECT")
    armature = bpy.data.armatures.new("BulkProbe")
    rig = bpy.data.objects.new("BulkProbe", armature)
    bpy.context.scene.collection.objects.link(rig)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for index, name in enumerate(("Driven", "Untouched")):
        bone = armature.edit_bones.new(name)
        bone.head = (0, 0, index * 2)
        bone.tail = (0, 0, index * 2 + 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.pose.bones[0].rotation_mode = "QUATERNION"
    rig.pose.bones[1].rotation_mode = "XYZ"
    rig.pose.bones[1].location = (0.125, 0, 0)
    mesh = bpy.data.meshes.new("BulkProbeMesh")
    mesh.from_pydata([(0, 0, 0), (0, 0, 2)], [], [])
    mesh_object = bpy.data.objects.new("BulkProbeMesh", mesh)
    bpy.context.scene.collection.objects.link(mesh_object)
    for index, name in enumerate(("Driven", "Untouched")):
        group = mesh_object.vertex_groups.new(name=name)
        group.add([index], 1.0, "REPLACE")
    modifier = mesh_object.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    bpy.context.view_layer.update()
    return rig, mesh_object


def vertices(mesh_object: Any) -> list[list[float]]:
    """Read evaluated positions after the caller's declared update barrier."""
    evaluated = mesh_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [list(vertex.co) for vertex in evaluated.data.vertices]


def probe_pose(rig: Any, mesh_object: Any) -> dict[str, Any]:
    """Check bulk roundtrips and whether owner tags propagate deformation."""
    capabilities: dict[str, Any] = {}
    for property_name, width in (
        ("location", 3),
        ("rotation_euler", 3),
        ("rotation_quaternion", 4),
        ("scale", 3),
        ("matrix_basis", 16),
        ("matrix", 16),
    ):
        buffer = array("f", [0]) * (width * len(rig.pose.bones))
        try:
            rig.pose.bones.foreach_get(property_name, buffer)
            result = {"get": True, "values": list(buffer)}
        except (TypeError, ValueError, RuntimeError, AttributeError) as error:
            capabilities[property_name] = {"get": False, "error": str(error)}
            continue
        if property_name != "matrix":
            try:
                rig.pose.bones.foreach_set(property_name, buffer)
                result["set"] = True
            except (TypeError, ValueError, RuntimeError, AttributeError) as error:
                result.update(set=False, error=str(error))
        capabilities[property_name] = result
    rig.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()
    before = vertices(mesh_object)
    buffer = array("f", [0]) * (3 * len(rig.pose.bones))
    rig.pose.bones.foreach_get("location", buffer)
    buffer[0] = 0.75
    rig.pose.bones.foreach_set("location", buffer)
    bpy.context.view_layer.update()
    without_tag = vertices(mesh_object)
    rig.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()
    with_tag = vertices(mesh_object)
    assert abs(with_tag[0][0] - 0.75) < 1e-6, with_tag
    assert with_tag[1] == before[1], (before, with_tag)
    return {"capabilities": capabilities, "before": before, "without_tag": without_tag, "with_tag": with_tag}


def probe_keys(mesh_object: Any) -> dict[str, Any]:
    """Check numeric key collection writes against evaluated mesh positions."""
    mesh_object.shape_key_add(name="Basis")
    shape = mesh_object.shape_key_add(name="Driven")
    shape.data[0].co.x += 0.5
    shape.value = 0.0
    bpy.context.view_layer.update()
    before = vertices(mesh_object)
    keys = mesh_object.data.shape_keys
    values = array("f", [0, 0.5])
    keys.key_blocks.foreach_set("value", values)
    keys.update_tag()
    mesh_object.update_tag(refresh={"DATA"})
    bpy.context.view_layer.update()
    after = vertices(mesh_object)
    assert abs(after[0][0] - before[0][0] - 0.25) < 1e-6, (before, after)
    return {"before": before, "after": after}


def probe_sockets() -> dict[str, Any]:
    """Test homogeneous and heterogeneous node socket collections separately."""
    material = bpy.data.materials.new("BulkProbe")
    material.use_nodes = True
    group = bpy.data.node_groups.new("BulkProbe", "ShaderNodeTree")
    for name in ("First", "Second"):
        group.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocketFloat")
    node = material.node_tree.nodes.new("ShaderNodeGroup")
    node.node_tree = group
    result: dict[str, Any] = {}
    for name in ("homogeneous", "heterogeneous"):
        if name == "heterogeneous":
            group.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
        width = sum(1 if socket.type == "VALUE" else 4 for socket in node.inputs)
        values = array("f", [0.25]) * width
        try:
            node.inputs.foreach_set("default_value", values)
            readback = array("f", [0]) * width
            node.inputs.foreach_get("default_value", readback)
            assert readback == values, (readback, values)
            result[name] = {"supported": True, "values": list(readback)}
        except (TypeError, ValueError, RuntimeError, AttributeError) as error:
            result[name] = {"supported": False, "error": str(error)}
    return result


def main() -> None:
    """Write reproducible capability evidence; assertions fail the process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    rig, mesh_object = make_rig()
    result = {
        "runtime": manifest,
        "pose": probe_pose(rig, mesh_object),
        "keys": probe_keys(mesh_object),
        "sockets": probe_sockets(),
        "passed": True,
        "scope": "Bulk API and pose/key geometry only; no render or simulation certification",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
