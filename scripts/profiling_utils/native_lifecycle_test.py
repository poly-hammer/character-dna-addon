"""Verify persisted native bindings in a fresh Blender without the add-on."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np


def prepare(fixture: Path, output: Path) -> None:
    from profiling_utils.native_frame_benchmark import _load, bind_native, snapshot

    instance = _load(fixture)
    bind_native(instance)
    samples = {}
    frames = (1, 17, 83, 120, 17, 1)
    for index, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        for name, values in snapshot(instance, graph).items():
            samples[f"{index}:{name}"] = values
    blend_path = output.with_suffix(".blend")
    snapshot_path = output.with_suffix(".npz")
    if blend_path.exists() or snapshot_path.exists():
        raise FileExistsError("Lifecycle fixture already exists")
    np.savez_compressed(snapshot_path, **samples)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), relative_remap=False)
    report = {
        "head": instance.head_rig.name,
        "body": instance.body_rig.name,
        "material": instance.head_material.name,
        "node": instance.head_texture_masks_node.name,
        "maps": instance.head_animated_map_plan,
        "frames": frames,
        "blend": str(blend_path),
        "snapshots": str(snapshot_path),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"NATIVE LIFECYCLE FIXTURE: {output}", flush=True)


def verify(fixture: Path, output: Path) -> None:
    from profiling_utils.native_frame_benchmark import snapshot

    report = json.loads(fixture.read_text(encoding="utf-8"))
    bpy.ops.wm.open_mainfile(filepath=report["blend"])
    if "character_dna" in bpy.context.preferences.addons:
        raise AssertionError("Persistence test must run without the add-on enabled")
    instance = SimpleNamespace(
        head_rig=bpy.data.objects[report["head"]],
        body_rig=bpy.data.objects[report["body"]],
        head_material=bpy.data.materials[report["material"]],
        head_texture_masks_node=SimpleNamespace(name=report["node"]),
        head_animated_map_plan=report["maps"],
    )
    errors = []
    with np.load(report["snapshots"]) as expected:
        for index, frame in enumerate(report["frames"]):
            bpy.context.scene.frame_set(frame)
            graph = bpy.context.evaluated_depsgraph_get()
            maximum = 0.0
            for name, actual in snapshot(instance, graph).items():
                reference = expected[f"{index}:{name}"]
                if reference.shape != actual.shape:
                    raise AssertionError(f"Reload changed output layout: {name}")
                maximum = max(maximum, float(np.max(np.abs(actual - reference), initial=0.0)))
            errors.append({"frame": frame, "max_error": maximum, "passed": maximum <= 1e-5})
            print(f"RELOAD frame={frame} error={maximum:.9g}", flush=True)
    result = {
        "scope": "native_fresh_process_reload",
        "addon_enabled": False,
        "frames": errors,
        "passed": all(item["passed"] for item in errors),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    if not result["passed"]:
        raise AssertionError("Native reload output parity failed")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "verify"])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (prepare if args.stage == "prepare" else verify)(args.fixture, args.output)
