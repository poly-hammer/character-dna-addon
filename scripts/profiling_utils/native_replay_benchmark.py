"""Measure downstream scene cost using prerecorded same-frame SDK outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from array import array
from pathlib import Path

import bpy
import numpy as np


def run(args: argparse.Namespace) -> None:
    from character_dna import rig_instance
    from character_dna.utilities import get_addon_window_manager_properties
    from profiling_utils.native_frame_benchmark import _hash, _load, bind_native, snapshot

    instance = _load(args.fixture)
    get_addon_window_manager_properties().evaluate_dependency_graph = True
    rig_instance.start_listening()
    scene = bpy.context.scene
    captured = {"head": [], "body": []}
    for frame in range(1, 121):
        scene.frame_set(frame)
        bpy.context.evaluated_depsgraph_get()
        for component, rows in captured.items():
            sdk = getattr(instance, f"{component}_instance")
            rows.append(
                {
                    "joints": array("f", sdk.getJointOutputs()),
                    "shapes": array("f", sdk.getBlendShapeOutputs()),
                    "maps": array("f", sdk.getAnimatedMapOutputs()),
                }
            )
    bind_native(instance)
    for component, rows in captured.items():
        rig = getattr(instance, f"{component}_rig")
        settings = rig[f"_riglogic_{component}"]
        settings["replay"] = rows
        settings["replay_applied"] = 0
        for frame in range(1, 121):
            settings["replay_frame"] = frame
            rig.keyframe_insert(data_path=f'["_riglogic_{component}"]["replay_frame"]', frame=frame)
    bpy.app.riglogic.rebuild_bindings()
    for index in range(args.warmup):
        scene.frame_set(index % 120 + 1)
        bpy.context.evaluated_depsgraph_get()
    samples = []
    for index in range(args.frames):
        start = time.perf_counter_ns()
        scene.frame_set((index + args.warmup) % 120 + 1)
        graph = bpy.context.evaluated_depsgraph_get()
        samples.append(time.perf_counter_ns() - start)
    for component in captured:
        rig = getattr(instance, f"{component}_rig").evaluated_get(graph)
        if rig[f"_riglogic_{component}"]["replay_applied"] != 1:
            raise AssertionError(f"Output replay did not run for {component}")
    actual = snapshot(instance, graph)
    errors = {}
    with np.load(args.reference) as expected:
        if set(expected.files) != set(actual):
            raise AssertionError("Replay target list differs")
        for name, values in actual.items():
            error = float(np.max(np.abs(values - expected[name]), initial=0.0))
            errors[name] = error
            if error > 1e-5:
                raise AssertionError(f"Replay parity failed: {name}, {error}")
    report = {
        "scope": "output_replay_downstream_diagnostic_not_native_backend",
        "fixture_sha256": _hash(args.fixture),
        "binary_sha256": _hash(Path(bpy.app.binary_path)),
        "warmup": args.warmup,
        "frames": args.frames,
        "last_frame": scene.frame_current,
        "mean_ms": statistics.mean(samples) / 1e6,
        "p95_ms": float(np.percentile(samples, 95)) / 1e6,
        "evaluated_fps": len(samples) * 1e9 / sum(samples),
        "times_ns": samples,
        "max_output_error": max(errors.values()),
        "passed": True,
        "caveat": "Includes replay-array lookup and animated replay index; not a mathematical upper bound.",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"REPLAY: {report['mean_ms']:.4f}ms, parity error={report['max_output_error']:.9g}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/addons"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--frames", type=int, default=1200)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
