"""Measure native multi-character scene scaling with output-isolation checks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from pathlib import Path

import bpy
import numpy as np


def run(args: argparse.Namespace) -> None:
    from profiling_utils.native_frame_benchmark import _hash
    from profiling_utils.native_isolation_test import append_character, pose

    bpy.ops.wm.open_mainfile(filepath=str(args.fixture))
    characters = [
        (
            next(obj for obj in bpy.data.objects if "_riglogic_head" in obj),
            next(obj for obj in bpy.data.objects if "_riglogic_body" in obj),
        )
    ]
    scene = bpy.context.scene
    results = []
    for count in (1, 2, 4, 8):
        while len(characters) < count:
            head, body, _root = append_character(args.fixture)
            characters.append((head, body))
        bpy.app.riglogic.rebuild_bindings()
        for index in range(args.warmup):
            scene.frame_set(index % 120 + 1)
            bpy.context.evaluated_depsgraph_get()
        times = []
        for index in range(args.frames):
            start = time.perf_counter_ns()
            scene.frame_set((index + args.warmup) % 120 + 1)
            graph = bpy.context.evaluated_depsgraph_get()
            times.append(time.perf_counter_ns() - start)
        reference_head = pose(characters[0][0], graph)
        reference_body = pose(characters[0][1], graph)
        for head, body in characters:
            if not np.allclose(pose(head, graph), reference_head, rtol=0, atol=1e-5):
                raise AssertionError("Multi-character head parity failed")
            if not np.allclose(pose(body, graph), reference_body, rtol=0, atol=1e-5):
                raise AssertionError("Multi-character body parity failed")
        result = {
            "characters": count,
            "mean_ms": statistics.mean(times) / 1e6,
            "p95_ms": float(np.percentile(times, 95)) / 1e6,
            "evaluated_fps": len(times) * 1e9 / sum(times),
            "times_ns": times,
            "pose_parity_passed": True,
        }
        results.append(result)
        print(f"NATIVE SCALING {count}: {result['mean_ms']:.3f}ms, {result['evaluated_fps']:.2f} evalFPS", flush=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "scope": "native_multi_character_scene_scaling_no_viewport",
                "fixture_sha256": _hash(args.fixture),
                "binary_sha256": _hash(Path(bpy.app.binary_path)),
                "warmup": args.warmup,
                "frames": args.frames,
                "results": results,
                "caveat": "One trial per size; duplicated identical actions, independent data and solver state.",
            },
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--frames", type=int, default=1200)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
