"""Capture evaluated animation traces for parity, independently of timed runs."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import bpy
import numpy as np


def run(args: argparse.Namespace) -> None:
    from profiling_utils.native_frame_benchmark import _hash, _load, bind_native, snapshot

    instance = _load(args.fixture)
    if args.backend == "native":
        bind_native(instance)
    else:
        from character_dna import rig_instance
        from character_dna.utilities import get_addon_window_manager_properties

        get_addon_window_manager_properties().evaluate_dependency_graph = True
        rig_instance.start_listening()
    frames = [(frame, 0.0) for frame in range(1, 121)] + [(40, 0.5), (9, 0.25), (120, 0.0), (1, 0.0), (57, 0.0)]
    arrays = {}
    for index, (frame, subframe) in enumerate(frames):
        bpy.context.scene.frame_set(frame, subframe=subframe)
        graph = bpy.context.evaluated_depsgraph_get()
        for name, values in snapshot(instance, graph).items():
            arrays[f"{index}:{name}"] = values
    snapshot_path = args.output.with_suffix(".npz")
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)
    np.savez_compressed(snapshot_path, **arrays)
    errors = {}
    if args.reference:
        with np.load(args.reference) as expected:
            if set(expected.files) != set(arrays):
                raise AssertionError("Trace targets differ")
            for name, actual in arrays.items():
                reference = expected[name]
                if reference.shape != actual.shape or not np.isfinite(reference).all():
                    raise AssertionError(f"Invalid reference trace: {name}")
                error = float(np.max(np.abs(actual - reference), initial=0.0))
                if error > 1e-5:
                    errors[name] = error
    report = {
        "scope": "untimed_125_frame_geometry_pose_mask_trace",
        "backend": args.backend,
        "frames": frames,
        "fixture_sha256": _hash(args.fixture),
        "binary_sha256": _hash(Path(bpy.app.binary_path)),
        "reference": str(args.reference) if args.reference else None,
        "snapshot": str(snapshot_path),
        "errors": errors,
        "passed": not errors,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if errors:
        raise AssertionError(f"Animation trace differs on {len(errors)} arrays; see {args.output}")
    print(f"FRAME TRACE {args.backend} PASS: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["python", "native"], required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
