"""Validate matched frame trials and report scene-evaluation throughput."""

from __future__ import annotations

import argparse
import json
import statistics

from pathlib import Path

import numpy as np


def summarize(folder: Path, output: Path) -> None:
    """Reject mismatched workloads or outputs before aggregating timing samples."""
    trials = {backend: sorted(folder.glob(f"frames-{backend}-[0-9][0-9].json")) for backend in ("python", "native")}
    if len(trials["python"]) != len(trials["native"]) or not trials["python"]:
        raise ValueError("Expected matching nonempty trial sets")
    reports = {
        backend: [json.loads(path.read_text(encoding="utf-8")) for path in paths] for backend, paths in trials.items()
    }
    baseline = reports["python"][0]
    for backend, items in reports.items():
        for item in items:
            if item["backend"] != backend:
                raise AssertionError("Mislabeled backend")
            for field in (
                "scope",
                "fixture_sha256",
                "binary_sha256",
                "script_sha256",
                "frame_count",
                "warmup",
                "last_frame",
                "workload",
            ):
                if item[field] != baseline[field]:
                    raise AssertionError(f"Incompatible trial field: {field}")
            if len(item["times_ns"]) != item["frame_count"] or min(item["times_ns"]) <= 0:
                raise AssertionError("Invalid timing sample count or duration")
    maximum_error = 0.0
    for reference, native in zip(reports["python"], reports["native"], strict=True):
        with np.load(reference["output_snapshot"]) as expected, np.load(native["output_snapshot"]) as actual:
            if set(expected.files) != set(actual.files):
                raise AssertionError("Output target lists differ")
            for name in expected.files:
                if expected[name].shape != actual[name].shape:
                    raise AssertionError(f"Output shape differs: {name}")
                if not np.isfinite(expected[name]).all() or not np.isfinite(actual[name]).all():
                    raise AssertionError(f"Non-finite output: {name}")
                error = float(np.max(np.abs(expected[name] - actual[name]), initial=0.0))
                maximum_error = max(maximum_error, error)
                if error > 1e-5:
                    raise AssertionError(f"Output parity failed: {name}, error={error}")
    metrics = {}
    for backend, items in reports.items():
        samples = np.asarray([sample for item in items for sample in item["times_ns"]], dtype=np.float64)
        trial_means = [statistics.mean(item["times_ns"]) / 1e6 for item in items]
        metrics[backend] = {
            "mean_ms": float(samples.mean() / 1e6),
            "median_ms": float(np.median(samples) / 1e6),
            "p95_ms": float(np.percentile(samples, 95) / 1e6),
            "p99_ms": float(np.percentile(samples, 99) / 1e6),
            "evaluated_fps": float(len(samples) * 1e9 / samples.sum()),
            "trial_mean_ms": trial_means,
            "trial_mean_range_ms": [min(trial_means), max(trial_means)],
            "frame_count": len(samples),
        }
        if len(trial_means) == 5:
            margin = 2.776445 * statistics.stdev(trial_means) / (5**0.5)
            mean = statistics.mean(trial_means)
            metrics[backend]["trial_mean_95_percent_t_interval_ms"] = [mean - margin, mean + margin]
    report = {
        "schema_version": 1,
        "scope": baseline["scope"],
        "workload": baseline["workload"],
        "fixture_sha256": baseline["fixture_sha256"],
        "binary_sha256": baseline["binary_sha256"],
        "trials_per_backend": len(trials["python"]),
        "output_parity_passed": True,
        "max_output_error": maximum_error,
        "parity_scope": "final-frame snapshots plus separately recorded pose/geometry parity tests",
        "metrics": metrics,
        "speedup": metrics["python"]["mean_ms"] / metrics["native"]["mean_ms"],
        "viewport_fps": None,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


def summarize_viewport(folder: Path, output: Path) -> None:
    """Aggregate compatible draw observations without claiming presented FPS."""
    reports = {
        backend: [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(folder.glob(f"viewport-{backend}-[0-9][0-9].json"))
        ]
        for backend in ("python", "native")
    }
    if not reports["python"] or len(reports["python"]) != len(reports["native"]):
        raise ValueError("Expected paired viewport captures")
    baseline = reports["python"][0]
    metrics = {}
    for backend, captures in reports.items():
        for capture in captures:
            if not capture["passed"] or capture["elapsed_observer_seconds"] <= 0:
                raise AssertionError("Invalid viewport capture")
            for field in (
                "scope",
                "fixture_sha256",
                "binary_sha256",
                "shading",
                "window_pixels",
                "region_pixels",
                "warmup_seconds",
                "requested_seconds",
                "playback_fps_target",
                "sync_mode",
            ):
                if capture[field] != baseline[field]:
                    raise AssertionError(f"Incompatible viewport setting: {field}")
        total_frames = sum(capture["distinct_drawn_frames"] - 1 for capture in captures)
        total_seconds = sum(capture["elapsed_observer_seconds"] for capture in captures)
        rates = [capture["drawn_animation_frames_per_second"] for capture in captures]
        metrics[backend] = {
            "drawn_animation_fps": total_frames / total_seconds,
            "trial_fps": rates,
            "range_fps": [min(rates), max(rates)],
            "observed_frames": total_frames,
            "observed_seconds": total_seconds,
        }
    report = {
        "scope": baseline["scope"],
        "shading": baseline["shading"],
        "window_pixels": baseline["window_pixels"],
        "region_pixels": baseline["region_pixels"],
        "trials_per_backend": len(reports["python"]),
        "metrics": metrics,
        "draw_throughput_speedup": metrics["native"]["drawn_animation_fps"] / metrics["python"]["drawn_animation_fps"],
        "presented_fps": None,
        "caveat": (
            "Post-pixel draw observer, not monitor presentation. "
            "Fixture has topology/mask images, not production skin textures."
        ),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewport", action="store_true")
    args = parser.parse_args()
    (summarize_viewport if args.viewport else summarize)(args.folder, args.output)
