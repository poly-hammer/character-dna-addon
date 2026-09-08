"""Validate and summarize saved full-scene stock Blender benchmark pairs."""

from __future__ import annotations

import argparse
import json
import statistics

from pathlib import Path

import numpy as np


def main() -> None:
    """Require matching provenance and snapshot parity before computing speedup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    pairs = []
    pooled = {"python": [], "carrier": []}
    for reference_path in sorted(args.root.glob("python-*.json")):
        suffix = reference_path.stem.removeprefix("python-")
        candidate_path = args.root / f"carrier-{suffix}.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        for key in ("fixture_sha256", "module_sha256", "frame_count", "warmup"):
            assert reference[key] == candidate[key], (key, reference_path)
        assert reference["runtime"] == candidate["runtime"]
        with (
            np.load(reference_path.with_suffix(".npz")) as expected,
            np.load(candidate_path.with_suffix(".npz")) as actual,
        ):
            assert set(expected.files) == set(actual.files)
            errors = {}
            for name in expected.files:
                assert expected[name].shape == actual[name].shape
                assert np.isfinite(expected[name]).all() and np.isfinite(actual[name]).all()
                errors[name] = float(np.max(np.abs(expected[name] - actual[name]), initial=0))
        maximum = max(errors.values(), default=0)
        assert maximum < 1e-5, (reference_path, maximum)
        pairs.append(
            {
                "trial": suffix,
                "python_ms": reference["mean_ms"],
                "carrier_ms": candidate["mean_ms"],
                "ratio": reference["mean_ms"] / candidate["mean_ms"],
                "snapshot_max_error": maximum,
            }
        )
        pooled["python"].extend(reference["samples_ms"])
        pooled["carrier"].extend(candidate["samples_ms"])
    assert len(pairs) == 5
    result = {"trials": pairs, "backends": {}}
    for backend, samples in pooled.items():
        ordered = sorted(samples)
        means = [pair[f"{backend}_ms"] for pair in pairs]
        result["backends"][backend] = {
            "samples": len(samples),
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": ordered[int(len(ordered) * 0.95)],
            "p99_ms": ordered[int(len(ordered) * 0.99)],
            "trial_mean_stdev_ms": statistics.stdev(means),
            "trial_mean_range_ms": [min(means), max(means)],
        }
    result["speedup"] = result["backends"]["python"]["mean_ms"] / result["backends"]["carrier"]["mean_ms"]
    result["performance_target_met"] = result["speedup"] >= 7
    result["parity_passed"] = True
    result["scope"] = (
        "Head/body/shape/map scene frames, eye aim OFF, no production preference or lifecycle certification"
    )
    output = args.root / "summary.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
