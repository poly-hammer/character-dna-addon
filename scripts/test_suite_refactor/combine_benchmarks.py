"""Require three complete matched pairs on every supported CI lane."""

import argparse
import json
import math
import os
import statistics

from pathlib import Path


LANES = (
    ("windows-latest", "4.5"),
    ("macos-latest", "4.5"),
    ("ubuntu-latest", "5.2"),
    ("windows-latest", "5.2"),
    ("macos-latest", "5.2"),
)
ENVIRONMENT_KEYS = ("python", "blender", "coverage", "pytest", "ci_corpus")


def read_pair(folder: Path, repeat: int) -> dict:
    summary = json.loads((folder / "summary.json").read_text())
    pairs = summary["pairs"]
    if len(pairs) != 1 or pairs[0]["repeat"] != repeat:
        raise ValueError("Expected exactly the requested repetition")
    pair = pairs[0]
    comparison = json.loads((folder / f"comparison-{repeat}.json").read_text())
    if not summary["accepted"] or not pair["accepted"] or not comparison["accepted"]:
        raise ValueError("Case, coverage, environment or runtime comparison was rejected")
    durations = [pair[f"{variant}_seconds"] for variant in ("baseline", "candidate")]
    if not all(math.isfinite(value) and value > 0 for value in durations):
        raise ValueError("Invalid process duration")
    if durations[1] / durations[0] > 0.25:
        raise ValueError("Process runtime exceeds 25% of baseline")
    if pair["cache_policy"] != "cold":
        raise ValueError("CI acceptance requires cold temporary caches")
    if not all(isinstance(pair[key], str) and pair[key] for key in ("baseline_source", "candidate_source")):
        raise ValueError("Missing source fingerprints")
    if not isinstance(pair["environment"], dict):
        raise ValueError("Missing environment metadata")
    return pair


def combine(root: Path) -> dict:
    errors, lanes = [], []
    for system, blender in LANES:
        pairs = []
        for repeat in (1, 2, 3):
            name = f"test-refactor-{system}-{blender}-{repeat}"
            try:
                pairs.append(read_pair(root / name, repeat))
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"{name}: {error}")
        if len(pairs) != 3:
            continue
        errors.extend(
            f"{system} {blender}: {key} differs between repetitions"
            for key in ("baseline_source", "candidate_source")
            if len({pair[key] for pair in pairs}) != 1
        )
        for key in ENVIRONMENT_KEYS:
            values = [pair["environment"].get(key) for pair in pairs]
            if any(value is None for value in values) or len(set(values)) != 1:
                errors.append(f"{system} {blender}: {key} differs or is missing")
        environment = pairs[0]["environment"]
        python = "3.11" if blender == "4.5" else "3.13"
        for key, expected in (("blender", blender), ("python", python)):
            if not str(environment.get(key, "")).startswith(expected):
                errors.append(f"{system} {blender}: expected {key} {expected}")
        if environment.get("ci_corpus") is not True:
            errors.append(f"{system} {blender}: expected the complete CI corpus")
        lane = {"os": system, "blender": blender}
        for variant in ("baseline", "candidate"):
            values = [pair[f"{variant}_seconds"] for pair in pairs]
            lane[variant] = {
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        lane["ratio"] = lane["candidate"]["median_seconds"] / lane["baseline"]["median_seconds"]
        lanes.append(lane)
    result = {"accepted": not errors and len(lanes) == len(LANES), "lanes": lanes, "errors": errors}
    if lanes:
        medians = {
            variant: [lane[variant]["median_seconds"] for lane in lanes] for variant in ("baseline", "candidate")
        }
        result["total_process_ratio"] = sum(medians["candidate"]) / sum(medians["baseline"])
        result["critical_path_process_ratio"] = max(medians["candidate"]) / max(medians["baseline"])
        result["accepted"] &= result["total_process_ratio"] <= 0.25 and result["critical_path_process_ratio"] <= 0.25
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = combine(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    output = json.dumps(result, indent=2)
    (args.root / "acceptance.json").write_text(output, encoding="utf-8")
    print(output)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(f"### Complete matrix acceptance\n\n```json\n{output}\n```\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
