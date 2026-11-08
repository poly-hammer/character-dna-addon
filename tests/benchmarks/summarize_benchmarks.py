"""Gate complete matched repetitions and retain detailed coverage/case differences."""

import argparse
import json
import os
import statistics

from pathlib import Path

from compare_runs import compare_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=3)
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text())
    groups = {}
    for run in runs:
        group = groups.setdefault(run["repeat"], {})
        if run["variant"] in group:
            raise ValueError("Duplicate variant in a repetition")
        group[run["variant"]] = run
    summaries = []
    for repeat, pair in sorted(groups.items()):
        if set(pair) != {"baseline", "candidate"}:
            summaries.append({"repeat": repeat, "accepted": False, "error": "Incomplete pair"})
            continue
        before, after = pair["baseline"], pair["candidate"]
        report_paths = [Path(run["report"]) for run in (before, after)]
        if any(not path.is_file() for path in report_paths):
            summaries.append({"repeat": repeat, "accepted": False, "error": "Missing completion report"})
            continue
        coverage_paths = [path.parent / f".coverage.{path.stem}" for path in report_paths]
        comparison = compare_reports(
            *(json.loads(path.read_text()) for path in report_paths),
            Path(before["root"]),
            Path(after["root"]),
            *coverage_paths,
        )
        ratio = after["process_seconds"] / before["process_seconds"]
        comparison["process_ratio"] = ratio
        comparison["source_unchanged"] = all(run["source_unchanged"] for run in pair.values())
        comparison["cache_policy_matches"] = (
            before.get("cache_policy") == after.get("cache_policy") and before.get("cache_policy") is not None
        )
        comparison["accepted"] &= (
            ratio <= 0.25
            and comparison["source_unchanged"]
            and comparison["cache_policy_matches"]
            and all(run["exit_code"] in (0, 1) for run in pair.values())
        )
        output = args.runs.parent / f"comparison-{repeat}.json"
        output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        summaries.append(
            {
                "repeat": repeat,
                "accepted": comparison["accepted"],
                "baseline_seconds": before["process_seconds"],
                "candidate_seconds": after["process_seconds"],
                "ratio": ratio,
                "environment": json.loads(report_paths[1].read_text())["environment"],
                "baseline_source": before["source_sha256"],
                "candidate_source": after["source_sha256"],
                "cache_policy": after["cache_policy"],
                "details": output.name,
            }
        )
    summary = {
        "pairs": summaries,
        "accepted": len(summaries) >= args.minimum_pairs and all(pair["accepted"] for pair in summaries),
    }
    for variant in ("baseline", "candidate"):
        values = [pair[f"{variant}_seconds"] for pair in summaries if "ratio" in pair]
        if values:
            summary[variant] = {
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
    if "baseline" in summary and "candidate" in summary:
        summary["median_ratio"] = summary["candidate"]["median_seconds"] / summary["baseline"]["median_seconds"]
    output = json.dumps(summary, indent=2)
    (args.runs.parent / "summary.json").write_text(output, encoding="utf-8")
    print(output)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(f"### Complete suite comparison\n\n```json\n{output}\n```\n")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
