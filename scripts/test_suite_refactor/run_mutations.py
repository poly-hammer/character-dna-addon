"""Verify unmodified tests pass and detect targeted faults in production behavior."""

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SWING = "tests/blender/test_swing_twist_editor.py::"
TRANSITIONS = "tests/blender/test_editor_transition_contracts.py"
CASES = {
    "resume": (
        [TRANSITIONS, "tests/blender/test_editor_runtime_lifecycle.py::test_failed_editor_entry_restores_runtime"],
        16,
        16,
    ),
    "rollback": (
        [
            TRANSITIONS,
            "tests/blender/test_editor_runtime_lifecycle.py::test_failed_merged_entry_restores_surface_and_removes_snapshots",
        ],
        13,
        7,
    ),
    "abort": (
        [
            TRANSITIONS,
            "tests/blender/test_editor_runtime_lifecycle.py::test_raw_editor_failed_entry_restores_bone_locks",
        ],
        14,
        14,
    ),
    "destination": (
        ["tests/blender/test_component_editing.py::test_component_selection_routes_and_locks_sessions"],
        1,
        1,
    ),
    "axis": ([SWING + "test_calf_back_90_preview_matches_complete_rbf_pose"], 1, 1),
    "units": ([SWING + "test_calf_back_90_preview_matches_complete_rbf_pose"], 1, 1),
    "rbf": ([SWING + "test_calf_back_90_preview_matches_complete_rbf_pose"], 1, 1),
    "shape": ([SWING + "test_preview_updates_rbf_shape_keys_and_restores_values"], 2, 2),
    "visibility": ([SWING + "test_collection_visibility_is_revealed_then_restored"], 4, 4),
    "publication": (
        [
            "tests/blender/test_document_transactions.py::test_native_document_write_publishes_once",
            SWING + "test_combined_commit_preserves_head_and_surface",
        ],
        3,
        3,
    ),
    "reader": (
        [
            "tests/blender/test_document_transactions.py::test_document_read_rechecks_file_revision",
            "tests/blender/test_document_transactions.py::test_native_document_read_refreshes_after_publication",
        ],
        2,
        2,
    ),
}


def run(folder: Path, label: str, nodes: list[str], fault: str | None = None) -> dict:
    report = folder / f"{label}.json"
    evidence = folder / f"{label}-evidence.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *nodes,
        "-p",
        "mutation_plugin",
        "-o",
        "addopts=",
        "-o",
        "log_cli=false",
        "--tb=short",
        "-q",
        f"--test-run-report={report}",
        f"--mutation-evidence={evidence}",
    ]
    if fault:
        command.append(f"--contract-fault={fault}")
    env = dict(os.environ, RUNNING_CI="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(Path(__file__).parent))
    with (folder / f"{label}.log").open("w", encoding="utf-8") as output:
        process = subprocess.run(  # noqa: S603 - explicit local diagnostic, no shell or source-file edits
            command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    data = json.loads(report.read_text(encoding="utf-8"))
    outcomes = data["outcomes"]
    result = {
        "fault": fault,
        "exit_code": process.returncode,
        "cases": len(outcomes),
        "detected": sum(phases.get("call") == "failed" for phases in outcomes.values()),
        "wall_seconds": data["wall_seconds"],
        "environment": data["environment"],
        "evidence": json.loads(evidence.read_text(encoding="utf-8")),
    }
    assert all(p.get("setup") == p.get("teardown") == "passed" for p in outcomes.values()), result
    assert all(p.get("call") in {"passed", "failed"} for p in outcomes.values()), result
    if fault:
        _, expected_cases, expected_failures = CASES[fault]
        assert result["cases"] == expected_cases and result["detected"] == expected_failures, result
        assert process.returncode == 1, result
        assert len(result["evidence"]) == expected_cases, result
        assert all(record["calls"] > 0 for record in result["evidence"]), result
        expected_exception = "EditorError" if fault == "destination" else "AssertionError"
        failures = [record for record in result["evidence"] if "exception" in record]
        assert len(failures) == expected_failures, result
        assert all(record["exception"] == expected_exception for record in failures), result
    else:
        assert process.returncode == 0 and result["detected"] == 0, result
    print(
        json.dumps({key: value for key, value in result.items() if key not in {"environment", "evidence"}}), flush=True
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=tuple(CASES), action="append")
    parser.add_argument(
        "--output", type=Path, required=True, help="New evidence directory; existing results are never replaced"
    )
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    faults = args.fault or list(CASES)
    nodes = list(dict.fromkeys(node for fault in faults for node in CASES[fault][0]))
    results = [run(folder, "control", nodes)]
    for fault in faults:
        results.append(run(folder, fault, CASES[fault][0], fault))
        (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
