"""Reproduce the planning pilots sequentially; never changes the normal test suite."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOTS = Path(__file__).resolve().parent
REPORTS = ROOT / "reports/test-suite-refactor"
CASES = {
    "rna": (
        "minimal_rna_plugin",
        [
            "tests/test_output_component_selection.py::test_legacy_output_selection_and_session_locks",
            "tests/test_output_component_selection.py::test_component_switch_refreshes_output_in_both_editions",
        ],
    ),
    "snapshots": (
        "snapshot_plugin",
        [
            "tests/test_editor_runtime_lifecycle.py::test_failed_editor_entry_restores_runtime",
            "tests/test_merge_split.py::test_scene_merge_noop_preserves_sources_and_shape_keys",
            "tests/test_merge_split.py::test_scene_split_rejects_shape_ownership_collision_without_mutation",
        ],
    ),
    "rbf": (
        "body_character_plugin",
        [
            "tests/rbf_editor/test_shape_driven.py::test_create_delete_and_revert_keys",
            "tests/rbf_editor/test_shape_driven.py::test_new_shape_commit_then_delete_revert",
            "tests/rbf_editor/test_shape_driven.py::test_apply_shape_value_and_mirror_membership",
        ],
    ),
    "native": (
        "slim_reader_plugin",
        [
            "tests/test_component_editing.py::test_rbf_noop_preserves_behavior",
            "tests/test_component_editing.py::test_removing_solver_preserves_surviving_pose_routes",
        ],
    ),
}
CASES["rbf-cached"] = CASES["rbf"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[*CASES, "contracts", "all"], default="rna")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if (ROOT / "tests/blender").is_dir():
        parser.error(
            "These historical substitutions require the pre-refactor checkout. "
            "Use run_benchmarks.py for the implemented fixtures; "
            "do not label the refactored suite as the old baseline."
        )
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    REPORTS.mkdir(parents=True, exist_ok=True)
    records = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    groups = [*CASES, "contracts"] if args.group == "all" else [args.group]
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("Git executable was not found")
    revision = subprocess.check_output(  # noqa: S603 - local Git, fixed read-only arguments
        [git, "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    for group in groups:
        for repeat in range(args.repeat):
            variants = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
            if group == "contracts":
                variants = ["contracts", "missing-resume-mutant"]
            for variant in variants:
                label = f"{stamp}-{group}-{repeat + 1}-{variant}"
                env = os.environ.copy()
                env.update(
                    RUNNING_CI="1",
                    PYTHONUNBUFFERED="1",
                    PYTHONPATH=str(PILOTS),
                    PILOT_REPORT=str(REPORTS / f"{label}.json"),
                    COVERAGE_FILE=str(REPORTS / f".coverage.{label}"),
                )
                env["PILOT_BODY_SNAPSHOT"] = "1" if group == "rbf-cached" and variant == "candidate" else "0"
                if group == "contracts":
                    command = [
                        sys.executable,
                        str(PILOTS / "run_pilot_pytest.py"),
                        str(PILOTS / "test_lifecycle_contract_pilot.py"),
                    ]
                    if variant == "contracts":
                        command += [str(PILOTS / "test_reader_layers_pilot.py")]
                    else:
                        command += ["-p", "mutation_plugin"]
                else:
                    plugin, cases = CASES[group]
                    command = [sys.executable, "-m", "pytest", *cases]
                    if variant == "candidate":
                        command += ["-p", plugin]
                command += [
                    "-q",
                    "-o",
                    "addopts=",
                    "-o",
                    "log_cli=false",
                    "-p",
                    "measure_plugin",
                    "--cov-branch",
                    "--cov-config=pyproject.toml",
                    "--cov=src/addons/character_dna",
                    f"--cov-report=xml:{REPORTS / f'coverage-{label}.xml'}",
                ]
                print(f"Running {label}", flush=True)
                start = time.perf_counter()
                with (REPORTS / f"{label}.log").open("w", encoding="utf-8") as output:
                    run = subprocess.run(  # noqa: S603 - explicit local test invocation, no shell
                        command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=False
                    )
                expected = 1 if variant == "missing-resume-mutant" else 0
                records.append(
                    {
                        "label": label,
                        "commit": revision,
                        "group": group,
                        "variant": variant,
                        "process_seconds": time.perf_counter() - start,
                        "exit_code": run.returncode,
                        "expected_exit_code": expected,
                        "command": command,
                    }
                )
                (REPORTS / f"{stamp}-runs.json").write_text(json.dumps(records, indent=2))
                print(f"Finished in {records[-1]['process_seconds']:.2f}s; exit {run.returncode}", flush=True)
                if run.returncode != expected:
                    return 1
                if variant == "missing-resume-mutant":
                    result = json.loads((REPORTS / f"{label}.json").read_text())
                    calls = [r for r in result["reports"] if r["when"] == "call"]
                    assert len(calls) == 12 and all(r["outcome"] == "failed" for r in calls), (
                        "Mutant was not caught by all twelve contracts"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
