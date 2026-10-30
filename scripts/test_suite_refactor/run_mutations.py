"""Prove lifecycle contracts detect omitted resume, surface rollback and cleanup."""

import json
import os
import subprocess
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT / "reports/test-suite-refactor/mutations"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for fault, expected_failures in (("resume", 12), ("rollback", 6), ("abort", 12)):
        report = folder / f"{fault}.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/blender/test_editor_transition_contracts.py",
            "-p",
            "mutation_plugin",
            f"--lifecycle-fault={fault}",
            "-o",
            "addopts=",
            "-o",
            "log_cli=false",
            "-q",
            f"--test-run-report={report}",
        ]
        env = dict(os.environ, RUNNING_CI="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(Path(__file__).parent))
        with (folder / f"{fault}.log").open("w", encoding="utf-8") as output:
            run = subprocess.run(  # noqa: S603 - explicit local mutation test invocation, no shell
                command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=False
            )
        data = json.loads(report.read_text())
        calls = [phases.get("call") for phases in data["outcomes"].values()]
        result = {
            "fault": fault,
            "exit_code": run.returncode,
            "cases": len(calls),
            "detected": calls.count("failed"),
            "expected": expected_failures,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        if run.returncode != 1 or len(calls) != 12 or calls.count("failed") != expected_failures:
            raise RuntimeError(f"Unexpected mutation result: {result}")
    (folder / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
