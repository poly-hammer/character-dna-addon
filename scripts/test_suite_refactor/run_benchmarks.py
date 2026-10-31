"""Measure complete, sequential baseline/candidate runs with the same interpreter."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def python_executable(value: str) -> str | None:
    """Keep the virtual environment entry point, even when it is a symlink."""
    executable = Path(value)
    return str(executable.absolute()) if executable.is_file() else shutil.which(value)


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for folder in (root / "src/addons/character_dna", root / "tests"):
        sources = []
        for directory, children, files in os.walk(folder):
            # Prune generated solver environments before walking them; rejecting
            # their files after rglob still traverses thousands of dependencies.
            children[:] = [
                name for name in children if name not in {"__pycache__", "temp", "test_files", "extra_test_files"}
            ]
            sources.extend(Path(directory) / name for name in files if name.endswith(".py"))
        for path in sorted(sources):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    for path in sorted((root / "tests/test_files").rglob("*")):
        if path.is_file() and path.suffix in {".dna", ".json", ".fbx", ".blend"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cleanup_temporary(temporary: Path, output: Path, label: str) -> float:
    """Remove one runner-created directory after validating absolute ownership."""
    start = time.perf_counter()
    temporary = temporary.resolve()
    if temporary.parent != output.resolve() or temporary.name != label + "-temporary":
        raise ValueError(f"Refusing to remove an unexpected temporary directory: {temporary}")
    shutil.rmtree(temporary)
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path, help="Isolated checkout of the original suite; omit for fresh-import comparison"
    )
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--repeat-start", type=int, default=1, help="First repetition number for distributed CI pairs")
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--case", action="append", help="Optional focused group; omit for complete suite")
    parser.add_argument("--variant", choices=("both", "baseline", "candidate"), default="both")
    parser.add_argument("--first", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument(
        "--cache-policy",
        choices=("cold", "shared"),
        default="cold",
        help="Cold isolates temporary solver environments and pip caches for every run",
    )
    parser.add_argument("--no-coverage", action="store_true", help="Diagnostic runs only, not acceptance measurements")
    parser.add_argument("--keep-temporary", action="store_true", help="Retain cold-run files for failure investigation")
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat_start < 1:
        parser.error("--repeat and --repeat-start must be positive")
    args.python = python_executable(args.python)
    if args.python is None:
        parser.error("Python executable was not found")
    output = ROOT / "reports/test-suite-refactor" / args.label
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for repeat in range(args.repeat_start - 1, args.repeat_start - 1 + args.repeat):
        variants = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
        if args.first == "candidate":
            variants.reverse()
        if args.variant != "both":
            variants = [args.variant]
        for variant in variants:
            root = (args.baseline_root or args.candidate_root) if variant == "baseline" else args.candidate_root
            root = root.resolve()
            # A few original diagnostics write into reports without creating it.
            # Normal CI's HTML reporter created it before those tests ran.
            (root / "reports").mkdir(exist_ok=True)
            label = f"{repeat + 1}-{variant}"
            report = output / f"{label}.json"
            env = dict(
                os.environ,
                RUNNING_CI="1",
                PYTHONUNBUFFERED="1",
                PYTHONPATH=str(ROOT / "tests"),
                COVERAGE_FILE=str(output / f".coverage.{label}"),
            )
            if args.cache_policy == "cold":
                temporary = output / (label + "-temporary")
                temporary.mkdir()
                env.update(
                    TMP=str(temporary),
                    TEMP=str(temporary),
                    TMPDIR=str(temporary),
                    PIP_CACHE_DIR=str(temporary / "pip-cache"),
                )
            command = [
                args.python,
                "-m",
                "pytest",
                *(args.case or ["tests"]),
                "-p",
                "performance",
                "-o",
                "addopts=",
                "-o",
                "log_cli=false",
                "-o",
                "tmp_path_retention_policy=failed",
                "--tb=short",
                "-vv",
                f"--test-run-report={report}",
                f"--junitxml={output / (label + '.xml')}",
            ]
            if variant == "baseline" and args.baseline_root is None:
                command.append("--fresh-rigs")
            if not args.no_coverage:
                command.extend(
                    [
                        "--cov-branch",
                        "--cov=src/addons/character_dna",
                        "--cov-config=pyproject.toml",
                        f"--cov-report=xml:{output / ('coverage-' + label + '.xml')}",
                    ]
                )
            print(f"Starting {label} in {root}", flush=True)
            fingerprint = source_fingerprint(root)
            with (output / (label + ".log")).open("w", encoding="utf-8") as log:
                start = time.perf_counter()
                completed = subprocess.run(  # noqa: S603 - explicit local test invocation, no shell
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
                process_seconds = time.perf_counter() - start
            record = {
                "variant": variant,
                "repeat": repeat + 1,
                "root": str(root),
                "source_sha256": fingerprint,
                "source_unchanged": fingerprint == source_fingerprint(root),
                "exit_code": completed.returncode,
                "process_seconds": process_seconds,
                "report": str(report),
                "command": command,
                "cache_policy": args.cache_policy,
            }
            runs.append(record)
            # Preserve completed process evidence even if a locked temporary
            # file prevents cleanup on Windows.
            (output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
            record["cleanup_seconds"] = (
                cleanup_temporary(temporary, output, label)
                if args.cache_policy == "cold" and not args.keep_temporary
                else 0.0
            )
            (output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
            print(f"Finished {label}: {record['process_seconds']:.2f}s; exit {completed.returncode}", flush=True)
            if not record["source_unchanged"]:
                raise RuntimeError("Source changed during measurement; discard the timing and repeat")
    return int(any(record["exit_code"] for record in runs))


if __name__ == "__main__":
    raise SystemExit(main())
