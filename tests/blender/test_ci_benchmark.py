"""The retained CI benchmark reports native measurements and compares like workloads."""

import ast
import importlib
import json

from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("profiling_utils.exporters")


def _snapshot(path: Path, schema: int, value: float) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"benchmark_schema": schema, "hardware": {}},
                "summary": {"native_head_ms": value, "native_body_ms": value, "full_evaluation_ms": value * 3},
            }
        ),
        encoding="utf-8",
    )


def test_native_benchmark_detects_regressions(benchmark, tmp_path):
    baseline, current = tmp_path / "baseline.json", tmp_path / "current.json"
    _snapshot(baseline, 2, 1.0)
    _snapshot(current, 2, 1.5)
    result = benchmark.compare_snapshots(baseline, current, threshold_pct=10)
    assert result["has_regressions"]
    assert {item["name"] for item in result["regressions"]} == {
        "native_head_ms",
        "native_body_ms",
        "full_evaluation_ms",
    }


def test_native_benchmark_rejects_obsolete_workload(benchmark, tmp_path):
    baseline, current = tmp_path / "baseline.json", tmp_path / "current.json"
    _snapshot(baseline, 1, 1.0)
    _snapshot(current, 2, 1.0)
    with pytest.raises(ValueError, match="workloads differ"):
        benchmark.compare_snapshots(baseline, current)


def test_ci_profiling_package_has_no_local_probe_dependencies(benchmark):
    folder = Path(benchmark.__file__).parent
    required = {
        "__init__.py",
        "ci_benchmark.py",
        "profile_rig_evaluation.py",
        "exporters.py",
    }
    assert required <= {path.name for path in folder.iterdir() if path.is_file()}
    # Local profiling tools may live beside the CI package. What matters is
    # whether CI imports them, including through a transitive relative import.
    pending = [folder / name for name in required]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        assert path.name in required, f"CI depends on local probe: {path.name}"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                local = folder / (name.split(".")[-1] + ".py")
                if local.is_file():
                    pending.append(local)
