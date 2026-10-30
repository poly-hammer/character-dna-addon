"""Acceptance must reject a fast run that silently loses existing checks."""

import json
import os
import subprocess
import venv

import pytest

from coverage import CoverageData
from scripts.test_suite_refactor.compare_runs import (
    SOURCE,
    canonical_node,
    compare_cases,
    compare_coverage,
    covered_path,
)
from scripts.test_suite_refactor.run_benchmarks import cleanup_temporary, python_executable


def report(nodes):
    return {"items": list(nodes), "outcomes": nodes}


def test_benchmark_interpreter_preserves_virtual_environment(tmp_path):
    environment = tmp_path / "isolated-python"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    completed = subprocess.run(  # noqa: S603 - newly created local Python, fixed probe, no shell
        [python_executable(str(executable)), "-c", "import json, sys; print(json.dumps(sys.prefix))"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert environment.samefile(json.loads(completed.stdout))


def test_benchmark_cleanup_refuses_unowned_directory(tmp_path):
    output = tmp_path / "measurement"
    output.mkdir()
    sibling = tmp_path / "unrelated"
    sibling.mkdir()
    sentinel = sibling / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="unexpected temporary"):
        cleanup_temporary(sibling, output, "1-candidate")
    with pytest.raises(ValueError, match="unexpected temporary"):
        cleanup_temporary(output, output, "1-candidate")
    owned = output / "1-candidate-temporary"
    owned.mkdir()
    (owned / "generated.txt").write_text("temporary")
    assert cleanup_temporary(owned, output, "1-candidate") >= 0
    assert not owned.exists()
    assert sentinel.read_text() == "keep"


def test_case_comparison_detects_missing_skipped_failed_and_unfinished_cases():
    old = {
        f"old.py::test_{name}": {"setup": "passed", "call": "passed", "teardown": "passed"}
        for name in ("retained", "missing", "skip", "fail", "unfinished")
    }
    new = {name.replace("old.py", "new.py"): dict(value) for name, value in old.items() if "missing" not in name}
    new["new.py::test_skip"]["call"] = "skipped"
    new["new.py::test_fail"]["call"] = "failed"
    del new["new.py::test_unfinished"]["teardown"]
    result = compare_cases(report(old), report(new), {"tests": {"old.py": "new.py"}})
    assert result["missing"] == ["new.py::test_missing"]
    assert result["new_skips"] == ["new.py::test_skip"]
    assert result["regressions"] == ["new.py::test_fail"]
    assert result["unfinished"] == ["new.py::test_unfinished"]


def test_known_failures_stay_visible_without_becoming_new_regressions():
    phases = {"setup": "passed", "call": "failed", "teardown": "passed"}
    old = report({"old.py::test_known": phases})
    migration = {"tests": {"old.py": "new.py"}}
    unchanged = compare_cases(old, report({"new.py::test_known": phases}), migration)
    assert not unchanged["regressions"] and not unchanged["fixed"]
    fixed = compare_cases(old, report({"new.py::test_known": dict(phases, call="passed")}), migration)
    assert fixed["fixed"] == ["new.py::test_known"]


def test_cancelled_baseline_cannot_serve_as_complete_comparison():
    old = report({"old.py::test_incomplete": {"setup": "passed"}})
    new = report({"new.py::test_incomplete": {"setup": "passed", "call": "passed", "teardown": "passed"}})
    result = compare_cases(old, new, {"tests": {"old.py": "new.py"}})
    assert result["baseline_unfinished"] == ["new.py::test_incomplete"]
    assert not result["unfinished"]


def test_existing_assertion_failure_cannot_hide_a_new_fixture_failure():
    old = report({"case": {"setup": "passed", "call": "failed", "teardown": "passed"}})
    new = report({"case": {"setup": "failed", "teardown": "passed"}})
    result = compare_cases(old, new, {"tests": {}})
    assert result["regressions"] == ["case"]


def test_inserted_statement_mapping_does_not_hide_a_missing_conditional_branch():
    executed = {(2, 3), (3, 4), (4, 5), (5, -1)}
    assert covered_path((2, 4), executed, {2, 4, 5})
    assert not covered_path((2, 5), executed, {2, 4, 5})
    assert not covered_path((2, 6), executed, {2, 4, 6})


def test_live_dna_guards_and_naming_matrices_keep_distinct_destinations():
    migration = {"tests": {"names.py": {"matrix": "unit/names.py", "live_dna": "blender/names.py"}}}
    assert canonical_node("names.py::test_names[left]", migration) == "unit/names.py::test_names[left]"
    assert canonical_node("names.py::test_matches_live_dna", migration) == "blender/names.py::test_matches_live_dna"


def test_coverage_comparison_detects_lost_branch_after_line_shift(tmp_path):
    roots = [tmp_path / name for name in ("before", "after")]
    source = "def choose(value):\n    if value:\n        return 1\n    return 0\n"
    data_paths = []
    for index, root in enumerate(roots):
        folder = root / SOURCE
        folder.mkdir(parents=True)
        (folder / "naming.py").write_text("")
        module = folder / "domain.py"
        module.write_text(source if index == 0 else "# moved down\n" + source)
        data_path = tmp_path / f".coverage.{index}"
        data = CoverageData(basename=str(data_path))
        arcs = {(-1, 2), (2, 3), (3, -1), (2, 4), (4, -1)}
        if index:
            arcs = {(-2, 3), (3, 4), (4, -2)}
        data.add_arcs({str(module): arcs})
        data.write()
        data_paths.append(data_path)
    result = compare_coverage(*data_paths, *roots, {"tests": {}, "functions": {}})
    assert result["missing_lines"] == [["domain.py", 4, "domain.py", 5]]
    assert len(result["missing_arcs"]) == 2
    assert not result["unmapped_lines"] and not result["unmapped_arcs"]


def test_deferred_import_mapping_requires_the_import_to_execute(tmp_path):
    roots = [tmp_path / name for name in ("before", "after")]
    sources = [
        "# module\nfrom helpers import lookup\nVALUE = 1\n",
        "# module\nVALUE = 1\ndef resolve():\n    from helpers import lookup\n",
    ]
    paths = []
    for index, root in enumerate(roots):
        folder = root / SOURCE
        folder.mkdir(parents=True)
        (folder / "naming.py").write_text("")
        module = folder / "domain.py"
        module.write_text(sources[index])
        path = tmp_path / f".coverage.{index}"
        data = CoverageData(basename=str(path))
        # Isolate the original import -> constant sequencing edge. The new
        # import executes only within resolve(), outside module sequencing.
        data.add_arcs({str(module): {(2, 3)} if index == 0 else {(2, 3), (3, -1), (-3, 4), (4, -3)}})
        data.write()
        paths.append(path)
    migration = {
        "tests": {},
        "functions": {},
        "declarations": [{"before": "domain.py", "after": "domain.py", "name": "VALUE"}],
        "imports": [
            {
                "before": ["domain.py", "from helpers import lookup"],
                "after": ["domain.py", "from helpers import lookup"],
            }
        ],
    }
    result = compare_coverage(*paths, *roots, migration)
    assert result["structural_declaration_arcs"] == [["domain.py", 2, 3]]
    assert not result["missing_lines"] and not result["missing_arcs"]

    data = CoverageData(basename=str(paths[1]))
    data.erase()
    data.add_arcs({str(roots[1] / SOURCE / "domain.py"): {(2, 3), (3, -1)}})
    data.write()
    result = compare_coverage(*paths, *roots, migration)
    assert result["missing_lines"] == [["domain.py", 2, "domain.py", 4]]
    assert result["missing_arcs"]
