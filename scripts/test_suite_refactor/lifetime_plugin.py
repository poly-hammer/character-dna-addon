"""Opt-in repeated lease canaries with state counts and resident-memory evidence."""

import gc
import hashlib
import inspect
import json
import logging
import sys

from pathlib import Path
from statistics import linear_regression, median
from typing import Any

import pytest


ALLOWED = {
    "test_rig_leases_restore_pristine_state_and_own_paths",
    "test_failed_edit_does_not_leave_suspended_runtime_or_editor_cache",
}
SAMPLES = pytest.StashKey[list[dict[str, Any]]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--lease-repetitions", type=int, default=10)
    parser.addoption("--lease-evidence", required=True)


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("lease_repetitions") < 8:
        raise pytest.UsageError("Use at least eight repetitions to distinguish warm-up from repeated ownership.")
    output = Path(config.getoption("lease_evidence"))
    if output.exists() or output.with_suffix(".jsonl").exists():
        raise pytest.UsageError("Choose a new lease evidence path; previous results are preserved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    config.stash[SAMPLES] = []


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if {item.name for item in items} != ALLOWED or len(items) != 2:
        raise pytest.UsageError("Select exactly the two existing lease canaries from test_rig_fixtures.py.")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    import bpy

    from character_dna.runtime import engine

    reporting = pyfuncitem.config.pluginmanager.get_plugin("performance")
    assert reporting is not None
    output = Path(pyfuncitem.config.getoption("lease_evidence"))
    arguments = {name: pyfuncitem.funcargs[name] for name in inspect.signature(pyfuncitem.obj).parameters}
    folder = arguments["tmp_path"]
    assert isinstance(folder, Path)
    source = hashlib.sha256(inspect.getsource(pyfuncitem.obj).encode()).hexdigest()
    baseline = None
    for index in range(pyfuncitem.config.getoption("lease_repetitions")):
        arguments["tmp_path"] = folder / str(index)
        pyfuncitem.obj(**arguments)
        gc.collect()
        state: dict[str, Any] = {
            name: len(getattr(bpy.data, name)) for name in ("objects", "meshes", "armatures", "shape_keys")
        }
        state["handlers"] = {
            name: [(callback.__module__, callback.__qualname__) for callback in getattr(bpy.app.handlers, name)]
            for name in ("load_pre", "load_post", "undo_pre", "undo_post", "frame_change_post", "depsgraph_update_post")
        }
        if baseline is None:
            baseline = state
        assert state == baseline, "Blender IDs or handler registrations accumulated across released leases"
        tracked = gc.get_objects()
        captured_logs = [value for value in tracked if isinstance(value, logging.LogRecord)]
        allocations = {
            "allocated_blocks": sys.getallocatedblocks(),
            "gc_objects": len(tracked),
            "gc_garbage": len(gc.garbage),
            "log_records": len(captured_logs),
            "log_tracebacks": sum(value.exc_info is not None for value in captured_logs),
            "sample_log_messages": [value.getMessage()[:500] for value in captured_logs[-2:]],
            "binding_wrappers": sum(".bindings" in str(getattr(type(value), "__module__", "")) for value in tracked),
            "native_owner_counts": {
                name: len(getattr(engine, name)) for name in ("_records", "_models", "_model_files", "_undo_models")
            },
        }
        del tracked, captured_logs
        sample = {
            "nodeid": pyfuncitem.nodeid,
            "iteration": index,
            "source_sha256": source,
            "state": state,
            "allocations": allocations,
            **reporting.memory_usage(),
        }
        pyfuncitem.config.stash[SAMPLES].append(sample)
        with output.with_suffix(".jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample) + "\n")
    return True


def pytest_sessionfinish(session: pytest.Session) -> None:
    samples = session.config.stash[SAMPLES]
    summary = {}
    for nodeid in dict.fromkeys(sample["nodeid"] for sample in samples):
        current = [sample["rss_bytes"] for sample in samples if sample["nodeid"] == nodeid and "rss_bytes" in sample]
        warm = current[2:]
        summary[nodeid] = {
            "rss_samples": current,
            "warmup_repetitions": 2,
            "warm_rss_range_bytes": max(warm) - min(warm) if warm else None,
            "warm_rss_slope_bytes_per_repetition": linear_regression(range(len(warm)), warm).slope
            if len(warm) > 1
            else None,
            "last_minus_first_window_bytes": median(warm[-3:]) - median(warm[:3]) if len(warm) >= 6 else None,
        }
    Path(session.config.getoption("lease_evidence")).write_text(
        json.dumps({"exitstatus": int(session.exitstatus), "samples": samples, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
