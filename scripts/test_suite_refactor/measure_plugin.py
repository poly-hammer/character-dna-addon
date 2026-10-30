"""Opt-in pilot instrumentation; never imported by the normal suite."""

import json
import os
import pathlib
import platform
import sys
import time

from collections.abc import Iterator

import pytest


START = time.perf_counter()
DATA = {"items": [], "fixtures": [], "reports": []}
OUT = pathlib.Path(os.environ.get("PILOT_REPORT", "reports/test-suite-refactor/collection.json"))


def flush():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(DATA, indent=2), encoding="utf-8")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        fixtures = []
        for name in item.fixturenames:
            defs = item._fixtureinfo.name2fixturedefs.get(name)  # noqa: SLF001 - opt-in fixture instrumentation
            if defs:
                d = defs[-1]
                fixtures.append(
                    {
                        "name": name,
                        "scope": d.scope,
                        "file": d.func.__code__.co_filename,
                        "line": d.func.__code__.co_firstlineno,
                    }
                )
        DATA["items"].append({"nodeid": item.nodeid, "fixtures": fixtures})
    DATA["collection_seconds"] = time.perf_counter() - START
    flush()


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest) -> Iterator[None]:
    start = time.perf_counter()
    yield
    DATA["fixtures"].append(
        {
            "name": fixturedef.argname,
            "nodeid": request.node.nodeid,
            "scope": fixturedef.scope,
            "seconds": time.perf_counter() - start,
        }
    )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    DATA["reports"].append(
        {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome, "seconds": report.duration}
    )
    if report.when == "teardown":
        flush()


def pytest_sessionfinish(exitstatus: int) -> None:
    DATA["exitstatus"] = int(exitstatus)
    DATA["wall_seconds"] = time.perf_counter() - START
    DATA["environment"] = {"python": sys.version, "platform": platform.platform()}
    if "bpy" in sys.modules:
        DATA["environment"]["blender"] = sys.modules["bpy"].app.version_string
    if "coverage" in sys.modules:
        DATA["environment"]["coverage"] = sys.modules["coverage"].__version__
    flush()
