"""Opt-in, append-only timings; cancellation retains every completed phase."""

import importlib.metadata
import json
import os
import platform
import sys
import time

from pathlib import Path

import pytest


def memory_usage():
    """Process bytes, without a platform-specific package or subprocess."""
    if sys.platform == "win32":
        import ctypes

        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_rss",
                    "rss",
                    "peak_paged_pool",
                    "paged_pool",
                    "peak_nonpaged_pool",
                    "nonpaged_pool",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if function(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"rss_bytes": counters.rss, "peak_rss_bytes": counters.peak_rss}
        return {}
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}


def pytest_addoption(parser):
    parser.addoption("--test-run-report", help="JSON report path (phase records use the adjacent .jsonl file)")
    parser.addoption("--fresh-rigs", action="store_true", help="Import each rig lease for matched fixture benchmarks")
    parser.addoption("--test-order-seed", type=int, help="Deterministically shuffle tests to check fixture isolation")
    parser.addoption("--test-profile", help="Optional cProfile output; diagnostic runs only")


def pytest_configure(config):
    path = config.getoption("--test-run-report")
    if path:
        config.pluginmanager.register(RunReport(Path(path)), "test-run-report")
    profile_path = config.getoption("--test-profile")
    if profile_path:
        config.pluginmanager.register(ProfileReport(Path(profile_path)), "test-profile")


class ProfileReport:
    def __init__(self, path):
        import cProfile

        self.path = path
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def pytest_sessionfinish(self):
        # Blender's forced process exit bypasses cProfile's normal atexit path.
        self.profiler.disable()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.dump_stats(str(self.path))


def pytest_collection_modifyitems(config, items):
    seed = config.getoption("--test-order-seed")
    if seed is not None:
        import random

        random.Random(seed).shuffle(items)  # noqa: S311 - reproducible test order, not security


class RunReport:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.with_suffix(".jsonl").open("w", encoding="utf-8", buffering=1)
        self.start = time.perf_counter()
        self.items = []
        self.outcomes = {}
        self.totals = {"setup": 0.0, "call": 0.0, "teardown": 0.0}
        self.processes = 0
        self.rig_profiles = {}
        sys.addaudithook(self.audit)

    def audit(self, event, _args):
        if event == "subprocess.Popen":
            self.processes += 1

    def pytest_fixture_post_finalizer(self, fixturedef, request):
        if fixturedef.argname == "rig_factory" and fixturedef.cached_result:
            factory = fixturedef.cached_result[0]
            self.rig_profiles = {"builds": factory.build_count, "profiles": len(factory.artifacts)}

    def emit(self, event):
        self.stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def pytest_collection_modifyitems(self, items):
        for item in items:
            fixtures = []
            for name in item.fixturenames:
                definitions = item._fixtureinfo.name2fixturedefs.get(name)
                if definitions:
                    fixture = definitions[-1]
                    fixtures.append({"name": name, "scope": fixture.scope, "source": fixture.func.__code__.co_filename})
            self.items.append(item.nodeid)
            self.emit({"event": "collection", "nodeid": item.nodeid, "fixtures": fixtures})

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        start = time.perf_counter()
        yield
        self.emit(
            {
                "event": "fixture",
                "name": fixturedef.argname,
                "scope": fixturedef.scope,
                "nodeid": request.node.nodeid,
                "seconds": time.perf_counter() - start,
            }
        )

    def pytest_runtest_logreport(self, report):
        self.totals[report.when] += report.duration
        self.outcomes.setdefault(report.nodeid, {})[report.when] = report.outcome
        self.emit(
            {
                "event": "phase",
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "seconds": report.duration,
            }
        )
        if report.when == "teardown":
            self.emit({"event": "memory", "nodeid": report.nodeid, **memory_usage()})

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "ci_corpus": bool(os.getenv("RUNNING_CI")),
            "cpu_count": os.cpu_count(),
            "processor": platform.processor(),
            "pytest": pytest.__version__,
        }
        try:
            environment["coverage"] = importlib.metadata.version("coverage")
        except importlib.metadata.PackageNotFoundError:
            environment["coverage"] = None
        if "bpy" in sys.modules:
            environment["blender"] = sys.modules["bpy"].app.version_string
        data = {
            "schema": 1,
            "environment": environment,
            "exitstatus": int(session.exitstatus),
            "wall_seconds": time.perf_counter() - self.start,
            "phase_seconds": self.totals,
            "items": self.items,
            "outcomes": self.outcomes,
            "fresh_rigs": session.config.getoption("--fresh-rigs"),
            "child_processes": self.processes,
            "rig_profiles": self.rig_profiles,
            "memory": memory_usage(),
            "cpu_seconds": {"user": os.times().user, "system": os.times().system},
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.stream.close()
