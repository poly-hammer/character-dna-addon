"""Shared configuration stays importable in Python without Blender."""

import sys

import pytest


pytest_plugins = ["performance"]


def pytest_addoption(parser):
    parser.addoption("--require-no-blender", action="store_true", help="Fail if the Python-only run imports bpy")


def pytest_sessionfinish(session):
    if session.config.getoption("--require-no-blender") and "bpy" in sys.modules:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        session.config.get_terminal_writer().line("Python-only test run imported bpy", red=True)


def pytest_collection_modifyitems(items):
    for item in items:
        layer = "unit" if "/unit/" in str(item.path).replace("\\", "/") else "blender"
        item.add_marker(getattr(pytest.mark, layer))
