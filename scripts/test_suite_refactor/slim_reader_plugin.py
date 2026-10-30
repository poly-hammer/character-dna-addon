"""Opt-in smaller fixtures for behavior-only tests; never export geometry with these."""

import inspect

from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest) -> object:
    if (
        fixturedef.argname != "component_reader"
        or Path(inspect.getfile(fixturedef.func)).name != "test_component_editing.py"
    ):
        return None
    assert request.node.originalname in {
        "test_rbf_noop_preserves_behavior",
        "test_removing_solver_preserves_surviving_pose_routes",
    }
    from test_reader_layers_pilot import ROOT, read

    from character_dna.dna_io import release_dna_handle

    path = next((ROOT / "tests/test_files/dna/default").rglob(f"{request.param}.dna"))
    reader = read(path, True)
    request.addfinalizer(lambda: release_dna_handle(reader))
    value = (request.param, reader)
    fixturedef.cached_result = (value, fixturedef.cache_key(request), None)
    return value
