"""Experimental fixture substitution for selected unchanged regression tests.

Only opt in with -p snapshot_plugin. Canonical per-session DNA paths are a pilot
limitation; tests asserting per-test path ownership must not use this prototype.
"""

import inspect
import pathlib
import shutil
import time

from contextlib import suppress

import pytest


CACHES = {}
EVENTS = []


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest) -> object:
    source = pathlib.Path(inspect.getfile(fixturedef.func)).name
    if fixturedef.argname != "character" or source not in {"test_merge_split.py", "test_editor_runtime_lifecycle.py"}:
        return None
    assert request.node.originalname in {
        "test_failed_editor_entry_restores_runtime",
        "test_scene_merge_noop_preserves_sources_and_shape_keys",
        "test_scene_split_rejects_shape_ownership_collision_without_mutation",
    }, "Snapshot prototype is validated only for the selected benchmark cases"
    import bpy

    from character_dna.utilities import get_active_rig_instance
    from constants import TEST_DNA_FOLDER

    start = time.perf_counter()
    key = (source, getattr(request, "param", None))
    cold = key not in CACHES
    if cold:
        directory = request.getfixturevalue("tmp_path_factory").mktemp("snapshot")
        kwargs = {
            arg: (directory if arg == "tmp_path" else request.getfixturevalue(arg)) for arg in fixturedef.argnames
        }
        generator = fixturedef.func(**kwargs)
        instance = next(generator)
        snapshot = directory / "pristine.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(snapshot), check_existing=False)
        CACHES[key] = (directory, snapshot)

        def teardown():
            with suppress(StopIteration):
                next(generator)
    else:
        directory, snapshot = CACHES[key]
        bpy.ops.wm.read_homefile(use_empty=True)
        for name in ("head.dna", "body.dna", "ExportManifest.json"):
            shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, directory / name)
        bpy.ops.wm.open_mainfile(filepath=str(snapshot))
        instance = get_active_rig_instance()
        # Blend reload restores portable bindings but intentionally does not keep
        # Python authoring readers alive. Recreate them before scene-merge tests.
        from character_dna.runtime import controller

        with controller.preserve_bindings():
            instance.initialize()

        def teardown():
            bpy.ops.wm.read_homefile(use_empty=True)

    request.addfinalizer(teardown)
    from character_dna.runtime import engine

    assert instance.head_dna_reader is not None and instance.body_dna_reader is not None
    assert engine.binding_issues(instance) == []
    assert engine.active(instance)
    EVENTS.append(
        {"nodeid": request.node.nodeid, "profile": str(key), "cold": cold, "seconds": time.perf_counter() - start}
    )
    fixturedef.cached_result = (instance, fixturedef.cache_key(request), None)
    return instance


def pytest_sessionfinish() -> None:
    import json
    import os

    path = pathlib.Path(os.environ["PILOT_REPORT"]).with_suffix(".snapshots.json")
    path.write_text(json.dumps(EVENTS, indent=2))
