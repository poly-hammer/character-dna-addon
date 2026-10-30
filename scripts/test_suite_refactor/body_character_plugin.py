"""Use a complete real body rig for body-only RBF assertions."""

import inspect
import os
import shutil

from pathlib import Path

import pytest


NAMES = {
    "test_create_delete_and_revert_keys",
    "test_new_shape_commit_then_delete_revert",
    "test_apply_shape_value_and_mirror_membership",
}
SNAPSHOT = None


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest) -> object:
    global SNAPSHOT
    if fixturedef.argname != "character" or Path(inspect.getfile(fixturedef.func)).name != "test_merge_split.py":
        return None
    assert request.node.originalname in NAMES and not hasattr(request, "param")
    import bpy

    from character_dna.utilities import get_active_rig_instance
    from constants import TEST_DNA_FOLDER

    directory = request.getfixturevalue("tmp_path")
    bpy.ops.wm.read_homefile(use_empty=True)
    request.addfinalizer(lambda: bpy.ops.wm.read_homefile(use_empty=True))
    if os.environ.get("PILOT_BODY_SNAPSHOT") == "1" and SNAPSHOT:
        directory, path = SNAPSHOT
        shutil.copyfile(TEST_DNA_FOLDER / "ada/body.dna", directory / "body.dna")
        bpy.ops.wm.open_mainfile(filepath=str(path))
        instance = get_active_rig_instance()
        from character_dna.runtime import controller, engine

        with controller.preserve_bindings():
            instance.initialize()
        assert instance.body_dna_reader is not None and engine.binding_issues(instance) == []
        assert engine.active(instance)
        fixturedef.cached_result = (instance, fixturedef.cache_key(request), None)
        return instance
    for name in ("body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, directory / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(directory / "body.dna"),
        include_body=False,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{i}": i == 0 for i in range(8)},
    )
    instance = get_active_rig_instance()
    instance.output.folder_path = str(directory / "output")
    assert instance.body_rig is not None and instance.head_rig is None
    if os.environ.get("PILOT_BODY_SNAPSHOT") == "1":
        path = directory / "body-pristine.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
        SNAPSHOT = directory, path
    fixturedef.cached_result = (instance, fixturedef.cache_key(request), None)
    return instance
