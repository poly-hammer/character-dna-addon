"""Keep real RNA callbacks and meshes; omit DNA for selection-only assertions."""

import inspect

from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest) -> object:
    if (
        fixturedef.argname != "character"
        or Path(inspect.getfile(fixturedef.func)).name != "test_output_component_selection.py"
    ):
        return None
    assert request.node.originalname in {
        "test_legacy_output_selection_and_session_locks",
        "test_component_switch_refreshes_output_in_both_editions",
    }
    import bpy

    bpy.ops.wm.read_homefile(use_empty=True)
    request.addfinalizer(lambda: bpy.ops.wm.read_homefile(use_empty=True))
    scene = bpy.context.scene.character_dna
    instance = scene.rig_instance_list.add()
    instance["name"] = "MinimalComponentRig"
    instance["auto_evaluate"] = False
    for kind in ("head", "body"):
        rig = bpy.data.objects.new(f"{instance.name}_{kind}_rig", bpy.data.armatures.new(f"{kind}Rig"))
        bpy.context.scene.collection.objects.link(rig)
        mesh = bpy.data.meshes.new(f"{kind}Mesh")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new(f"{instance.name}_{kind}_lod0_mesh", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.modifiers.new("Rig", "ARMATURE").object = rig
        setattr(instance, f"{kind}_rig", rig)
        setattr(instance, f"{kind}_mesh", obj)
    instance.editor_component = "head"
    assert instance.output.head_item_list and instance.output.body_item_list
    fixturedef.cached_result = (instance, fixturedef.cache_key(request), None)
    return instance
