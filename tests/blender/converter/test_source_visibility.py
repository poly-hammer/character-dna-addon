"""Only consumed inputs are hidden, and only after successful conversion."""

from types import SimpleNamespace

import bpy
import pytest

from character_dna import utilities
from character_dna.editors.converter.lifecycle import ConversionLifecycle
from character_dna.editors.converter.operators import CHARACTER_DNA_OT_convert_to_dna


@pytest.mark.parametrize("layout", ["SPLIT", "JOINED"])
def test_source_collection_includes_materials_without_validating_their_transforms(layout):
    head, body, joined, material, unused = [object() for _ in range(5)]
    props = SimpleNamespace(
        input_layout=layout,
        joined_mesh=joined,
        extra_meshes=[
            SimpleNamespace(role="head", fit_method="WRAP", scene_object=head, material_source=material),
            SimpleNamespace(role="body", fit_method="WRAP", scene_object=body, material_source=material),
            SimpleNamespace(role="teeth", fit_method="RELATIVE", scene_object=unused, material_source=head),
        ],
    )
    collect = CHARACTER_DNA_OT_convert_to_dna._iter_input_meshes
    targets = {head, body} if layout == "SPLIT" else {joined}
    assert set(collect(props)) == targets
    sources = collect(props, include_materials=True)
    assert set(sources) == targets | {head, material}
    assert len(sources) == len(set(sources))
    assert unused not in sources


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_cleanup_hides_sources_only_on_success(outcome):
    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("Source visibility")
    bpy.context.window.scene = scene
    mesh = bpy.data.meshes.new("Source visibility mesh")
    objects = [bpy.data.objects.new(name, mesh) for name in ("Wrap", "Material source", "Unrelated")]
    for obj in objects:
        scene.collection.objects.link(obj)
    target, material, unrelated = objects
    try:
        target.hide_render = True
        material.hide_set(True)
        target.select_set(True)
        unrelated.select_set(True)
        bpy.context.view_layer.objects.active = target
        visibility = [(obj, obj.hide_get(), obj.hide_render) for obj in (target, material)]
        snapshot = {
            "visibility": visibility,
            "selected": [target, unrelated],
            "active": target,
            "evaluate": utilities.get_addon_window_manager_properties().evaluate_dependency_graph,
            "instance_index": scene.character_dna.rig_instance_list_active_index,
        }
        operator = SimpleNamespace(
            _snapshot=snapshot,
            _pipeline=None,
            _rig_instance=None,
            _reporter=None,
            _properties=SimpleNamespace(new_name="Uncreated conversion"),
            _owned={name: set() for name in ("objects", "meshes", "armatures", "collections", "materials", "images")},
            _remove_timer=lambda _context: None,
        )
        target.hide_set(True)  # Temporary hiding while fitting is in progress.
        if outcome == "cancel":
            operator._restore = lambda context, rollback: ConversionLifecycle._restore(operator, context, rollback)
            ConversionLifecycle.cancel(operator, bpy.context)
        else:
            ConversionLifecycle._restore(operator, bpy.context, rollback=outcome == "failure")
        for obj, hidden, render in visibility:
            assert obj.hide_get() == (True if outcome == "success" else hidden)
            assert obj.hide_render == (True if outcome == "success" else render)
        assert not unrelated.hide_get() and not unrelated.hide_render
        assert unrelated.select_get()
        assert bpy.context.view_layer.objects.active == (None if outcome == "success" else target)
        assert operator._snapshot is None
    finally:
        bpy.context.window.scene = previous
        bpy.data.scenes.remove(scene)
        for obj in objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
