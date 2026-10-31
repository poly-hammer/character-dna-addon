"""A new scene or DNA merge consistently presents and selects LOD0."""

from types import SimpleNamespace

import bpy
import pytest

from test_merge_split import character as character_fixture

from character_dna.editors.converter import scene_merge
from character_dna.ui import callbacks


character = character_fixture
pytestmark = pytest.mark.rig_profile(compact=True)


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
@pytest.mark.parametrize("mode", ["SCENE", "DNA"])
def test_new_merge_resets_lod_and_selection(character, mode):
    instance = character
    instance.converter_merge_mode = mode
    instance.view_options.active_lod = "lod5"
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    assert instance.view_options.active_lod == "lod0"
    surfaces = scene_merge.surfaces(instance)
    assert set(surfaces) == set(range(8))
    assert bpy.context.view_layer.objects.active == surfaces[0]
    assert surfaces[0].select_get()
    assert all(proxy.hide_get() == (lod != 0) for lod, proxy in surfaces.items())
    assert all(proxy.hide_render == (lod != 0) for lod, proxy in surfaces.items())
    # Accessory meshes follow the same LOD; the previous eyes/teeth cannot remain visible.
    for index, obj in instance.head_mesh_index_lookup.items():
        name = instance.head_dna_reader.getMeshName(index)
        if not name.startswith("head_lod") and name.endswith("_lod5_mesh"):
            assert obj.hide_get()
    instance.view_options.active_lod = "lod6"
    assert not surfaces[6].hide_get()
    assert all(proxy.hide_get() for lod, proxy in surfaces.items() if lod != 6)
    if mode == "SCENE":
        assert bpy.ops.character_dna.split_components() == {"FINISHED"}
        assert instance.view_options.active_lod == "lod6"
        for lod in range(8):
            obj = bpy.data.objects[f"{instance.name}_head_lod{lod}_mesh"]
            assert obj.hide_get() == (lod != 6)
        assert not bpy.data.objects[f"{instance.name}_body_lod3_mesh"].hide_get()


def test_merged_lod_enum_uses_registry_and_preserves_sparse_indices(monkeypatch):
    registry = {"0": object(), "5": object()}
    instance = SimpleNamespace(name="MergedLodEnumProbe", get=lambda _key, _default=None: registry)
    monkeypatch.setattr(callbacks, "get_active_rig_instance", lambda: instance)
    first = callbacks.get_head_mesh_lod_items(None, None)
    assert [(item[0], item[4]) for item in first] == [("lod0", 0), ("lod5", 5)]
    assert callbacks.get_head_mesh_lod_items(None, None) is first
    # A registry update can change available LODs without changing the object count.
    registry["6"] = registry.pop("5")
    updated = callbacks.get_head_mesh_lod_items(None, None)
    assert [(item[0], item[4]) for item in updated] == [("lod0", 0), ("lod6", 6)]


def test_merged_visibility_ignores_sources_outside_view_layer():
    surface = bpy.data.objects.new("MergedVisibilityProbe", None)
    source = bpy.data.objects.new("MergedWorkingSourceProbe", None)
    bpy.context.scene.collection.objects.link(surface)
    source["dna_working_source"] = True
    surface["dna_merge_head"] = source
    instance = SimpleNamespace(
        view_options=SimpleNamespace(active_lod="lod0"),
        get=lambda _key: {"0": surface},
    )
    try:
        assert source.name not in bpy.context.view_layer.objects
        scene_merge.sync_visibility(instance)
        assert not surface.hide_get()
        assert source.hide_render
    finally:
        bpy.data.objects.remove(surface, do_unlink=True)
        bpy.data.objects.remove(source)
