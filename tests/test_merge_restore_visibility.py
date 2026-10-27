"""DNA conversion backups restore the intended LOD without exposing source meshes."""

import json

import bpy
import pytest

from character_dna.editors.backup_manager import core as backups
from character_dna.editors.converter import scene_merge
from test_dna_operation_backups import configure_backups, entries
from test_merge_split import character as character_fixture


character = character_fixture


def visibility(instance):
    bpy.context.view_layer.update()
    managed = {
        obj.name: obj for kind in ("head", "body") for obj in instance.component(kind).mesh_index_lookup.values()
    }
    sources = {
        obj.name: {
            "hidden": obj.hide_get(),
            "viewport": obj.hide_viewport,
            "render": obj.hide_render,
            "visible": obj.visible_get(),
        }
        for obj in managed.values()
    }
    return {
        "lod": instance.view_options.active_lod,
        "sources": sources,
        "surfaces": {
            lod: {
                "name": obj.name,
                "hidden": obj.hide_get(),
                "viewport": obj.hide_viewport,
                "render": obj.hide_render,
                "visible": obj.visible_get(),
            }
            for lod, obj in scene_merge.surfaces(instance).items()
        },
        "active_object": bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None,
    }


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
def test_dna_merge_pre_post_restore_has_consistent_lod_visibility(character, tmp_path, monkeypatch):
    configure_backups(monkeypatch, tmp_path)
    instance = character
    instance.converter_merge_mode = "DNA"
    instance.view_options.active_lod = "lod5"
    accessory = next(
        obj
        for obj in instance.head_mesh_index_lookup.values()
        if obj.name.endswith("_lod5_mesh") and "_head_" not in obj.name
    )
    accessory.hide_set(True)
    accessory.hide_render = True
    states = {"before": visibility(instance)}
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    states["merged"] = visibility(instance)
    saved = entries(instance)
    pre = saved[backups.BackupType.PRE_DNA_MERGE.value][0]
    post = saved[backups.BackupType.POST_DNA_MERGE.value][0]
    # User changes LOD after merging; restoring a snapshot must recover its LOD.
    instance.view_options.active_lod = "lod7"
    assert backups.restore_backup(instance, pre)
    states["pre"] = visibility(instance)
    assert backups.restore_backup(instance, post)
    states["post"] = visibility(instance)
    (tmp_path / "visibility.json").write_text(json.dumps(states, indent=2))
    assert states["merged"]["lod"] == "lod0"
    assert states["pre"]["lod"] == "lod5"
    assert not states["pre"]["surfaces"]
    assert states["pre"]["sources"] == states["before"]["sources"]
    assert states["pre"]["active_object"] == f"{instance.name}_head_lod5_mesh"
    assert states["post"]["lod"] == "lod0"
    for label in ("merged", "post"):
        assert {lod for lod, state in states[label]["surfaces"].items() if state["visible"]} == {0}
        assert not any(
            state["visible"]
            for name, state in states[label]["sources"].items()
            if name.startswith((f"{instance.name}_head_lod", f"{instance.name}_body_lod"))
        )
        assert states[label]["active_object"] == f"{instance.name}_merged_lod0_mesh"
    assert states["post"]["sources"] == states["merged"]["sources"]
