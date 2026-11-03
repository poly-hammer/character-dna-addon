"""Replacement meshes must own the current scene's DNA shape bindings."""

from types import SimpleNamespace

import bpy
import pytest


@pytest.mark.parametrize("kind", ["head", "body"])
def test_replacement_main_mesh_wins_over_retained_template(minimal_character, kind):
    instance = minimal_character
    mesh = getattr(instance, f"{kind}_mesh")
    rig = getattr(instance, f"{kind}_rig")
    template = mesh.copy()
    template["ca_dna_mesh_index"] = 0
    template["ca_dna_component"] = kind
    template["ca_dna_rig"] = rig
    reader = SimpleNamespace(getMeshCount=lambda: 1, getMeshName=lambda _: f"{kind}_lod0_mesh")
    component = instance.component(kind)
    assert component.meshes_for_reader(reader) == {0: mesh}
    bpy.context.scene.collection.objects.link(template)
    assert component.meshes_for_reader(reader) == {0: mesh}


@pytest.mark.parametrize("kind", ["head", "body"])
def test_secondary_mesh_binding_ignores_unlinked_ids_and_keeps_renamed_tags(minimal_character, kind):
    instance = minimal_character
    mesh = getattr(instance, f"{kind}_mesh")
    rig = getattr(instance, f"{kind}_rig")
    secondary = mesh.copy()
    secondary.name = f"{instance.name}_secondary_lod0_mesh"
    secondary["ca_dna_mesh_index"] = 1
    secondary["ca_dna_component"] = kind
    secondary["ca_dna_rig"] = rig
    reader = SimpleNamespace(
        getMeshCount=lambda: 2,
        getMeshName=lambda index: f"{kind}_lod0_mesh" if index == 0 else "secondary_lod0_mesh",
    )
    component = instance.component(kind)
    assert component.meshes_for_reader(reader) == {0: mesh}
    bpy.context.scene.collection.objects.link(secondary)
    secondary.name = "ArtistRenamedSecondaryMesh"
    assert component.meshes_for_reader(reader) == {0: mesh, 1: secondary}
    bpy.context.scene.collection.objects.unlink(secondary)
    assert component.meshes_for_reader(reader) == {0: mesh}
