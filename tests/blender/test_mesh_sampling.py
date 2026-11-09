"""Read-only canonical samples must agree with the converter's eventual Split."""

import bpy
import numpy as np
import pytest

from character_dna.editors.converter.scene import merge
from test_merge_split import character as character_fixture, coordinates


character = character_fixture


def mapped_vertex(proxy, kind, index):
    return next(i for i, v in enumerate(proxy.data.attributes[f"DNA_{kind}_vertex"].data) if v.value == index)


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
@pytest.mark.rig_profile(compact=True)
def test_sampling_combines_shared_lod_edits_and_matches_split(character):
    instance = character
    component = instance.component("body")
    before = component.sample_mesh(0)
    merge.merge(instance)
    surfaces = merge.surfaces(instance)
    body = surfaces[0]["dna_merge_body"]
    assert body == surfaces[1]["dna_merge_body"]
    original = coordinates(body.data.vertices)
    first = mapped_vertex(surfaces[0], "body", 0)
    second = mapped_vertex(surfaces[1], "body", 1)
    surfaces[0].data.shape_keys.reference_key.data[first].co.x += 0.02
    surfaces[1].data.shape_keys.reference_key.data[second].co.y += 0.03
    for obj in surfaces.values():
        obj.data.update()
    ids = (set(bpy.data.objects.keys()), set(bpy.data.meshes.keys()))
    actual = component.sample_mesh(0)
    expected = before.positions.copy()
    expected[0, 0] += 0.02
    expected[1, 1] += 0.03
    np.testing.assert_allclose(actual.positions, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(coordinates(body.data.vertices), original)
    assert (set(bpy.data.objects.keys()), set(bpy.data.meshes.keys())) == ids
    merge.split(instance)
    np.testing.assert_allclose(component.sample_mesh(0).positions, actual.positions, rtol=0, atol=1e-6)


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
@pytest.mark.rig_profile(compact=True)
def test_sampling_rejects_conflicting_shared_lod_edits(character):
    instance = character
    component = instance.component("body")
    merge.merge(instance)
    surfaces = merge.surfaces(instance)
    for lod, distance in [(0, 0.02), (1, 0.04)]:
        proxy = surfaces[lod]
        proxy.data.shape_keys.reference_key.data[mapped_vertex(proxy, "body", 0)].co.x += distance
        proxy.data.update()
    with pytest.raises(ValueError, match="Conflicting edits"):
        component.sample_mesh(0)
    with pytest.raises(ValueError, match="Conflicting edits"):
        merge.prepare_mesh_edits(instance)


@pytest.mark.rig_profile(compact=True)
def test_sampling_new_merged_shape_matches_split_and_preserves_original_keys(character):
    instance = character
    component = instance.component("body")
    original = component.sample_mesh(0)
    source = component.mesh
    keys = tuple(source.data.shape_keys.key_blocks.keys()) if source.data.shape_keys else ()
    proxy = merge.merge(instance)
    key = proxy.shape_key_add(name="body_lod0_mesh__landmark_sample", from_mix=False)
    index = mapped_vertex(proxy, "body", 0)
    key.data[index].co.x += 0.02
    key.value = 0.5
    proxy.data.update()
    sample = component.sample_mesh(0)
    expected = original.positions.copy()
    expected[0, 0] += 0.01
    np.testing.assert_allclose(sample.positions, expected, rtol=0, atol=1e-6)
    assert (tuple(source.data.shape_keys.key_blocks.keys()) if source.data.shape_keys else ()) == keys
    merge.split(instance)
    np.testing.assert_allclose(component.sample_mesh(0).positions, sample.positions, rtol=0, atol=1e-6)


@pytest.mark.rig_profile(compact=True)
def test_sampling_rejects_topology_modifiers(character):
    component = character.component("body")
    sample = component.sample_mesh(0)
    assert sample.position_ids == (0, 1, 2)
    assert not sample.positions.flags.writeable
    modifier = component.mesh.modifiers.new("Changed topology", "SUBSURF")
    try:
        with pytest.raises(ValueError, match="DNA has|changed DNA face topology"):
            component.sample_mesh(0)
    finally:
        component.mesh.modifiers.remove(modifier)
    np.testing.assert_array_equal(component.sample_mesh(0).positions, sample.positions)


@pytest.mark.rig_profile(compact=True)
def test_sampling_rejects_unflushed_merged_edit_mode(character):
    component = character.component("body")
    proxy = merge.merge(character)
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        with pytest.raises(ValueError, match="Object Mode"):
            component.sample_mesh(0)
        assert proxy.mode == "EDIT"
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
