"""Rebuilding source geometry must retain authored shape-key animation."""

import bpy
import pytest

from character_dna.editors.converter.source_animation import copy_animation
from character_dna.utilities.reference import remap_id_users


@pytest.fixture
def keys():
    bpy.ops.wm.read_homefile(use_empty=True)
    result = []
    for name in ("Original", "Rebuilt"):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(0, 0, 0)], [], [])
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.shape_key_add(name="Basis")
        obj.shape_key_add(name="head_lod0_mesh__smile")
        result.append(mesh.shape_keys)
    try:
        yield result
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def _animate(owner):
    block = owner.key_blocks[-1]
    for frame, value in ((1, 0.2), (10, 0.8)):
        block.value = value
        block.keyframe_insert("value", frame=frame)
    return owner.animation_data


def test_copy_action_slot_and_native_driver(keys):
    source, target = keys
    animation = _animate(source)
    animation.action_influence = 0.35
    curve = source.key_blocks[-1].driver_add("value")
    curve.driver.type = "SCRIPTED"
    variable = curve.driver.variables.new()
    variable.name, variable.type = "weight", "SINGLE_PROP"
    variable.targets[0].id_type = "KEY"
    variable.targets[0].id = source
    variable.targets[0].data_path = 'key_blocks["Basis"].value'
    curve.driver.expression = "weight * 0.7"
    curve.mute = True
    curve.modifiers.new("NOISE").strength = 0.3

    copy_animation(source, target)

    copied = target.animation_data
    assert copied.action == animation.action
    assert copied.action_slot == animation.action_slot
    assert copied.action_influence == pytest.approx(0.35)
    driver = copied.drivers[0]
    assert driver != curve
    assert driver.data_path == curve.data_path
    assert driver.mute
    assert driver.driver.expression == "weight * 0.7"
    assert driver.driver.variables[0].targets[0].id == source
    assert driver.modifiers[-1].strength == pytest.approx(0.3)
    original_mesh = source.user
    rebuilt_mesh = target.user
    remap_id_users(source, target)
    assert driver.driver.variables[0].targets[0].id == target
    assert original_mesh.shape_keys == source
    bpy.data.meshes.remove(original_mesh, do_unlink=True)
    assert rebuilt_mesh.shape_keys == target


def test_copy_nla_timing_and_animated_influence(keys):
    source, target = keys
    animation = _animate(source)
    track = animation.nla_tracks.new()
    track.name = "Authored shapes"
    strip = track.strips.new("Smile", 20, animation.action)
    strip.action_slot = animation.action_slot
    strip.scale = 1.5
    strip.repeat = 2
    strip.use_reverse = True
    strip.use_animated_influence = True
    for frame, value in ((20, 0.1), (40, 0.9)):
        strip.influence = value
        strip.keyframe_insert("influence", frame=frame)
    curve = strip.fcurves.find("influence")
    curve.modifiers.new("NOISE").strength = 0.05
    track.lock, track.mute = True, True

    copy_animation(source, target)

    copied = target.animation_data.nla_tracks[0]
    assert copied.name == track.name
    assert copied.lock and copied.mute
    result = copied.strips[0]
    assert result.action == strip.action
    assert result.action_slot == strip.action_slot
    for name in ("frame_start", "frame_end", "scale", "repeat"):
        assert getattr(result, name) == pytest.approx(getattr(strip, name))
    assert result.use_reverse
    assert result.use_animated_influence
    for frame in (20, 25, 35, 40):
        assert result.fcurves.find("influence").evaluate(frame) == pytest.approx(curve.evaluate(frame))
