"""Unreal Control Rig reference poses and real RigLogic eye-aim integration."""

from __future__ import annotations

import json

from pathlib import Path

import bpy
import pytest

from mathutils import Matrix, Quaternion, Vector

from character_dna.runtime import authoring, controller, engine, eyes
from character_dna.utilities import get_active_rig_instance


REFERENCE = json.loads((Path(__file__).parent / "test_files/json/eye_convergence_unreal.json").read_text())
NAMES = {
    "CTRL_C_eyesAim": eyes.CENTER_NAME,
    "CTRL_L_eyeAim_space": "GRP_L_eyeAim",
    "CTRL_R_eyeAim_space": "GRP_R_eyeAim",
    "CTRL_L_eyeAim": "CTRL_L_eyeAim",
    "CTRL_R_eyeAim": "CTRL_R_eyeAim",
}


def _matrix(transform):
    x, y, z, w = transform["rotation"]
    basis = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    return (
        basis
        @ Matrix.LocRotScale(Vector(transform["location"]) * 0.01, Quaternion((w, x, y, z)), Vector(transform["scale"]))
        @ basis
    )


def _update(face):
    face.update_tag()
    bpy.context.view_layer.update()
    return face.evaluated_get(bpy.context.evaluated_depsgraph_get())


@pytest.fixture
def reference_face():
    """Use the Unreal hierarchy's measured rest transforms, independent of DNA fitting."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    armature = bpy.data.armatures.new("ConvergenceReference")
    face = bpy.data.objects.new("ConvergenceReference", armature)
    bpy.context.collection.objects.link(face)
    bpy.context.view_layer.objects.active = face
    face.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for source, name in NAMES.items():
        bone = armature.edit_bones.new(name)
        bone.length = 0.01
        bone.matrix = _matrix(REFERENCE["initial"][source])
        if "_space" in source:
            bone.parent = armature.edit_bones[eyes.CENTER_NAME]
        elif source != eyes.CENTER_NAME:
            bone.parent = armature.edit_bones[f"GRP_{source.split('_')[1]}_eyeAim"]
    switch = armature.edit_bones.new(eyes.SWITCH_NAME)
    switch.length = 0.01
    bpy.ops.object.mode_set(mode="OBJECT")
    limit = face.pose.bones[eyes.SWITCH_NAME].constraints.new("LIMIT_LOCATION")
    limit.use_min_x = limit.use_max_x = True
    limit.max_x = 0.02
    limit.owner_space = "LOCAL"
    eyes.install(face)
    yield face
    bpy.data.objects.remove(face, do_unlink=True)
    bpy.data.armatures.remove(armature)


@pytest.mark.parametrize("case", ["neutral", "moved_center", "eye_offsets"])
def test_convergence_matches_unreal(reference_face, case):
    face = reference_face
    samples = [s for s in REFERENCE["samples"] if s["case"] == case]
    face.pose.bones[eyes.CENTER_NAME].matrix = _matrix(samples[0]["transforms"][eyes.CENTER_NAME])
    _update(face)
    for side in ("L", "R"):
        name = f"CTRL_{side}_eyeAim"
        face.pose.bones[name].matrix = _matrix(samples[0]["transforms"][name])
    for sample in samples:
        face.pose.bones[eyes.SWITCH_NAME].location.x = sample["value"] * 0.02
        evaluated = _update(face)
        for source, name in NAMES.items():
            actual = evaluated.pose.bones[name].matrix
            expected = _matrix(sample["transforms"][source])
            assert (
                max(abs(a - b) for ar, er in zip(actual, expected, strict=True) for a, b in zip(ar, er, strict=True))
                < 2e-6
            ), name


def test_convergence_is_native_idempotent_and_clamped(reference_face):
    face = reference_face
    eyes.install(face)
    for side in ("L", "R"):
        assert len(face.pose.bones[f"GRP_{side}_eyeAim"].constraints) == 1
    assert len(face.animation_data.drivers) == 2
    assert all(c.driver.is_simple_expression and not c.driver.use_self for c in face.animation_data.drivers)
    for x, expected in [(-0.02, 0.0), (0.01, 0.5), (0.04, 1.0)]:
        face.pose.bones[eyes.SWITCH_NAME].location.x = x
        evaluated = _update(face)
        assert evaluated.pose.bones["GRP_L_eyeAim"].constraints[eyes.CONSTRAINT_NAME].influence == pytest.approx(
            expected
        )


def test_copied_board_rebinds_convergence(reference_face):
    source = reference_face
    copy = source.copy()
    copy.data = source.data.copy()
    copy.animation_data_clear()
    bpy.context.collection.objects.link(copy)
    try:
        eyes.install(copy)
        copy.pose.bones[eyes.SWITCH_NAME].location.x = 0.02
        evaluated = _update(copy)
        for side in ("L", "R"):
            group = evaluated.pose.bones[f"GRP_{side}_eyeAim"]
            assert (group.head - evaluated.pose.bones[eyes.CENTER_NAME].head).length < 1e-6
        assert source.pose.bones[eyes.SWITCH_NAME].location.x == 0.0
    finally:
        data = copy.data
        bpy.data.objects.remove(copy, do_unlink=True)
        bpy.data.armatures.remove(data)


def test_convergence_updates_native_eye_solve_and_animation(load_head_only_dna):
    instance = get_active_rig_instance()
    face = instance.face_board
    switch = face.pose.bones[eyes.SWITCH_NAME]
    face.pose.bones["CTRL_lookAtSwitch"].location.y = 1.0
    for frame, value in [(1, 0.0), (3, 0.02)]:
        switch.location.x = value
        switch.keyframe_insert("location", index=0, frame=frame)
    snapshots = []
    for frame in (1, 2, 3, 1):
        bpy.context.scene.frame_set(frame)
        _update(face)
        record = next(r for r in engine._records.values() if r["component"] == "head")
        context = record["last_context"]
        live = list(engine.native_module().control_snapshot(context["session"])["gui"])
        state = authoring.sample(instance, "head")
        sampled = [state.getGUIControl(i) for i in range(instance.head_dna_reader.getGUIControlCount())]
        assert sampled == pytest.approx(live, abs=2e-6)
        snapshots.append(live)
    assert max(abs(a - b) for a, b in zip(snapshots[0], snapshots[2], strict=True)) > 0.05
    assert snapshots[0] == pytest.approx(snapshots[3], abs=2e-6)
    # With eye aim off the slider moves the targets, but never changes DNA GUI input.
    face.pose.bones["CTRL_lookAtSwitch"].location.y = 0.0
    _update(face)
    state = authoring.sample(instance, "head")
    before = [state.getGUIControl(i) for i in range(instance.head_dna_reader.getGUIControlCount())]
    bpy.context.scene.frame_set(3)
    _update(face)
    state = authoring.sample(instance, "head")
    assert before == pytest.approx([state.getGUIControl(i) for i in range(len(before))], abs=2e-6)
    controller.rebuild(instance)
    assert all(len(face.pose.bones[f"GRP_{side}_eyeAim"].constraints) == 1 for side in ("L", "R"))


def test_current_files_gain_convergence_on_load(load_head_only_dna, tmp_path):
    """A pre-feature native rig gains portable constraints without rig migration."""
    face = get_active_rig_instance().face_board
    for side in ("L", "R"):
        bone = face.pose.bones[f"GRP_{side}_eyeAim"]
        constraint = bone.constraints[eyes.CONSTRAINT_NAME]
        constraint.driver_remove("influence")
        bone.constraints.remove(constraint)
    path = tmp_path / "before_convergence.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path))
    instance = get_active_rig_instance()
    face = instance.face_board
    face.pose.bones[eyes.SWITCH_NAME].location.x = 0.02
    evaluated = _update(face)
    center = evaluated.pose.bones[eyes.CENTER_NAME].head
    assert all((evaluated.pose.bones[f"GRP_{side}_eyeAim"].head - center).length < 1e-6 for side in ("L", "R"))
    assert engine.active(instance)
