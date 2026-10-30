"""Saved Blender 4.5 target metadata must follow Blender 5's converted drivers."""

import hashlib

import bpy
import pytest

from character_dna.runtime import bindings, controller, engine
from character_dna.utilities import get_active_rig_instance, migration
from fixtures.rigs import RigProfile


pytestmark = pytest.mark.skipif(
    "hide" not in bpy.types.PoseBone.bl_rna.properties, reason="Pose visibility was introduced in Blender 5"
)


def _switch_carrier():
    return next(c for c in engine.carriers(get_active_rig_instance()) if c["component"] == "switches")


def _metadata(carrier):
    return [(t["owner"].id_type, t["owner"].name, t["path"], t["index"], t["channel"]) for t in carrier["targets"]]


def _assert_visibility(names):
    face = get_active_rig_instance().face_board
    for frame, hidden in ((1, True), (2, False), (3, True), (2, False)):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = face.evaluated_get(bpy.context.evaluated_depsgraph_get())
        assert all(evaluated.pose.bones[name].hide == hidden for name in names)


@pytest.fixture(scope="module")
def saved_visibility_rig(addon, tmp_path_factory, rig_factory):
    folder = tmp_path_factory.mktemp("eye_visibility")
    profile = RigProfile(
        components=("head",), meshes=False, face_board=True, pose_bones=("FACIAL_L_Eye", "FACIAL_R_Eye")
    )
    with rig_factory.lease(profile, folder):
        instance = get_active_rig_instance()
        face = instance.face_board
        carrier = _switch_carrier()
        names = []
        for target in carrier["targets"]:
            if target["path"].endswith(".hide"):
                bone = face.path_resolve(target["path"].removesuffix(".hide"))
                names.append(bone.name)
                # Blender moves the real FCurve but leaves the addon's ID-property
                # descriptor untouched. Reproduce that state without an old binary.
                target["owner"] = face.data
                target["path"] = bone.bone.path_from_id("hide")
        assert len(names) == 7
        switch = face.pose.bones["CTRL_lookAtSwitch"]
        for frame, value in ((1, 0.0), (2, 1.0), (3, 0.5)):
            switch.location.y = value
            switch.keyframe_insert("location", index=1, frame=frame)
        path = folder / "converted.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        return path, names, instance.name


@pytest.fixture
def converted_rig(saved_visibility_rig, monkeypatch):
    monkeypatch.setattr(bpy.context.preferences.filepaths, "use_scripts_auto_execute", False)
    bpy.ops.wm.open_mainfile(filepath=str(saved_visibility_rig[0]), use_scripts=False)
    yield saved_visibility_rig
    bpy.ops.wm.read_homefile(app_template="")


def test_converted_visibility_evaluates_and_survives_reopen(converted_rig, tmp_path):
    _source, names, _name = converted_rig
    carrier = _switch_carrier()
    before = _metadata(carrier)
    assert not engine.carrier_issues(carrier)
    assert not migration.detect_runtime_migration(bpy.context.scene)
    _assert_visibility(names)
    saved = tmp_path / "reopened.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    bpy.ops.wm.open_mainfile(filepath=str(saved), use_scripts=False)
    assert not engine.carrier_issues(_switch_carrier())
    assert _metadata(_switch_carrier()) == before
    _assert_visibility(names)
    assert not bpy.app.autoexec_fail


def test_converted_visibility_mute_remove_preserve_edit_mode_driver(converted_rig):
    _source, names, _name = converted_rig
    carrier = _switch_carrier()
    face = carrier["face"]
    pose_path = face.pose.bones[names[0]].path_from_id("hide")
    edit_path = face.data.bones[names[0]].path_from_id("hide")
    user_driver = face.data.driver_add(edit_path)
    user_driver.driver.expression = "0.0"
    bindings.mute_targets(carrier, True)
    assert face.animation_data.drivers.find(pose_path).mute
    assert not user_driver.mute
    bindings.mute_targets(carrier, False)
    assert not face.animation_data.drivers.find(pose_path).mute
    _assert_visibility(names)
    bindings.remove_targets(carrier)
    assert all(not face.animation_data.drivers.find(face.pose.bones[name].path_from_id("hide")) for name in names)
    assert face.data.animation_data.drivers.find(edit_path).driver.expression == "0.0"


def test_converted_visibility_rebuild_replaces_owned_drivers(converted_rig):
    _source, names, _name = converted_rig
    instance = get_active_rig_instance()
    controller.rebuild(instance)
    assert not engine.binding_issues(instance)
    assert all(t["owner"] == instance.face_board for t in _switch_carrier()["targets"])
    _assert_visibility(names)


@pytest.mark.parametrize("problem", ["replacement", "missing"])
def test_converted_visibility_does_not_mask_conflicts(converted_rig, problem):
    _source, names, _name = converted_rig
    carrier = _switch_carrier()
    face = carrier["face"]
    path = face.pose.bones[names[0]].path_from_id("hide")
    if problem == "replacement":
        face.animation_data.drivers.find(path).driver.expression = "0.0"
        with pytest.raises(ValueError, match="conflicting driver"):
            migration.preflight_runtime_migration([get_active_rig_instance()])
    else:
        face.driver_remove(path)
    assert any(path in issue for issue in engine.carrier_issues(carrier))
    with pytest.raises(ValueError, match="missing or conflicting output driver"):
        engine.adopt(get_active_rig_instance())
    bindings.mute_targets(carrier, True)
    bindings.remove_targets(carrier)
    if problem == "replacement":
        curve = face.animation_data.drivers.find(path)
        assert curve.driver.expression == "0.0"
        assert not curve.mute


@pytest.mark.parametrize("mode", ["APPEND", "LINK", "EDITABLE_LINK"])
def test_converted_visibility_reference_transport(converted_rig, mode, tmp_path):
    """Exercise the standalone descriptor reader as well as runtime hydration."""
    source, names, name = converted_rig
    fingerprint = hashlib.sha256(source.read_bytes()).digest()
    bpy.ops.wm.read_homefile(app_template="")
    assert bpy.ops.character_dna.append_or_link_metahuman(
        filepath=str(source),
        operation_type="APPEND" if mode == "APPEND" else "LINK",
        editable_rig=mode == "EDITABLE_LINK",
        meta_human_names=name,
    ) == {"FINISHED"}
    before = _metadata(_switch_carrier())
    for reopen in (False, True):
        if reopen:
            saved = tmp_path / f"{mode}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(saved))
            bpy.ops.wm.open_mainfile(filepath=str(saved), use_scripts=False)
        assert not engine.carrier_issues(_switch_carrier())
        assert _metadata(_switch_carrier()) == before
        _assert_visibility(names)
    assert hashlib.sha256(source.read_bytes()).digest() == fingerprint
