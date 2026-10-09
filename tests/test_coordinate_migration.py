"""Legacy animation must preserve evaluated skinning, not just FCurve values."""

from pathlib import Path
from types import SimpleNamespace

import bpy
import pytest

from mathutils import Euler, Matrix, Vector

from character_dna.runtime import controller, engine
from character_dna.utilities import migration
from constants import TEST_DNA_FOLDER


@pytest.mark.parametrize(
    ("angle", "translation", "scale", "world_scale", "accepted"),
    [
        (0.001, 0, 1, 1, True),  # Celeste-sized finger-axis drift, amplified in the matrix translation.
        (0, 0.00009, 1, 1, True),
        (0, 0.00011, 1, 1, False),
        (0.002, 0, 1, 1, False),
        (0, 0, 1.0002, 1, False),
        (0, 0.000002, 1, 100, False),  # Armature/scene scale must affect the physical tolerance.
    ],
)
def test_control_motion_tolerance_uses_distance_angle_and_scale(angle, translation, scale, world_scale, accepted):
    head = Vector((1, 0, 1))
    bone = SimpleNamespace(head_local=head, tail_local=head + Vector((0, 0.03, 0)))
    actual = (
        Matrix.Translation(head + Vector((translation, 0, 0)))
        @ Matrix.Rotation(angle, 4, "Z")
        @ Matrix.Scale(scale, 4)
        @ Matrix.Translation(-head)
    )
    errors = migration._control_motion_error(Matrix.Identity(4), actual, bone, Matrix.Scale(world_scale, 3))
    assert (
        all(error <= limit for error, limit in zip(errors, migration._CONTROL_MOTION_LIMITS, strict=True)) == accepted
    )
    if angle == 0.001:
        assert migration._matrix_error(Matrix.Identity(4), actual) > 0.0002
        assert errors[0] < 0.00004
        assert errors[1] == pytest.approx(angle, abs=1e-6)


@pytest.fixture
def legacy_body(addon):
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.character_dna.import_dna(
        filepath=str(TEST_DNA_FOLDER / "ada" / "body.dna"),
        import_mesh=False,
        import_bones=True,
        import_face_board=False,
        include_body=False,
        import_materials=False,
    )
    instance = bpy.context.scene.character_dna.rig_instance_list[0]
    rig = instance.body_rig
    native = {bone.name: bone.matrix_local.copy() for bone in rig.data.bones}
    wm = bpy.context.window_manager.character_dna
    wm.evaluate_dependency_graph = False
    with controller.preserve_bindings():
        engine.discard(instance)
        migration._activate(rig)
        bpy.ops.object.mode_set(mode="EDIT")
        for bone in rig.data.edit_bones:
            bone.matrix = bone.matrix @ migration.BASIS
        bpy.ops.object.mode_set(mode="OBJECT")
        del rig["dna_coordinate_version"]
    yield instance, native
    bpy.ops.wm.read_homefile(use_empty=True)


def matrix_values(matrix):
    return [value for row in matrix for value in row]


def skinning(rig, name):
    evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return matrix_values(evaluated.pose.bones[name].matrix @ rig.data.bones[name].matrix_local.inverted())


def animate(rig, name="upperarm_l", mode="QUATERNION"):
    bone = rig.pose.bones[name]
    bone.rotation_mode = mode
    for frame, angles in ((1, (0, 0, 0)), (4.5, (0.24, -0.38, 0.49)), (9, (-0.41, 0.37, 0.17))):
        if mode == "QUATERNION":
            bone.rotation_quaternion = Euler(angles).to_quaternion()
            bone.keyframe_insert("rotation_quaternion", frame=frame)
        else:
            bone.rotation_euler = angles
            bone.keyframe_insert("rotation_euler", frame=frame)
        bone.location = (angles[0] * 0.1, angles[1] * 0.1, angles[2] * 0.1)
        bone.keyframe_insert("location", frame=frame)
    return rig.animation_data.action


@pytest.mark.parametrize("mode", ["QUATERNION", "XYZ"])
def test_in_place_reimport_preserves_skinning_and_actions(legacy_body, mode):
    instance, native = legacy_body
    rig = instance.body_rig
    original = animate(rig, mode=mode)
    original_handle = migration.slot_handle(rig.animation_data)
    original_points = [
        (curve.data_path, curve.array_index, [tuple(p.co) for p in curve.keyframe_points])
        for bag in migration.channel_containers(original, original_handle)
        for curve in bag.fcurves
    ]
    frames = (1, 2, 4.5, 7, 9)
    expected = {}
    for frame in frames:
        bpy.context.scene.frame_set(int(frame), subframe=frame - int(frame))
        expected[frame] = {name: skinning(rig, name) for name in ("upperarm_l", "lowerarm_l", "hand_l", "index_01_l")}
    bpy.context.scene.frame_set(5, subframe=0.25)
    pointer = rig.as_pointer()
    assert migration.migrate_runtime_data(bpy.context) == ("default", 1, 0)
    assert rig.as_pointer() == pointer
    assert (bpy.context.scene.frame_current, bpy.context.scene.frame_subframe) == (5, 0.25)
    assert not bpy.context.window_manager.character_dna.evaluate_dependency_graph
    assert rig.get("dna_coordinate_version") == 1
    assert not engine.binding_issues(instance)
    for name, matrix in native.items():
        assert matrix_values(rig.data.bones[name].matrix_local) == pytest.approx(matrix_values(matrix), abs=2e-5)
    assert rig.animation_data.action != original
    assert original.use_fake_user
    assert [
        (curve.data_path, curve.array_index, [tuple(p.co) for p in curve.keyframe_points])
        for bag in migration.channel_containers(original, original_handle)
        for curve in bag.fcurves
    ] == original_points
    for frame in frames:
        bpy.context.scene.frame_set(int(frame), subframe=frame - int(frame))
        for name, matrix in expected[frame].items():
            assert skinning(rig, name) == pytest.approx(matrix, abs=4e-5), (frame, name)
    migrated = rig.animation_data.action
    assert migration.migrate_runtime_data(bpy.context) == ("default", 0, 1)
    assert rig.animation_data.action == migrated


@pytest.mark.parametrize("mode", ["QUATERNION", "XYZ"])
def test_nla_and_shared_action_are_preserved(legacy_body, mode):
    instance, _ = legacy_body
    rig = instance.body_rig
    original = animate(rig, mode=mode)
    handle = migration.slot_handle(rig.animation_data)
    rig.animation_data.action = None
    strip = rig.animation_data.nla_tracks.new().strips.new("Authored clip", 20, original)
    migration.set_action(strip, original, handle)
    strip.scale, strip.repeat, strip.blend_type = 1.5, 2, "REPLACE"
    other = rig.copy()
    bpy.context.scene.collection.objects.link(other)
    before = (
        strip.frame_start,
        strip.frame_end,
        strip.scale,
        strip.repeat,
        strip.action_frame_start,
        strip.action_frame_end,
    )
    samples = {}
    for frame in (20, 24, 27, 31):
        bpy.context.scene.frame_set(frame)
        samples[frame] = skinning(rig, "hand_l")
    migration.migrate_runtime_data(bpy.context)
    assert strip.action != original
    assert other.animation_data.nla_tracks[0].strips[0].action == original
    assert before == (
        strip.frame_start,
        strip.frame_end,
        strip.scale,
        strip.repeat,
        strip.action_frame_start,
        strip.action_frame_end,
    )
    for frame, sample in samples.items():
        bpy.context.scene.frame_set(frame)
        assert skinning(rig, "hand_l") == pytest.approx(sample, abs=4e-5)


def test_declined_actions_are_retained_but_not_played(legacy_body):
    instance, _ = legacy_body
    rig = instance.body_rig
    original = animate(rig)
    strip = rig.animation_data.nla_tracks.new().strips.new("Stashed", 20, original)
    migration.migrate_runtime_data(bpy.context, migrate_actions=False)
    assert original.use_fake_user
    assert rig.animation_data.action is None
    assert strip.action == original and strip.mute


def test_control_rig_requires_confirmation_and_integrations(legacy_body):
    instance, _ = legacy_body
    control = bpy.data.objects.new("Controls", bpy.data.armatures.new("Controls"))
    bpy.context.scene.collection.objects.link(control)
    instance.control_rig = control
    original = instance.body_rig.data
    with pytest.raises(ValueError, match="You must remove the control rig"):
        migration.migrate_runtime_data(bpy.context)
    with pytest.raises(ValueError, match="You must remove the control rig"):
        migration.migrate_runtime_data(bpy.context, rebuild_controls=True)
    assert instance.body_rig.data == original


def test_raw_legacy_record_reimports_before_metadata_is_discarded(legacy_body):
    instance, _ = legacy_body
    rig = instance.body_rig
    action = animate(rig)
    scene = bpy.context.scene
    with controller.preserve_bindings():
        scene["meta_human_dna"] = {
            "rig_logic_instance_list": [
                {
                    "instance_name": instance.name,
                    "body_rig": rig,
                    "body_dna_file_path": instance.body_dna_file_path,
                }
            ]
        }
        scene.character_dna.rig_instance_list.clear()
    assert migration.migrate_runtime_data(bpy.context)[1:] == (1, 0)
    assert "meta_human_dna" not in scene
    assert scene.character_dna.rig_instance_list[0].body_rig == rig
    assert rig["dna_coordinate_version"] == 1
    assert rig.animation_data.action != action


def test_failure_restores_original_rig_and_action(legacy_body, monkeypatch):
    instance, _ = legacy_body
    rig = instance.body_rig
    action = animate(rig)
    data = rig.data
    all_actions = set(bpy.data.actions)
    all_armatures = set(bpy.data.armatures)
    monkeypatch.setattr(controller, "rebuild", lambda _: (_ for _ in ()).throw(RuntimeError("rebuild failed")))
    with pytest.raises(RuntimeError, match="rebuild failed"):
        migration.migrate_runtime_data(bpy.context)
    assert rig.data == data and rig.get("dna_coordinate_version", 0) == 0
    assert rig.animation_data.action == action
    assert set(bpy.data.actions) == all_actions
    assert set(bpy.data.armatures) == all_armatures


def test_bone_child_of_attachment_stays_in_place(legacy_body):
    instance, _ = legacy_body
    rig = instance.body_rig
    follower = bpy.data.objects.new("Attachment", None)
    bpy.context.scene.collection.objects.link(follower)
    constraint = follower.constraints.new("CHILD_OF")
    constraint.target, constraint.subtarget = rig, "head"
    constraint.inverse_matrix = rig.data.bones["head"].matrix_local.inverted()
    follower.location = (0.1, 0.3, 1.5)
    bpy.context.view_layer.update()
    expected = matrix_values(follower.matrix_world)
    migration.migrate_runtime_data(bpy.context)
    bpy.context.view_layer.update()
    assert matrix_values(follower.matrix_world) == pytest.approx(expected, abs=3e-5)


@pytest.mark.parametrize("relative", [False, True])
def test_bone_parented_attachment_keeps_animation(legacy_body, relative):
    instance, _ = legacy_body
    rig = instance.body_rig
    animate(rig)
    rig.data.bones["hand_l"].use_relative_parent = relative
    follower = bpy.data.objects.new("Weapon", None)
    bpy.context.scene.collection.objects.link(follower)
    follower.parent, follower.parent_type, follower.parent_bone = rig, "BONE", "hand_l"
    follower.location = (0.1, 0.2, 0.3)
    expected = {}
    for frame in (1, 4, 9):
        bpy.context.scene.frame_set(frame)
        expected[frame] = matrix_values(follower.matrix_world)
    migration.migrate_runtime_data(bpy.context)
    for frame, matrix in expected.items():
        bpy.context.scene.frame_set(frame)
        assert matrix_values(follower.matrix_world) == pytest.approx(matrix, abs=3e-5)


def test_other_action_slots_are_not_converted(legacy_body):
    from character_dna.fbx.writer import ensure_action_channelbag

    instance, _ = legacy_body
    rig = instance.body_rig
    action = animate(rig)
    extra_slot = action.slots.new("OBJECT", name="Unrelated Object")
    bag = ensure_action_channelbag(action, extra_slot)
    curve = bag.fcurves.new("location", index=1)
    curve.keyframe_points.insert(1, 17)
    handle = extra_slot.handle
    migration.migrate_runtime_data(bpy.context)
    bags = migration.channel_containers(rig.animation_data.action, handle)
    assert len(bags) == 1
    assert bags[0].fcurves[0].data_path == "location"
    assert bags[0].fcurves[0].keyframe_points[0].co.y == 17


def test_sparse_quaternion_preserves_unkeyed_component_sign(legacy_body):
    instance, _ = legacy_body
    rig = instance.body_rig
    bone = rig.pose.bones["upperarm_l"]
    bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = (-0.8, 0.05, 0.1, 0.2)
    samples = {}
    for frame, value in ((1, 0.1), (8, 0.3)):
        bone.rotation_quaternion.y = value
        bone.keyframe_insert("rotation_quaternion", index=2, frame=frame)
        bpy.context.scene.frame_set(frame)
        samples[frame] = skinning(rig, "hand_l")
    migration.migrate_runtime_data(bpy.context)
    for frame, values in samples.items():
        bpy.context.scene.frame_set(frame)
        assert skinning(rig, "hand_l") == pytest.approx(values, abs=3e-5)


def test_custom_bone_constraint_is_rejected_before_mutation(legacy_body):
    instance, _ = legacy_body
    rig = instance.body_rig
    data = rig.data
    constraint = rig.pose.bones["upperarm_l"].constraints.new("LIMIT_ROTATION")
    constraint.owner_space = "LOCAL"
    with pytest.raises(ValueError, match="bake custom bone constraints"):
        migration.migrate_runtime_data(bpy.context)
    assert rig.data == data and not rig.get("dna_coordinate_version")


def test_full_character_migration_preserves_face_animation_and_links(addon):
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.character_dna.import_dna(
        filepath=str(TEST_DNA_FOLDER / "ada" / "head.dna"),
        import_mesh=False,
        import_bones=True,
        import_face_board=True,
        include_body=True,
        import_materials=False,
    )
    instance = bpy.context.scene.character_dna.rig_instance_list[0]
    face, head, body = instance.face_board, instance.head_rig, instance.body_rig
    jaw = face.pose.bones["CTRL_C_jaw"]
    for frame, value in ((1, 0), (5, 0.8)):
        jaw.location.y = value
        jaw.keyframe_insert("location", frame=frame)
    action = face.animation_data.action
    bpy.context.scene.frame_set(5)
    expected = skinning(head, "FACIAL_C_Jaw")
    native = {rig: {bone.name: bone.matrix_local.copy() for bone in rig.data.bones} for rig in (head, body)}
    bpy.context.window_manager.character_dna.evaluate_dependency_graph = False
    with controller.preserve_bindings():
        engine.discard(instance)
        for rig in (head, body):
            migration._activate(rig)
            bpy.ops.object.mode_set(mode="EDIT")
            for bone in rig.data.edit_bones:
                bone.matrix = bone.matrix @ migration.BASIS
            bpy.ops.object.mode_set(mode="OBJECT")
            del rig["dna_coordinate_version"]
            for bone in rig.pose.bones:
                bone.matrix_basis = Matrix.Identity(4)
        for constraint in face.constraints:
            if constraint.type == "CHILD_OF" and constraint.target in (head, body):
                constraint.inverse_matrix = migration.BASIS.inverted() @ constraint.inverse_matrix
    assert migration.migrate_runtime_data(bpy.context) == ("default", 1, 0)
    assert instance.head_rig == head and instance.body_rig == body and instance.face_board == face
    assert face.animation_data.action == action
    assert skinning(head, "FACIAL_C_Jaw") == pytest.approx(expected, abs=5e-5)
    for rig, bones in native.items():
        for name, matrix in bones.items():
            assert matrix_values(rig.data.bones[name].matrix_local) == pytest.approx(matrix_values(matrix), abs=3e-5)
    bpy.ops.wm.read_homefile(use_empty=True)


@pytest.fixture
def control_addons(addon):
    import importlib
    import sys

    repo = Path(__file__).resolve().parents[2]
    directories = [
        repo / name / "src" / "addons" for name in ("character-control-rig-addon", "character-assembly-addon")
    ]
    if not all(path.exists() for path in directories):
        pytest.skip("Optional control-rig migration integration requires the sibling addon checkouts")
    rigify = Path("C:/Program Files/Blender Foundation/Blender 5.2/5.2/scripts/addons_core")
    if rigify.exists():
        sys.path.append(str(rigify))
    modules = []
    for directory, name in zip(directories, ("character_control_rig", "character_assembly"), strict=True):
        sys.path.append(str(directory))
        module = importlib.import_module(name)
        module.register()
        modules.append(module)
        if name not in bpy.context.preferences.addons:
            bpy.context.preferences.addons.new().module = name
    yield
    for module in reversed(modules):
        module.unregister()


@pytest.mark.parametrize("failure", [False, "verify", "motion"])
def test_confirmed_control_rebuild_retains_ik_and_fk_motion(  # noqa: PLR0915
    control_addons, legacy_body, monkeypatch, failure, tmp_path
):
    from character_control_rig.rig_builder import get_builder
    from character_control_rig.rig_builder.mappings import load_source_to_deform_mappings
    from character_control_rig.utilities import get_or_create_rig_instance_proxy

    instance, _ = legacy_body
    body = instance.body_rig
    builder = get_builder("RIGIFY")
    builder.validate_runtime(bpy.context)
    meta = builder.create_metarig(instance.name, body, builder.load_snap_data("metahuman"))
    control = builder.generate_control_rig(instance.name, meta)
    builder.constrain_to_source(body, control, load_source_to_deform_mappings("rigify", "metahuman"), instance.name)
    instance.control_rig = control
    proxy = get_or_create_rig_instance_proxy(bpy.context.scene.character_control_rig, instance.name)
    proxy.framework = "RIGIFY"
    proxy.snap_bones_to_mesh = False
    ik = control.pose.bones["hand_ik.L"]
    finger = control.pose.bones["f_index.01.L"]
    for frame, value in ((1, 0), (5, 0.08), (10, -0.03)):
        ik.location = (value, value * 0.5, -value)
        ik.keyframe_insert("location", frame=frame)
        finger.rotation_mode = "XYZ"
        finger.rotation_euler.x = value * 3
        finger.keyframe_insert("rotation_euler", frame=frame)
    action = control.animation_data.action
    nla_action = action.copy()
    nla_action.name = "Other control clip"
    strip = control.animation_data.nla_tracks.new().strips.new("Control clip", 20, nla_action)
    migration.set_action(strip, nla_action, migration.slot_handle(control.animation_data))
    strip.frame_start, strip.scale, strip.repeat = 20.25, 1.25, 2
    nla_timing = (strip.frame_start, strip.frame_end, strip.scale, strip.repeat)
    control.animation_data.action = None
    active_strip = control.animation_data.nla_tracks.new().strips.new("First control clip", 1, action)
    migration.set_action(active_strip, action, migration.slot_handle(strip))
    original_name = control.name
    expected = {}
    for frame in (1, 3, 5, 8, 10, 23, 30):
        bpy.context.scene.frame_set(frame)
        expected[frame] = {name: skinning(body, name) for name in ("upperarm_l", "hand_l", "index_01_l")}
    assert expected[1] != expected[5]
    if failure:
        from character_dna.utilities.migration import ControlUpgrade

        original_data = body.data
        original_objects = set(bpy.data.objects)
        if failure == "motion":
            original_apply = ControlUpgrade.apply

            def move_rebuilt_root(plan):
                original_apply(plan)
                plan.new.pose.bones["root"].location.x += 0.01
                plan.new.update_tag()

            monkeypatch.setattr(ControlUpgrade, "apply", move_rebuilt_root)
        else:
            monkeypatch.setattr(
                ControlUpgrade, "verify", lambda _: (_ for _ in ()).throw(ValueError("Injected failure"))
            )
        with pytest.raises(ValueError, match="frame.*mm" if failure == "motion" else "Injected failure"):
            migration.migrate_runtime_data(bpy.context, rebuild_controls=True)
        assert instance.control_rig == control
        assert control.animation_data.action is None
        assert body.data == original_data and not body.get("dna_coordinate_version")
        assert not engine.carriers(instance)
        assert set(bpy.data.objects) == original_objects
        for frame, values in expected.items():
            bpy.context.scene.frame_set(frame)
            for name, matrix in values.items():
                assert skinning(body, name) == pytest.approx(matrix, abs=3e-5), (frame, name)
        return
    original_file = tmp_path / "Legacy.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(original_file))
    original_bytes = original_file.read_bytes()
    assert bpy.ops.character_dna.migrate_legacy_data(rebuild_controls=True) == {"FINISHED"}
    assert original_file.read_bytes() == original_bytes
    assert bpy.data.filepath == str(original_file)
    backups = list(tmp_path.glob("Legacy.pre-migration-*.blend"))
    assert len(backups) == 1
    with bpy.data.libraries.load(str(backups[0])) as (saved, _):
        assert action.name in saved.actions
        assert not any(".Migrated" in name for name in saved.actions)
    assert instance.control_rig.name == original_name
    assert instance.control_rig.animation_data.action is None
    assert action.use_fake_user
    new_strip = instance.control_rig.animation_data.nla_tracks[0].strips[0]
    assert new_strip.action != nla_action
    assert (new_strip.frame_start, new_strip.frame_end, new_strip.scale, new_strip.repeat) == nla_timing
    for frame, values in expected.items():
        bpy.context.scene.frame_set(frame)
        for name, matrix in values.items():
            assert skinning(body, name) == pytest.approx(matrix, abs=3e-5), (frame, name)
