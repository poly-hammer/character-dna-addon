import bpy
import pytest

from character_dna.runtime import controller, engine
from character_dna.utilities import detect_legacy_data, detect_runtime_migration, migrate_legacy_data, misc


@pytest.fixture
def empty_scene():
    """Provide a clean, empty scene with the addon enabled and reset afterward."""
    bpy.ops.wm.read_homefile(use_empty=True)
    yield bpy.context.scene
    bpy.ops.wm.read_homefile(use_empty=True)


def _make_sibling_instance(scene: bpy.types.Scene) -> dict:
    """Populate the read-only ``character_dna_pro`` sibling pointer to mimic a
    .blend saved by the other edition."""
    arm = bpy.data.objects.new("Ada_head_rig", bpy.data.armatures.new("Ada_head_rig"))
    mesh = bpy.data.objects.new("Ada_head_mesh", bpy.data.meshes.new("Ada_head_mesh"))
    body = bpy.data.objects.new("Ada_body_rig", bpy.data.armatures.new("Ada_body_rig"))
    mat = bpy.data.materials.new("Ada_head_shader")
    for obj in (arm, mesh, body):
        scene.collection.objects.link(obj)

    instance = scene.character_dna_pro.rig_instance_list.add()
    instance.name = "Ada"
    instance.head_rig = arm
    instance.head_mesh = mesh
    instance.body_rig = body
    instance.head_material = mat
    instance.head_dna_file_path = "//head.dna"
    instance.body_dna_file_path = "//body.dna"
    instance.output.folder_path = "//out"
    return {"head_rig": arm, "head_mesh": mesh, "body_rig": body, "head_material": mat}


def test_detect_cross_edition_data(empty_scene: bpy.types.Scene):
    _make_sibling_instance(empty_scene)
    assert detect_legacy_data(empty_scene) == ("character_dna_pro", "rig_instance_list")


def test_no_migration_for_current_edition_data(empty_scene: bpy.types.Scene):
    instance = empty_scene.character_dna.rig_instance_list.add()
    instance.name = "Ada"
    assert detect_legacy_data(empty_scene) is None
    assert not detect_runtime_migration(empty_scene)


def test_empty_armatures_are_not_runtime_migration(empty_scene: bpy.types.Scene):
    _make_sibling_instance(empty_scene)
    migrate_legacy_data(bpy.context)
    assert not detect_runtime_migration(empty_scene)


def test_cross_edition_migration_copies_all_fields(empty_scene: bpy.types.Scene):
    objects = _make_sibling_instance(empty_scene)

    result = migrate_legacy_data(bpy.context)
    assert result == "cross_edition"

    instances = list(empty_scene.character_dna.rig_instance_list)
    assert len(instances) == 1
    migrated = instances[0]
    assert migrated.name == "Ada"
    assert migrated.head_rig == objects["head_rig"]
    assert migrated.head_mesh == objects["head_mesh"]
    assert migrated.body_rig == objects["body_rig"]
    assert migrated.head_material == objects["head_material"]
    assert migrated.head_dna_file_path == "//head.dna"
    assert migrated.body_dna_file_path == "//body.dna"
    assert migrated.output.folder_path == "//out"

    # The migrated sibling list is cleared so it is neither re-detected nor re-saved.
    assert len(empty_scene.character_dna_pro.rig_instance_list) == 0
    assert detect_legacy_data(empty_scene) is None


def test_cross_edition_migration_skips_existing_names(empty_scene: bpy.types.Scene):
    _make_sibling_instance(empty_scene)
    existing = empty_scene.character_dna.rig_instance_list.add()
    existing.name = "Ada"

    migrate_legacy_data(bpy.context)

    # The existing instance is not duplicated.
    names = [instance.name for instance in empty_scene.character_dna.rig_instance_list]
    assert names.count("Ada") == 1


def test_detection_survives_group_without_rig_instance_list(empty_scene: bpy.types.Scene):
    """A registered group from another addon may not define ``rig_instance_list``.

    The old ``meta_human_dna`` prototype's scene group exposes ``bl_rna`` but stores its
    instances under a different name, so reading the attribute unguarded raised
    AttributeError on every panel redraw. See issue #341's sibling report and
    CHARACTER-DNA-ADDON-MHQ.
    """
    from character_dna.utilities.misc import _rig_instance_sources

    class ForeignGroup:
        bl_rna = object()

        def get(self, _key, default=None):
            return default

    assert _rig_instance_sources(ForeignGroup(), "rig_instance_list") == []
    assert detect_legacy_data(empty_scene) is None


def _make_runtime_instance(scene: bpy.types.Scene, name: str = "Ada"):
    armature = bpy.data.armatures.new(f"{name}_head_rig")
    rig = bpy.data.objects.new(armature.name, armature)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature.edit_bones.new("FACIAL_C_Jaw")
    bone.tail.z = 0.1
    bpy.ops.object.mode_set(mode="OBJECT")
    instance = scene.character_dna.rig_instance_list.add()
    instance.name = name
    instance.head_rig = rig
    return instance


@pytest.fixture
def migration_contract(monkeypatch: pytest.MonkeyPatch):
    """Isolate migration orchestration from the runtime implementation in progress."""
    monkeypatch.setattr(engine, "SCHEMA_VERSION", 2, raising=False)
    monkeypatch.setattr(engine, "binding_issues", lambda _instance: ["Missing saved drivers"], raising=False)
    monkeypatch.setattr(engine, "carriers", lambda _instance: [])
    monkeypatch.setattr(engine, "capability", lambda: (True, "available"))
    monkeypatch.setattr(controller, "rebuild", lambda _instance: pytest.fail("Unexpected rebuild"), raising=False)
    monkeypatch.setattr(controller, "reconcile", lambda: pytest.fail("Migration must not reconcile all rigs"))


@pytest.mark.parametrize("enabled", [True, False])
def test_detection_uses_saved_graph_not_session(
    empty_scene, migration_contract, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    instance = _make_runtime_instance(empty_scene)
    instance.auto_evaluate = enabled
    monkeypatch.setattr(engine, "active", lambda _instance: pytest.fail("Detection must not inspect active sessions"))
    monkeypatch.setattr(engine, "binding_issues", lambda _instance: [])
    assert not detect_runtime_migration(empty_scene)
    monkeypatch.setattr(engine, "binding_issues", lambda _instance: ["Scene-bound native carrier"])
    assert detect_runtime_migration(empty_scene)
    assert detect_legacy_data(empty_scene) is None


@pytest.mark.parametrize("previous_flag", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_metadata_preserves_evaluation_state(
    empty_scene, monkeypatch: pytest.MonkeyPatch, previous_flag: bool, failure: bool
) -> None:
    _make_sibling_instance(empty_scene)
    properties = bpy.context.window_manager.character_dna
    properties.evaluate_dependency_graph = previous_flag
    transition = controller._transitioning
    if failure:

        def fail_copy(*args):
            raise ValueError("metadata failure")

        monkeypatch.setattr(misc, "_copy_rig_instance_fields", fail_copy)
        with pytest.raises(ValueError, match="metadata failure"):
            migrate_legacy_data(bpy.context)
    else:
        migrate_legacy_data(bpy.context)
    assert properties.evaluate_dependency_graph == previous_flag
    assert controller._transitioning == transition


def test_cross_edition_preserves_settings_and_identity(empty_scene) -> None:
    _make_sibling_instance(empty_scene)
    source = empty_scene.character_dna_pro.rig_instance_list[0]
    source.auto_evaluate = False
    source.auto_evaluate_head = False
    source.evaluate_bones = False
    source.evaluate_shape_keys = False
    source["native_runtime_id"] = "saved-native-identity"
    migrate_legacy_data(bpy.context)
    migrated = empty_scene.character_dna.rig_instance_list[0]
    assert not migrated.auto_evaluate
    assert not migrated.auto_evaluate_head
    assert not migrated.evaluate_bones
    assert not migrated.evaluate_shape_keys
    assert migrated["native_runtime_id"] == "saved-native-identity"


def test_missing_dna_preflights_entire_batch(empty_scene, migration_contract, tmp_path) -> None:
    first = _make_runtime_instance(empty_scene)
    second = _make_runtime_instance(empty_scene, "Other")
    path = tmp_path / "head.dna"
    path.touch()
    first.head_dna_file_path = str(path)
    objects = {obj.as_pointer() for obj in bpy.data.objects}
    with pytest.raises(ValueError, match="Other: missing head DNA file path"):
        misc.migrate_runtime_data(bpy.context)
    assert {obj.as_pointer() for obj in bpy.data.objects} == objects
    assert second.head_dna_file_path == ""


def test_missing_legacy_dna_does_not_consume_metadata(empty_scene, migration_contract) -> None:
    instance = _make_runtime_instance(empty_scene)
    rig = instance.head_rig
    empty_scene.character_dna.rig_instance_list.clear()
    empty_scene["meta_human_dna"] = {
        "rig_logic_instance_list": [{"instance_name": "Ada", "head_rig": rig, "head_dna_file_path": ""}]
    }
    with pytest.raises(ValueError, match="missing head DNA file path"):
        misc.migrate_runtime_data(bpy.context)
    assert "meta_human_dna" in empty_scene
    assert len(empty_scene.character_dna.rig_instance_list) == 0


def test_future_schema_rejected_before_rebuild(
    empty_scene, migration_contract, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    instance = _make_runtime_instance(empty_scene)
    path = tmp_path / "head.dna"
    path.touch()
    instance.head_dna_file_path = str(path)
    carrier = bpy.data.objects.new("future_carrier", None)
    carrier["schema_version"] = 3
    monkeypatch.setattr(engine, "carriers", lambda _instance: [carrier])
    with pytest.raises(ValueError, match="unsupported future runtime schema 3"):
        misc.migrate_runtime_data(bpy.context)
    assert carrier["schema_version"] == 3


def test_linked_source_rejected_before_rebuild(
    empty_scene, migration_contract, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    instance = _make_runtime_instance(empty_scene)
    path = tmp_path / "head.dna"
    path.touch()
    instance.head_dna_file_path = str(path)
    library_path = tmp_path / "linked.blend"
    source_carrier = bpy.data.objects.new("linked_carrier", None)
    source_carrier["schema_version"] = 1
    bpy.data.libraries.write(str(library_path), {source_carrier})
    with bpy.data.libraries.load(str(library_path), link=True) as (_source, target):
        target.objects = [source_carrier.name]
    linked = target.objects[0]
    monkeypatch.setattr(engine, "carriers", lambda _instance: [linked])
    with pytest.raises(ValueError, match=r"open the source \.blend"):
        misc.migrate_runtime_data(bpy.context)
    assert linked.library is not None
    assert linked["schema_version"] == 1


@pytest.mark.parametrize("writer", ["driver", "keyframe"])
def test_conflicting_saved_output_rejected_before_rebuild(
    empty_scene, migration_contract, monkeypatch: pytest.MonkeyPatch, tmp_path, writer: str
) -> None:
    instance = _make_runtime_instance(empty_scene)
    path = tmp_path / "head.dna"
    path.touch()
    instance.head_dna_file_path = str(path)
    rig = instance.head_rig
    bone = rig.pose.bones["FACIAL_C_Jaw"]
    output_path = bone.path_from_id("location")
    if writer == "driver":
        curve = rig.driver_add(output_path, 0)
        curve.driver.expression = "0.25"
    else:
        bone.keyframe_insert(data_path="location", index=0, frame=1)
    carrier = bpy.data.objects.new("old_carrier", None)
    carrier["schema_version"] = 1
    carrier["targets"] = [{"owner": rig, "path": output_path, "index": 0, "channel": 0}]
    monkeypatch.setattr(engine, "carriers", lambda _instance: [carrier])
    with pytest.raises(ValueError, match=r"conflicting driver|keyframed native output"):
        misc.migrate_runtime_data(bpy.context)
    assert carrier["schema_version"] == 1
    if writer == "driver":
        assert rig.animation_data.drivers.find(output_path, index=0).driver.expression == "0.25"
    else:
        assert rig.animation_data.action is not None


def test_rebuild_is_targeted_and_idempotent(
    empty_scene, migration_contract, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    instance = _make_runtime_instance(empty_scene)
    other = _make_runtime_instance(empty_scene, "Other")
    path = tmp_path / "head.dna"
    path.touch()
    instance.head_dna_file_path = str(path)
    pending = {instance.name}
    calls = []

    def rebuild(target):
        calls.append(target.name)
        pending.remove(target.name)

    monkeypatch.setattr(engine, "binding_issues", lambda target: ["Old drivers"] if target.name in pending else [])
    monkeypatch.setattr(controller, "rebuild", rebuild)
    assert misc.migrate_runtime_data(bpy.context) == ("default", 1, 1)
    assert misc.migrate_runtime_data(bpy.context) == ("default", 0, 2)
    assert calls == [instance.name]
    assert other.head_dna_file_path == ""


def test_migration_operator_supports_undo() -> None:
    from character_dna.operators import MigrateLegacyData

    assert {"REGISTER", "UNDO"} <= MigrateLegacyData.bl_options


@pytest.fixture
def portable_head(request: pytest.FixtureRequest):
    """Load the real head only once the parallel runtime API is available."""
    if getattr(engine, "SCHEMA_VERSION", 0) != 2 or not hasattr(controller, "rebuild"):
        pytest.skip("Requires the schema-v2 engine and targeted controller.rebuild implementation")
    request.getfixturevalue("load_head_only_dna")
    return bpy.context.scene.character_dna.rig_instance_list[0]


def _action_snapshot(owner):
    animation = owner.animation_data
    action = animation.action
    if action.is_action_layered:
        curves = [
            curve
            for layer in action.layers
            for strip in layer.strips
            for bag in strip.channelbags
            for curve in bag.fcurves
        ]
    else:
        curves = action.fcurves
    return (
        action.as_pointer(),
        animation.action_slot.handle,
        tuple(
            (
                curve.data_path,
                curve.array_index,
                tuple((tuple(point.co), point.interpolation) for point in curve.keyframe_points),
            )
            for curve in curves
        ),
    )


@pytest.mark.parametrize("layout", ["handler", "scene_bound"])
def test_real_migration_preserves_animation_and_evaluation(portable_head, layout: str) -> None:
    instance = portable_head
    scene = bpy.context.scene
    face = instance.face_board
    head = instance.head_rig
    jaw = face.pose.bones["CTRL_C_jaw"]
    for frame, value in ((1, 0.0), (10, 0.8)):
        jaw.location.y = value
        jaw.keyframe_insert(data_path="location", index=1, frame=frame)
    authored_action = _action_snapshot(face)
    unrelated = head.driver_add("location", 0)
    unrelated.driver.expression = "0.125"
    unrelated_pointer = unrelated.as_pointer()
    if layout == "handler":
        engine.discard(instance)
    else:
        for carrier in engine.carriers(instance):
            carrier["schema_version"] = 1
            carrier["scene"] = scene
    scene.frame_set(4, subframe=0.25)
    previous_frame = (scene.frame_current, scene.frame_subframe)
    properties = bpy.context.window_manager.character_dna
    properties.evaluate_dependency_graph = False
    assert detect_runtime_migration(scene)
    assert bpy.ops.character_dna.migrate_legacy_data() == {"FINISHED"}
    assert engine.binding_issues(instance) == []
    assert not detect_runtime_migration(scene)
    assert (scene.frame_current, scene.frame_subframe) == previous_frame
    assert not properties.evaluate_dependency_graph
    assert instance.face_board == face
    assert instance.head_rig == head
    assert _action_snapshot(face) == authored_action
    assert head.animation_data.drivers.find("location", index=0).as_pointer() == unrelated_pointer
    carriers = {carrier.as_pointer() for carrier in engine.carriers(instance)}
    objects = {obj.as_pointer() for obj in bpy.data.objects}
    assert all(carrier["schema_version"] == 2 and "scene" not in carrier for carrier in engine.carriers(instance))
    assert bpy.ops.character_dna.migrate_legacy_data() == {"FINISHED"}
    assert {carrier.as_pointer() for carrier in engine.carriers(instance)} == carriers
    assert {obj.as_pointer() for obj in bpy.data.objects} == objects
    assert _action_snapshot(face) == authored_action
    scene.frame_set(1)
    neutral = head.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones["FACIAL_C_Jaw"].matrix.copy()
    scene.frame_set(10)
    opened = head.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones["FACIAL_C_Jaw"].matrix.copy()
    assert neutral != opened


def test_current_schema_without_session_is_not_legacy(portable_head) -> None:
    instance = portable_head
    identity = instance["native_runtime_id"]
    carriers = {carrier.as_pointer() for carrier in engine.carriers(instance)}
    instance.auto_evaluate = False
    engine.release_records(instance)
    assert not engine.active(instance)
    assert not detect_runtime_migration(bpy.context.scene)
    assert misc.migrate_runtime_data(bpy.context)[1] == 0
    assert instance["native_runtime_id"] == identity
    assert {carrier.as_pointer() for carrier in engine.carriers(instance)} == carriers


def test_raw_legacy_record_resolves_carriers_in_native_scene(portable_head) -> None:
    """Raw migration records can be inspected alongside registered native instances."""
    instance = portable_head
    legacy = {"head_rig": instance.head_rig, "face_board": instance.face_board}
    assert engine.carriers(legacy) == engine.carriers(instance)
    legacy["head_rig"] = instance.head_rig.name
    assert engine.carriers(legacy) == engine.carriers(instance)
