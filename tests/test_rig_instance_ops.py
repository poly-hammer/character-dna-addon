import hashlib

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import bpy
import pytest


@pytest.mark.parametrize(
    ("metahuman_name",),
    [
        ("ada_copy",),
    ],
)
def test_duplicate_rig_instance(load_dna_for_rig_instance_ops, temp_folder: Path, metahuman_name: str):
    temp_folder.mkdir(parents=True, exist_ok=True)

    bpy.ops.character_dna.duplicate_rig_instance(  # type: ignore
        new_name=metahuman_name, new_folder=str(temp_folder)
    )

    instances = list(bpy.context.scene.character_dna.rig_instance_list)  # type: ignore
    instance_names = [instance.name for instance in instances]

    assert metahuman_name in instance_names, f"Rig instance {metahuman_name} should be present in the scene"

    for instance in instances:
        assert instance.body_rig is not None, f"Body rig should be set for {instance.name}"
        assert instance.body_mesh is not None, f"Body mesh should be set for {instance.name}"
        assert instance.body_dna_file_path is not None, f"Body DNA file path should be set for {instance.name}"
        assert instance.head_rig is not None, f"Head rig should be set for {instance.name}"
        assert instance.head_mesh is not None, f"Head mesh should be set for {instance.name}"
        assert instance.head_dna_file_path is not None, f"Head DNA file path should be set for {instance.name}"


@pytest.mark.parametrize(
    ("direction", "name", "initial_index", "expected_index"),
    [
        ("UP", "ada", 0, 1),
        ("DOWN", "ada", 1, 0),
    ],
)
def test_rig_instance_entry_move(
    load_dna_for_rig_instance_ops, direction: str, name: str, initial_index: int, expected_index: int
):
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == initial_index, (
        f"Rig instance {name} should be at index {initial_index} before move"
    )
    bpy.ops.character_dna.rig_instance_entry_move(active_index=initial_index, direction=direction)  # type: ignore
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == expected_index, (
        f"Rig instance {name} should be at index {expected_index} after move"
    )


def test_rig_instance_entry_add():
    name = "Untitled1"
    # open default scene
    bpy.ops.wm.read_homefile(app_template="")
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert len(instance_names) == 0, "Rig instance list should be empty before add"
    bpy.ops.character_dna.rig_instance_entry_add(active_index=0)  # type: ignore
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == 0, f"Rig instance {name} should be at index 0 before move"


def test_rig_instance_entry_remove():
    name = "Untitled1"
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == 0, f'Rig instance {name} should be at index 0 from the previous "add" test'
    bpy.ops.character_dna.rig_instance_entry_remove(active_index=0)  # type: ignore
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert len(instance_names) == 0, "Rig instance list should be empty after remove"


@pytest.fixture
def removal_scene() -> Iterator[bpy.types.Scene]:
    from character_dna.runtime import engine

    original_scene = bpy.context.window.scene
    before = set(bpy.data.user_map())
    scene = bpy.data.scenes.new("RigInstanceRemoval")
    bpy.context.window.scene = scene
    try:
        yield scene
    finally:
        for created_scene in set(bpy.data.scenes) - before:
            for instance in created_scene.character_dna.rig_instance_list:
                engine.release_records(instance)
        bpy.context.window.scene = original_scene
        bpy.data.batch_remove(ids=set(bpy.data.user_map()) - before)


def test_remove_reference_root_with_name_collision(removal_scene: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from character_dna import operators
    from character_dna.runtime import engine
    from character_dna.ui.callbacks import get_active_rig_instance

    properties = removal_scene.character_dna
    instance = properties.rig_instance_list.add()
    instance["name"] = "RemoveReference"
    unrelated = bpy.data.collections.new(instance.name)
    removal_scene.collection.children.link(unrelated)
    sentinel = bpy.data.objects.new("RemovalSentinel", None)
    unrelated.objects.link(sentinel)
    root = bpy.data.collections.new(instance.name)
    removal_scene.collection.children.link(root)
    nested = bpy.data.collections.new("RemovalNested")
    root.children.link(nested)
    owned = bpy.data.objects.new("RemovalOwned", None)
    nested.objects.link(owned)
    helper = bpy.data.objects.new("RemovalParent", None)
    nested.objects.link(helper)
    owned.parent = helper
    helper_name = helper.name
    instance["reference_root"] = root
    root_name, nested_name, owned_name = root.name, nested.name, owned.name
    assert root_name != instance.name
    released = []
    original_release = engine.release_records

    def release_records(removing: Any) -> None:
        assert removing == properties.rig_instance_list[0]
        assert removing.get("reference_root") == root
        assert bpy.data.collections.get(root_name) == root
        assert nested.objects.get(owned_name) == owned
        released.append(removing.name)
        original_release(removing)

    monkeypatch.setattr(engine, "release_records", release_records)
    monkeypatch.setattr(engine, "discard", lambda *_args: pytest.fail("Removal must not discard shared drivers"))

    assert bpy.ops.character_dna.rig_instance_entry_remove(active_index=0) == {"FINISHED"}

    assert released == ["RemoveReference"]
    assert "UNDO" in operators.UILIST_RIG_INSTANCE_OT_entry_remove.bl_options
    assert bpy.data.collections.get("RemoveReference") == unrelated
    assert unrelated.objects.get(sentinel.name) == sentinel
    assert bpy.data.collections.get(root_name) is None
    assert bpy.data.collections.get(nested_name) is None
    assert bpy.data.objects.get(owned_name) is None
    assert bpy.data.objects.get(helper_name) is None
    assert len(properties.rig_instance_list) == 0
    assert get_active_rig_instance() is None


@pytest.mark.parametrize("mode", ["APPEND", "LINK"])
def test_remove_imported_reference_shared_with_other_scene(removal_scene: Any, tmp_path: Path, mode: str) -> None:
    from character_dna.ui.callbacks import get_active_rig_instance
    from character_dna.utilities import reference

    source_root = bpy.data.collections.new("RemovalImport")
    removal_scene.collection.children.link(source_root)
    source_mesh = bpy.data.meshes.new("RemovalImportMesh")
    source_object = bpy.data.objects.new("RemovalImportObject", source_mesh)
    source_root.objects.link(source_object)
    descriptor = reference.describe_instance(
        removal_scene, "character_dna", {"name": source_root.name, "head_mesh": source_object}
    )
    assert not descriptor["issues"]
    source = tmp_path / "removal_source.blend"
    bpy.data.libraries.write(str(source), {source_root})
    source_digest = hashlib.sha256(source.read_bytes()).digest()
    bpy.data.batch_remove(ids={source_root, source_object, source_mesh})

    instance = reference.import_characters(bpy.context, str(source), [descriptor], mode, False)[0]
    root = instance["reference_root"]
    root_name = root.name
    shared = instance.head_mesh
    instance.output.head_item_list.add().scene_object = shared
    other_scene = bpy.data.scenes.new("RemovalOtherScene")
    bpy.context.window.scene = other_scene
    other_instance = reference.import_characters(bpy.context, str(source), [descriptor], "LINK", False)[0]
    other_root = other_instance["reference_root"]
    other_root_name = other_root.name
    if mode == "APPEND":
        other_root.objects.link(shared)
    assert shared in other_root.objects.values()
    if mode == "LINK":
        assert other_instance.head_mesh == shared
        assert shared.library is not None
    bpy.context.window.scene = removal_scene

    assert bpy.ops.character_dna.rig_instance_entry_remove(active_index=0) == {"FINISHED"}

    assert bpy.data.collections.get(root_name) is None
    assert bpy.data.collections.get(other_root_name) == other_root
    assert shared in other_root.objects.values()
    assert shared in other_scene.objects.values()
    assert shared not in removal_scene.objects.values()
    assert other_instance.head_mesh is not None
    assert len(other_scene.character_dna.rig_instance_list) == 1
    assert len(removal_scene.character_dna.rig_instance_list) == 0
    assert get_active_rig_instance() is None
    assert hashlib.sha256(source.read_bytes()).digest() == source_digest


@pytest.mark.parametrize("delete_associated_data", [False, True])
def test_remove_reference_without_data_or_with_missing_root(
    removal_scene: Any, monkeypatch: pytest.MonkeyPatch, delete_associated_data: bool
) -> None:
    from character_dna.runtime import engine
    from character_dna.ui.callbacks import get_active_rig_instance

    properties = removal_scene.character_dna
    instance = properties.rig_instance_list.add()
    instance["name"] = "RemovalMissing"
    unrelated = bpy.data.collections.new(instance.name)
    removal_scene.collection.children.link(unrelated)
    root = bpy.data.collections.new(instance.name)
    removal_scene.collection.children.link(root)
    instance["reference_root"] = root
    root_name = root.name
    if delete_associated_data:
        bpy.data.collections.remove(root)
        assert "reference_root" in instance
        assert instance.get("reference_root") is None
    released = []
    monkeypatch.setattr(engine, "release_records", lambda removing: released.append(removing.name))

    assert bpy.ops.character_dna.rig_instance_entry_remove(
        active_index=0, delete_associated_data=delete_associated_data
    ) == {"FINISHED"}

    assert released == ["RemovalMissing"]
    assert bpy.data.collections.get("RemovalMissing") == unrelated
    if not delete_associated_data:
        assert bpy.data.collections.get(root_name) == root
    assert len(properties.rig_instance_list) == 0
    assert get_active_rig_instance() is None


@pytest.mark.parametrize("reference_mode", ["", "APPEND"])
def test_cancelled_duplicate_preserves_native_bindings(
    load_head_only_dna: Any, tmp_path: Path, reference_mode: str
) -> None:
    from character_dna.runtime import engine
    from character_dna.utilities import get_active_rig_instance, get_addon_window_manager_properties

    instance = get_active_rig_instance()
    instance["reference_mode"] = reference_mode
    instance.output.folder_path = str(tmp_path / "missing_source_output")
    assert engine.active(instance)
    carrier_pointers = {carrier.as_pointer() for carrier in engine.carriers()}
    window_properties = get_addon_window_manager_properties(bpy.context)
    previous_evaluation = window_properties.evaluate_dependency_graph

    with pytest.raises(RuntimeError, match="Folder not found"):
        bpy.ops.character_dna.duplicate_rig_instance(new_name="CancelledCopy", new_folder=str(tmp_path / "missing"))

    assert {carrier.as_pointer() for carrier in engine.carriers()} == carrier_pointers
    assert engine.active(instance)
    assert window_properties.evaluate_dependency_graph == previous_evaluation
    assert bpy.context.scene.character_dna.rig_instance_list.get("CancelledCopy") is None


@pytest.mark.parametrize("reference_mode", ["LINK", "EDITABLE_LINK"])
def test_duplicate_refuses_linked_source_before_mutation(
    removal_scene: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_mode: str
) -> None:
    from character_dna.runtime import engine
    from character_dna.utilities import get_addon_window_manager_properties

    instances = removal_scene.character_dna.rig_instance_list
    source = instances.add()
    source["name"] = "LinkedDuplicateSource"
    source["reference_mode"] = reference_mode
    before = set(bpy.data.user_map())
    window_properties = get_addon_window_manager_properties(bpy.context)
    previous_evaluation = window_properties.evaluate_dependency_graph
    monkeypatch.setattr(engine, "discard", lambda *_args: pytest.fail("Linked duplicate must not discard bindings"))

    with pytest.raises(RuntimeError, match="Append the character from its source file first"):
        bpy.ops.character_dna.duplicate_rig_instance(new_name="LinkedCopy", new_folder=str(tmp_path))

    assert set(bpy.data.user_map()) == before
    assert len(instances) == 1
    assert not list(tmp_path.iterdir())
    assert window_properties.evaluate_dependency_graph == previous_evaluation


def test_duplicate_failure_restores_appended_native_source(
    load_head_only_dna: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from character_dna import utilities
    from character_dna.runtime import controller, engine

    source = utilities.get_active_rig_instance()
    source["reference_mode"] = "APPEND"
    source.output.folder_path = str(tmp_path / "missing_source_output")
    assert engine.active(source)
    source_name = source.name
    rebuild_names = []
    original_rebuild = controller.rebuild

    def rebuild(instance: Any) -> None:
        rebuild_names.append(instance.name)
        original_rebuild(instance)

    def fail_copy(**_kwargs: Any) -> None:
        assert not engine.carriers(source)
        raise RuntimeError("Duplicate copy failed")

    monkeypatch.setattr(controller, "rebuild", rebuild)
    monkeypatch.setattr(utilities, "copy_mesh", fail_copy)
    window_properties = utilities.get_addon_window_manager_properties(bpy.context)
    previous_evaluation = window_properties.evaluate_dependency_graph
    window_properties.evaluate_dependency_graph = False
    try:
        with pytest.raises(RuntimeError, match="Duplicate copy failed"):
            bpy.ops.character_dna.duplicate_rig_instance(new_name="FailedCopy", new_folder=str(tmp_path))

        assert engine.active(source)
        assert rebuild_names == [source_name]
        assert not window_properties.evaluate_dependency_graph
        assert bpy.context.scene.character_dna.rig_instance_list.get("FailedCopy") is None
    finally:
        window_properties.evaluate_dependency_graph = previous_evaluation


def test_duplicate_preserves_unrelated_native_records_and_appended_source(  # noqa: PLR0915
    load_head_only_dna: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from character_dna import operators, utilities
    from character_dna.runtime import controller, engine

    def duplicate(name: str) -> set[str]:
        operator = SimpleNamespace(new_name=name, new_folder=str(tmp_path), copy_face_board=True)
        return operators.DuplicateRigInstance._duplicate(operator, bpy.context)

    properties = bpy.context.scene.character_dna
    source = utilities.get_active_rig_instance()
    source_name = source.name
    source["reference_mode"] = "APPEND"
    source.output.folder_path = str(tmp_path / "missing_source_output")
    source.auto_evaluate = False
    controller.reconcile()
    assert not engine.carriers(source)
    source.auto_evaluate = True
    controller.reconcile()
    assert engine.active(source)

    rebuild_names = []
    original_rebuild = controller.rebuild

    def rebuild(instance: Any) -> None:
        rebuild_names.append(instance.name)
        original_rebuild(instance)

    monkeypatch.setattr(controller, "rebuild", rebuild)
    monkeypatch.setattr(controller, "reconcile", lambda: pytest.fail("Duplicate must not reconcile other rigs"))
    monkeypatch.setattr(engine, "invalidate", lambda: pytest.fail("Duplicate must not invalidate other records"))
    window_properties = utilities.get_addon_window_manager_properties(bpy.context)
    previous_evaluation = window_properties.evaluate_dependency_graph
    window_properties.evaluate_dependency_graph = False
    try:
        assert duplicate("UnrelatedCopy") == {"FINISHED"}
        assert rebuild_names == [source_name, "UnrelatedCopy"]
        assert not window_properties.evaluate_dependency_graph
        instances = properties.rig_instance_list
        assert engine.active(instances[source_name])
        assert engine.active(instances["UnrelatedCopy"])
        unrelated_pointers = {carrier.as_pointer() for carrier in engine.carriers(instances["UnrelatedCopy"])}
        unrelated_records = {pointer: engine._records[pointer] for pointer in unrelated_pointers}
        with controller.preserve_bindings():
            older = instances.add()
            older["name"] = "OlderWithoutChannels"
            instances.move(len(instances) - 1, 0)
        properties.rig_instance_list_active_index = instances.find(source_name)
        rebuild_names.clear()

        assert duplicate("TargetedCopy") == {"FINISHED"}

        assert {carrier.as_pointer() for carrier in engine.carriers(instances["UnrelatedCopy"])} == unrelated_pointers
        assert all(engine._records[pointer] is record for pointer, record in unrelated_records.items())
        assert rebuild_names == [source_name, "TargetedCopy"]
        assert not engine.carriers(instances["OlderWithoutChannels"])
        assert not window_properties.evaluate_dependency_graph
        assert engine.active(instances[source_name])
        assert engine.active(instances["TargetedCopy"])
        assert engine.instance_id(instances[source_name]) != engine.instance_id(instances["TargetedCopy"])

        for name, value in ((source_name, 0.8), ("TargetedCopy", 0.2)):
            instance = instances[name]
            instance.face_board.pose.bones["CTRL_C_jaw"].location.y = value
            instance.face_board.update_tag()
        bpy.context.view_layer.update()
        for name, value in ((source_name, 0.8), ("TargetedCopy", 0.2)):
            instance = instances[name]
            reader = instance.head_dna_reader
            index = next(
                index
                for index in range(reader.getRawControlCount())
                if reader.getRawControlName(index) == "CTRL_expressions.jawOpen"
            )
            native_value = engine.ui_raw_control_value(instance, index, bpy.context.view_layer.depsgraph)
            assert native_value == pytest.approx(value)
    finally:
        window_properties.evaluate_dependency_graph = previous_evaluation
