import hashlib

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import bpy
import pytest

from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


@pytest.fixture
def load_head_only_dna(rig_factory, tmp_path, compact_dna):
    """Duplicate real native ownership and meshes without dense geometry."""
    profile = RigProfile(
        components=("head",), asset=str(compact_dna), pose_bones=STATE_BONES, face_board=True, materials=True
    )
    with rig_factory.lease(profile, tmp_path):
        yield


@pytest.mark.parametrize(
    ("metahuman_name",),
    [
        ("ada_copy",),
    ],
)
def test_duplicate_rig_instance(load_dna_for_rig_instance_ops, temp_folder: Path, metahuman_name: str):
    from character_dna.dna_io.coordinates import COORDINATE_VERSION

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
        assert instance.head_rig.get("dna_coordinate_version") == COORDINATE_VERSION
        assert instance.body_rig.get("dna_coordinate_version") == COORDINATE_VERSION


@pytest.mark.parametrize(
    ("direction", "name", "initial_index", "expected_index"),
    [
        ("UP", "ada", 0, 1),
        ("DOWN", "ada", 1, 0),
    ],
)
def test_rig_instance_entry_move(
    empty_character_scene, direction: str, name: str, initial_index: int, expected_index: int
):
    names = ["ada", "ada_copy"] if initial_index == 0 else ["ada_copy", "ada"]
    objects = {}
    for index, item_name in enumerate(names):
        instance = empty_character_scene.character_dna.rig_instance_list.add()
        instance["name"] = item_name
        for component in ("head_rig", "body_rig", "face_board"):
            obj = bpy.data.objects.new(f"{item_name}_{component}", bpy.data.armatures.new(component))
            empty_character_scene.collection.objects.link(obj)
            instance[component] = obj
            obj.location.x = index * 3.0 + (0.75 if component == "face_board" else 0.0)
            objects[item_name, component] = obj
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == initial_index, (
        f"Rig instance {name} should be at index {initial_index} before move"
    )
    bpy.ops.character_dna.rig_instance_entry_move(active_index=initial_index, direction=direction)  # type: ignore
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == expected_index, (
        f"Rig instance {name} should be at index {expected_index} after move"
    )
    for index, item_name in enumerate(names):
        for component in ("head_rig", "body_rig", "face_board"):
            expected_x = (1 - index) * 3.0 + (0.75 if component == "face_board" else 0.0)
            assert objects[item_name, component].location.x == pytest.approx(expected_x)


def test_rig_instance_entry_add(empty_character_scene):
    name = "Untitled1"
    # open default scene
    bpy.ops.wm.read_homefile(app_template="")
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert len(instance_names) == 0, "Rig instance list should be empty before add"
    bpy.ops.character_dna.rig_instance_entry_add(active_index=0)  # type: ignore
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == 0, f"Rig instance {name} should be at index 0 before move"


def test_rig_instance_entry_remove(empty_character_scene):
    name = "Untitled1"
    assert bpy.ops.character_dna.rig_instance_entry_add(active_index=0) == {"FINISHED"}
    instance_names = [instance.name for instance in bpy.context.scene.character_dna.rig_instance_list]  # type: ignore
    assert instance_names.index(name) == 0, f"Rig instance {name} should be at index 0 before removal"
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


def _world_vertices(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [tuple(evaluated.matrix_world @ vertex.co) for vertex in evaluated.data.vertices]


@pytest.fixture
def skinned_removal_asset(removal_scene):
    """Two armatures, driven shape keys/material, and a hidden secondary LOD."""
    instance = removal_scene.character_dna.rig_instance_list.add()
    instance["name"] = "MeshRemoval"
    root = bpy.data.collections.new(instance.name)
    removal_scene.collection.children.link(root)
    lods = [bpy.data.collections.new(f"{instance.name}_lod{i}") for i in range(2)]
    for lod in lods:
        root.children.link(lod)
    rigs = []
    for i in range(2):
        rig = bpy.data.objects.new(f"RemovalRig{i}", bpy.data.armatures.new(f"RemovalArmature{i}"))
        root.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bone = rig.data.edit_bones.new(f"Bone{i}")
        bone.tail = (0, 0, 1)
        bpy.ops.object.mode_set(mode="OBJECT")
        rig.pose.bones[0].rotation_mode = "XYZ"
        rig.pose.bones[0].rotation_euler.y = 0.25 * (i + 1)
        rig.select_set(False)
        rigs.append(rig)
    material = bpy.data.materials.new("RemovalMaterial")
    if not material.node_tree:
        material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    node.inputs["Roughness"].driver_add("default_value").driver.expression = "0.37"
    group = bpy.data.node_groups.new("RemovalDrivenGroup", "ShaderNodeTree")
    value = group.nodes.new("ShaderNodeValue")
    value.outputs[0].driver_add("default_value").driver.expression = "0.73"
    material.node_tree.nodes.new("ShaderNodeGroup").node_tree = group
    meshes = []
    for i, lod in enumerate(lods):
        mesh = bpy.data.meshes.new(f"RemovalMeshData{i}")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 1)], [], [(0, 1, 2)])
        mesh.uv_layers.new(name="DiffuseUV")
        for loop, uv in zip(mesh.uv_layers[0].data, [(0, 0), (1, 0), (0, 1)], strict=True):
            loop.uv = uv
        mesh.materials.append(material)
        obj = bpy.data.objects.new(f"RemovalMesh{i}", mesh)
        lod.objects.link(obj)
        obj.parent = rigs[0]
        obj.location.x = i + 0.5
        obj.shape_key_add(name="Basis")
        shape = obj.shape_key_add(name="Expression")
        shape.data[1].co.z += 0.5
        shape.driver_add("value").driver.expression = "0.6"
        for j, rig in enumerate(rigs):
            obj.vertex_groups.new(name=f"Bone{j}").add([0, 1, 2], 1.0, "REPLACE")
            mod = obj.modifiers.new(f"Skin{j}", "ARMATURE")
            mod.object = rig
        meshes.append(obj)
    bpy.context.view_layer.update()
    return instance, root, lods, rigs, meshes, material


def test_remove_apply_mesh_shape_preserves_deformation_and_only_lods(skinned_removal_asset):
    _instance, root, lods, rigs, meshes, material = skinned_removal_asset
    # A trailing modifier must not be baked and then applied a second time.
    meshes[0].modifiers.new("Subdivision", "SUBSURF").levels = 1
    bpy.context.view_layer.update()
    expected = {obj.name: _world_vertices(obj) for obj in meshes}
    names = [obj.name for obj in meshes]
    obsolete_names = [rig.name for rig in rigs]
    collection_names = [root.name, *(lod.name for lod in lods)]
    original_data = [obj.data.name for obj in meshes]
    helper = bpy.data.objects.new("RemovalHelper", None)
    root.objects.link(helper)
    obsolete_names.append(helper.name)
    meshes[0]["dna_merge_head"] = helper
    meshes[1].hide_set(True)
    bpy.context.view_layer.layer_collection.children[root.name].children[lods[1].name].exclude = True
    material_name = material.name

    assert bpy.ops.character_dna.rig_instance_entry_remove(
        active_index=0, delete_associated_data=False, apply_mesh_shape=True
    ) == {"FINISHED"}
    bpy.context.view_layer.update()

    assert not bpy.context.scene.character_dna.rig_instance_list
    assert all(bpy.data.objects.get(name) is None for name in obsolete_names)
    assert all(bpy.data.collections.get(name) is None for name in collection_names)
    assert all(bpy.data.meshes.get(name) is None for name in original_data)
    assert bpy.data.materials.get(material_name) is None
    assert bpy.data.node_groups.get("RemovalDrivenGroup") is None
    for name in names:
        obj = bpy.data.objects[name]
        assert list(obj.users_collection) == [bpy.context.scene.collection]
        assert obj.parent is None
        assert not obj.animation_data
        assert not obj.data.shape_keys
        assert all(mod.type != "ARMATURE" for mod in obj.modifiers)
        assert "dna_merge_head" not in obj
        assert obj.data.uv_layers[0].name == "DiffuseUV"
        assert tuple(obj.data.uv_layers[0].data[1].uv) == (1, 0)
        assert not obj.active_material.node_tree.animation_data
        group = next(node.node_tree for node in obj.active_material.node_tree.nodes if node.type == "GROUP")
        assert not group.animation_data
        assert group.nodes[0].outputs[0].default_value == pytest.approx(0.73)
        roughness = obj.active_material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value
        assert roughness == pytest.approx(0.37)
        for actual, before in zip(_world_vertices(obj), expected[name], strict=True):
            assert actual == pytest.approx(before, abs=1e-6)
    assert bpy.data.objects[names[1]].hide_get()
    assert len(bpy.data.objects[names[0]].modifiers) == 1


def test_remove_apply_mesh_failure_preserves_asset(skinned_removal_asset, monkeypatch):
    from character_dna.utilities import rig_removal

    _instance, _root, _lods, _rigs, meshes, _material = skinned_removal_asset
    before = set(bpy.data.user_map())
    originals = {obj: obj.data for obj in meshes}

    def fail(*_args):
        raise ValueError("Capture failed")

    monkeypatch.setattr(rig_removal, "_static_material", fail)
    with pytest.raises(RuntimeError, match="Capture failed"):
        bpy.ops.character_dna.rig_instance_entry_remove(
            active_index=0, delete_associated_data=False, apply_mesh_shape=True
        )
    assert set(bpy.data.user_map()) == before
    assert len(bpy.context.scene.character_dna.rig_instance_list) == 1
    assert all(obj.data == mesh for obj, mesh in originals.items())
    assert all(len(obj.modifiers) == 2 for obj in meshes)


def test_remove_apply_mesh_preserves_shared_material(skinned_removal_asset):
    _instance, _root, _lods, _rigs, meshes, material = skinned_removal_asset
    unrelated = bpy.data.objects.new("UnrelatedMesh", meshes[0].data)
    bpy.context.scene.collection.objects.link(unrelated)
    original_data = unrelated.data
    meshes[0].material_slots[0].link = "OBJECT"
    meshes[0].material_slots[0].material = material
    assert bpy.ops.character_dna.rig_instance_entry_remove(
        active_index=0, delete_associated_data=False, apply_mesh_shape=True
    ) == {"FINISHED"}
    assert unrelated.data == original_data
    assert original_data.shape_keys is not None
    assert unrelated.active_material == material
    assert material.node_tree.animation_data.drivers
    assert meshes[0].data != original_data
    assert meshes[0].active_material != material
    assert not meshes[0].active_material.node_tree.animation_data


@pytest.mark.parametrize("delete_all", [False, True])
def test_remove_apply_mesh_is_explicit_and_delete_all_takes_precedence(skinned_removal_asset, delete_all):
    _instance, root, _lods, _rigs, meshes, _material = skinned_removal_asset
    name = meshes[0].name
    root_name = root.name
    assert bpy.ops.character_dna.rig_instance_entry_remove(
        active_index=0, delete_associated_data=delete_all, apply_mesh_shape=delete_all
    ) == {"FINISHED"}
    if delete_all:
        assert bpy.data.objects.get(name) is None
        assert bpy.data.collections.get(root_name) is None
    else:
        assert bpy.data.objects.get(name) == meshes[0]
        assert meshes[0].data.shape_keys is not None
        assert len(meshes[0].modifiers) == 2
        assert bpy.data.collections.get(root_name) == root


def test_remove_apply_mesh_preserves_native_facial_pose(load_head_only_dna):
    from character_dna.runtime import engine
    from character_dna.utilities import get_active_rig_instance

    instance = get_active_rig_instance()
    instance.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.7
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    root = bpy.data.collections[instance.name]
    meshes = [obj for child in root.children if "_lod" in child.name for obj in child.all_objects if obj.type == "MESH"]
    expected = {obj.name: _world_vertices(obj) for obj in meshes}
    assert expected
    carriers = [obj.name for obj in engine.carriers(instance)]
    assert carriers
    assert bpy.ops.character_dna.rig_instance_entry_remove(
        active_index=0, delete_associated_data=False, apply_mesh_shape=True
    ) == {"FINISHED"}
    bpy.context.view_layer.update()
    assert all(bpy.data.objects.get(name) is None for name in carriers)
    for name, coordinates in expected.items():
        obj = bpy.data.objects[name]
        assert list(obj.users_collection) == [bpy.context.scene.collection]
        assert not obj.data.shape_keys
        for before, after in zip(coordinates, _world_vertices(obj), strict=True):
            assert after == pytest.approx(before, abs=1e-6)


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

    # Restore the operation guards before the owned rig fixture releases native state.
    with monkeypatch.context() as operation_patches:
        operation_patches.setattr(controller, "rebuild", rebuild)
        operation_patches.setattr(
            controller, "reconcile", lambda: pytest.fail("Duplicate must not reconcile other rigs")
        )
        operation_patches.setattr(
            engine, "invalidate", lambda: pytest.fail("Duplicate must not invalidate other records")
        )
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

            assert {
                carrier.as_pointer() for carrier in engine.carriers(instances["UnrelatedCopy"])
            } == unrelated_pointers
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
