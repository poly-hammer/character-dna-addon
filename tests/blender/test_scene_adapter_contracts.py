"""Scene adapters retain their repeat, missing-component and cleanup contracts."""

import json

from types import SimpleNamespace
from unittest.mock import Mock

import bpy
import pytest

from character_dna.components.body import CharacterComponentBody
from character_dna.components.head import CharacterComponentHead
from character_dna.constants import MASKS_TEXTURE
from character_dna.runtime import engine
from character_dna.utilities import constrain_face_board_to_head, get_topology_texture_constants
from character_dna.utilities.action import flush_texture_mask_keyframes_to_action, get_channel_bag


@pytest.mark.parametrize("kind", ["head", "body"])
def test_driven_names_match_native_outputs_with_one_name_lookup_per_joint(kind):
    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from character_dna.rig_instance import ATTR_COUNT_PER_EULER_JOINT, RigInstance
    from constants import TEST_DNA_FOLDER

    reader = get_dna_reader(TEST_DNA_FOLDER / "ada" / f"{kind}.dna")
    try:
        expected = {
            reader.getJointName(attribute // ATTR_COUNT_PER_EULER_JOINT)
            for solver in range(reader.getRBFSolverCount())
            for pose in reader.getRBFSolverPoseIndices(solver)
            for attribute in reader.getRBFPoseJointOutputIndices(pose)
        }
        observed = Mock(spec=type(reader), wraps=reader)
        instance = SimpleNamespace(
            data={}, cache_key=lambda component, name: (component, name), **{f"{kind}_dna_reader": observed}
        )
        query = getattr(RigInstance, f"{kind}_driven_bone_names").fget
        assert set(query(instance)) == expected
        assert observed.getJointName.call_count <= reader.getJointCount()
        calls = observed.getJointName.call_count
        assert set(query(instance)) == expected
        assert observed.getJointName.call_count == calls
        instance.data.clear()
        assert set(query(instance)) == expected
        assert observed.getJointName.call_count == 2 * calls
    finally:
        release_dna_handle(reader)


def armature(name, bone_name):
    bpy.ops.object.armature_add()
    obj = bpy.context.object
    obj.name = name
    obj.data.bones[0].name = bone_name
    return obj


def test_document_view_cache_evicts_oldest_and_retains_one_oversized_document(monkeypatch):
    from collections import OrderedDict

    from character_dna.dna_io import document

    monkeypatch.setattr(document, "_read_cache", OrderedDict())
    monkeypatch.setattr(document, "MAX_VIEW_CACHE_BYTES", 5)
    document._cache_read("first", {"id": "A"}, {"head": b"123"})
    document._cache_read("second", {"id": "B"}, {"body": b"456"})
    assert list(document._read_cache) == ["second"]
    manifest, parts = {"id": "B", "values": [1]}, {"head": b"123456"}
    document._cache_read("replacement", manifest, parts)
    manifest["values"].append(2)
    parts["head"] = b"changed"
    assert list(document._read_cache) == ["replacement"]
    assert document._read_cache["replacement"] == ({"id": "B", "values": [1]}, {"head": b"123456"})


def test_source_reload_preserves_deferred_channels(empty_character_scene, compact_two_target_asset):
    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from character_dna.dna_io.misc import create_shape_key
    from character_dna.editors.converter import source_views

    objects = []
    for name in ("BeforeReload", "AfterReload"):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new(name, mesh)
        empty_character_scene.collection.objects.link(obj)
        objects.append(obj)
    old, new = objects
    reader = get_dna_reader(compact_two_target_asset / "head.dna")
    try:
        channel = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, 0))
        key = create_shape_key(0, 0, old, reader, channel, prefix="head_lod0_mesh__", linear_modifier=0.01)
        expected = [tuple(vertex.co) for vertex in key.data]
        source_views._restore_keys(old, new, reader, 0)
        assert list(new.data.shape_keys.key_blocks.keys()) == ["Basis", key.name]
        assert [tuple(vertex.co) for vertex in new.data.shape_keys.key_blocks[key.name].data] == expected
        assert reader.getBlendShapeTargetCount(0) == 2
    finally:
        release_dna_handle(reader)


def test_body_import_ignores_include_body_option(empty_character_scene, compact_dna_asset):
    from character_dna.utilities import get_active_rig_instance

    source = compact_dna_asset / "body.dna"
    assert bpy.ops.character_dna.import_dna(
        filepath=str(source),
        include_body=True,
        import_bones=False,
        import_mesh=False,
        import_face_board=False,
        import_shape_keys=False,
        import_materials=False,
    ) == {"FINISHED"}
    instance = get_active_rig_instance()
    assert source.samefile(bpy.path.abspath(instance.body_dna_file_path))
    assert instance.head_rig is None and instance.head_dna_reader is None


def test_head_only_pose_presets_and_absent_body_are_safe(minimal_character, tmp_path):
    instance = minimal_character
    instance.body_rig = None
    instance["head_rig"] = armature("PresetHead", "head")
    instance["face_board"] = armature("PresetBoard", "CTRL_C_jaw")
    pose = tmp_path / "pose.json"
    pose.write_text(json.dumps({"face_board": {"CTRL_C_jaw": {"location": [0, 0.6, 0]}}, "body": {}}))
    head = CharacterComponentHead(rig_instance=instance, dna_file_path=tmp_path / "missing.dna")
    assert head.dna_reader is None and head.dna_importer is None
    head.scene_properties = SimpleNamespace(face_board=SimpleNamespace(face_pose_previews=str(tmp_path / "pose.png")))
    head.set_pose()
    assert instance.face_board.pose.bones["CTRL_C_jaw"].location.y == pytest.approx(0.6)
    body = CharacterComponentBody(rig_instance=instance, dna_file_path=tmp_path / "missing.dna", component_type="body")
    body.set_pose()
    assert instance.body_rig is None


def test_texture_reimport_purges_previous_placeholder_images(minimal_character, tmp_path):
    component = CharacterComponentHead(rig_instance=minimal_character, dna_file_path=tmp_path / "missing.dna")
    topology = get_topology_texture_constants()
    assert topology is not None
    names = (MASKS_TEXTURE, topology.HEAD_TOPOLOGY_TEXTURE)
    for name in names:
        bpy.data.images.new(name, width=1, height=1)
    component._purge_existing_materials()
    assert all(bpy.data.images.get(name) is None for name in names)
    component._purge_existing_materials()
    assert all(bpy.data.images.get(name) is None for name in names)


def test_reconstraining_face_board_reuses_child_of_and_preserves_other_constraints(empty_character_scene):
    head = armature("ConstraintHead", "head")
    board = armature("ConstraintBoard", "control")
    bone = board.pose.bones["control"]
    other = bone.constraints.new("LIMIT_LOCATION")
    for _ in range(2):
        constrain_face_board_to_head(head, None, board, "control")
        constraints = list(bone.constraints)
        assert len(constraints) == 2 and constraints[0] == other
        assert constraints[1].type == "CHILD_OF" and constraints[1].target == head
        assert constraints[1].subtarget == "head"


@pytest.mark.parametrize("has_record", [False, True])
def test_paused_native_driver_preserves_epoch_without_solving(monkeypatch, has_record):
    original = SimpleNamespace(as_pointer=lambda: 123, get={"authoring_paused": True}.get)
    owner = SimpleNamespace(original=original, get={"epoch": 8.5}.get)
    monkeypatch.setattr(engine, "_records", {123: {"carrier": original}} if has_record else {})
    assert engine.solve(owner, None) == 8.5


def test_rebaking_texture_mask_replaces_curve_in_existing_action(empty_character_scene):
    tree = bpy.data.node_groups.new("MaskOwner", "ShaderNodeTree")
    group = bpy.data.node_groups.new("MaskInputs", "ShaderNodeTree")
    group.interface.new_socket(name="mask", in_out="INPUT", socket_type="NodeSocketFloat")
    node = tree.nodes.new("ShaderNodeGroup")
    node.node_tree = group
    action = None
    for samples in ([(1, 0.0), (2, 0.6)], [(1, 0.8), (3, 0.2)]):
        result = flush_texture_mask_keyframes_to_action(node, {"mask": samples}, "RebakedMask")
        if action is not None:
            assert result == action
        action = result
        bag = get_channel_bag(action, tree.name, "NODETREE")
        assert len(bag.fcurves) == 1
        curve = bag.fcurves[0]
        assert curve.data_path == node.inputs["mask"].path_from_id("default_value")
        assert [value for point in curve.keyframe_points for value in point.co] == pytest.approx(
            [value for sample in samples for value in sample]
        )


@pytest.mark.parametrize("method,align", [("overwrite", True), ("calibrate", False)])
def test_calibration_respects_disabled_seam_alignment(minimal_character, method, align):
    from character_dna.dna_io.calibrator import DNACalibrator

    minimal_character.output.method = method
    minimal_character.output.align_head_and_body = align
    calibrator = object.__new__(DNACalibrator)
    calibrator._instance = minimal_character
    calibrator._component_type = "head"
    calibrator._seam_follower = "head"
    positions = {0: [[1.0, 2.0, 3.0]]}
    assert calibrator._get_body_mesh_lookup(0, f"{minimal_character.name}_head_lod0_mesh", {"0": {0: 0}}) == {}
    calibrator._align_seams(positions, positions)
    assert positions == {0: [[1.0, 2.0, 3.0]]}


def test_finishing_shape_import_without_session_is_idempotent(empty_character_scene):
    from character_dna.editors.shape_key_editor.operators import ImportShapeKeys

    operation = SimpleNamespace(_timer=None)
    for _ in range(2):
        assert ImportShapeKeys.finish(operation, bpy.context) == {"FINISHED"}
        assert operation._import_session is None
        assert bpy.context.window_manager.character_dna.progress == 1


@pytest.mark.parametrize("include_body", [False, True])
def test_head_import_with_missing_optional_body_is_valid(
    empty_character_scene, compact_dna_asset, tmp_path, include_body
):
    from character_dna.utilities import get_active_rig_instance

    head = tmp_path / "head.dna"
    head.write_bytes((compact_dna_asset / "head.dna").read_bytes())
    assert not (tmp_path / "body.dna").exists()
    assert bpy.ops.character_dna.import_dna(
        filepath=str(head),
        include_body=include_body,
        import_bones=False,
        import_mesh=False,
        import_face_board=False,
        import_shape_keys=False,
        import_materials=False,
    ) == {"FINISHED"}
    instance = get_active_rig_instance()
    assert head.samefile(bpy.path.abspath(instance.head_dna_file_path))
    assert instance.head_rig is None
    assert instance.body_dna_reader is None
    assert instance.body_rig is None and instance.body_mesh is None


@pytest.mark.parametrize("previous_mesh", [False, True])
def test_head_only_import_spacing_uses_previous_mesh_bounds(
    empty_character_scene, compact_dna_asset, previous_mesh, tmp_path
):
    from character_dna.utilities import (
        get_active_rig_instance,
        get_bounding_box_left_x,
        get_bounding_box_width,
    )

    (tmp_path / "head.dna").write_bytes((compact_dna_asset / "head.dna").read_bytes())
    (tmp_path / "ExportManifest.json").write_text(json.dumps({"metaHumanName": "PreviousHead"}))
    options = {
        "filepath": str(tmp_path / "head.dna"),
        "include_body": False,
        "import_face_board": False,
        "import_shape_keys": False,
        "import_materials": False,
    }
    assert bpy.ops.character_dna.import_dna(**options, import_bones=False, import_mesh=previous_mesh) == {"FINISHED"}
    previous = get_active_rig_instance()
    previous_name = previous.name
    expected = (
        get_bounding_box_left_x(previous.head_mesh) - get_bounding_box_width(previous.head_mesh) / 2
        if previous_mesh
        else 0
    )
    options["filepath"] = str(compact_dna_asset / "head.dna")
    assert bpy.ops.character_dna.import_dna(**options, import_bones=True, import_mesh=True) == {"FINISHED"}
    current = get_active_rig_instance()
    previous = bpy.context.scene.character_dna.rig_instance_list[previous_name]
    assert current != previous
    assert current.head_rig.location.x == pytest.approx(expected)
    assert current.body_rig is None and previous.body_rig is None


def test_texture_masks_copy_native_outputs_to_matching_sockets(empty_character_scene):
    from character_dna.rig_component import RigComponent

    material = bpy.data.materials.new("NativeMaskContract")
    material.use_nodes = True
    group = bpy.data.node_groups.new("NativeMaskInputs", "ShaderNodeTree")
    for name in ("cheek", "jaw", "outside_outputs", "wm1.head_wm1_jawOpen_msk"):
        group.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocketFloat")
    node = material.node_tree.nodes.new("ShaderNodeGroup")
    node.name = "head_texture_logic"
    node.node_tree = group
    reader = SimpleNamespace(
        getAnimatedMapCount=lambda: 4,
        getAnimatedMapName=lambda index: ("cheek", "jaw", "missing", "outside_outputs")[index],
    )
    component = RigComponent(SimpleNamespace(head_material=material, head_dna_reader=reader), "head")
    state = SimpleNamespace(getAnimatedMapOutputs=lambda: (0.25, 0.75, 0.5))
    assert component.update_texture_masks(state=state) == [("cheek", 0.25), ("jaw", 0.75)]
    assert [node.inputs[name].default_value for name in ("cheek", "jaw", "outside_outputs")] == [0.25, 0.75, 0.0]
