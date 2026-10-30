"""DNA identities survive Blender's key-name limit and runtime binding setup."""

import hashlib
import json

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from test_merge_split import character as _character, coordinates

from character_dna import shape_names
from character_dna.dna_io.misc import create_shape_key
from character_dna.dna_io.shape_migration import recover_mesh
from character_dna.editors.converter import scene_merge
from character_dna.runtime import controller, engine
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


character = _character


@pytest.fixture
def head_character(rig_factory, tmp_path):
    # This fixture checks shape-channel values, not skin deformation. Keep all
    # original mesh/target geometry and native behavior, with only the scene
    # input bones and observed outputs needed to bind the runtime.
    with rig_factory.lease(RigProfile(components=("head",), pose_bones=STATE_BONES), tmp_path) as instance:
        yield instance


@pytest.fixture
def mesh():
    data = bpy.data.meshes.new("name_test")
    data.from_pydata([(0, 0, 0), (1, 0, 0)], [], [])
    obj = bpy.data.objects.new("name_test", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    yield obj
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(data)


@pytest.mark.parametrize("channel", ["brow_down_L", "a" * 47, "a" * 48, "é" * 30, "funnel__" + "L" * 60])
def test_lossless_names_and_mesh_ownership(mesh, channel):
    keys = mesh.data.shape_keys
    name = "head_lod0_mesh__" + channel
    stored = shape_names.ensure_name(keys, name)
    block = mesh.shape_key_add(name=stored)
    assert block.name == stored
    assert len(stored.encode()) <= 63
    assert stored.startswith("head_lod0_mesh__")
    assert shape_names.logical_name(keys, stored) == name
    assert shape_names.find_key(keys, name) == block
    assert shape_names.ensure_name(keys, name) == stored
    assert shape_names.mesh_name(stored, {"head_lod0_mesh": "head", "body_lod0_mesh": "body"}) == "head_lod0_mesh"
    assert (stored == name) == (len(name.encode()) <= 63)
    copy = mesh.data.copy()
    try:
        assert shape_names.logical_name(copy.shape_keys, stored) == name
    finally:
        bpy.data.meshes.remove(copy)


@pytest.mark.parametrize("literal_first", [True, False])
def test_literal_hash_collision_preserves_both_identities(mesh, literal_first):
    keys = mesh.data.shape_keys
    long = "head_lod0_mesh__" + "corrective" * 10
    literal = "head_lod0_mesh__h_" + hashlib.sha256(long.encode()).hexdigest()[:12]
    names = [literal, long] if literal_first else [long, literal]
    for index, name in enumerate(names):
        block = mesh.shape_key_add(name=shape_names.ensure_name(keys, name))
        block.data[0].co.x = index + 1
    assert shape_names.find_key(keys, names[0]).data[0].co.x == 1
    assert shape_names.find_key(keys, names[1]).data[0].co.x == 2
    assert len(keys.key_blocks) == 3
    assert not any(key.name.endswith(".001") for key in keys.key_blocks)


def test_hash_collision_extends_digest(mesh, monkeypatch):
    monkeypatch.setattr(shape_names.hashlib, "sha256", lambda _data: SimpleNamespace(hexdigest=lambda: "a" * 64))
    keys = mesh.data.shape_keys
    names = ["head_lod0_mesh__" + c * 80 for c in "xy"]
    blocks = [mesh.shape_key_add(name=shape_names.ensure_name(keys, name)) for name in names]
    assert blocks[0].name != blocks[1].name
    assert [shape_names.logical_name(keys, block.name) for block in blocks] == names


def test_rename_keeps_animation_and_external_drivers(mesh):
    key = mesh.shape_key_add(name="old")
    key.keyframe_insert("value", frame=1)
    other = mesh.shape_key_add(name="other")
    driver = other.driver_add("value").driver
    var = driver.variables.new()
    var.targets[0].id_type = "KEY"
    var.targets[0].id = key.id_data
    var.targets[0].data_path = key.path_from_id("value")
    shape_names.rename_key(key.id_data, key, "head_lod0_mesh__" + "long" * 20)
    assert var.targets[0].data_path == key.path_from_id("value")
    animation = key.id_data.animation_data
    action = animation.action
    paths = [
        curve.data_path
        for layer in action.layers
        for strip in layer.strips
        for bag in strip.channelbags
        for curve in bag.fcurves
    ]
    assert key.path_from_id("value") in paths


def test_recovery_does_not_guess_modified_or_ambiguous_keys(mesh):
    full = "head_lod0_mesh__" + "x" * 60
    key = mesh.shape_key_add(name=full)
    key.data[0].co.x += 0.3
    reader = SimpleNamespace(
        getVertexPositionCount=lambda _mesh: 2,
        getMeshName=lambda _mesh: "head_lod0_mesh",
        getBlendShapeTargetCount=lambda _mesh: 1,
        getBlendShapeChannelIndex=lambda *_args: 0,
        getBlendShapeChannelName=lambda _channel: full.split("__", 1)[1],
        getBlendShapeTargetVertexIndices=lambda *_args: [0],
        getBlendShapeTargetDeltaXs=lambda *_args: [0.1],
        getBlendShapeTargetDeltaYs=lambda *_args: [0.0],
        getBlendShapeTargetDeltaZs=lambda *_args: [0.0],
    )
    before = coordinates(key.data)
    assert recover_mesh(mesh, reader, 0, 1.0) == {}
    np.testing.assert_array_equal(coordinates(key.data), before)
    assert shape_names.find_key(mesh.data.shape_keys, full) is None


def test_all_42_long_head_targets_import_recover_bind_and_split(character, monkeypatch):
    instance = character
    reader = instance.head_dna_reader
    mesh = instance.head_mesh
    index = next(i for i, obj in instance.component("head").mesh_index_lookup.items() if obj == mesh)
    prefix = reader.getMeshName(index) + "__"
    targets = [
        (target, reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(index, target)))
        for target in range(reader.getBlendShapeTargetCount(index))
        if len((prefix + reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(index, target))).encode())
        > 63
    ]
    assert len(targets) == 42
    with controller.authoring_operation(instance):
        for target, name in targets:
            create_shape_key(target, index, mesh, reader, name, prefix=prefix, linear_modifier=0.01)
        keys = mesh.data.shape_keys
        before = {prefix + name: coordinates(shape_names.find_key(keys, prefix + name).data) for _, name in targets}
        # Re-import is idempotent, with no Blender truncation/.001 duplicates.
        for target, name in targets:
            create_shape_key(target, index, mesh, reader, name, prefix=prefix, linear_modifier=0.01)
        assert len(keys.key_blocks) == 43
        # Reproduce the old importer, including Blender's .001 collisions.
        keys[shape_names.PROPERTY] = "{}"
        for _, name in targets:
            block = next(
                block
                for block in list(keys.key_blocks)[1:]
                if np.array_equal(coordinates(block.data), before[prefix + name])
            )
            block.name = prefix + name
        recovered = recover_mesh(mesh, reader, index, 0.01)
        assert len(recovered) == 42
        for name, coords in before.items():
            np.testing.assert_array_equal(coordinates(shape_names.find_key(keys, name).data), coords)
        proxy = scene_merge.merge(instance)
        raw = json.loads(proxy["dna_merge_metadata"])
        assert raw["version"] == 3
        assert all("kind" not in entry for entry in raw["keys"].values())
        assert all(key.name.startswith(prefix) for key in list(proxy.data.shape_keys.key_blocks)[1:])
        scene_merge.split(instance)
        instance.destroy_references()
    assert engine.active(instance)
    assert sum(len(plan[1]) for plan in instance.component("head").shape_key_apply_plan) == 42
    assert engine.binding_issues(instance) == []
    for name, coords in before.items():
        np.testing.assert_array_equal(coordinates(shape_names.find_key(keys, name).data), coords)
    plan = instance.component("head").shape_key_apply_plan
    assert len(instance.component("head").shape_key_blocks_by_name) == 42
    full = prefix + targets[0][1]
    channel = instance.component("head").channel_name_to_index_lookup[full]
    block = shape_names.find_key(keys, full)

    def fail(*_args, **_kwargs):
        raise AssertionError("Name resolution reached a warmed frame evaluation")

    monkeypatch.setattr(shape_names, "logical_name", fail)
    monkeypatch.setattr(shape_names, "stored_name", fail)
    monkeypatch.setattr(shape_names, "find_key", fail)
    monkeypatch.setattr(shape_names, "ensure_name", fail)
    for frame in range(1, 5):
        assert engine.preview_shape(instance, channel, frame / 5)
        bpy.context.view_layer.update()
        assert keys.evaluated_get(bpy.context.view_layer.depsgraph).key_blocks[block.name].value == pytest.approx(
            frame / 5
        )
        assert instance.shape_key_editor.shape_key_list[full].value == pytest.approx(frame / 5)
        bpy.context.scene.frame_set(frame)
        instance.component("head").update_shape_keys()
        assert instance.component("head").shape_key_apply_plan is plan


def test_legacy_surface_upgrade_preserves_uncommitted_sculpt(character):
    instance = character
    obj = instance.head_mesh
    reader = instance.head_dna_reader
    target = next(
        i
        for i in range(reader.getBlendShapeTargetCount(0))
        if len(("head_lod0_mesh__" + reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, i))).encode())
        > 63
    )
    full = "head_lod0_mesh__" + reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, target))
    with controller.authoring_operation(instance):
        key = create_shape_key(
            target, 0, obj, reader, full.split("__", 1)[1], prefix="head_lod0_mesh__", linear_modifier=0.01
        )
        key.name = full  # legacy Blender truncation
        obj.data.shape_keys[shape_names.PROPERTY] = "{}"
        proxy = scene_merge.merge(instance)
        metadata = scene_merge.read_metadata(proxy)
        block = proxy.data.shape_keys.key_blocks[1]
        record = metadata["keys"].pop(block.name)
        block.name = "head_1234567890_head_lod0_mesh__long_old_alias"
        metadata["keys"][block.name] = record
        metadata["version"] = 2
        metadata["indices"] = {"0": 0}  # old skin-weight loop overwrote these indices
        proxy["dna_merge_metadata"] = json.dumps(metadata)
        block.data[-1].co.x += 0.02
        sculpt = coordinates(block.data)
        source = coordinates(key.data)
        instance.destroy_references()
    migrated = shape_names.find_key(proxy.data.shape_keys, full)
    assert migrated is not None
    assert instance.shape_key_editor.shape_key_list.find(full) >= 0
    np.testing.assert_array_equal(coordinates(migrated.data), sculpt)
    source_key = shape_names.find_key(obj.data.shape_keys, full)
    np.testing.assert_array_equal(coordinates(source_key.data), source)
    driver = proxy.data.shape_keys.animation_data.drivers.find(migrated.path_from_id("value")).driver
    assert driver.variables[0].targets[0].id == obj.data.shape_keys
    assert driver.variables[0].targets[0].data_path == source_key.path_from_id("value")
    metadata = scene_merge.read_metadata(proxy)
    assert metadata["version"] == 3
    assert metadata["indices"] == {"head": 0, "body": 0}
    assert metadata["keys"][migrated.name]["kind"] == "head"
    with controller.authoring_operation(instance):
        scene_merge.validate_split(instance)
        scene_merge.split(instance, apply=False)


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("method", ["export", "calibrate"])
def test_long_shape_written_with_full_channel_identity(mesh, method, kind):
    from character_dna.bindings import dna
    from character_dna.dna_io.calibrator import DNACalibrator
    from character_dna.dna_io.exporter import DNAExporter
    from character_dna.dna_io.misc import release_dna_handle

    mesh_name = f"{kind}_lod0_mesh"
    channel_name = "a_long_corrective__" + "expression" * 10
    full = mesh_name + "__" + channel_name
    mesh.name = "test_" + mesh_name
    block = mesh.shape_key_add(name=shape_names.ensure_name(mesh.data.shape_keys, full))
    block.data[0].co.x += 0.025
    reader = SimpleNamespace(
        getLODCount=lambda: 1,
        getBlendShapeChannelIndicesForLOD=lambda _lod: [0],
        getMeshIndicesForLOD=lambda _lod: [0],
        getMeshName=lambda _mesh: mesh_name,
        getBlendShapeTargetCount=lambda _mesh: 1,
        getBlendShapeChannelIndex=lambda *_args: 0,
        getBlendShapeChannelName=lambda _channel: channel_name,
    )
    stream = dna.MemoryStream()
    writer = dna.BinaryStreamWriter(stream)
    result = None
    try:
        writer.setMeshName(0, mesh_name)
        writer.setBlendShapeChannelName(0, channel_name)
        writer.setBlendShapeChannelIndex(0, 0, 0)
        exporter = object.__new__(DNAExporter if method == "export" else DNACalibrator)
        exporter._dna_reader = reader
        exporter._dna_writer = writer
        exporter._component_type = kind
        exporter._prefix = "test"
        exporter._linear_modifier = 0.01
        exporter._mesh_blend_shape_channel_pairs = []
        if method == "export":
            exporter._write_mesh_shape_keys_from_scene(0, 0, mesh, mesh_name)
        else:
            exporter.calibrate_shape_keys()
        writer.write()
        assert dna.Status.isOk(), dna.Status.get().message
        stream.seek(0)
        result = dna.BinaryStreamReader(stream, dna.DataLayer_All)
        result.read()
        assert dna.Status.isOk(), dna.Status.get().message
        assert result.getBlendShapeChannelName(0) == channel_name
        assert list(result.getBlendShapeTargetVertexIndices(0, 0)) == [0]
        assert list(result.getBlendShapeTargetDeltaXs(0, 0)) == pytest.approx([2.5])
    finally:
        release_dna_handle(result)
        release_dna_handle(writer)


def test_frame_cost_uses_same_numeric_bindings_for_compact_and_hashed_names(head_character):
    """Name storage must preserve every native channel at distinct sampled frames."""
    instance = head_character
    obj = instance.head_mesh
    reader = instance.head_dna_reader
    prefix = "head_lod0_mesh__"
    with controller.authoring_operation(instance):
        for target in range(reader.getBlendShapeTargetCount(0)):
            name = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, target))
            if len((prefix + name).encode()) > 63:
                create_shape_key(target, 0, obj, reader, name, prefix=prefix, linear_modifier=0.01)
        instance.destroy_references()
    keys = obj.data.shape_keys
    originals = [(block, shape_names.logical_name(keys, block.name), block.name) for block in list(keys.key_blocks)[1:]]
    assert len(originals) == 42
    channels = [
        instance.component("head").channel_name_to_index_lookup[logical] for _block, logical, _name in originals
    ]
    raw_control = next(
        index
        for index in range(reader.getRawControlCount())
        if reader.getRawControlName(index) == "CTRL_expressions.jawOpen"
    )
    for mode in ("hashed", "compact", "compact", "hashed"):
        with controller.authoring_operation(instance):
            mapping = {}
            for index, (block, logical, hashed) in enumerate(originals):
                block.name = hashed if mode == "hashed" else prefix + "s_" + str(index)
                mapping[block.name] = logical
            keys[shape_names.PROPERTY] = json.dumps(mapping)
            instance.destroy_references()
        assert sum(len(plan[1]) for plan in instance.component("head").shape_key_apply_plan) == 42
        # The old loop timed 340 frames but asserted no frame outputs. Keep
        # both transitions and idempotent renaming, with an actual SDK oracle
        # for all 42 channels at each sample. Complete-suite timing lives in
        # the dedicated benchmark runner rather than this correctness test.
        for frame, value in ((1, 0.0), (17, 0.35), (42, 0.8)):
            state = instance.head_instance
            state.setRawControl(raw_control, value)
            instance.head_manager.calculate(state)
            expected = np.asarray(state.getBlendShapeOutputs())[channels]
            bpy.context.scene.frame_set(frame)
            assert engine.preview_controls(instance, state)
            bpy.context.view_layer.update()
            evaluated = keys.evaluated_get(bpy.context.view_layer.depsgraph)
            actual = [shape_names.find_key(evaluated, logical).value for _block, logical, _name in originals]
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


def test_alias_registry_survives_blend_reload(tmp_path):
    bpy.ops.wm.read_homefile(use_empty=True)
    data = bpy.data.meshes.new("portable_names")
    data.from_pydata([(0, 0, 0)], [], [])
    obj = bpy.data.objects.new("portable_names", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    name = "body_lod0_mesh__" + "long_Δ_corrective__" * 6
    block = obj.shape_key_add(name=shape_names.ensure_name(data.shape_keys, name))
    block.data[0].co.x = 0.123
    stored = block.name
    path = tmp_path / "portable_names.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path))
    try:
        obj = bpy.data.objects["portable_names"]
        block = shape_names.find_key(obj.data.shape_keys, name)
        assert block.name == stored
        assert shape_names.logical_name(obj.data.shape_keys, stored) == name
        assert block.data[0].co.x == pytest.approx(0.123)
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def test_long_surface_edits_use_logical_names_and_rollback(character):
    from character_dna.editors.converter import surface_session

    instance = character
    obj = instance.head_mesh
    reader = instance.head_dna_reader
    target = next(
        i
        for i in range(reader.getBlendShapeTargetCount(0))
        if len("head_lod0_mesh__" + reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, i))) > 63
    )
    name = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, target))
    full = "head_lod0_mesh__" + name
    with controller.authoring_operation(instance):
        key = create_shape_key(target, 0, obj, reader, name, prefix="head_lod0_mesh__", linear_modifier=0.01)
        proxy = scene_merge.merge(instance)
        snapshots = scene_merge.snapshot_sources(instance)
        before = coordinates(key.data)
        block = shape_names.find_key(proxy.data.shape_keys, full)
        head_ids = [item.value for item in proxy.data.attributes["DNA_head_vertex"].data]
        body_ids = [item.value for item in proxy.data.attributes["DNA_body_vertex"].data]
        vertex = next(i for i, value in enumerate(head_ids) if value >= 0 and body_ids[i] < 0)
        block.data[vertex].co.x += 0.01
        _sources, _metadata, updates = scene_merge._prepare_surface(instance, proxy)
        assert [name for _obj, name, _coords, _settings in updates] == [full]
        scene_merge.apply_edits(instance, publish=False)
        assert coordinates(key.data)[head_ids[vertex], 0] == pytest.approx(before[head_ids[vertex], 0] + 0.01)
        scene_merge.restore_sources(snapshots)
        np.testing.assert_array_equal(coordinates(key.data), before)
        assert shape_names.find_key(obj.data.shape_keys, full) == key
        surface_session.refresh(instance)
        scene_merge.split(instance)
        assert instance.component("head").shape_key_blocks_by_name[full] == key
        obj.shape_key_remove(key)
        instance.destroy_references()
        assert full not in instance.component("head").shape_key_blocks_by_name


def test_new_surface_shape_uses_mesh_prefix_without_discarding_foreign_edits(character):
    instance = character
    with controller.authoring_operation(instance):
        proxy = scene_merge.merge(instance)
        block = proxy.shape_key_add(name="head_lod0_mesh__new_sculpt", from_mix=False)
        heads = [item.value for item in proxy.data.attributes["DNA_head_vertex"].data]
        bodies = [item.value for item in proxy.data.attributes["DNA_body_vertex"].data]
        head_vertex = next(i for i, value in enumerate(heads) if value >= 0 and bodies[i] < 0)
        body_vertex = next(i for i, value in enumerate(bodies) if value >= 0 and heads[i] < 0)
        block.data[head_vertex].co.x += 0.01
        original = block.data[body_vertex].co.copy()
        block.data[body_vertex].co.x += 0.02
        with pytest.raises(ValueError, match="cross-component sculpt"):
            scene_merge.split(instance)
        assert scene_merge.proxy_for(instance) == proxy
        assert shape_names.find_key(instance.head_mesh.data.shape_keys, block.name) is None
        block.data[body_vertex].co = original
        scene_merge.split(instance)
        key = shape_names.find_key(instance.head_mesh.data.shape_keys, "head_lod0_mesh__new_sculpt")
        assert key is not None
        assert shape_names.find_key(instance.body_mesh.data.shape_keys, "head_lod0_mesh__new_sculpt") is None
