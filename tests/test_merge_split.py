"""Reversible welded editing and standalone component DNA documents."""

import json
import shutil

from pathlib import Path

import bpy
import numpy as np
import pytest

from character_dna.constants import HEAD_TO_BODY_EDGE_LOOP_FILE_PATH
from character_dna.dna_io import document, get_dna_reader, get_dna_writer, release_dna_handle
from character_dna.editors.converter import scene_merge
from character_dna.runtime import controller, engine
from character_dna.utilities import get_active_rig_instance
from constants import TEST_DNA_FOLDER


@pytest.fixture
def character(tmp_path, request):
    bpy.ops.wm.read_homefile(use_empty=True)
    for name in ("head.dna", "body.dna", "ExportManifest.json"):
        shutil.copyfile(TEST_DNA_FOLDER / "ada" / name, tmp_path / name)
    bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "head.dna"),
        include_body=True,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 or getattr(request, "param", None) == "all_lods" for index in range(8)},
    )
    instance = get_active_rig_instance()
    try:
        yield instance
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def coordinates(data):
    values = np.empty(len(data) * 3, dtype=np.float32)
    data.foreach_get("co", values)
    return values.reshape(-1, 3)


def evaluated_coordinates(obj):
    """Read the visible result, including shape keys and armature deformation."""
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return coordinates(mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def add_keys(instance):
    for kind in ("head", "body"):
        obj = instance.component(kind).mesh
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="custom")
        key.data[0].co.x += 0.015
        key.slider_min = -1
        key.slider_max = 2
        second = obj.shape_key_add(name="relative")
        second.relative_key = key
        second.data[1].co.y += 0.01


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
def test_merge_all_scene_lods_and_switch_without_splitting(character):
    instance = character
    sources = scene_merge.surface_pairs(instance)
    assert len(sources) == 8
    original = {
        obj: (obj.data.as_pointer(), coordinates(obj.data.vertices).copy())
        for _, pair, _ in sources
        for obj in pair.values()
    }
    instance.view_options.active_lod = "lod3"
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    registry = scene_merge.surfaces(instance)
    assert set(registry) == set(range(8))
    for lod, proxy in registry.items():
        assert proxy.hide_get() == (lod != 3)
        assert len(proxy.modifiers) == 2
        assert all(proxy[f"dna_merge_{kind}"].hide_get() for kind in ("head", "body"))
    assert registry[0]["dna_merge_body"] == registry[1]["dna_merge_body"]
    instance.view_options.active_lod = "lod6"
    assert not registry[6].hide_get()
    assert registry[3].hide_get()
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    for obj, (identity, coords) in original.items():
        assert obj.data.as_pointer() == identity
        np.testing.assert_array_equal(coordinates(obj.data.vertices), coords)


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
def test_lod_failure_rolls_back_and_late_lods_reuse_existing_surface(character, monkeypatch):
    instance = character
    pairs = scene_merge.surface_pairs(instance)
    visibility = {obj: obj.hide_get() for _, sources, _ in pairs for obj in sources.values()}
    original_merge = scene_merge._merge_pair

    def failing_merge(instance, lod, sources, indices, states):
        if lod == 4:
            raise ValueError("Simulated LOD4 failure")
        return original_merge(instance, lod, sources, indices, states)

    with monkeypatch.context() as patch:
        patch.setattr(scene_merge, "_merge_pair", failing_merge)
        with pytest.raises(ValueError, match="LOD4"):
            scene_merge.merge(instance)
    assert not scene_merge.surfaces(instance)
    assert all(obj.hide_get() == hidden for obj, hidden in visibility.items())
    with monkeypatch.context() as patch:
        patch.setattr(scene_merge, "surface_pairs", lambda _instance: pairs[:1])
        first = scene_merge.merge(instance)
    scene_merge.adopt_imported_lods(instance)
    assert len(scene_merge.surfaces(instance)) == 8
    assert scene_merge.surfaces(instance)[0] == first
    scene_merge.split(instance)


@pytest.mark.parametrize("commit", [False, True], ids=["revert", "commit"])
def test_raw_jaw_edit_keeps_merged_surface_and_canonical_target(character, commit):
    from character_dna.editors.raw_control_editor import utilities
    from character_dna.editors.raw_control_editor.editor import RawControlEditor

    instance = character
    instance.editor_component = "head"
    instance.raw_control_editor.raw_controls_active_index = instance.raw_control_editor.raw_controls.find(
        "CTRL_expressions.jawOpen"
    )
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    identity = (proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer())
    posed = evaluated_coordinates(proxy)
    assert np.max(np.linalg.norm(posed - coordinates(proxy.data.vertices), axis=1)) > 0.01
    editor = RawControlEditor.for_instance(instance)
    editor.enter(bpy.context)
    assert bpy.context.object == proxy
    np.testing.assert_allclose(evaluated_coordinates(proxy), posed, atol=1e-6)
    for mode in ("POSE", "EDIT", "OBJECT", "SCULPT"):
        instance.raw_control_editor.view_mode = mode
        np.testing.assert_allclose(evaluated_coordinates(proxy), posed, atol=1e-6, err_msg=mode)
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))
    target = next(
        row.scene_object
        for row in instance.raw_control_editor.target_meshes
        if row.scene_object.get("dna_edit_surface")
    )
    assert len(target.data.vertices) == len(instance.head_mesh.data.vertices)
    bpy.ops.object.mode_set(mode="OBJECT")
    reference = proxy.data.shape_keys.key_blocks["DNA_edit_reference"]
    assert reference.lock_shape
    reference_positions = coordinates(reference.data)
    before = coordinates(target.data.vertices).copy()
    index = next(index for index, item in enumerate(proxy.data.attributes["DNA_head_vertex"].data) if item.value == 0)
    proxy.active_shape_key.data[index].co.x += 0.01
    expected = posed.copy()
    expected[index, 0] += 0.01
    np.testing.assert_allclose(evaluated_coordinates(proxy), expected, atol=1e-6)
    np.testing.assert_array_equal(coordinates(reference.data), reference_positions)
    utilities._flush_sculpt_to_mesh(target)
    np.testing.assert_allclose(coordinates(target.data.vertices)[0], before[0] + [0.01, 0, 0], atol=1e-6)
    if commit:
        editor.commit(bpy.context)
    else:
        editor.revert(bpy.context)
    assert scene_merge.proxy_for(instance) == proxy
    assert (proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer()) == identity
    assert "DNA_edit_preview" not in proxy.data.shape_keys.key_blocks
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))
    assert all(mod.show_viewport for mod in proxy.modifiers)
    # A second session must not accumulate a baked pose from the first one.
    instance.raw_control_editor.raw_controls_active_index = instance.raw_control_editor.raw_controls.find(
        "CTRL_expressions.jawOpen"
    )
    posed = evaluated_coordinates(proxy)
    assert np.max(np.linalg.norm(posed - coordinates(proxy.data.vertices), axis=1)) > 0.01
    editor.enter(bpy.context)
    np.testing.assert_allclose(evaluated_coordinates(proxy), posed, atol=1e-6)
    editor.revert(bpy.context)


def test_failed_raw_publication_keeps_merged_sculpt_targets_for_retry(character, monkeypatch):
    from contextlib import contextmanager

    from character_dna.editors.raw_control_editor.editor import RawControlEditor

    instance = character
    instance.raw_control_editor.raw_controls_active_index = instance.raw_control_editor.raw_controls.find(
        "CTRL_expressions.jawOpen"
    )
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    editor = RawControlEditor.for_instance(instance)
    editor.enter(bpy.context)
    targets = [row.scene_object for row in instance.raw_control_editor.target_meshes]
    path = Path(instance.head_dna_file_path)
    before = path.read_bytes()
    transaction = document.edit_transaction

    @contextmanager
    def failed_transaction(instance):
        with transaction(instance):
            yield
            raise OSError("Simulated publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(document, "edit_transaction", failed_transaction)
        with pytest.raises(OSError, match="publication failure"):
            editor.commit(bpy.context)
    assert editor.is_editing
    assert scene_merge.proxy_for(instance) == proxy
    assert [row.scene_object for row in instance.raw_control_editor.target_meshes] == targets
    assert all(target.name in bpy.data.objects for target in targets)
    assert path.read_bytes() == before
    editor.commit(bpy.context)
    assert not editor.is_editing
    assert scene_merge.proxy_for(instance) == proxy
    assert not instance.raw_control_editor.target_meshes
    assert engine.binding_issues(instance) == []


def test_scene_merge_noop_preserves_sources_and_shape_keys(character):
    instance = character
    add_keys(instance)
    objects = [instance.head_mesh, instance.body_mesh]
    original = {
        obj.name: {key.name: coordinates(key.data).copy() for key in obj.data.shape_keys.key_blocks} for obj in objects
    }
    identities = [(obj.as_pointer(), obj.data.as_pointer()) for obj in objects]
    seam = json.loads(HEAD_TO_BODY_EDGE_LOOP_FILE_PATH.read_text())["0"]
    for _cycle in range(2):
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
        proxy = scene_merge.proxy_for(instance)
        assert len(proxy.data.vertices) == sum(len(obj.data.vertices) for obj in objects) - len(seam)
        assert [mod.object for mod in proxy.modifiers] == [instance.head_rig, instance.body_rig]
        assert [mod.vertex_group for mod in proxy.modifiers] == ["DNA_head", "DNA_body"]
        masks = [proxy.vertex_groups[name].index for name in ("DNA_head", "DNA_body")]
        for vertex in proxy.data.vertices:
            assert sum(group.weight for group in vertex.groups if group.group in masks) == pytest.approx(1)
        assert all(obj.hide_get() for obj in objects)
        assert bpy.ops.character_dna.split_components() == {"FINISHED"}
        assert [(obj.as_pointer(), obj.data.as_pointer()) for obj in objects] == identities
        for obj in objects:
            for key in obj.data.shape_keys.key_blocks:
                np.testing.assert_array_equal(coordinates(key.data), original[obj.name][key.name])
            assert obj.data.shape_keys.key_blocks["relative"].relative_key.name == "custom"
        assert engine.binding_issues(instance) == []


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("commit", [False, True], ids=["revert", "commit"])
def test_raw_rest_sculpt_preserves_merge_and_writes_selected_component(character, kind, commit):
    from character_dna.editors.raw_control_editor.editor import RawControlEditor

    instance = character
    instance.editor_component = kind
    instance.raw_control_editor.raw_controls_active_index = 0
    paths = {name: Path(instance.component(name).dna_file_path) for name in ("head", "body")}
    originals = {name: path.read_bytes() for name, path in paths.items()}
    before = coordinates(instance.component(kind).mesh.data.vertices).copy()
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    identity = proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer()
    editor = RawControlEditor.for_instance(instance)
    editor.enter(bpy.context)
    bpy.ops.object.mode_set(mode="OBJECT")
    vertex = next(
        index for index, item in enumerate(proxy.data.attributes[f"DNA_{kind}_vertex"].data) if item.value == 0
    )
    proxy.active_shape_key.data[vertex].co.x += 0.01
    if commit:
        editor.commit(bpy.context)
    else:
        editor.revert(bpy.context)
    expected = before.copy()
    if commit:
        expected[0, 0] += 0.01
    # The rest-pose target is sampled through Blender's float32 armature stack.
    np.testing.assert_allclose(coordinates(instance.component(kind).mesh.data.vertices), expected, atol=5e-6)
    other = "body" if kind == "head" else "head"
    assert paths[other].read_bytes() == originals[other]
    if not commit:
        assert paths[kind].read_bytes() == originals[kind]
    assert identity == (proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer())
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))
    assert engine.binding_issues(instance) == []


@pytest.mark.parametrize("method", ["calibrate", "overwrite"])
def test_export_merged_surface_keeps_presentation_and_source_files(character, tmp_path, method):
    from character_dna.dna_io.calibrator import DNACalibrator
    from character_dna.dna_io.exporter import DNAExporter

    instance = character
    originals = {kind: Path(instance.component(kind).dna_file_path).read_bytes() for kind in ("head", "body")}
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    vertex = next(index for index, item in enumerate(proxy.data.attributes["DNA_head_vertex"].data) if item.value == 0)
    proxy.data.shape_keys.reference_key.data[vertex].co.x += 0.01
    instance.output.folder_path = str(tmp_path / "export")
    instance.output.run_validations = False
    instance.output.auto_update_lods = False
    exporter_type = DNACalibrator if method == "calibrate" else DNAExporter
    exporter = exporter_type(
        instance, 100, file_name="head.dna", component_type="head", textures=False, vertex_colors=False
    )
    try:
        assert exporter.validate_scene()[0]
        assert exporter.run()[0]
    finally:
        release_dna_handle(exporter._dna_writer)
        release_dna_handle(exporter._dna_reader)
    reader = get_dna_reader(tmp_path / "export" / "head.dna")
    try:
        assert reader.getVertexPositionXs(0)[0] == pytest.approx(
            instance.head_mesh.data.vertices[0].co.x * 100, abs=2e-4
        )
    finally:
        release_dna_handle(reader)
    assert scene_merge.proxy_for(instance) == proxy
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))
    for kind, content in originals.items():
        assert Path(instance.component(kind).dna_file_path).read_bytes() == content


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("commit", [False, True], ids=["revert", "commit"])
def test_rbf_keeps_merged_surface(character, kind, commit):
    from character_dna.editors.rbf_editor.editor import RBFEditor

    instance = character
    instance.editor_component = kind
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    identity = proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer()
    editor = RBFEditor.for_instance(instance)
    editor.enter(bpy.context)
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))
    if commit:
        editor.commit(bpy.context)
    else:
        editor.revert(bpy.context)
    assert scene_merge.proxy_for(instance) == proxy
    assert identity == (proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer())
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))


def test_scene_split_rejects_shape_ownership_collision_without_mutation(character):
    instance = character
    add_keys(instance)
    head_index, body_index = next(iter(json.loads(HEAD_TO_BODY_EDGE_LOOP_FILE_PATH.read_text())["0"].items()))
    head_index = int(head_index)
    head_before = instance.head_mesh.data.shape_keys.key_blocks["custom"].data[head_index].co.copy()
    with controller.authoring_operation(instance):
        proxy = scene_merge.merge(instance)
        metadata = json.loads(proxy["dna_merge_metadata"])
        name = next(name for name, owner in metadata["keys"].items() if owner == {"kind": "head", "name": "custom"})
        # A collision must fail before modifying either original mesh.
        proxy.data.shape_keys.key_blocks[name].data[body_index].co.x += 0.02
        with pytest.raises(ValueError, match="ownership collision"):
            scene_merge.split(instance)
        np.testing.assert_array_equal(
            instance.head_mesh.data.shape_keys.key_blocks["custom"].data[head_index].co, head_before
        )
        scene_merge.split(instance, apply=False)


def test_scene_split_preserves_seam_edit_and_source_indices_after_reordering(character):
    instance = character
    add_keys(instance)
    instance.body_mesh.data.shape_keys.key_blocks["custom"].name = "body_custom"
    head_index, body_index = next(iter(json.loads(HEAD_TO_BODY_EDGE_LOOP_FILE_PATH.read_text())["0"].items()))
    head_index = int(head_index)
    instance.head_mesh.data.shape_keys.key_blocks["custom"].data[head_index].co.x += 0.005
    before = instance.head_mesh.data.shape_keys.key_blocks["custom"].data[head_index].co.copy()
    body_before = instance.body_mesh.data.shape_keys.key_blocks[0].data[body_index].co.copy()
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.sort_elements(type="REVERSE", elements={"VERT"})
    bpy.ops.object.mode_set(mode="OBJECT")
    ids = [item.value for item in proxy.data.attributes["DNA_head_vertex"].data]
    proxy.data.shape_keys.key_blocks["custom"].data[ids.index(head_index)].co.x += 0.02
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    np.testing.assert_allclose(
        instance.head_mesh.data.shape_keys.key_blocks["custom"].data[head_index].co,
        np.asarray(before) + np.array([0.02, 0, 0]),
        atol=1e-6,
    )
    counterpart = instance.body_mesh.data.shape_keys.key_blocks["custom"]
    np.testing.assert_allclose(
        counterpart.data[body_index].co, np.asarray(body_before) + np.array([0.025, 0, 0]), atol=1e-6
    )
    assert counterpart.id_data.animation_data.drivers.find(counterpart.path_from_id("value"))
    # A second merge must not add the same boundary displacement twice.
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    metadata = json.loads(proxy["dna_merge_metadata"])
    recipient_name = next(name for name, owner in metadata["keys"].items() if owner.get("exclude_seam"))
    np.testing.assert_allclose(
        proxy.data.shape_keys.key_blocks[recipient_name].data[body_index].co,
        proxy.data.shape_keys.key_blocks[0].data[body_index].co,
        atol=1e-6,
    )
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}


def test_document_noop_restores_exact_bytes_and_rejects_collisions(tmp_path, monkeypatch):
    paths = {kind: TEST_DNA_FOLDER / "ada" / f"{kind}.dna" for kind in document.PARTITIONS}
    merged = tmp_path / "Bruce.dna"
    document.merge_files(paths, merged, "Bruce")
    assert merged.read_bytes().startswith(b"DNA")
    document._read_cache.clear()
    assert document.is_document(merged)
    outputs = document.split_files(merged, tmp_path / "split")
    for kind in document.PARTITIONS:
        assert outputs[kind].read_bytes() == paths[kind].read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        document.merge_files(paths, merged, "Bruce")
    with pytest.raises(ValueError, match="already exists"):
        document.split_files(merged, tmp_path / "split")

    _, first = document.materialize(merged)
    _, second = document.materialize(merged)
    document.publish_partition(first["head"])
    # A real revision change invalidates another independently opened working copy.
    manifest, parts = document.read_document(merged)
    manifest["name"] = "changed_elsewhere"
    document.write_document(merged, manifest, parts)
    with pytest.raises(ValueError, match="changed on disk"):
        document.validate_write(second["head"])

    from character_dna.bindings import dna

    _, working = document.materialize(merged)
    head_before = working["head"].read_bytes()
    document_before = merged.read_bytes()
    reader = get_dna_reader(working["head"])
    writer = get_dna_writer(working["head"])
    writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.setName("failed_edit")
    release_dna_handle(reader)

    def fail_publish(*_args, **_kwargs):
        raise OSError("Simulated document replacement failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(document, "write_document", fail_publish)
            with pytest.raises(OSError, match="Simulated"):
                writer.write()
    finally:
        release_dna_handle(writer)
    assert working["head"].read_bytes() == head_before
    assert merged.read_bytes() == document_before


def test_uv_reconstruction_is_scoped_and_requires_complete_unambiguous_coverage():
    created = []

    def mesh_object(name, heights):
        mesh = bpy.data.meshes.new(name)
        points = [(x, y, z) for z in heights for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        mesh.from_pydata(points, [], [list(range(i * 4, i * 4 + 4)) for i in range(len(heights))])
        uv = mesh.uv_layers.new()
        for loop in mesh.loops:
            uv.data[loop.index].uv = mesh.vertices[loop.vertex_index].co[:2]
        obj = bpy.data.objects.new(name, mesh)
        created.append(obj)
        return obj

    try:
        proxy = mesh_object("proxy", [0, 10])
        original = mesh_object("original", [0])
        scene_merge._attribute(proxy.data, scene_merge.PARTITION, [0, 1], "FACE")
        values = coordinates(proxy.data.vertices)
        values[0, 2] += 0.25
        np.testing.assert_allclose(scene_merge._transfer(proxy, original, "head", values), values[:4], atol=1e-6)
        proxy.data.attributes[scene_merge.PARTITION].data[1].value = 0
        with pytest.raises(ValueError, match="Ambiguous"):
            scene_merge._transfer(proxy, original, "head", values)
        proxy.data.attributes[scene_merge.PARTITION].data[1].value = 1
        for item in original.data.uv_layers.active.data:
            item.uv.x += 3
        with pytest.raises(ValueError, match="Incomplete"):
            scene_merge._transfer(proxy, original, "head", values)
    finally:
        for obj in created:
            mesh = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)


@pytest.mark.parametrize("commit", [False, True], ids=["revert", "commit"])
def test_shape_editor_uses_welded_surface_and_restores_canonical_order(character, commit):
    from character_dna.dna_io.misc import create_shape_key
    from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor

    instance = character
    instance.editor_component = "head"
    reader = instance.head_dna_reader
    mesh_index = next(index for index, obj in instance.head_mesh_index_lookup.items() if obj == instance.head_mesh)
    channel = reader.getBlendShapeChannelIndex(mesh_index, 0)
    name = reader.getBlendShapeChannelName(channel)
    with controller.authoring_operation(instance):
        key = create_shape_key(
            0,
            mesh_index,
            instance.head_mesh,
            reader,
            name,
            prefix=f"{reader.getMeshName(mesh_index)}__",
            linear_modifier=0.01,
        )
        instance.destroy_references()
        instance.sync_shape_key_list()
    instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(key.name)
    original = coordinates(key.data).copy()
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    editor = ShapeKeyEditor.for_instance(instance)
    editor.enter(bpy.context)
    proxy = scene_merge.proxy_for(instance)
    assert bpy.context.object == proxy
    bpy.ops.object.mode_set(mode="OBJECT")
    ids = [item.value for item in proxy.data.attributes["DNA_head_vertex"].data]
    index = ids.index(0)
    proxy.active_shape_key.data[index].co.x += 0.01
    if commit:
        editor.commit(bpy.context)
    else:
        editor.revert(bpy.context)
    assert scene_merge.proxy_for(instance) == proxy
    assert instance.head_mesh.hide_get()
    assert instance.body_mesh.hide_get()
    restored = instance.head_mesh.data.shape_keys.key_blocks[key.name]
    expected = original.copy()
    if commit:
        expected[0, 0] += 0.01
    np.testing.assert_allclose(coordinates(restored.data), expected, atol=2e-6)
    assert engine.binding_issues(instance) == []


def test_merged_surface_and_document_recover_after_blend_reload(character, tmp_path):
    instance = character
    instance.converter_merge_mode = "DNA"
    instance.converter_dna_directory = str(tmp_path / "combined")
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    original_head = Path(instance.head_dna_file_path)
    instance_name = instance.name
    blend = tmp_path / "merged.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.read_homefile(use_empty=True)
    original_head.unlink()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    restored = get_active_rig_instance()
    assert restored.name == instance_name
    assert Path(restored.head_dna_file_path).exists()
    assert scene_merge.proxy_for(restored) is not None
    restored.converter_dna_directory = str(tmp_path / "split")
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    assert engine.binding_issues(restored) == []


def test_combined_document_import_and_partition_edit(character, tmp_path):
    instance = character
    instance.converter_merge_mode = "DNA"
    instance.converter_dna_directory = str(tmp_path / "combined")
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    merged = Path(instance["combined_dna_path"])
    from character_dna.editors.raw_control_editor.editor import RawControlEditor
    from character_dna.editors.rbf_editor.editor import RBFEditor

    proxy = scene_merge.proxy_for(instance)
    identity = proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer()
    instance.editor_component = "head"
    instance.raw_control_editor.raw_controls_active_index = instance.raw_control_editor.raw_controls.find(
        "CTRL_expressions.jawOpen"
    )
    posed = evaluated_coordinates(proxy)
    raw_editor = RawControlEditor.for_instance(instance)
    raw_editor.enter(bpy.context)
    np.testing.assert_allclose(evaluated_coordinates(proxy), posed, atol=1e-6)
    raw_editor.commit(bpy.context)
    original_head = document.read_document(merged)[1]["head"]
    instance.editor_component = "body"
    rbf_editor = RBFEditor.for_instance(instance)
    rbf_editor.enter(bpy.context)
    radius = instance.rbf_editor.rbf_solver_list[0].radius + 0.25
    instance.rbf_editor.rbf_solver_list[0].radius = radius
    rbf_editor.commit(bpy.context)
    assert document.read_document(merged)[1]["head"] == original_head
    assert instance.body_dna_reader.getRBFSolverRadius(0) == pytest.approx(radius)
    assert identity == (proxy.as_pointer(), proxy.data.as_pointer(), proxy.data.shape_keys.as_pointer())
    original_body = document.read_document(merged)[1]["body"]
    path = Path(instance.head_dna_file_path)
    with controller.authoring_operation(instance):
        reader = get_dna_reader(path)
        writer = get_dna_writer(path)
        from character_dna.bindings import dna

        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        writer.setName("edited_head")
        release_dna_handle(reader)
        instance.destroy()
        writer.write()
        release_dna_handle(writer)
    assert document.read_document(merged)[1]["body"] == original_body
    portable = tmp_path / "portable" / "Bruce.dna"
    portable.parent.mkdir()
    shutil.copyfile(merged, portable)
    bpy.ops.wm.read_homefile(use_empty=True)
    assert bpy.ops.character_dna.import_dna(
        filepath=str(portable),
        include_body=False,
        import_face_board=False,
        import_mesh=False,
        import_bones=True,
        import_materials=False,
    ) == {"FINISHED"}
    restored = get_active_rig_instance()
    assert restored.head_rig and restored.body_rig
    assert restored.head_dna_reader.getName() == "edited_head"
    assert engine.binding_issues(restored) == []
    restored.converter_dna_directory = str(tmp_path / "restored")
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}
    assert not restored.get("combined_dna_path")
    assert Path(restored.body_dna_file_path).read_bytes() == original_body


def test_scene_split_rolls_back_failed_bridge_publication(character, monkeypatch):
    instance = character
    add_keys(instance)
    instance.body_mesh.data.shape_keys.key_blocks["custom"].name = "body_custom"
    before = {kind: scene_merge._shape_signature(instance.component(kind).mesh) for kind in ("head", "body")}
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    body_index = next(iter(json.loads(HEAD_TO_BODY_EDGE_LOOP_FILE_PATH.read_text())["0"].values()))
    proxy.data.shape_keys.key_blocks["custom"].data[body_index].co.x += 0.02

    def fail_publish(_instance):
        raise OSError("Simulated document write failure")

    with monkeypatch.context() as patch:
        patch.setattr(document, "publish_scene_bridges", fail_publish)
        with pytest.raises(OSError, match="Simulated"):
            scene_merge.split(instance)
    assert scene_merge.proxy_for(instance) == proxy
    for kind in ("head", "body"):
        assert scene_merge._shape_signature(instance.component(kind).mesh) == before[kind]
    assert bpy.ops.character_dna.split_components() == {"FINISHED"}


def test_failed_split_rebinding_keeps_merged_scene_and_file_ownership(character, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from character_dna.dna_io import shape_dependencies
    from character_dna.editors.converter.merge_split import SplitComponents

    instance = character
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    sources = {kind: Path(instance.component(kind).dna_file_path) for kind in ("head", "body")}
    instance.converter_merge_mode = "DNA"
    instance.converter_dna_directory = str(tmp_path / "split")
    instance["combined_dna_path"] = str(tmp_path / "combined.dna")
    instance["combined_dna_id"] = "test-document"
    outputs = {kind: tmp_path / "split" / f"{kind}.dna" for kind in sources}

    def staged_outputs(_path, directory):
        directory.mkdir()
        for kind, path in outputs.items():
            shutil.copyfile(sources[kind], path)
        return outputs

    def failed_binding(_instance, _bridges):
        raise ValueError("Simulated binding failure")

    monkeypatch.setattr(document, "validate_write", lambda _path: None)
    monkeypatch.setattr(document, "publish_scene_bridges", lambda _instance: None)
    monkeypatch.setattr(document, "read_document", lambda _path: ({"scene_bridges": []}, {}))
    monkeypatch.setattr(document, "split_files", staged_outputs)
    monkeypatch.setattr(shape_dependencies, "rebind_scene", failed_binding)
    messages = []
    operation = SimpleNamespace(report=lambda _level, message: messages.append(message))
    assert SplitComponents.execute(operation, bpy.context) == {"CANCELLED"}
    assert messages == ["Simulated binding failure"]
    assert scene_merge.proxy_for(instance) == proxy
    assert instance.get("combined_dna_id") == "test-document"
    assert all(Path(instance.component(kind).dna_file_path) == path for kind, path in sources.items())
    assert not any(path.exists() for path in outputs.values())
    assert all(obj.hide_get() for obj in (instance.head_mesh, instance.body_mesh))
    assert engine.binding_issues(instance) == []


def test_combined_neckline_shape_is_portable_and_backup_restores_document(  # noqa: PLR0915
    character, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from character_dna.dna_io.misc import create_shape_key
    from character_dna.editors.backup_manager import core as backups
    from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor
    from character_dna.editors.shape_key_editor.operators import ImportShapeKeys

    instance = character
    reader = instance.head_dna_reader
    mesh = next(index for index, obj in instance.head_mesh_index_lookup.items() if obj == instance.head_mesh)
    channel = reader.getBlendShapeChannelIndex(mesh, 0)
    with controller.authoring_operation(instance):
        key = create_shape_key(
            0,
            mesh,
            instance.head_mesh,
            reader,
            reader.getBlendShapeChannelName(channel),
            prefix=f"{reader.getMeshName(mesh)}__",
            linear_modifier=0.01,
        )
        instance.destroy_references()
        instance.sync_shape_key_list()
    name = key.name
    instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(name)
    instance.converter_merge_mode = "DNA"
    instance.converter_dna_directory = str(tmp_path / "combined")
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    merged = Path(instance["combined_dna_path"])
    original_document = merged.read_bytes()
    monkeypatch.setattr(backups, "is_auto_backup_enabled", lambda: True)
    backup_id = backups.create_backup(instance, backups.BackupType.MANUAL)
    assert backup_id
    editor = ShapeKeyEditor.for_instance(instance)
    editor.enter(bpy.context)
    proxy = scene_merge.proxy_for(instance)
    bpy.ops.object.mode_set(mode="OBJECT")
    body_index = next(iter(json.loads(HEAD_TO_BODY_EDGE_LOOP_FILE_PATH.read_text())["0"].values()))
    proxy.active_shape_key.data[body_index].co.x += 0.012
    editor.commit(bpy.context)
    expected = coordinates(instance.body_mesh.data.shape_keys.key_blocks[name].data).copy()
    assert document.read_document(merged)[0]["scene_bridges"]
    portable = tmp_path / "portable.dna"
    shutil.copyfile(merged, portable)
    outputs = document.split_files(merged, tmp_path / "resolved")
    recipient = get_dna_reader(outputs["body"])
    try:
        assert recipient.getBlendShapeTargetCount(0) == 1
        assert recipient.getRawControlCount() > 176
    finally:
        release_dna_handle(recipient)
    assert backups.restore_backup(instance, backup_id)
    assert merged.read_bytes() == original_document
    assert not instance.body_mesh.get("dna_shape_bridges")
    assert not instance.body_mesh.data.shape_keys or name not in instance.body_mesh.data.shape_keys.key_blocks
    assert engine.binding_issues(instance) == []
    bpy.ops.wm.read_homefile(use_empty=True)
    assert bpy.ops.character_dna.import_dna(
        filepath=str(portable),
        include_body=False,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    ) == {"FINISHED"}
    restored = get_active_rig_instance()
    body_key = restored.body_mesh.data.shape_keys.key_blocks[name]
    np.testing.assert_array_equal(coordinates(body_key.data), expected)
    driver = body_key.id_data.animation_data.drivers.find(body_key.path_from_id("value")).driver
    assert driver.variables[0].targets[0].id == restored.head_mesh.data.shape_keys
    assert engine.binding_issues(restored) == []
    # Even an interrupted import which has only cleared the source Key ID must
    # restore the source channel needed by the surviving cross-component driver.
    was_active = controller.suspend(restored)
    restored.head_mesh.shape_key_clear()
    operation = SimpleNamespace(_timer=None, _import_instance=restored, _native_was_active=was_active)
    assert ImportShapeKeys.finish(operation, bpy.context) == {"FINISHED"}
    driver = body_key.id_data.animation_data.drivers.find(body_key.path_from_id("value")).driver
    assert driver.variables[0].targets[0].id == restored.head_mesh.data.shape_keys
    assert name in restored.head_mesh.data.shape_keys.key_blocks
    assert engine.binding_issues(restored) == []
    # Missing mesh outputs cannot be interpreted as intentional relationship deletion.
    bpy.ops.wm.read_homefile(use_empty=True)
    assert bpy.ops.character_dna.import_dna(
        filepath=str(portable),
        include_body=False,
        import_face_board=False,
        import_mesh=False,
        import_bones=True,
        import_materials=False,
    ) == {"FINISHED"}
    bones_only = get_active_rig_instance()
    before = portable.read_bytes()
    document.publish_scene_bridges(bones_only)
    assert portable.read_bytes() == before
    assert document.split_files(portable, tmp_path / "bones_only_split")["body"].exists()


def test_document_rejects_corruption_before_materializing(tmp_path):
    invalid = tmp_path / "bad.dna"
    invalid.write_bytes(document.MAGIC + b"broken")
    with pytest.raises(ValueError, match="Invalid combined"):
        document.materialize(invalid)


def test_scene_corrective_metadata_belongs_to_author_and_survives_import(character, tmp_path):
    from character_dna.dna_io.misc import create_shape_key
    from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor

    instance = character
    reader = instance.head_dna_reader
    channel = reader.getBlendShapeChannelIndex(0, 0)
    with controller.authoring_operation(instance):
        key = create_shape_key(
            0,
            0,
            instance.head_mesh,
            reader,
            reader.getBlendShapeChannelName(channel),
            prefix="head_lod0_mesh__",
            linear_modifier=0.01,
        )
        instance.destroy_references()
        instance.sync_shape_key_list()
    name = key.name
    instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(name)
    original_body = (tmp_path / "body.dna").read_bytes()
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy = scene_merge.proxy_for(instance)
    editor = ShapeKeyEditor.for_instance(instance)
    editor.enter(bpy.context)
    bpy.ops.object.mode_set(mode="OBJECT")
    proxy.active_shape_key.data[0].co.x += 0.01
    editor.commit(bpy.context)
    assert scene_merge.proxy_for(instance) == proxy
    assert (tmp_path / "body.dna").read_bytes() == original_body
    assert document.scene_extensions(instance)["scene_bridges"]
    expected = coordinates(instance.body_mesh.data.shape_keys.key_blocks[name].data).copy()
    bpy.ops.wm.read_homefile(use_empty=True)
    assert bpy.ops.character_dna.import_dna(
        filepath=str(tmp_path / "head.dna"),
        include_body=True,
        import_face_board=False,
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    ) == {"FINISHED"}
    restored = get_active_rig_instance()
    np.testing.assert_array_equal(coordinates(restored.body_mesh.data.shape_keys.key_blocks[name].data), expected)
    # The DNA metadata must carry the same relationship when no scene is available.
    bpy.ops.wm.read_homefile(use_empty=True)
    combined = tmp_path / "offline.dna"
    document.merge_files({kind: tmp_path / f"{kind}.dna" for kind in ("head", "body")}, combined, "offline")
    document._read_cache.clear()
    assert document.read_document(combined)[0]["scene_bridges"]
    reader = get_dna_reader(combined)
    try:
        assert reader.getBlendShapeTargetDeltaCount(0, 0) > 0
    finally:
        release_dna_handle(reader)


def test_merged_masks_match_posed_components_with_object_transforms(character):
    from mathutils import Matrix

    instance = character
    with controller.authoring_operation(instance):
        offset = Matrix.Translation((0.3, -0.2, 0.7)) @ Matrix.Rotation(0.2, 4, "Z")
        for kind in ("head", "body"):
            obj = instance.component(kind).mesh
            obj.matrix_world = offset @ obj.matrix_world
        bone = instance.body_rig.pose.bones["neck_01"]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler.z = 0.3
        bpy.context.view_layer.update()
        proxy = scene_merge.merge(instance)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = coordinates(proxy.evaluated_get(graph).data.vertices)
        for kind in ("head", "body"):
            obj = instance.component(kind).mesh
            expected = scene_merge.transform_positions(
                coordinates(obj.evaluated_get(graph).data.vertices),
                proxy.matrix_world.inverted() @ obj.matrix_world,
            )
            attribute = proxy.data.attributes[f"DNA_{kind}_vertex"]
            ids = np.array([item.value for item in attribute.data])
            valid = ids >= 0
            if kind == "head":
                valid &= np.array([item.value for item in proxy.data.attributes["DNA_body_vertex"].data]) < 0
            np.testing.assert_allclose(evaluated[valid], expected[ids[valid]], atol=2e-6)
        scene_merge.split(instance)
