"""Lossless component data without source-file recovery copies."""

import bpy
import numpy as np
import pytest

from character_dna.dna_io import document, merged_graph, partitions
from constants import TEST_DNA_FOLDER
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


@pytest.fixture
def character(rig_factory, tmp_path, request, compact_dna, compact_two_target_asset):
    assert request.param == "all_lods"
    profile = RigProfile(asset=str(compact_two_target_asset), pose_bones=STATE_BONES, lods=tuple(range(8)))
    with rig_factory.lease(profile, tmp_path) as instance:
        yield instance


def test_compact_projection(tmp_path):
    paths = {kind: TEST_DNA_FOLDER / "ada" / f"{kind}.dna" for kind in document.PARTITIONS}
    graphs = {kind: merged_graph.from_bytes(path.read_bytes()) for kind, path in paths.items()}
    native, info = merged_graph.compose(graphs, "Compact")
    records = partitions.describe(graphs, native, info)
    projected = partitions.project(native, info, records)
    from test_native_merged_graph import first_difference

    for kind, graph in graphs.items():
        assert first_difference(projected[kind], graph) is None
    output = tmp_path / "Compact.dna"
    document.merge_files(paths, output, "Compact")
    assert output.stat().st_size < sum(path.stat().st_size for path in paths.values()) * 1.25
    payload = document._payload(output)
    assert "recovery" not in payload
    assert "bridges" not in payload["indices"]
    document._read_cache.clear()
    split = document.split_files(output, tmp_path / "split")
    for kind, path in split.items():
        assert first_difference(merged_graph.from_bytes(path.read_bytes()), graphs[kind]) is None


def test_target_commit_uses_native_arrays_and_survives_cold_reopen(tmp_path, monkeypatch):
    from test_native_merged_graph import first_difference

    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from character_dna.dna_io.coordinates import write_shape_targets

    output = tmp_path / "Target.dna"
    document.merge_files(
        {kind: TEST_DNA_FOLDER / "ada" / f"{kind}.dna" for kind in document.PARTITIONS}, output, "Target"
    )
    _, paths = document.materialize(output)
    reader = get_dna_reader(paths["head"])
    indices = list(reader.getBlendShapeTargetVertexIndices(0, 0))
    values = [
        [float(x), float(y), float(z)]
        for x, y, z in zip(
            reader.getBlendShapeTargetDeltaXs(0, 0),
            reader.getBlendShapeTargetDeltaYs(0, 0),
            reader.getBlendShapeTargetDeltaZs(0, 0),
            strict=True,
        )
    ]
    values[0][0] += 0.125
    release_dna_handle(reader)

    def reject(*args, **kwargs):
        raise AssertionError("A target edit rebuilt the document graph")

    with monkeypatch.context() as patch:
        for method in ("compose", "from_bytes", "to_bytes"):
            patch.setattr(merged_graph, method, reject)
        write_shape_targets(paths["head"], {(0, 0): (indices, values)})
    reader = get_dna_reader(paths["head"])
    try:
        np.testing.assert_array_equal(reader.getBlendShapeTargetVertexIndices(0, 0), indices)
        np.testing.assert_allclose(
            np.column_stack([getattr(reader, "getBlendShapeTargetDelta" + axis + "s")(0, 0) for axis in "XYZ"]),
            values,
            atol=1e-6,
        )
    finally:
        release_dna_handle(reader)
    expected = {kind: merged_graph.from_bytes(path.read_bytes()) for kind, path in paths.items()}
    document._read_cache.clear()
    _, parts = document.read_document(output)
    for kind, content in parts.items():
        difference = first_difference(merged_graph.from_bytes(content), expected[kind])
        assert difference is None, difference

    from character_dna.dna_io import document_edits

    before_document, before_part = output.read_bytes(), paths["head"].read_bytes()
    values[0][0] += 0.125

    def fail_publish(*args, **kwargs):
        raise OSError("Simulated target publication failure")

    monkeypatch.setattr(document_edits, "write_targets", fail_publish)
    with pytest.raises(OSError, match="Simulated target"):
        write_shape_targets(paths["head"], {(0, 0): (indices, values)})
    assert output.read_bytes() == before_document
    assert paths["head"].read_bytes() == before_part


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
def test_scoped_commit_keeps_other_keys_and_all_lod_revisions(character, monkeypatch):
    from test_merge_split import coordinates

    from character_dna.dna_io.misc import create_shape_key
    from character_dna.editors.converter import scene_merge
    from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor
    from character_dna.runtime import controller

    instance = character
    with controller.authoring_operation(instance):
        reader = instance.head_dna_reader
        names = []
        for index in (0, 1):
            channel = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, index))
            key = create_shape_key(
                index, 0, instance.head_mesh, reader, channel, prefix="head_lod0_mesh__", linear_modifier=0.01
            )
            names.append(key.name)
        instance.destroy_references()
        instance.sync_shape_key_list()
    instance.converter_merge_mode = "DNA"
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(names[0])
    editor = ShapeKeyEditor.for_instance(instance)
    unrelated = instance.head_mesh.data.shape_keys.key_blocks[names[1]]
    pointer, original = unrelated.as_pointer(), coordinates(unrelated.data)

    def reject(*args, **kwargs):
        raise AssertionError("Ordinary editor commit recomposed the full document")

    monkeypatch.setattr(document, "write_document", reject)
    for _ in range(2):
        editor.enter(bpy.context)
        snapshot = instance.head_mesh["dna_edit_snapshot"]
        assert list(snapshot.shape_keys.key_blocks.keys()) == ["Basis", names[0]]
        proxy = scene_merge.proxy_for(instance)
        bpy.ops.object.mode_set(mode="OBJECT")
        vertex = next(
            index for index, item in enumerate(proxy.data.attributes["DNA_head_vertex"].data) if item.value == 0
        )
        proxy.active_shape_key.data[vertex].co.x += 0.001
        editor.commit(bpy.context)
        assert unrelated.as_pointer() == pointer
        np.testing.assert_array_equal(coordinates(unrelated.data), original)
        scene_merge.validate_split(instance)
