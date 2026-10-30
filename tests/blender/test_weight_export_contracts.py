"""Weight lookup shortcuts preserve native order, zeros and every influence."""

from types import SimpleNamespace

import bpy
import pytest

from character_dna.bindings import dna
from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
from character_dna.dna_io.calibrator import DNACalibrator
from character_dna.dna_io.exporter import DNAExporter
from character_dna.dna_io.importer import DNAImporter


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("method", ["overwrite", "calibrate"])
def test_weight_export_preserves_native_rows_and_replaced_groups(
    empty_character_scene, compact_dna_asset, tmp_path, kind, method
):
    reader = get_dna_reader(compact_dna_asset / f"{kind}.dna")
    writer = result = None
    try:
        mesh = bpy.data.meshes.new("WeightContract")
        mesh.vertices.add(reader.getVertexPositionCount(0))
        obj = bpy.data.objects.new("WeightContract_" + reader.getMeshName(0), mesh)
        empty_character_scene.collection.objects.link(obj)
        DNAImporter.set_vertex_groups(SimpleNamespace(_dna_reader=reader), 0, obj)
        bones = {reader.getJointName(index): index for index in range(reader.getJointCount())}
        root = obj.vertex_groups.get(reader.getJointName(0)) or obj.vertex_groups.new(name=reader.getJointName(0))
        root.add([0], 0.0, "REPLACE")
        root.add([1], 0.375, "REPLACE")
        extra = obj.vertex_groups.new(name="NotADNAJoint")
        extra.add([0, 1], 0.5, "REPLACE")

        for iteration in range(2):
            # Retain the original RNA name/weight queries as the independent oracle.
            groups = {group.index: group for group in obj.vertex_groups}
            expected = []
            grouped = {group.name: [] for group in obj.vertex_groups}
            for vertex in mesh.vertices:
                indices, values = [], []
                for membership in vertex.groups:
                    group = groups[membership.group]
                    name = group.name
                    weight = group.weight(vertex.index)
                    if weight > 0:
                        grouped[name].append((vertex.index, weight))
                    if name in bones and (method == "calibrate" or weight > 0):
                        indices.append(bones[name])
                        values.append(weight)
                expected.append((indices, values))
            assert DNAExporter.get_mesh_vertex_groups(obj) == grouped
            path = tmp_path / f"{method}-{iteration}.dna"
            writer = get_dna_writer(path)
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            if method == "overwrite":
                operation = SimpleNamespace(_dna_writer=writer, _bone_index_lookup=bones)
                DNAExporter.set_dna_vertex_groups(operation, 0, obj)
            else:
                operation = SimpleNamespace(
                    _dna_reader=reader,
                    _dna_writer=writer,
                    _component_type=kind,
                    _prefix="WeightContract",
                    _auto_lod_enabled=True,
                )
                DNACalibrator.calibrate_vertex_groups(operation)
            writer.write()
            release_dna_handle(writer)
            writer = None
            result = get_dna_reader(path)
            assert result.getSkinWeightsCount(0) == len(expected)
            for vertex, (indices, values) in enumerate(expected):
                assert list(result.getSkinWeightsJointIndices(0, vertex)) == indices
                assert list(result.getSkinWeightsValues(0, vertex)) == values
            release_dna_handle(result)
            result = None
            # A following operation must not reuse removed RNA group wrappers.
            obj.vertex_groups.remove(extra)
            obj.vertex_groups.remove(root)
            root = obj.vertex_groups.new(name=reader.getJointName(0))
            root.add([0, 1], 0.125, "REPLACE")
            extra = obj.vertex_groups.new(name=reader.getJointName(0) + "_replacement")
            extra.add([0], 0.25, "REPLACE")
    finally:
        for handle in (result, writer, reader):
            release_dna_handle(handle)
