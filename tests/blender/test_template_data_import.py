"""Internal template import restores editable masks without changing normal imports."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from test_vertex_color_roundtrip import write_surface

from character_dna.dna_io import DNAImporter, get_dna_reader, get_dna_writer, mesh_data, metadata, release_dna_handle


@pytest.mark.parametrize("kind", ["head", "body"])
@pytest.mark.parametrize("topology_type", ["MetaHuman", "3D Scan Store"])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_template_masks_restore_sparse_dna_positions(tmp_path, kind, topology_type, enabled):
    source = write_surface(tmp_path / "source.dna", component=kind)
    path = tmp_path / f"{kind}.dna"
    reader, writer = get_dna_reader(source), get_dna_writer(path)
    faces = [[0, 2, 5], [0, 5, 7]]
    record = {
        "mesh_index": 0,
        "name": f"{kind}_lod0_mesh",
        "lods": [0],
        "topology": mesh_data.topology(8, faces),
        "colors": [],
        "neckline": None,
        "exclusions": {"EXCLUDE_Tongue": [2, 7], "EXCLUDE_Empty": []},
        "preservations": {
            "preserve_MouthPocket": {"vertices": [0, 5], "weights": [0.015362793579697609, 0.713427]},
            "PRESERVE_Empty": {"vertices": [], "weights": []},
        },
    }
    try:
        from character_dna.bindings import dna

        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        writer.setVertexPositions(
            0, [[0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], [0, 0, 0], [100, 100, 0], [0, 0, 0], [0, 100, 0]]
        )
        writer.setVertexLayouts(0, [[p, i, 0] for i, p in enumerate([5, 0, 7, 2, 0, 5])])
        writer.setFaceVertexLayoutIndices(0, 0, [1, 3, 0])
        writer.setFaceVertexLayoutIndices(0, 1, [4, 5, 2])
        metadata.write_many(
            writer,
            reader,
            {f"{mesh_data.PREFIX}.{kind}": {"schema": 1, "component": kind, "space": "component", "meshes": [record]}},
            {mesh_data.type_key(kind): topology_type},
        )
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)
    options = SimpleNamespace(
        import_vertex_groups=False,
        import_vertex_colors=False,
        import_normals=False,
        **{f"import_lod{i}": True for i in range(8)},
    )
    if enabled is not None:
        options.import_template_data = enabled
    instance = SimpleNamespace(
        name="Authoring", head_dna_file_path=str(path), body_dna_file_path=str(path), output=SimpleNamespace()
    )
    importer = DNAImporter(instance, options, 0.01, component_type=kind)
    obj = None
    try:
        importer.initialize_dna_data()
        obj = importer.create_mesh_object(0, f"{kind}_lod0_mesh")
        if not enabled:
            assert not obj.vertex_groups
            return
        assert set(obj.vertex_groups.keys()) == {*record["exclusions"], *record["preservations"]}
        captured = mesh_data.capture(obj, 0, record["name"], 8, faces, [0], colors=False)
        assert captured["exclusions"] == record["exclusions"]
        for name, mask in record["preservations"].items():
            assert captured["preservations"][name]["vertices"] == mask["vertices"]
            np.testing.assert_array_equal(
                captured["preservations"][name]["weights"], np.array(mask["weights"], dtype=np.float32)
            )
    finally:
        if obj is not None:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        release_dna_handle(importer._dna_reader)


def test_operator_flag_is_hidden_nonpersistent_and_reaches_both_components(
    empty_character_scene, compact_dna_asset, tmp_path
):
    from character_dna.dna_io import merged_graph

    flag = bpy.ops.character_dna.import_dna.get_rna_type().properties["import_template_data"]
    assert flag.is_hidden and flag.is_skip_save and flag.default is False
    for kind in ("head", "body"):
        graph = merged_graph.from_bytes((compact_dna_asset / f"{kind}.dna").read_bytes())
        name = f"{kind}_lod0_mesh"
        index = graph["defn1.1"]["meshNames"].index(name)
        record = {
            "mesh_index": index,
            "name": name,
            "lods": [0],
            "topology": mesh_data.topology(3, [[0, 1, 2]]),
            "colors": [],
            "neckline": None,
            "exclusions": {"EXCLUDE_Test": [0, 2]},
            "preservations": {"PRESERVE_Test": {"vertices": [0, 1], "weights": [0.125, 0.875]}},
        }
        mesh_data.set_graph_payload(
            graph, kind, {"schema": 1, "component": kind, "space": "component", "meshes": [record]}, "3D Scan Store"
        )
        (tmp_path / f"{kind}.dna").write_bytes(merged_graph.to_bytes(graph))
    options = dict(
        filepath=str(tmp_path / "head.dna"),
        include_body=True,
        import_mesh=True,
        import_bones=True,
        import_vertex_groups=False,
        import_materials=False,
        import_face_board=False,
        import_vertex_colors=False,
        import_shape_keys=False,
        **{f"import_lod{i}": i == 0 for i in range(8)},
    )
    assert bpy.ops.character_dna.import_dna(import_template_data=True, **options) == {"FINISHED"}
    instance = bpy.context.scene.character_dna.rig_instance_list[-1]
    for obj in (instance.head_mesh, instance.body_mesh):
        assert obj.vertex_groups["EXCLUDE_Test"].weight(2) == 1
        assert obj.vertex_groups["PRESERVE_Test"].weight(1) == 0.875
    # A subsequent ordinary API import must not inherit the authoring flag.
    assert bpy.ops.character_dna.import_dna(**options) == {"FINISHED"}
    instance = bpy.context.scene.character_dna.rig_instance_list[-1]
    for obj in (instance.head_mesh, instance.body_mesh):
        assert not any(
            mesh_data.is_exclusion_group(g.name) or mesh_data.is_preservation_group(g.name) for g in obj.vertex_groups
        )
