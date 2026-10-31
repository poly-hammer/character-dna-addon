"""Unit coverage for the Paste Shape Key Values operator's utilities.

The paste helper copies a source shape key's delta (source - basis) onto the live
edited key block, is selection-aware in Edit mode, and can optionally mirror the
delta across the symmetry plane. The verbatim (non-mirror) path and the source
enumeration / default-twin helpers need no rig instance, so they are exercised
directly against a scratch Blender mesh.
"""

import pytest

from character_dna.constants import SHAPE_KEY_BASIS_NAME


# Shape-key editing lives in the Pro ``editors`` submodule; skip cleanly on a Free
# checkout where it is absent.
utilities = pytest.importorskip("character_dna.editors.shape_key_editor.utilities")


def _make_mesh_with_keys(name: str, key_names: "list[str]"):
    """Create a scratch quad mesh with a basis plus ``key_names`` shape keys and
    return ``(mesh_object, {key_name: key_block})``."""
    import bpy

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)], [], [(0, 1, 3, 2)])
    mesh.update()
    mesh_object = bpy.data.objects.new(f"{name}_object", mesh)
    bpy.context.scene.collection.objects.link(mesh_object)
    mesh_object.shape_key_add(name=SHAPE_KEY_BASIS_NAME, from_mix=False)
    blocks = {key_name: mesh_object.shape_key_add(name=key_name, from_mix=False) for key_name in key_names}
    return mesh_object, blocks


def test_same_mesh_paste_source_blocks_excludes_basis_and_edited():
    mesh_object, _blocks = _make_mesh_with_keys(
        "paste_sources", ["head__brow_down_L", "head__brow_down_R", "head__jaw_open"]
    )
    try:
        sources = utilities.same_mesh_paste_source_blocks(mesh_object, "head__brow_down_L")
        names = {block.name for block in sources}
        assert names == {"head__brow_down_R", "head__jaw_open"}
        assert SHAPE_KEY_BASIS_NAME not in names
    finally:
        import bpy

        bpy.data.objects.remove(mesh_object, do_unlink=True)


def test_default_paste_source_name_returns_mirror_twin():
    source_names = {"head__brow_down_R", "head__jaw_open"}
    assert utilities.default_paste_source_name("head__brow_down_L", source_names) == "head__brow_down_R"


def test_default_paste_source_name_none_when_no_twin_present():
    assert utilities.default_paste_source_name("head__brow_down_L", {"head__jaw_open"}) is None
    assert utilities.default_paste_source_name("head__jaw_open", {"head__brow_down_R"}) is None


def test_paste_verbatim_copies_source_delta_onto_edited_key():
    mesh_object, blocks = _make_mesh_with_keys("paste_verbatim", ["edited", "source"])
    try:
        source = blocks["source"]
        edited = blocks["edited"]
        # Sculpt the source key away from the basis.
        source.data[0].co.x += 0.5
        source.data[2].co.z -= 0.25

        changed = utilities.paste_shape_key_values(None, mesh_object, edited, source, mirror=False)

        assert changed == 2
        for i in range(len(mesh_object.data.vertices)):
            assert tuple(edited.data[i].co) == pytest.approx(tuple(source.data[i].co))
    finally:
        import bpy

        bpy.data.objects.remove(mesh_object, do_unlink=True)


def test_paste_verbatim_selection_aware_in_edit_mode():
    import bmesh
    import bpy

    mesh_object, blocks = _make_mesh_with_keys("paste_selection", ["edited", "source"])
    try:
        source = blocks["source"]
        edited = blocks["edited"]
        # Sculpt the source on two vertices; only vertex 0 will be selected.
        source.data[0].co.x += 0.5
        source.data[1].co.y += 0.3

        mesh = mesh_object.data
        bpy.context.view_layer.objects.active = mesh_object
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bm = bmesh.from_edit_mesh(mesh)
            bm.verts.ensure_lookup_table()
            for vertex in bm.verts:
                vertex.select = False
            bm.verts[0].select = True
            bmesh.update_edit_mesh(mesh)

            changed = utilities.paste_shape_key_values(None, mesh_object, edited, source, mirror=False)
        finally:
            if mesh_object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

        # Only the selected vertex 0 received the paste; vertex 1 stayed at basis.
        assert changed == 1
        assert tuple(edited.data[0].co) == pytest.approx(tuple(source.data[0].co))
        assert edited.data[1].co.y == pytest.approx(0.0)
    finally:
        bpy.data.objects.remove(mesh_object, do_unlink=True)
