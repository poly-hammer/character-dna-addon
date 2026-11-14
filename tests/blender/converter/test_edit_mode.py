"""Conversion must capture pending Edit Mode geometry and UV edits."""

import bmesh
import bpy
import numpy as np
import pytest

from mathutils import Vector

from character_dna.editors.converter import templates
from character_dna.editors.converter.native.template import TemplateReader
from character_dna.editors.converter.native.uv import read as read_uv
from character_dna.editors.converter.properties import get_converter_properties


@pytest.mark.parametrize(
    ("edit_roles", "validate_teeth_uvs"),
    [
        pytest.param(("teeth",), False, id="teeth-validation-disabled"),
        pytest.param(("head", "teeth"), True, id="multiple-meshes-validation-enabled"),
    ],
)
def test_conversion_captures_pending_mesh_edits(empty_character_scene, tmp_path, edit_roles, validate_teeth_uvs):
    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / "metahuman_head_advanced.json")
    source = TemplateReader(str(templates.current_dna_files(props)["head"]), "head")
    inputs = {}
    for item in templates.input_rows(props):
        shape = source.meshes[item.dna_mesh_name]
        faces = [face.tolist() for face in np.split(shape.corners, np.cumsum(shape.sizes)[:-1])]
        mesh = bpy.data.meshes.new(item.role)
        mesh.from_pydata(shape.positions, [], faces)
        mesh.uv_layers.new(name="DiffuseUV").data.foreach_set("uv", shape.uv.ravel())
        obj = bpy.data.objects.new(item.role, mesh)
        bpy.context.scene.collection.objects.link(obj)
        item.scene_object = obj
        inputs[item.role] = (obj, shape)
        if item.role == "teeth":
            item.validate_uvs = validate_teeth_uvs
    props.new_name, props.new_folder = "EditModeTest", str(tmp_path)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for role in edit_roles:
        inputs[role][0].select_set(True)
    bpy.context.view_layer.objects.active = inputs["teeth"][0]
    bpy.ops.object.mode_set(mode="EDIT")
    offset = np.array([0.0, -0.001, 0.002])
    uv_offset = np.array([0.0001, -0.0001])
    try:
        for role in edit_roles:
            obj, _shape = inputs[role]
            mesh = bmesh.from_edit_mesh(obj.data)
            layer = mesh.loops.layers.uv.active
            for vertex in mesh.verts:
                vertex.co += Vector(offset)
            for face in mesh.faces:
                for loop in face.loops:
                    loop[layer].uv += Vector(uv_offset)
            assert obj.data.is_editmode
        # Deliberately leave the edits in BMesh. The operator must commit them.
        assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
        assert bpy.context.mode == "OBJECT"
        output = TemplateReader(str(tmp_path / "head.dna"), "head")
        for role, (obj, shape) in inputs.items():
            assert not obj.data.is_editmode
            expected = shape.positions + offset if role in edit_roles else shape.positions
            np.testing.assert_allclose(output.meshes[shape.name].positions, expected, atol=3e-5, rtol=0)
            expected_uv = shape.uv + uv_offset if role in edit_roles else shape.uv
            np.testing.assert_allclose(read_uv(obj.data), expected_uv, atol=1e-7, rtol=0)
    finally:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
