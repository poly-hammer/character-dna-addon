"""Head-only preset contracts, tag browsing, and real DNA conversion."""

import json

from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.dna_io import mesh_data
from character_dna.editors.converter import catalog, templates, ui
from character_dna.editors.converter.native.template import TemplateReader
from character_dna.editors.converter.properties import get_converter_properties

from . import test_templates
from .test_templates import Layout, read_package_recipe


setup_scene = test_templates.setup_scene


HEAD_INPUTS = {
    "metahuman_head_basic.json": {"head"},
    "metahuman_head_advanced.json": {"head", "eye_left", "eye_right", "teeth"},
    "3d_scan_store_head_basic.json": {"head"},
    "3d_scan_store_head_advanced.json": {
        "head",
        "eye_left",
        "eye_right",
        "lens_left",
        "lens_right",
        "teeth",
        "eye_edge",
        "brows",
        "eyelashes",
    },
}


@pytest.mark.parametrize("filename,inputs", HEAD_INPUTS.items())
def test_head_inputs_and_component_ui(setup_scene, filename, inputs):
    props, _ = setup_scene
    templates.load_template(props, templates.TEMPLATE_FOLDER / filename)
    assert {item.role for item in templates.input_rows(props)} == inputs
    assert len(props.extra_meshes) == 9
    assert set(templates.current_dna_files(props)) == {"head"}
    assert all(path.is_file() for path in templates.current_dna_files(props).values())
    assert props.scene_layout == props.input_layout == "SPLIT" and props.dna_layout == "SEPARATE"
    assert props.pose_normalization == "OFF" and not (props.adjust_seam or props.constrain_head_to_body)
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    fields = [args[1] for name, args, _ in layout.calls if name == "prop"]
    assert fields.count("scene_object") == len(inputs)
    assert not {"input_layout", "scene_layout", "dna_layout", "joined_mesh"} & set(fields)
    panel = SimpleNamespace(
        layout=layout, _draw_active_extra_mesh=ui.CHARACTER_DNA_PT_converter_config_meshes._draw_active_extra_mesh
    )
    ui.CHARACTER_DNA_PT_converter_config_footer.draw(panel, bpy.context)
    fields = [args[1] for name, args, _ in layout.calls if name == "prop"]
    assert not {"pose_normalization", "constrain_head_to_body", "adjust_seam"} & set(fields)
    recipe = templates.from_properties(props)
    templates.validate_template(recipe)
    if filename == "3d_scan_store_head_basic.json":
        assert not recipe["meshes"][0].get("preserve_clearance_references")


def test_filter_combination_does_not_change_selection_or_targets(setup_scene, monkeypatch):
    from character_dna.editors.converter.callbacks import get_templates

    props, obj = setup_scene
    props.extra_meshes[0].scene_object = obj
    catalog.sync_filters(props)
    before = props.template_file, props.template_data
    props.template_tag_filters["head"].enabled = True
    props.template_tag_filters["advanced"].enabled = True
    selected = catalog.items(props, filtered=True)
    assert {Path(row[0]).name for row in selected} == {
        "metahuman_head_advanced.json",
        "3d_scan_store_head_advanced.json",
    }
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    assert any(
        name == "label" and kwargs.get("text") == "4 filtered from view" and kwargs.get("icon") == "INFO"
        for name, _, kwargs in layout.calls
    )
    props.template_tag_match = "ANY"
    assert len(catalog.items(props, filtered=True)) == 6
    props.template_tag_match = "ALL"
    props.template_tag_filters["full body"].enabled = True
    assert catalog.items(props, filtered=True) == []
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    assert any(name == "label" and kwargs.get("text") == "6 filtered from view" for name, _, kwargs in layout.calls)
    assert [row[0] for row in get_templates(props, bpy.context)] == [props.template_file]
    assert props.template == props.template_file
    assert (props.template_file, props.template_data) == before
    assert props.extra_meshes[0].scene_object == obj
    assert bpy.ops.character_dna.converter_template_tag(action="CLEAR") == {"FINISHED"}
    assert len(catalog.items(props, filtered=True)) == 7
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    assert not any(kwargs.get("text", "").endswith(" filtered from view") for _, _, kwargs in layout.calls)

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("Cached dropdown drawing must not read template files")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    first = catalog.items(props, filtered=True)
    assert all(catalog.items(props, filtered=True) is first for _ in range(20))


def test_tag_edit_save_load_and_filtered_enum(setup_scene, tmp_path):
    props, obj = setup_scene
    props.extra_meshes[0].scene_object = obj
    assert bpy.ops.character_dna.converter_template_tag(action="ADD") == {"FINISHED"}
    props.template_tags[-1].value = "  Studio Test  "
    assert props.template_tags[-1].value == "Studio Test"
    assert "studio test" in props.template_tag_filters
    path = tmp_path / "custom.zip"
    templates.export_template(props, path)
    assert "Studio Test" in read_package_recipe(path)["tags"]
    key = props.template_id
    installed = Path(props.template_file)
    identifier = next(row[4] for row in catalog.items(props) if row[0] == key)
    templates.load_template(props, templates.TEMPLATE_FOLDER / "metahuman_head_basic.json")
    props.template_tag_filters["studio test"].enabled = True
    assert [row[0] for row in catalog.items(props, filtered=True)] == [key]
    props.template = key
    assert props.extra_meshes[0].scene_object == obj
    assert props.template_tags[-1].value == "Studio Test"
    assert catalog.items(props, filtered=True)[0][4] == identifier
    props.template_tags[-1].value = "Custom"
    assert props.template_tags[-1].value == "Custom"
    props.template_tags_index = 3
    assert bpy.ops.character_dna.converter_template_tag(action="REMOVE") == {"FINISHED"}
    assert len(props.template_tags) == 3
    # Metadata changes made externally become visible only on explicit refresh.
    saved = json.loads(installed.read_text())
    saved["tags"] = ["External"]
    installed.write_text(json.dumps(saved))
    templates.load_template(props, templates.TEMPLATE_FOLDER / "metahuman_head_basic.json")
    assert bpy.ops.character_dna.converter_template_tag(action="REFRESH") == {"FINISHED"}
    assert "external" in props.template_tag_filters


def test_tags_and_filter_state_survive_blend_reload(empty_character_scene, tmp_path):
    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / "metahuman_head_advanced.json")
    bpy.ops.character_dna.converter_template_tag(action="ADD")
    props.template_tags[-1].value = "Studio"
    template = tmp_path / "studio.zip"
    templates.export_template(props, template)
    key, installed = props.template_id, props.template_file
    props.template_tag_filters["head"].enabled = True
    props.template_tag_filters["studio"].enabled = True
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / "tags.blend"))
    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / "tags.blend"))
    props = get_converter_properties()
    assert props.template_file == installed
    assert props.template_id == key
    assert props.template_tags[-1].value == "Studio"
    assert {item.name for item in props.template_tag_filters if item.enabled} == {"head", "studio"}
    assert [row[0] for row in catalog.items(props, filtered=True)] == [key]


@pytest.mark.parametrize("filename", HEAD_INPUTS)
def test_head_conversion_preserves_primary_uv_and_relative_anatomy(empty_character_scene, tmp_path, filename):
    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / filename)
    templates.export_template(props, tmp_path / "head-template.zip")
    source = TemplateReader(str(templates.current_dna_files(props)["head"]), "head")
    offset = np.array([0.0, -0.002, 0.003])
    for item in templates.input_rows(props):
        if item.mesh_part == "RIGHT":
            item.scene_object = next(
                row.scene_object for row in props.extra_meshes if row.role == item.role.replace("right", "left")
            )
            continue
        names = [item.dna_mesh_name]
        if item.mesh_part == "LEFT":
            # Scan Store presets accept the usual joined pair, then select each side.
            names.append(
                next(row.dna_mesh_name for row in props.extra_meshes if row.role == item.role.replace("left", "right"))
            )
        shapes = [source.meshes[name] for name in names]
        faces, vertex_offset = [], 0
        for shape in shapes:
            faces.extend(
                (face + vertex_offset).tolist() for face in np.split(shape.corners, np.cumsum(shape.sizes)[:-1])
            )
            vertex_offset += len(shape.positions)
        mesh = bpy.data.meshes.new(item.role)
        mesh.from_pydata(np.concatenate([shape.positions for shape in shapes]) + offset, [], faces)
        mesh.uv_layers.new().data.foreach_set("uv", np.concatenate([shape.uv for shape in shapes]).ravel())
        obj = bpy.data.objects.new(item.role, mesh)
        bpy.context.scene.collection.objects.link(obj)
        item.scene_object = obj
    props.new_name, props.new_folder = "HeadPreset", str(tmp_path)
    assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
    assert {path.name for path in tmp_path.glob("*.dna")} == {"head.dna"}
    instance = bpy.context.scene.character_dna.rig_instance_list["HeadPreset"]
    assert instance.head_rig and instance.face_board and not instance.body_rig
    output = TemplateReader(str(tmp_path / "head.dna"), "head")
    for name, shape in source.meshes.items():
        result = output.meshes[name]
        np.testing.assert_array_equal(result.uv, shape.uv)
        replacement = next(item.fit_method == "REPLACE" for item in props.extra_meshes if item.dna_mesh_name == name)
        np.testing.assert_array_equal(
            result.position_ids, np.arange(len(shape.positions)) if replacement else shape.position_ids
        )
        keep = np.ones(len(shape.positions), dtype=bool)
        if filename == "3d_scan_store_head_advanced.json" and name == "head_lod0_mesh":
            # Supplied teeth can expand the painted mouth pocket during clearance.
            record = mesh_data.record_for(source, "head", shape.index)
            for mask in record["preservations"].values():
                vertices = [
                    index for index, weight in zip(mask["vertices"], mask["weights"], strict=True) if weight > 0
                ]
                keep[np.isin(shape.position_ids, vertices)] = False
            assert not keep.all()
        np.testing.assert_allclose(result.positions[keep], (shape.positions + offset)[keep], atol=3e-5)
        assert np.isfinite(result.positions).all()
    assert all(item.scene_object.hide_get() and item.scene_object.hide_render for item in templates.input_rows(props))
