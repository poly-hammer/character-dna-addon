"""File-backed template selection, required targets, migration and panel contracts."""

import json

from pathlib import Path
from types import SimpleNamespace

import bpy
import pytest

from character_dna.editors.converter import templates, ui
from character_dna.editors.converter.dna import components as dna_components
from character_dna.editors.converter.operators import CHARACTER_DNA_OT_convert_to_dna
from character_dna.editors.converter.properties import ConverterExtraMeshItem, get_converter_properties


@pytest.fixture
def setup_scene():
    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("Template tests")
    bpy.context.window.scene = scene
    props = get_converter_properties()
    templates.migrate(props)
    data = bpy.data.meshes.new("Template target")
    obj = bpy.data.objects.new("Template target", data)
    scene.collection.objects.link(obj)
    try:
        yield props, obj
    finally:
        bpy.context.window.scene = previous
        bpy.data.scenes.remove(scene)
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)


@pytest.mark.parametrize("preset,count", [("basic", 2), ("advanced", 5)])
def test_presets_share_dna_and_keep_all_relationships(setup_scene, preset, count):
    props, obj = setup_scene
    next(item for item in props.extra_meshes if item.role == "head").scene_object = obj
    path = templates.TEMPLATE_FOLDER / f"metahuman_{preset}.json"
    props.template = str(path)
    assert props.template_file == str(path)
    assert props.template_name == f"MetaHuman (Full Body {preset.title()})"
    assert len(props.extra_meshes) == 10
    assert sum(item.fit_method == "COPY" for item in props.extra_meshes) == count
    assert next(item for item in props.extra_meshes if item.role == "head").scene_object == obj
    files = templates.current_dna_files(props)
    assert files == templates.resolve_dna_files(templates.default_template())
    assert all(path.is_file() for path in files.values())


def test_scan_store_requires_six_accessory_copy_inputs(setup_scene):
    props, _ = setup_scene
    props.template = str(templates.BUNDLED_TEMPLATES[-1])
    assert props.input_layout == "JOINED"
    assert {item.dna_mesh_name for item in templates.input_rows(props)} == {
        "teeth_lod0_mesh",
        "eyeLeft_lod0_mesh",
        "eyeRight_lod0_mesh",
        "eyeLensLeft_lod0_mesh",
        "eyeLensRight_lod0_mesh",
        "eyeEdge_lod0_mesh",
    }
    assert {item.role for item in props.extra_meshes if item.fit_method == "RELATIVE"} == {"brows", "eyelashes"}
    assert all(item.fit_method == "COPY" for item in templates.input_rows(props))
    assert {item.role: item.mesh_part for item in templates.input_rows(props)} == {
        "eye_left": "LEFT",
        "eye_right": "RIGHT",
        "lens_left": "LEFT",
        "lens_right": "RIGHT",
        "teeth": "FULL",
        "eye_edge": "FULL",
    }
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    fields = [args[1] for name, args, _ in layout.calls if name == "prop"]
    assert fields.count("scene_object") == 6 and fields.count("joined_mesh") == 1


def test_bundled_enum_ids_survive_retiring_exact(setup_scene):
    from character_dna.editors.converter.callbacks import get_templates

    props, obj = setup_scene
    entries = get_templates(props, bpy.context)
    assert [entry[4] for entry in entries] == [0, 1, 4]
    assert [entry[1] for entry in entries] == [
        "MetaHuman (Full Body Basic)",
        "MetaHuman (Full Body Advanced)",
        "3D Scan Store (Full Body)",
    ]
    assert not (templates.TEMPLATE_FOLDER / "metahuman_exact.json").exists()
    with pytest.raises(ValueError, match="Unknown bundled template"):
        templates.default_template("exact")
    for item in props.extra_meshes:
        item.fit_method = "COPY"
        item.scene_object = obj
    props.template_file = str(templates.TEMPLATE_FOLDER / "metahuman_exact.json")
    props["template"] = 2
    files = templates.current_dna_files(props)
    templates.migrate(props)
    assert props.template == "__scene__" and props.template_file == ""
    assert all(item.fit_method == "COPY" and item.scene_object == obj for item in props.extra_meshes)
    assert templates.current_dna_files(props) == files


@pytest.mark.parametrize("path", templates.BUNDLED_TEMPLATES)
def test_uv_default_and_custom_override_round_trip(setup_scene, tmp_path, path):
    props, _ = setup_scene
    props.validate_uvs = False
    templates.load_template(props, path)
    assert props.validate_uvs
    props.validate_uvs = False
    saved = tmp_path / "uv-override.json"
    templates.save_template(props, saved)
    assert not json.loads(saved.read_text())["validate_uvs"]
    assert not props.validate_uvs


def test_legacy_json_method_loads_and_saves_as_copy(setup_scene, tmp_path):
    props, obj = setup_scene
    recipe = templates.default_template("advanced")
    for row in recipe["meshes"]:
        if row["method"] == "COPY":
            row["method"] = "WRAP"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(recipe))
    props.extra_meshes[0].scene_object = obj
    props.extra_meshes[0].material_source = obj
    templates.load_template(props, path)
    assert sum(item.fit_method == "COPY" for item in props.extra_meshes) == 5
    assert props.extra_meshes[0].scene_object == props.extra_meshes[0].material_source == obj
    assert '"WRAP"' in path.read_text()  # Loading does not overwrite the user's file.
    templates.save_template(props, path)
    saved = json.loads(path.read_text())
    assert {row["method"] for row in saved["meshes"]} == {"COPY", "RELATIVE"}


@pytest.mark.parametrize("stored_method", [0, "WRAP"])
def test_old_scene_methods_migrate_without_changing_bindings(setup_scene, stored_method):
    props, obj = setup_scene
    item = props.extra_meshes[0]
    item.scene_object = item.material_source = obj
    item.property_unset("fit_method")
    item["fit_method"] = stored_method
    item.hide_render = True
    props.template_data = props.template_data.replace('"COPY"', '"WRAP"')
    templates.migrate(props)
    assert item.fit_method == "COPY" and item.get("fit_method") == 0
    assert item.scene_object == item.material_source == obj and item.hide_render
    assert '"WRAP"' not in props.template_data
    assert props.extra_meshes[2].fit_method == "RELATIVE"


@pytest.mark.parametrize("preset", templates.BUNDLED_TEMPLATES)
def test_target_picker_excludes_other_assignments_and_releases_cleared_meshes(setup_scene, preset):
    props, obj = setup_scene
    props.template = str(preset)
    props.input_layout = "SPLIT"
    for item in props.extra_meshes:
        item.fit_method = "COPY"
        item.mesh_part = "FULL"
    poll = ConverterExtraMeshItem.__annotations__["scene_object"].keywords["poll"]
    for assigned in props.extra_meshes:
        assigned.scene_object = obj
        for candidate in props.extra_meshes:
            assert poll(candidate, obj) == (candidate == assigned)
        assigned.scene_object = None
        assert all(poll(candidate, obj) for candidate in props.extra_meshes)


@pytest.mark.parametrize("first", ["FULL", "LEFT", "RIGHT"])
@pytest.mark.parametrize("second", ["FULL", "LEFT", "RIGHT"])
def test_side_sharing_matches_picker_and_conversion_validation(setup_scene, first, second):
    props, obj = setup_scene
    a, b = props.extra_meshes[:2]
    a.mesh_part, b.mesh_part = first, second
    a.scene_object = obj
    poll = ConverterExtraMeshItem.__annotations__["scene_object"].keywords["poll"]
    allowed = first != "FULL" and second != "FULL"
    assert poll(b, obj) == allowed
    # Scripts can bypass PointerProperty polling; conversion must reject conflicts too.
    b.scene_object = obj
    if allowed:
        templates.validate_input_bindings(props)
    else:
        with pytest.raises(ValueError, match="Full inputs are exclusive"):
            templates.validate_input_bindings(props)
    a.scene_object = None
    assert poll(b, obj)


def test_merged_source_is_exclusive_and_inactive_rows_do_not_reserve_inputs(setup_scene):
    from character_dna.editors.converter.properties import ConverterProperties

    props, obj = setup_scene
    props.template = str(templates.BUNDLED_TEMPLATES[-1])
    rows = {item.role: item for item in props.extra_meshes}
    poll = ConverterExtraMeshItem.__annotations__["scene_object"].keywords["poll"]
    full_body_poll = ConverterProperties.__annotations__["joined_mesh"].keywords["poll"]
    rows["head"].scene_object = obj  # Stale split target, inactive in merged layout.
    rows["brows"].scene_object = obj  # Relative fits have no target binding.
    assert poll(rows["eye_left"], obj) and full_body_poll(props, obj)
    rows["eye_left"].scene_object = obj
    assert not full_body_poll(props, obj)
    props.joined_mesh = obj
    assert not poll(rows["eye_left"], obj)
    with pytest.raises(ValueError, match="Full inputs are exclusive"):
        templates.validate_input_bindings(props)


def test_mesh_part_defaults_and_legacy_scenes_remain_full(setup_scene):
    props, _ = setup_scene
    assert all(item.mesh_part == "FULL" for item in props.extra_meshes)
    for item in props.extra_meshes:
        item.property_unset("mesh_part")
    templates.migrate(props)
    assert all(row["mesh_part"] == "FULL" for row in templates.from_properties(props)["meshes"])
    recipe = templates.default_template()
    recipe["meshes"][0]["mesh_part"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid mesh part"):
        templates.validate_template(recipe)


def test_target_picker_rejects_non_mesh_objects(setup_scene):
    props, _ = setup_scene
    poll = ConverterExtraMeshItem.__annotations__["scene_object"].keywords["poll"]
    empty = bpy.data.objects.new("Non-mesh target", None)
    try:
        assert not poll(props.extra_meshes[0], empty)
    finally:
        bpy.data.objects.remove(empty)


def test_target_picker_uses_owning_scene_instead_of_active_scene(setup_scene):
    props, obj = setup_scene
    poll = ConverterExtraMeshItem.__annotations__["scene_object"].keywords["poll"]
    previous = bpy.context.window.scene
    other_scene = bpy.data.scenes.new("Other converter scene")
    try:
        bpy.context.window.scene = other_scene
        other_props = get_converter_properties()
        templates.migrate(other_props)
        props.extra_meshes[0].scene_object = obj
        assert not poll(props.extra_meshes[1], obj)
        assert poll(other_props.extra_meshes[1], obj)
        props.extra_meshes[0].scene_object = None
        other_props.extra_meshes[0].scene_object = obj
        assert poll(props.extra_meshes[1], obj)
        assert not poll(other_props.extra_meshes[1], obj)
    finally:
        bpy.context.window.scene = previous
        bpy.data.scenes.remove(other_scene)


def test_custom_template_save_load_and_relative_links(setup_scene, tmp_path):
    props, obj = setup_scene
    recipe = templates.default_template()
    recipe["name"] = "My scan"
    recipe["dna_files"] = {kind: str(tmp_path / f"{kind}.dna") for kind in ("head", "body")}
    for path in recipe["dna_files"].values():
        Path(path).write_bytes(b"DNA link test")
    templates.apply_template(props, recipe)
    next(item for item in props.extra_meshes if item.role == "head").scene_object = obj
    folder = tmp_path / "templates"
    folder.mkdir()
    path = folder / "scan.json"
    templates.save_template(props, path)
    assert props.template == str(path)
    assert props.template_files.get(str(path)).label == "My scan"
    assert json.loads(path.read_text())["dna_files"]["head"] == "../head.dna"
    assert str(obj.name) not in path.read_text()
    assert templates.current_dna_files(props)["head"] == tmp_path / "head.dna"
    props.extra_meshes[2].fit_method = "COPY"
    props.extra_meshes[2].mesh_part = "RIGHT"
    props.extra_meshes[2].material_source = obj
    assert bpy.ops.character_dna.converter_template(action="SAVE") == {"FINISHED"}
    props.template = str(templates.BUNDLED_TEMPLATES[0])
    props.template = str(path)
    assert props.extra_meshes[2].fit_method == "COPY"
    assert props.extra_meshes[2].mesh_part == "RIGHT"
    assert props.extra_meshes[2].material_source == obj
    assert "transfer_material" not in path.read_text()
    assert "material_source" not in path.read_text()
    assert next(item for item in props.extra_meshes if item.role == "head").scene_object == obj


def test_new_template_operator_links_arbitrary_dna_and_preserves_installation(setup_scene, tmp_path):
    props, _ = setup_scene
    bundled = templates.BUNDLED_TEMPLATES[0]
    before = bundled.read_bytes()
    with pytest.raises(RuntimeError, match="custom template"):
        bpy.ops.character_dna.converter_template(action="SAVE")
    with pytest.raises(ValueError, match="read-only"):
        templates.save_template(props, bundled)
    with pytest.raises(ValueError, match="read-only"):
        templates.save_template(props, templates.ADDON_FOLDER / "custom.json")
    path = tmp_path / "custom.json"
    assert bpy.ops.character_dna.converter_template(
        action="NEW",
        filepath=str(path),
        template_name="Custom DNA",
        head_dna_file=str(tmp_path / "my-head.dna"),
        body_dna_file=str(tmp_path / "my-body.dna"),
    ) == {"FINISHED"}
    assert templates.current_dna_files(props)["head"] == tmp_path / "my-head.dna"
    assert bundled.read_bytes() == before


def test_custom_template_selection_survives_blend_storage(setup_scene, tmp_path):
    props, source_object = setup_scene
    props.extra_meshes[0].material_source = source_object
    props.extra_meshes[0].hide_viewport = True
    props.extra_meshes[0].hide_render = False
    props.extra_meshes[0].mesh_part = "LEFT"
    path = tmp_path / "stored-template.json"
    templates.save_template(props, path)
    # Existing .blend files store the original operation as enum integer zero.
    props.extra_meshes[0]["fit_method"] = 0
    props.template_data = props.template_data.replace('"COPY"', '"WRAP"')
    blend = tmp_path / "template-scene.blend"
    bpy.data.libraries.write(str(blend), {bpy.context.scene})
    with bpy.data.libraries.load(str(blend)) as (source, destination):
        destination.scenes = source.scenes
    restored = destination.scenes[0]
    try:
        loaded = restored.character_dna.converter
        templates.migrate(loaded)
        assert loaded.extra_meshes[0].fit_method == "COPY"
        assert loaded.extra_meshes[0].mesh_part == "LEFT"
        assert '"WRAP"' not in loaded.template_data
        assert loaded.template == str(path)
        assert loaded.template_file == str(path)
        assert templates.current_dna_files(loaded) == templates.current_dna_files(props)
        assert len(loaded.extra_meshes) == 10
        assert loaded.extra_meshes[0].material_source in restored.objects.values()
        assert loaded.extra_meshes[0].material_source.type == "MESH"
        assert loaded.extra_meshes[0].hide_viewport and not loaded.extra_meshes[0].hide_render
        assert dna_components.raw_paths(loaded) == dna_components.raw_paths(props)
    finally:
        objects = list(restored.objects)
        bpy.data.scenes.remove(restored)
        for obj in objects:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)


@pytest.mark.parametrize("missing", [row["id"] for row in templates.default_template()["meshes"]])
def test_every_copy_target_is_required_before_import(setup_scene, tmp_path, monkeypatch, missing):
    props, obj = setup_scene
    recipe = templates.default_template()
    for row in recipe["meshes"]:
        row["method"] = "COPY"
    templates.apply_template(props, recipe)
    for item in props.extra_meshes:
        item.scene_object = obj if item.role != missing else None
    props.new_name, props.new_folder = "RequiredTargets", str(tmp_path)
    before = set(bpy.data.objects)

    def unexpected_import(*_args):
        pytest.fail("Missing targets must fail before DNA import")

    monkeypatch.setattr(CHARACTER_DNA_OT_convert_to_dna, "_import_component", unexpected_import)
    mesh_name = next(item.dna_mesh_name for item in props.extra_meshes if item.role == missing)
    with pytest.raises(RuntimeError, match=f"requires a target mesh for: {mesh_name}"):
        bpy.ops.character_dna.convert_to_dna()
    assert set(bpy.data.objects) == before
    assert not list(tmp_path.iterdir())
    assert not props.convert_is_running


@pytest.mark.parametrize("missing", ["teeth", "eye_left", "eye_right", "lens_left", "lens_right", "eye_edge"])
def test_scan_store_accessory_targets_are_required_before_import(setup_scene, tmp_path, monkeypatch, missing):
    props, obj = setup_scene
    props.template = str(templates.BUNDLED_TEMPLATES[-1])
    props.joined_mesh = obj
    for item in templates.input_rows(props):
        item.scene_object = None if item.role == missing else obj
    props.new_name, props.new_folder = "RequiredScanInputs", str(tmp_path)

    def unexpected_import(*_args):
        pytest.fail("Missing accessory targets must fail before DNA import")

    monkeypatch.setattr(CHARACTER_DNA_OT_convert_to_dna, "_import_component", unexpected_import)
    mesh = next(item.dna_mesh_name for item in props.extra_meshes if item.role == missing)
    with pytest.raises(RuntimeError, match=f"Copy to Target requires a target mesh for: {mesh}"):
        bpy.ops.character_dna.convert_to_dna()
    assert not list(tmp_path.iterdir())


def test_previous_primary_inputs_migrate_once_without_losing_settings(setup_scene):
    props, obj = setup_scene
    props.template = str(templates.BUNDLED_TEMPLATES[1])
    recipe = json.loads(props.template_data)
    recipe.pop("dna_files")
    props.template_data = json.dumps(recipe)
    for index in reversed(range(len(props.extra_meshes))):
        if props.extra_meshes[index].role in {"head", "body"}:
            props.extra_meshes.remove(index)
    props.head_mesh = props.body_mesh = obj
    props.head_dna_mesh = "scan_head"
    props.relationships_version = 0
    templates.migrate(props)
    templates.migrate(props)
    rows = {item.role: item for item in props.extra_meshes}
    assert len(rows) == len(props.extra_meshes) == 10
    assert rows["head"].scene_object == rows["body"].scene_object == obj
    assert rows["head"].dna_mesh_name == "scan_head"
    assert rows["teeth"].fit_method == "COPY"
    assert props.template == "__scene__"
    assert templates.from_properties(props)["dna_files"]


def test_missing_template_selection_retains_unsaved_setup(setup_scene, tmp_path):
    props, _ = setup_scene
    templates.apply_template(props, templates.default_template())
    before = props.template_data
    missing = tmp_path / "missing.json"
    templates.remember_template(props, missing, "Missing template")
    props.template = str(missing)
    assert props.template == "__scene__"
    assert props.template_data == before
    assert props.convert_has_error


class Layout:
    """Capture the panel's rendered controls using real RNA property data."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def draw(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return SimpleNamespace() if name == "operator" else self

        return draw


@pytest.mark.parametrize("method", ["COPY", "RELATIVE"])
@pytest.mark.parametrize("input_layout", ["SPLIT", "JOINED"])
def test_relationships_expose_material_and_visibility_settings(setup_scene, method, input_layout):
    props, _ = setup_scene
    props.input_layout = input_layout
    props.extra_meshes_index = 0
    props.extra_meshes[0].fit_method = method
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter_config._draw_active_extra_mesh(layout, props)
    fields = [args[1] for name, args, _ in layout.calls if name == "prop"]
    assert {"material_source", "hide_viewport", "hide_render"} <= set(fields)
    assert "transfer_material" not in fields
    assert ("mesh_part" in fields) == (method == "COPY" and input_layout == "SPLIT")


@pytest.mark.parametrize("input_layout", ["SPLIT", "JOINED"])
def test_previous_material_checkbox_migrates_once(setup_scene, input_layout):
    props, obj = setup_scene
    props.input_layout = input_layout
    props.joined_mesh = obj
    item = next(item for item in props.extra_meshes if item.role == "head")
    item.scene_object = obj if input_layout == "SPLIT" else None
    item.transfer_material = True
    templates.migrate(props)
    assert item.material_source == obj
    assert not item.transfer_material
    item.material_source = None
    templates.migrate(props)
    assert item.material_source is None


def test_material_picker_accepts_shared_input_source_and_rejects_non_mesh(setup_scene):
    props, obj = setup_scene
    poll = ConverterExtraMeshItem.__annotations__["material_source"].keywords["poll"]
    props.extra_meshes[0].scene_object = obj
    props.extra_meshes[0].material_source = obj
    assert poll(props.extra_meshes[1], obj)
    empty = bpy.data.objects.new("Non-mesh material source", None)
    try:
        assert not poll(props.extra_meshes[1], empty)
    finally:
        bpy.data.objects.remove(empty)


@pytest.mark.parametrize("preset", templates.BUNDLED_TEMPLATES)
def test_template_controls_eye_edge_visibility(setup_scene, preset, tmp_path):
    props, _ = setup_scene
    props.template = str(preset)
    item = next(item for item in props.extra_meshes if item.role == "eye_edge")
    expected = preset.name.startswith("metahuman_")
    assert item.hide_viewport == item.hide_render == expected
    # Save independent user choices, then reload and migrate without resetting them.
    item.hide_viewport, item.hide_render = True, False
    path = tmp_path / "visibility.json"
    templates.save_template(props, path)
    saved = next(row for row in json.loads(path.read_text())["meshes"] if row["id"] == "eye_edge")
    assert saved["hide_viewport"] is True and saved["hide_render"] is False
    templates.migrate(props)
    item = next(item for item in props.extra_meshes if item.role == "eye_edge")
    assert item.hide_viewport is True and item.hide_render is False


@pytest.mark.parametrize("field", ["hide_viewport", "hide_render"])
def test_visibility_configuration_rejects_non_boolean(field):
    recipe = templates.default_template()
    recipe["meshes"][0][field] = "false"
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        templates.validate_template(recipe)


@pytest.mark.parametrize("preset", templates.BUNDLED_TEMPLATES)
def test_old_scene_visibility_migrates_once_from_template(setup_scene, preset):
    props, _ = setup_scene
    props.template = str(preset)
    item = next(item for item in props.extra_meshes if item.role == "eye_edge")
    item.property_unset("hide_viewport")
    item.property_unset("hide_render")
    recipe = json.loads(props.template_data)
    for row in recipe["meshes"]:
        row.pop("hide_viewport", None)
        row.pop("hide_render", None)
    props.template_data = json.dumps(recipe)
    templates.migrate(props)
    assert item.hide_viewport == item.hide_render == preset.name.startswith("metahuman_")
    item.hide_viewport = item.hide_render = False
    templates.migrate(props)
    assert not item.hide_viewport and not item.hide_render


@pytest.mark.parametrize(
    "mesh,component", [("eyeEdge_lod0_mesh", "head"), ("body_lod0_mesh", "body"), ("custom_surface", "head")]
)
@pytest.mark.parametrize("hidden,render_disabled", [(False, False), (True, False), (False, True), (True, True)])
def test_output_visibility_survives_lod_switches(setup_scene, mesh, component, hidden, render_disabled):
    from character_dna.editors.converter.pipeline import ConversionPipeline
    from character_dna.ui.callbacks import set_active_lod

    _, obj = setup_scene
    obj.name = f"Visibility_{mesh}"
    instance = bpy.context.scene.character_dna.rig_instance_list.add()
    instance.name = "Visibility"
    obj.hide_viewport = obj.hide_render = True
    pipeline = SimpleNamespace(
        objects={"mesh": obj},
        rows={"mesh": {"component": component, "hide_viewport": hidden, "hide_render": render_disabled}},
        operator=SimpleNamespace(_rig_instance=instance),
    )
    ConversionPipeline.apply_visibility(pipeline)
    assert obj.hide_viewport == obj.hide_get() == hidden
    assert obj.hide_render == render_disabled
    # Exercise the LOD handler without loading DNA just to populate the picker.
    set_active_lod(instance.view_options, 2)
    assert obj.hide_get() and obj.hide_render
    set_active_lod(instance.view_options, 0)
    assert obj.hide_viewport == obj.hide_get() == hidden
    assert obj.hide_render == render_disabled


@pytest.mark.parametrize("hidden,render_disabled", [(False, False), (True, False), (False, True), (True, True)])
def test_merged_surface_obeys_component_visibility(setup_scene, hidden, render_disabled):
    from character_dna.editors.converter.scene.merge import sync_visibility

    _, head = setup_scene
    body, surface = head.copy(), head.copy()
    for obj in (body, surface):
        bpy.context.scene.collection.objects.link(obj)
    try:
        head["dna_converter_visibility"] = {"hide_viewport": hidden, "hide_render": render_disabled}
        surface["dna_merge_head"], surface["dna_merge_body"] = head, body
        registry = {"dna_merged_surfaces": {"0": surface}}
        instance = SimpleNamespace(get=registry.get, view_options=SimpleNamespace(active_lod="lod0"))
        sync_visibility(instance)
        assert surface.hide_viewport == surface.hide_get() == hidden
        assert surface.hide_render == render_disabled
        assert head.hide_get() and body.hide_get() and head.hide_render and body.hide_render
    finally:
        for obj in (body, surface):
            bpy.data.objects.remove(obj, do_unlink=True)


@pytest.mark.parametrize("preset,count", [(0, 2), (1, 5)])
def test_panel_exposes_copy_targets_and_only_custom_save(setup_scene, tmp_path, preset, count):
    props, _ = setup_scene
    props.template = str(templates.BUNDLED_TEMPLATES[preset])
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    fields = [args[1] for name, args, _ in layout.calls if name == "prop"]
    assert fields.count("scene_object") == count
    assert "template" in fields
    assert not {"base_dna", "template_name", "head_mesh", "body_mesh", "head_dna_mesh", "body_dna_mesh"} & set(fields)
    assert not any(kwargs.get("icon") == "FILE_TICK" for _, _, kwargs in layout.calls)
    templates.save_template(props, tmp_path / "copy.json")
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter.draw(SimpleNamespace(layout=layout), bpy.context)
    assert any(kwargs.get("icon") == "FILE_TICK" for _, _, kwargs in layout.calls)
    props.extra_meshes_index = 0
    layout = Layout()
    ui.CHARACTER_DNA_PT_converter_config._draw_active_extra_mesh(layout, props)
    fields = [args[1] for name, args, _ in layout.calls if name == "prop"]
    assert "scene_object" in fields
    assert not {"profile", "enabled", "max_bind_distance", "reference_dna_mesh_names"} & set(fields)


@pytest.mark.parametrize("style", ["absolute", "blend", "environment", "environment_blend", "template"])
def test_dna_paths_resolve_against_their_own_origins(tmp_path, monkeypatch, style):
    blend_file = tmp_path / "scene" / "character.blend"
    template_file = tmp_path / "templates" / "custom.json"
    monkeypatch.setenv("CHARACTER_DNA_TEST_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("CHARACTER_DNA_TEST_BLEND", "//assets")
    paths = {
        "absolute": (str(tmp_path / "head.dna"), tmp_path / "head.dna"),
        "blend": ("//head.dna", blend_file.parent / "head.dna"),
        "environment": ("{CHARACTER_DNA_TEST_ROOT}/head.dna", tmp_path / "assets" / "head.dna"),
        "environment_blend": ("{CHARACTER_DNA_TEST_BLEND}/head.dna", blend_file.parent / "assets" / "head.dna"),
        "template": ("../assets/head.dna", tmp_path / "assets" / "head.dna"),
    }
    value, expected = paths[style]
    assert dna_components.resolve_path(value, str(template_file), blend_file=str(blend_file)) == expected


def test_dna_paths_reject_unresolved_environment_and_unsaved_blend(monkeypatch):
    monkeypatch.delenv("CHARACTER_DNA_TEST_UNSET", raising=False)
    with pytest.raises(ValueError, match="CHARACTER_DNA_TEST_UNSET"):
        dna_components.resolve_path("{CHARACTER_DNA_TEST_UNSET}/head.dna")
    with pytest.raises(ValueError, match="Save the blend"):
        dna_components.resolve_path("//head.dna", blend_file="")


def test_component_changes_recache_real_dna_and_release_files(setup_scene, tmp_path, monkeypatch):
    from character_dna.bindings import dna
    from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle

    props, _ = setup_scene
    files = templates.current_dna_files(props)
    head = next(item for item in props.dna_components if item.component == "head")
    assert "head_lod0_mesh" in head.available_meshes
    path = tmp_path / "custom.dna"
    reader = get_dna_reader(files["head"])
    writer = get_dna_writer(path)
    try:
        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        writer.setMeshName(0, "custom_head")
        writer.write()
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)
    calls = []

    def read_definition(path, **kwargs):
        calls.append((path, kwargs["data_layer"]))
        return get_dna_reader(path, **kwargs)

    monkeypatch.setattr(dna_components, "get_dna_reader", read_definition)
    head.filepath = str(path)
    assert not head.error
    assert "custom_head" in head.available_meshes
    assert "head_lod0_mesh" not in head.available_meshes
    assert ("head", "custom_head") in dna_components.available_meshes(props)
    assert all(layer == "Definition" for _, layer in calls)
    count = len(calls)
    dna_components.refresh(props)
    assert len(calls) == count
    renamed = path.with_name("moved.dna")
    path.replace(renamed)
    assert bpy.ops.character_dna.converter_dna_component(action="REFRESH") == {"FINISHED"}
    assert not head.available_meshes
    assert "not found" in head.error
    head.filepath = str(renamed)
    assert "custom_head" in head.available_meshes
    assert templates.current_dna_files(props)["head"] == renamed


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "environment", "duplicate"])
def test_invalid_components_remove_stale_choices(setup_scene, tmp_path, monkeypatch, invalid):
    props, _ = setup_scene
    head = next(item for item in props.dna_components if item.component == "head")
    assert head.available_meshes
    if invalid == "duplicate":
        head.component = "body"
        assert all(item.error for item in props.dna_components)
    elif invalid == "environment":
        monkeypatch.delenv("CHARACTER_DNA_TEST_UNSET", raising=False)
        head.filepath = "{CHARACTER_DNA_TEST_UNSET}/head.dna"
    else:
        path = tmp_path / "invalid.dna"
        if invalid == "corrupt":
            path.write_bytes(b"Not DNA")
        head.filepath = str(path)
    assert head.error
    assert not head.available_meshes


def test_component_list_edits_persist_and_do_not_restore_removed_links(setup_scene):
    props, _ = setup_scene
    assert {item.component for item in props.dna_components} == {"head", "body"}
    props.dna_components_index = 1
    assert bpy.ops.character_dna.converter_dna_component(action="REMOVE") == {"FINISHED"}
    templates.migrate(props)
    assert set(templates.current_dna_files(props)) == {"head"}
    assert bpy.ops.character_dna.converter_dna_component(action="ADD") == {"FINISHED"}
    body = props.dna_components[1]
    assert body.component == "body"
    assert not body.filepath
    assert body.error
    assert bpy.ops.character_dna.converter_dna_component(action="ADD") == {"CANCELLED"}
    for _ in range(2):
        assert bpy.ops.character_dna.converter_dna_component(action="REMOVE") == {"FINISHED"}
    templates.migrate(props)
    assert not props.dna_components
    with pytest.raises(ValueError, match="Add at least one"):
        templates.current_dna_files(props)


def test_template_save_preserves_dynamic_component_paths(setup_scene, tmp_path, monkeypatch):
    props, _ = setup_scene
    original = templates.current_dna_files(props)
    monkeypatch.setenv("CHARACTER_DNA_TEST_HEAD", str(original["head"].parent))
    values = {"head": "{CHARACTER_DNA_TEST_HEAD}/head.dna", "body": "//body.dna"}
    for item in props.dna_components:
        item.filepath = values[item.component]
    target = tmp_path / "custom.json"
    templates.save_template(props, target)
    assert json.loads(target.read_text())["dna_files"] == values
    assert dna_components.raw_paths(props) == values
    assert dna_components.resolve_path(values["head"]) == original["head"]


@pytest.mark.parametrize("invalid", ["missing_file", "missing_mesh"])
def test_component_failure_stops_conversion_before_import(setup_scene, tmp_path, monkeypatch, invalid):
    props, obj = setup_scene
    for item in props.extra_meshes:
        if item.fit_method == "COPY":
            item.scene_object = obj
    props.new_name, props.new_folder = "InvalidComponent", str(tmp_path)
    if invalid == "missing_file":
        props.dna_components[0].filepath = str(tmp_path / "missing.dna")
        error = "DNA file not found"
    else:
        props.extra_meshes[0].dna_mesh_name = "absent_mesh"
        error = "absent_mesh.*not available"

    def unexpected_import(*_args):
        pytest.fail("Invalid components must fail before DNA import")

    monkeypatch.setattr(CHARACTER_DNA_OT_convert_to_dna, "_import_component", unexpected_import)
    with pytest.raises(RuntimeError, match=error):
        bpy.ops.character_dna.convert_to_dna()
    assert not list(tmp_path.iterdir())


def test_component_callbacks_use_the_owning_scene(setup_scene, tmp_path):
    props, _ = setup_scene
    previous = bpy.context.window.scene
    other = bpy.data.scenes.new("Other DNA components")
    try:
        bpy.context.window.scene = other
        other_props = get_converter_properties()
        templates.migrate(other_props)
        props.dna_components[0].filepath = str(tmp_path / "missing.dna")
        assert props.dna_components[0].error
        assert not props.dna_components[0].available_meshes
        assert other_props.dna_components[0].available_meshes
        assert not other_props.dna_components[0].error
    finally:
        bpy.context.window.scene = previous
        bpy.data.scenes.remove(other)


def test_old_template_links_migrate_and_cache_rebuilds_on_reload(setup_scene):
    from character_dna.editors.converter.properties import _initialize_active_template

    props, obj = setup_scene
    props.extra_meshes[0].scene_object = obj
    expected = templates.current_dna_files(props)
    props.dna_components.clear()
    props.components_initialized = False
    templates.migrate(props)
    assert templates.current_dna_files(props) == expected
    props.dna_components[0].available_meshes.clear()
    _initialize_active_template()
    assert props.dna_components[0].available_meshes
    assert props.extra_meshes[0].scene_object == obj


def test_configuration_exposes_components_before_relationships(setup_scene):
    props, _ = setup_scene
    layout = Layout()
    panel = SimpleNamespace(
        layout=layout, _draw_active_extra_mesh=ui.CHARACTER_DNA_PT_converter_config._draw_active_extra_mesh
    )
    ui.CHARACTER_DNA_PT_converter_config.draw(panel, bpy.context)
    lists = [args[3] for name, args, _ in layout.calls if name == "template_list"]
    assert lists == ["dna_components", "extra_meshes"]
    searches = [args for name, args, _ in layout.calls if name == "prop_search"]
    assert searches[0][1] == "dna_mesh_name"
    assert searches[0][2] == props.dna_components[0]
    assert searches[0][3] == "available_meshes"
