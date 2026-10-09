"""File-backed template selection, required targets, migration and panel contracts."""

import json

from pathlib import Path
from types import SimpleNamespace

import bpy
import pytest

from character_dna.editors.converter import dna_components, templates, ui
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


@pytest.mark.parametrize("preset,count", [("basic", 2), ("advanced", 5), ("exact", 10)])
def test_presets_share_dna_and_keep_all_relationships(setup_scene, preset, count):
    props, obj = setup_scene
    next(item for item in props.extra_meshes if item.role == "head").scene_object = obj
    path = templates.TEMPLATE_FOLDER / f"metahuman_{preset}.json"
    props.template = str(path)
    assert props.template_file == str(path)
    assert props.template_name == f"MetaHuman ({preset.title()})"
    assert len(props.extra_meshes) == 10
    assert sum(item.fit_method == "WRAP" for item in props.extra_meshes) == count
    assert next(item for item in props.extra_meshes if item.role == "head").scene_object == obj
    files = templates.current_dna_files(props)
    assert files == templates.resolve_dna_files(templates.default_template())
    assert all(path.is_file() for path in files.values())


@pytest.mark.parametrize("preset", templates.BUNDLED_TEMPLATES)
def test_target_picker_excludes_other_assignments_and_releases_cleared_meshes(setup_scene, preset):
    props, obj = setup_scene
    props.template = str(preset)
    poll = ConverterExtraMeshItem.__annotations__["scene_object"].keywords["poll"]
    for assigned in props.extra_meshes:
        assigned.scene_object = obj
        for candidate in props.extra_meshes:
            assert poll(candidate, obj) == (candidate == assigned)
        assigned.scene_object = None
        assert all(poll(candidate, obj) for candidate in props.extra_meshes)


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
    props.extra_meshes[2].fit_method = "WRAP"
    assert bpy.ops.character_dna.converter_template(action="SAVE") == {"FINISHED"}
    props.template = str(templates.BUNDLED_TEMPLATES[0])
    props.template = str(path)
    assert props.extra_meshes[2].fit_method == "WRAP"
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
    props, _ = setup_scene
    path = tmp_path / "stored-template.json"
    templates.save_template(props, path)
    blend = tmp_path / "template-scene.blend"
    bpy.data.libraries.write(str(blend), {bpy.context.scene})
    with bpy.data.libraries.load(str(blend)) as (source, destination):
        destination.scenes = source.scenes
    restored = destination.scenes[0]
    try:
        loaded = restored.character_dna.converter
        assert loaded.template == str(path)
        assert loaded.template_file == str(path)
        assert templates.current_dna_files(loaded) == templates.current_dna_files(props)
        assert len(loaded.extra_meshes) == 10
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
def test_every_wrap_target_is_required_before_import(setup_scene, tmp_path, monkeypatch, missing):
    props, obj = setup_scene
    props.template = str(templates.BUNDLED_TEMPLATES[2])
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
    assert rows["teeth"].fit_method == "WRAP"
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


@pytest.mark.parametrize("preset,count", [(0, 2), (1, 5), (2, 10)])
def test_panel_exposes_wrap_targets_and_only_custom_save(setup_scene, tmp_path, preset, count):
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
        if item.fit_method == "WRAP":
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
