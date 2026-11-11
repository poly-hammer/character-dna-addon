"""Portable template installation, isolation, and file transaction contracts."""

import hashlib
import json
import shutil

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import bpy
import pytest

from character_dna.editors.converter import catalog, packages, templates, ui
from character_dna.editors.converter.properties import _initialize_active_template, get_converter_properties
from character_dna.editors.converter.template_operators import CHARACTER_DNA_OT_delete_converter_template

from . import test_templates
from .test_templates import Layout, read_package_recipe


setup_scene = test_templates.setup_scene


@pytest.fixture
def exported(setup_scene, tmp_path):
    props, _ = setup_scene
    templates.load_template(props, templates.TEMPLATE_FOLDER / "3d_scan_store_head_basic.json")
    archive = tmp_path / "head.zip"
    templates.export_template(props, archive)
    return props, archive


def rewrite(archive, **changes):
    with ZipFile(archive) as package:
        contents = {name: package.read(name) for name in package.namelist()}
    contents.update(changes)
    if "template.json" in changes:
        manifest = json.loads(contents["manifest.json"])
        manifest["files"]["template.json"] = hashlib.sha256(contents["template.json"]).hexdigest()
        contents["manifest.json"] = json.dumps(manifest).encode()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=1) as package:
        for name, content in contents.items():
            if content is not None:
                package.writestr(name, content)


def test_default_and_custom_actions(exported):
    props, _ = exported
    for custom in (True, False):
        if not custom:
            templates.load_template(props, templates.BUNDLED_TEMPLATES[0])
        layout = Layout()
        ui.CHARACTER_DNA_MT_converter_template.draw(SimpleNamespace(layout=layout), bpy.context)
        actions = [args[0] for call, args, _ in layout.calls if call == "operator"]
        expected = ["character_dna.export_converter_template", "character_dna.load_converter_template"]
        assert actions == expected + (["character_dna.delete_converter_template"] if custom else [])
        assert CHARACTER_DNA_OT_delete_converter_template.poll(bpy.context) == custom


def test_delete_removes_installed_copy_only(exported):
    props, archive = exported
    folder = Path(props.template_file).parent
    source = templates.resolve_dna_files(
        json.loads((templates.TEMPLATE_FOLDER / "3d_scan_store_head_basic.json").read_text())
    )["head"]
    original = source.read_bytes()
    assert bpy.ops.character_dna.delete_converter_template() == {"FINISHED"}
    assert not folder.exists()
    assert archive.is_file() and source.read_bytes() == original
    assert props.template == str(templates.BUNDLED_TEMPLATES[0])
    templates.import_template(props, archive)
    assert folder.is_dir()


def test_defaults_cannot_be_deleted(setup_scene):
    props, _ = setup_scene
    before = templates.BUNDLED_TEMPLATES[0].read_bytes()
    with pytest.raises(ValueError, match="custom"):
        templates.delete_template(props)
    with pytest.raises(ValueError, match="custom"):
        packages.remove(str(templates.BUNDLED_TEMPLATES[0]))
    assert templates.BUNDLED_TEMPLATES[0].read_bytes() == before


def test_other_scene_dependency_prevents_deletion(exported):
    props, _ = exported
    scene = bpy.data.scenes.new("Uses template")
    try:
        templates.load_template(scene.character_dna.converter, Path(props.template_file))
        with pytest.raises(ValueError, match="Uses template"):
            templates.delete_template(props)
        assert Path(props.template_file).is_file()
    finally:
        bpy.data.scenes.remove(scene)


def test_duplicate_import_and_shared_catalog(exported):
    props, archive = exported
    key = props.template_id
    installed = Path(props.template_file)
    timestamp = installed.stat().st_mtime_ns
    assert bpy.ops.character_dna.load_converter_template(filepath=str(archive)) == {"FINISHED"}
    assert props.template_id == key and installed.stat().st_mtime_ns == timestamp
    scene = bpy.data.scenes.new("New template scene")
    try:
        other = scene.character_dna.converter
        templates.migrate(other)
        assert key in {row[0] for row in catalog.items(other)}
        other.template = key
        assert other.template_file == str(installed)
    finally:
        bpy.data.scenes.remove(scene)


def test_changed_package_requires_confirmation(exported):
    props, archive = exported
    installed = Path(props.template_file)
    before = installed.read_bytes()
    recipe = read_package_recipe(archive)
    recipe["name"] = "Updated Head"
    rewrite(archive, **{"template.json": json.dumps(recipe).encode()})
    with pytest.raises(packages.PackageExistsError) as caught:
        templates.import_template(props, archive)
    assert installed.read_bytes() == before
    assert props.template_name != "Updated Head"
    templates.import_template(props, archive, replace_digest=caught.value.digest)
    assert props.template_name == "Updated Head"
    assert json.loads(installed.read_text())["name"] == "Updated Head"


def test_reimport_can_restore_modified_installed_dna(exported):
    props, archive = exported
    installed = templates.current_dna_files(props)["head"]
    original = installed.read_bytes()
    installed.write_bytes(b"Locally modified")
    with pytest.raises(packages.PackageExistsError) as caught:
        templates.import_template(props, archive)
    assert installed.read_bytes() == b"Locally modified"
    templates.import_template(props, archive, replace_digest=caught.value.digest)
    assert installed.read_bytes() == original


def test_failed_replacement_rolls_back(exported, monkeypatch):
    props, archive = exported
    installed = Path(props.template_file)
    before = installed.read_bytes()
    recipe = read_package_recipe(archive)
    recipe["name"] = "Updated Head"
    rewrite(archive, **{"template.json": json.dumps(recipe).encode()})
    with pytest.raises(packages.PackageExistsError) as caught:
        templates.import_template(props, archive)
    original = Path.rename

    def fail_stage(path, target):
        if path.name == "package":
            raise OSError("simulated install failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(OSError, match="simulated"):
        templates.import_template(props, archive, replace_digest=caught.value.digest)
    assert installed.read_bytes() == before


@pytest.mark.parametrize("failure", ["traversal", "missing", "checksum", "version", "dna_path", "mesh"])
def test_invalid_archive_does_not_change_library(exported, failure):
    props, archive = exported
    before = props.template_id, props.template_data
    installed = Path(props.template_file)
    saved = installed.read_bytes()
    if failure == "traversal":
        rewrite(archive, **{"../outside.txt": b"bad"})
    elif failure == "missing":
        rewrite(archive, **{"dna/head.dna": None})
    elif failure == "checksum":
        rewrite(archive, **{"dna/head.dna": b"broken DNA"})
    elif failure == "version":
        with ZipFile(archive) as package:
            manifest = json.loads(package.read("manifest.json"))
        manifest["version"] = 99
        rewrite(archive, **{"manifest.json": json.dumps(manifest).encode()})
    else:
        recipe = read_package_recipe(archive)
        if failure == "dna_path":
            recipe["dna_files"]["head"] = "../../external.dna"
        else:
            recipe["meshes"][0]["mesh"] = "missing_head_mesh"
        rewrite(archive, **{"template.json": json.dumps(recipe).encode()})
    with pytest.raises(ValueError):
        templates.import_template(props, archive)
    assert (props.template_id, props.template_data) == before
    assert installed.read_bytes() == saved
    assert not list(packages.library_root().glob(".load-*"))


def test_export_failure_preserves_existing_archive(setup_scene, tmp_path):
    props, _ = setup_scene
    archive = tmp_path / "keep.zip"
    archive.write_bytes(b"Existing export")
    recipe = templates.from_properties(props)
    recipe["dna_files"]["head"] = str(tmp_path / "missing.dna")
    with pytest.raises(OSError):
        templates.export_template(props, archive, recipe)
    assert archive.read_bytes() == b"Existing export"
    assert not packages.installed()


def test_library_relocation_rebases_dna_without_resetting_edits(exported, tmp_path, monkeypatch):
    props, _ = exported
    key, old_file = props.template_id, Path(props.template_file)
    props.extra_meshes[0].uv_tolerance = 0.02
    new_data = tmp_path / "relocated"
    target = new_data / "converter" / "templates"
    shutil.copytree(packages.library_root(), target)
    monkeypatch.setattr(packages, "get_user_data_folder", lambda: new_data)
    _initialize_active_template()
    props = get_converter_properties()
    assert props.template_id == props.template == key
    assert Path(props.template_file) != old_file
    assert templates.current_dna_files(props)["head"].is_relative_to(target)
    assert props.extra_meshes[0].uv_tolerance == pytest.approx(0.02)


def test_same_display_name_has_distinct_package_ids(exported, tmp_path):
    props, _ = exported
    key = props.template_id
    templates.export_template(props, tmp_path / "second.zip")
    assert props.template_id != key
    assert len(packages.installed()) == 2


def test_custom_dropdown_uses_only_cached_metadata(exported, monkeypatch):
    props, _ = exported
    catalog.refresh(props)
    first = catalog.items(props)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Dropdown performed filesystem I/O")

    monkeypatch.setattr(Path, "read_text", unexpected)
    monkeypatch.setattr(Path, "glob", unexpected)
    assert all(catalog.items(props) is first for _ in range(50))
