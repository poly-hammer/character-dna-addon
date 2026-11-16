"""Append/Link discovery preserves literal paths and UTF-8 inspector results."""

import json
import shutil
import sys

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from character_dna.utilities import misc


@pytest.mark.parametrize(
    "binary",
    [
        "/Applications/Blender 4.5.app/Contents/MacOS/Blender",
        "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe",
        "/opt/Blender 4.5/blender",
        "",
    ],
)
def test_inspector_passes_paths_as_literal_arguments(monkeypatch, tmp_path, binary):
    scripts = PurePosixPath(
        "/Users/José/Library/Application Support/Blender/extensions/character_dna/resources/scripts"
    )
    source = PurePosixPath("/Users/José/Characters/Ada's pose & [draft] $name.blend")
    monkeypatch.setattr(misc, "bpy", SimpleNamespace(app=SimpleNamespace(binary_path=binary)))
    monkeypatch.setattr(misc, "SCRIPTS_FOLDER", scripts)
    monkeypatch.setattr(misc, "TEMP_FOLDER", tmp_path)
    calls = []

    def run(command, **options):
        calls.append(command)
        assert isinstance(command, list)
        assert not options.get("shell", False)
        assert options["encoding"] == "utf-8"
        assert command[command.index("--blend-file") + 1] == str(source)
        assert str(scripts / "save_rig_instance_data.py") in command
        if binary:
            assert command[0] == binary
            assert "--background" in command
            assert "--disable-autoexec" in command
        else:
            assert command[:2] == [sys.executable, str(scripts / "save_rig_instance_data.py")]
        Path(command[command.index("--data-file") + 1]).write_text('{"José": {"name": "José"}}', encoding="utf-8")
        # A teardown failure must not discard a successfully written descriptor.
        return SimpleNamespace(returncode=-11, stdout="native teardown")

    monkeypatch.setattr(misc.subprocess, "run", run)
    assert misc.extract_rig_instance_data_from_blend_file(source) == ({"José": {"name": "José"}}, "")
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["launch", "process", "inspector"])
def test_inspector_reports_failures_and_cleans_temporary_files(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(misc, "TEMP_FOLDER", tmp_path)

    def run(command, **_options):
        if failure == "launch":
            raise OSError("Cannot launch Blender")
        if failure == "inspector":
            output = Path(command[command.index("--data-file") + 1])
            output.write_text(json.dumps({"incomplete": {}}), encoding="utf-8")
            output.with_name(f"{output.stem}_error.log").write_text("Cannot read José's file", encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="Cannot inspect source")

    monkeypatch.setattr(misc.subprocess, "run", run)
    data, error = misc.extract_rig_instance_data_from_blend_file(tmp_path / "source.blend")
    assert not data
    assert "Cannot" in error
    if failure == "inspector":
        assert error == "Cannot read José's file"
    assert list(tmp_path.iterdir()) == []


def test_real_inspector_handles_application_support_and_unicode_paths(
    setup_reference_blend_file, tmp_path, monkeypatch
):
    # Copy the standalone inspector to the macOS-style installation directory.
    addon = tmp_path / "Library/Application Support/Blender/character_dna"
    scripts = addon / "resources/scripts"
    scripts.mkdir(parents=True)
    (addon / "utilities").mkdir()
    shutil.copyfile(misc.SCRIPTS_FOLDER / "save_rig_instance_data.py", scripts / "save_rig_instance_data.py")
    shutil.copyfile(Path(misc.__file__).with_name("reference.py"), addon / "utilities/reference.py")
    source = tmp_path / "José's character & poses [draft].blend"
    shutil.copyfile(setup_reference_blend_file, source)
    before = source.stat().st_mtime_ns, source.stat().st_size
    monkeypatch.setattr(misc, "SCRIPTS_FOLDER", scripts)
    monkeypatch.setattr(misc, "TEMP_FOLDER", tmp_path / "Inspection Output")
    data, error = misc.extract_rig_instance_data_from_blend_file(source)
    assert not error, error
    assert "ada" in data
    assert data["ada"]["carriers"]
    assert data["ada"]["pointers"]["control_rig"] == "OBJECT::ada_control_rig"
    assert (source.stat().st_mtime_ns, source.stat().st_size) == before
    assert list((tmp_path / "Inspection Output").iterdir()) == []
