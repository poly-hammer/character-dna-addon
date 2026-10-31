"""Preferences setup must work before any character or solver folder exists."""

import sys

from types import SimpleNamespace

import bpy
import pytest

from character_dna.editors.raw_control_editor.operators import CHARACTER_DNA_OT_setup_solver_environment
from character_dna.utilities import get_addon_preferences


def test_setup_poll_without_character_or_folder(tmp_path):
    bpy.ops.wm.read_homefile(use_empty=True)
    preferences = get_addon_preferences().raw_control_editor
    previous = preferences.solver_env_root
    try:
        preferences.solver_env_root = str(tmp_path / "missing" / "character-dna" / "solver-venvs")
        assert not bpy.context.scene.character_dna.rig_instance_list
        assert CHARACTER_DNA_OT_setup_solver_environment.poll(bpy.context)
    finally:
        preferences.solver_env_root = previous


@pytest.mark.parametrize("blank_root", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_setup_runs_without_character(tmp_path, monkeypatch, blank_root, failure):
    from character_dna.editors.raw_control_editor import dependency_extraction, operators, solver_worker
    from character_dna.utilities import get_addon_window_manager_properties

    bpy.ops.wm.read_homefile(use_empty=True)
    preferences = get_addon_preferences().raw_control_editor
    previous_root, previous_device = preferences.solver_env_root, preferences.solver_device
    root = tmp_path / "missing" / "character-dna" / "solver-venvs"
    monkeypatch.setattr(dependency_extraction, "default_solver_root", lambda: root)
    called = []

    def provision(path, *, device, force, progress_cb):
        called.append((path, device, force))
        assert not path.exists()
        progress_cb("Creating solver environment")
        if failure:
            raise RuntimeError("provision failed")
        # The provisioning boundary is stubbed: no network or PyTorch install.
        path.mkdir(parents=True)
        return {"torch": "test", "device": "cpu"}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

        def is_alive(self):
            return False

    monkeypatch.setattr(solver_worker, "provision", provision)
    monkeypatch.setattr(operators.threading, "Thread", ImmediateThread)
    timer = object()
    removed = []
    context = SimpleNamespace(
        window=bpy.context.window,
        screen=None,
        window_manager=SimpleNamespace(
            character_dna=get_addon_window_manager_properties(),
            event_timer_add=lambda *_args, **_kwargs: timer,
            event_timer_remove=removed.append,
            modal_handler_add=lambda _operator: None,
        ),
    )
    cls = CHARACTER_DNA_OT_setup_solver_environment
    reports = []
    operator = SimpleNamespace(_reinstall=False, report=lambda level, message: reports.append((level, message)))
    operator._finish = lambda context: cls._finish(operator, context)
    try:
        preferences.solver_env_root = "" if blank_root else str(root)
        preferences.solver_device = "cpu"
        assert cls.poll(bpy.context)
        assert cls.execute(operator, context) == {"RUNNING_MODAL"}
        assert not cls.poll(bpy.context), "Setup must not start twice without an active character"
        assert cls.modal(operator, context, SimpleNamespace(type="TIMER")) == (
            {"CANCELLED"} if failure else {"FINISHED"}
        )
        expected = root / f"python-{sys.version_info.major}.{sys.version_info.minor}" / "cpu"
        assert called == [(expected, "cpu", False)]
        assert dependency_extraction.resolve_solver_env_dir() == expected
        assert removed == [timer]
        assert cls.poll(bpy.context)
        assert get_addon_window_manager_properties().progress == 1.0
        if failure:
            assert "provision failed" in reports[-1][1]
        else:
            assert expected.is_dir()
            assert "solver ready" in reports[-1][1]
    finally:
        preferences.solver_env_root, preferences.solver_device = previous_root, previous_device
        cls._setup_running = False


def test_setup_stays_disabled_during_bone_matching():
    bpy.ops.wm.read_homefile(use_empty=True)
    instance = bpy.context.scene.character_dna.rig_instance_list.add()
    instance.raw_control_editor.match_is_running = True
    try:
        assert not CHARACTER_DNA_OT_setup_solver_environment.poll(bpy.context)
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)
