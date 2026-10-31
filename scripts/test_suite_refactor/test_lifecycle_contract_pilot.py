"""Cheap additional transition contracts; not a replacement for real binding tests."""

from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from character_dna.editors.converter import scene_merge, surface_session
from character_dna.editors.shared.editor import Editor
from character_dna.runtime import controller


class Instance(dict):
    name = "ContractRig"
    editor_component = "body"
    auto_evaluate_body = True


@pytest.mark.parametrize("failure_stage", ["enter", "tracking", "overlay"])
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_entry_contract(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str, merged: bool, cleanup_fails: bool
) -> None:
    """Keep the actual base lifecycle; fake only collaborators with separate integration coverage."""
    instance = Instance()
    events = []

    def fail(stage: str) -> None:
        events.append(stage)
        if stage == failure_stage:
            raise RuntimeError("original failure")

    def abort():
        events.append("abort")
        if cleanup_fails:
            raise ValueError("cleanup failure")

    suspend = create_autospec(controller.suspend, return_value=True)
    resume = create_autospec(controller.resume)
    begin = create_autospec(surface_session.begin)
    finish = create_autospec(surface_session.finish)
    monkeypatch.setattr(controller, "suspend", suspend)
    monkeypatch.setattr(controller, "resume", resume)
    monkeypatch.setattr(surface_session, "begin", begin)
    monkeypatch.setattr(surface_session, "finish", finish)
    monkeypatch.setattr(
        scene_merge, "surfaces", create_autospec(scene_merge.surfaces, return_value={0: object()} if merged else {})
    )
    monkeypatch.setattr(Editor, "active_editor", lambda _instance: None)
    component = SimpleNamespace(
        set_auto_evaluate=lambda inst, value: setattr(inst, "auto_evaluate_body", value),
        ensure_initialized=lambda _inst: events.append("initialize"),
    )
    editor = SimpleNamespace(
        instance=instance,
        is_editing=False,
        editor_id="contract",
        component=component,
        preserve_runtime_bindings=True,
        edits_surfaces=True,
        _capture_scene_context=lambda _context: {"selection": "original"},
        _store_scene_context=lambda _context: events.append("capture"),
        surface_selection=lambda: None,
        _enter=lambda _context: fail("enter"),
        _abort_enter=abort,
        _tracking_init_kwargs=lambda _context: {},
        tracking=SimpleNamespace(initialize=lambda **_kwargs: fail("tracking")),
        overlay=SimpleNamespace(register=lambda: fail("overlay")),
    )
    context = object()
    with pytest.raises(RuntimeError, match="original failure"):
        Editor.enter(editor, context)
    assert not editor.is_editing
    assert instance.auto_evaluate_body
    assert "native_editor_resume" not in instance
    assert events.count("abort") == 1
    suspend.assert_called_once_with(instance, preserve_bindings=True)
    resume.assert_called_once_with(instance, True)
    if merged:
        begin.assert_called_once_with(instance, selection=None)
        finish.assert_called_once_with(instance, commit=False)
    else:
        begin.assert_not_called()
        finish.assert_not_called()
