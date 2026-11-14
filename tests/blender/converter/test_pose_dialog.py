"""Pose confirmation cancels cleanly and only approves the displayed joint chains."""

import json

from types import SimpleNamespace

import bpy
import pytest

from character_dna.editors.converter import pose_dialog
from character_dna.editors.converter.lifecycle import ConversionLifecycle
from character_dna.editors.converter.native.session import GeometryJob, PoseRangeError


ISSUES = [{"joint": "indextoe_02_r", "angle": 77.44, "limit": 45.0}]


@pytest.fixture(autouse=True)
def clear_confirmation():
    pose_dialog.clear()
    yield
    pose_dialog.clear()


def test_native_range_error_retains_all_joints_and_closes_worker():
    calls = []

    def result():
        raise ValueError(json.dumps({"pose_range_error": ISSUES}, separators=(",", ":")))

    job = GeometryJob(
        SimpleNamespace(
            poll=lambda: {"status": "failed", "events": [], "dropped_events": 0},
            result=result,
            close=lambda: calls.append("closed"),
        ),
        {},
    )
    with pytest.raises(PoseRangeError) as error:
        job.commit()
    assert error.value.issues == ISSUES
    assert "Normalize to A-Pose" in str(error.value)
    assert "to Off" in str(error.value)
    assert calls == ["closed"]


def test_confirmation_is_offered_only_after_rollback(monkeypatch):
    calls = []
    monkeypatch.setattr(pose_dialog, "offer", lambda *args: calls.append(("dialog", args)))
    operator = SimpleNamespace(
        _interactive=True,
        preserve_pose_joints='["other_joint"]',
        _restore=lambda *_args, **_kwargs: calls.append("rollback"),
        _reporter=SimpleNamespace(cancel=lambda _message: calls.append("progress stopped")),
        report=lambda *_args: pytest.fail("An expected confirmation must not raise an error popup"),
    )
    assert ConversionLifecycle._failed(operator, bpy.context, PoseRangeError(ISSUES)) == {"CANCELLED"}
    assert calls[:2] == ["rollback", "progress stopped"]
    assert calls[2][1][:2] == (ISSUES, '["other_joint"]')


def test_cancel_expires_confirmation_without_starting_conversion():
    pose_dialog.offer(ISSUES, "[]", bpy.context)
    operator = SimpleNamespace(session=pose_dialog._pending.token)
    pose_dialog.CHARACTER_DNA_OT_confirm_partial_a_pose.cancel(operator, bpy.context)
    assert pose_dialog._pending is None
    assert not bpy.app.timers.is_registered(pose_dialog._open_dialog)


def test_proceed_restarts_with_only_confirmed_joints(monkeypatch):
    calls = []
    pose_dialog.offer(ISSUES, '["previous_joint"]', bpy.context)
    operator = SimpleNamespace(session=pose_dialog._pending.token)
    monkeypatch.setattr(
        pose_dialog,
        "bpy",
        SimpleNamespace(
            app=bpy.app,
            ops=SimpleNamespace(
                character_dna=SimpleNamespace(
                    convert_to_dna=lambda *args, **kwargs: calls.append((args, kwargs)) or {"RUNNING_MODAL"}
                )
            ),
        ),
    )
    assert pose_dialog.CHARACTER_DNA_OT_confirm_partial_a_pose.execute(operator, bpy.context) == {"FINISHED"}
    assert calls[0][0] == ("INVOKE_DEFAULT",)
    assert json.loads(calls[0][1]["preserve_pose_joints"]) == ["indextoe_02_r", "previous_joint"]
    assert pose_dialog._pending is None


def test_stale_dialog_cannot_approve_a_new_conversion():
    pose_dialog.offer(ISSUES, "[]", bpy.context)
    errors = []
    operator = SimpleNamespace(session=pose_dialog._pending.token, report=lambda *args: errors.append(args))
    pose_dialog.clear()
    pose_dialog.offer(ISSUES, "[]", bpy.context)
    assert pose_dialog.CHARACTER_DNA_OT_confirm_partial_a_pose.execute(operator, bpy.context) == {"CANCELLED"}
    assert errors and "expired" in errors[0][1]
    pose_dialog.CHARACTER_DNA_OT_confirm_partial_a_pose.cancel(operator, bpy.context)
    assert pose_dialog._pending is not None


def test_mode_change_expires_confirmation():
    props = pose_dialog.get_converter_properties()
    before = props.pose_normalization
    try:
        props.pose_normalization = "LIMBS"
        pose_dialog.offer(ISSUES, "[]", bpy.context)
        errors = []
        operator = SimpleNamespace(session=pose_dialog._pending.token, report=lambda *args: errors.append(args))
        props.pose_normalization = "SPINE"
        assert pose_dialog.CHARACTER_DNA_OT_confirm_partial_a_pose.execute(operator, bpy.context) == {"CANCELLED"}
        assert errors and "expired" in errors[0][1]
        assert pose_dialog._pending is None
    finally:
        props.pose_normalization = before
