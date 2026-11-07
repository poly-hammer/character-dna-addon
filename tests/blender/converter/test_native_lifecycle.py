"""Native jobs are closed before Blender invalidates captured scene identities."""

from types import SimpleNamespace

import bpy

from character_dna.editors.converter import lifecycle
from character_dna.editors.converter.native.jobs import StageJob


def test_cancelled_operator_rejects_late_timer_before_accessing_old_scene():
    calls = []
    operator = SimpleNamespace(_session_active=False, _advance=lambda _context: calls.append(True))
    result = lifecycle.ConversionLifecycle.modal(operator, None, SimpleNamespace(type="TIMER"))
    assert result == {"CANCELLED"}
    assert not calls


def test_shutdown_joins_worker_before_removing_timer_and_is_idempotent():
    order = []
    job = StageJob(SimpleNamespace(close=lambda: order.append("join")), "waiting")
    operator = SimpleNamespace(
        _session_active=True,
        _pipeline=SimpleNamespace(pose_jobs=[job]),
        _remove_timer=lambda _context: order.append("timer"),
        _snapshot=None,
        _reporter=None,
    )
    operator._restore = lambda context, rollback: lifecycle.ConversionLifecycle._restore(operator, context, rollback)
    operator.cancel = lambda context: lifecycle.ConversionLifecycle.cancel(operator, context)
    lifecycle._active_conversions[id(operator)] = operator
    lifecycle.cancel_active_conversions(bpy.context)
    lifecycle.cancel_active_conversions(bpy.context)
    assert order == ["join", "timer"]
    assert job.task is None
    assert not operator._session_active
    assert not lifecycle._active_conversions


def test_inactive_operator_cannot_cancel_a_new_conversion():
    calls = []
    old = SimpleNamespace(_session_active=False)
    current = SimpleNamespace(cancel=lambda _context: calls.append("current"))
    lifecycle._active_conversions[id(current)] = current
    try:
        assert lifecycle.ConversionLifecycle.modal(old, None, SimpleNamespace(type="ESC")) == {"CANCELLED"}
        assert not calls
        lifecycle.cancel_active_conversions(bpy.context)
        assert calls == ["current"]
    finally:
        lifecycle._active_conversions.clear()
