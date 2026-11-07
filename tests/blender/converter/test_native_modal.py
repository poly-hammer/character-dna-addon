"""Conversion publishes only the final native endpoint; timers never animate it."""

from types import SimpleNamespace

import pytest

from character_dna.editors.converter.lifecycle import ConversionLifecycle
from character_dna.editors.converter.native.jobs import StageJob


def test_modal_waits_for_worker_then_publishes_once_without_playback():
    calls = []
    status = ["running", "completed"]
    task = SimpleNamespace(
        poll=lambda: {"status": status.pop(0), "events": [], "dropped_events": 0},
        result=lambda: calls.append("result") or "final buffers",
        close=lambda: calls.append("join"),
    )
    job = StageJob(task, "Native conversion")

    def publish(_context):
        assert job.take_result() == "final buffers"
        calls.append("publish")
        operator._session_active = False
        return False

    operator = SimpleNamespace(
        _session_active=True,
        _transition=job,
        _advance=publish,
        _reporter=SimpleNamespace(update=lambda _label: calls.append("progress")),
    )
    tick = SimpleNamespace(type="TIMER")
    assert ConversionLifecycle.modal(operator, None, tick) == {"RUNNING_MODAL"}
    assert calls == ["progress"]
    assert ConversionLifecycle.modal(operator, None, tick) == {"FINISHED"}
    assert calls == ["progress", "result", "join", "publish"]
    assert ConversionLifecycle.modal(operator, None, tick) == {"CANCELLED"}
    assert calls.count("publish") == 1


@pytest.mark.parametrize("event", ["ESC", "RIGHTMOUSE"])
def test_modal_cancellation_cannot_publish_a_pending_result(event):
    calls = []
    operator = SimpleNamespace(_session_active=True, cancel=lambda _context: calls.append("cancel"))
    assert ConversionLifecycle.modal(operator, None, SimpleNamespace(type=event)) == {"CANCELLED"}
    assert calls == ["cancel"]


def test_modal_worker_failure_rolls_back_before_any_final_scene_commit():
    calls = []

    def failed_poll():
        raise ValueError("invalid native endpoint")

    def rollback(_context, error):
        calls.append(str(error))
        return {"CANCELLED"}

    operator = SimpleNamespace(
        _session_active=True,
        _transition=SimpleNamespace(poll=failed_poll),
        _failed=rollback,
        _advance=lambda _context: pytest.fail("A failed worker must not publish scene objects"),
    )
    assert ConversionLifecycle.modal(operator, None, SimpleNamespace(type="TIMER")) == {"CANCELLED"}
    assert calls == ["invalid native endpoint"]
