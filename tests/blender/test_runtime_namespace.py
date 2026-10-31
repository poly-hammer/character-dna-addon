"""Deferred runtime work must resolve the enabled Free or Pro operator namespace."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from character_dna.constants import ToolInfo
from character_dna.runtime import controller
from character_dna.utilities import misc


@pytest.mark.parametrize("edition", ["character_dna", "character_dna_pro"])
def test_deferred_sync_uses_current_edition(monkeypatch, edition, caplog):
    caplog.set_level("DEBUG")
    sync = Mock(return_value={"FINISHED"})
    timers = SimpleNamespace(is_registered=Mock(return_value=False), register=Mock())
    blender = SimpleNamespace(
        ops=SimpleNamespace(**{edition: SimpleNamespace(sync_native_runtime=sync)}),
        app=SimpleNamespace(timers=timers),
    )
    monkeypatch.setattr(ToolInfo, "NAME", edition)
    monkeypatch.setattr(controller, "bpy", blender)
    monkeypatch.setattr(misc, "bpy", blender)
    monkeypatch.setattr(controller, "_transitioning", False)
    monkeypatch.setattr(controller.ui_refresh, "invalidate_migration", Mock())

    controller.request_sync()
    callback = timers.register.call_args.args[0]
    callback()

    sync.assert_called_once_with()
    assert not caplog.records
