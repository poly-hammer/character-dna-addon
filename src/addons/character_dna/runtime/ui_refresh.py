"""Demand-driven sidebar redraws, independent of native rig evaluation."""

from __future__ import annotations

import time

from typing import Any

import bpy

from . import engine


PLAYBACK_INTERVAL = 1.0 / 20.0
IDLE_INTERVAL = 0.25
SUBSCRIPTION_LIFETIME = 0.75
_subscribers: dict[tuple[int, int, int], float] = {}
_migration: dict[int, bool] = {}
_migration_pending: set[int] = set()


def migration_needed(scene: Any) -> bool:
    """Read cached migration state and defer cache misses outside panel drawing."""
    key = scene.as_pointer()
    if key not in _migration:
        _migration_pending.add(key)
        if not bpy.app.timers.is_registered(_refresh_migration):
            bpy.app.timers.register(_refresh_migration, first_interval=0.0)
    return _migration.get(key, False)


def invalidate_migration() -> None:
    """Invalidate structural UI state without doing validation in the caller."""
    _migration.clear()


def _refresh_migration() -> None:
    from ..utilities import detect_legacy_data, detect_runtime_migration

    pending = _migration_pending.copy()
    _migration_pending.clear()
    for scene in bpy.data.scenes:
        key = scene.as_pointer()
        if key in pending:
            _migration[key] = detect_legacy_data(scene) is not None or detect_runtime_migration(scene)
    context = getattr(bpy, "context", None)
    manager = getattr(context, "window_manager", None)
    for window in getattr(manager, "windows", ()):
        if window.scene.as_pointer() in pending:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "UI":
                            region.tag_redraw()


def watch(context: Any, instance: Any) -> None:
    """Renew interest from an expanded panel body, never from poll or its header."""
    if not engine.active(instance):
        return
    window, area, region = context.window, context.area, context.region
    if not window or not area or not region or area.type != "VIEW_3D" or region.type != "UI":
        return
    if not area.spaces.active.show_region_ui or region.width <= 1 or region.height <= 1:
        return
    key = (window.as_pointer(), area.as_pointer(), region.as_pointer())
    _subscribers[key] = time.monotonic()
    if not bpy.app.timers.is_registered(_refresh):
        bpy.app.timers.register(_refresh, first_interval=PLAYBACK_INTERVAL)


def _refresh() -> float | None:
    now = time.monotonic()
    pending = {key: seen for key, seen in _subscribers.items() if now - seen <= SUBSCRIPTION_LIFETIME}
    _subscribers.clear()
    playback = False
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D" or not area.spaces.active.show_region_ui:
                continue
            for region in area.regions:
                key = (window.as_pointer(), area.as_pointer(), region.as_pointer())
                if key not in pending or region.type != "UI" or region.width <= 1 or region.height <= 1:
                    continue
                _subscribers[key] = pending[key]
                region.tag_redraw()
                playback |= screen.is_animation_playing
    if not _subscribers:
        return None
    return PLAYBACK_INTERVAL if playback else IDLE_INTERVAL


def clear() -> None:
    """Drop all UI identities before load, undo, add-on reload or disable."""
    _subscribers.clear()
    _migration.clear()
    _migration_pending.clear()
    if bpy.app.timers.is_registered(_refresh_migration):
        bpy.app.timers.unregister(_refresh_migration)
    if bpy.app.timers.is_registered(_refresh):
        bpy.app.timers.unregister(_refresh)
