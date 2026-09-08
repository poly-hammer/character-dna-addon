"""Opt-in native backend transitions and Blender lifecycle coordination."""

from __future__ import annotations

import functools
import logging
import uuid

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import bpy

from ..constants import ToolInfo
from . import engine, ui_refresh


logger = logging.getLogger(__name__)
_transitioning = False
_undoing = False
_suspended: set[str] = set()
_status = "Python backend"


def enabled() -> bool:
    """Read the opt-in preference without requiring native dependencies."""
    from ..utilities import get_addon_preferences

    preferences = get_addon_preferences()
    return bool(preferences and getattr(preferences, "experimental_native_riglogic", False))


def status() -> str:
    """Return the last requested backend transition result."""
    failures = engine.errors()
    return "Native evaluation error: " + "; ".join(failures) if failures else _status


def release(instance: Any) -> None:
    """Release native ownership before DNA replacement or component reinitialization."""
    if not _undoing and not _transitioning and engine.carriers(instance):
        engine.discard(instance)
        if enabled() and engine.instance_id(instance) not in _suspended:
            request_sync()


def reconcile() -> None:
    """Rebuild opted-in rigs in a write-safe operator context, one backend at a time."""
    global _transitioning, _status
    from .. import rig_instance

    if _transitioning:
        return
    if rig_instance.is_rendering() or bpy.app.is_job_running("RENDER"):
        raise RuntimeError("Change the native backend after rendering finishes")
    _transitioning = True
    failures = []
    bound = 0
    identities = set()
    try:
        engine.discard()
        available, reason = engine.capability() if enabled() else (False, "Python backend")
        for scene in bpy.data.scenes:
            properties = getattr(scene, ToolInfo.NAME, None)
            for instance in getattr(properties, "rig_instance_list", []):
                identity = engine.instance_id(instance)
                if identity and identity in identities:
                    instance["native_runtime_id"] = uuid.uuid4().hex
                identities.add(engine.instance_id(instance))
                if engine.instance_id(instance) in _suspended or instance.get("native_editor_resume", False):
                    continue
                try:
                    instance.initialize()
                    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
                        if instance.auto_evaluate:
                            for component in ("body", "head"):
                                if getattr(instance, f"auto_evaluate_{component}"):
                                    instance.evaluate(component=component)
                    if available:
                        engine.install(instance)
                        bound += int(engine.active(instance))
                except Exception as error:
                    engine.discard(instance)
                    failures.append(f"{instance.name}: {error}")
                    logger.exception("Native runtime transition failed for %s", instance.name)
        _status = "; ".join(failures) if failures else (f"Native: {bound} rig(s)" if available else reason)
    finally:
        _transitioning = False


class CHARACTER_DNA_OT_sync_native_runtime(bpy.types.Operator):
    """Rebuild or remove the experimental native bindings."""

    bl_idname = f"{ToolInfo.NAME}.sync_native_runtime"
    bl_label = "Rebuild Native Evaluation"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    def execute(self, _context: Any) -> set[str]:
        try:
            reconcile()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


def _apply_requested() -> None:
    try:
        bpy.ops.character_dna.sync_native_runtime()
    except Exception:
        logger.exception("Could not apply the requested native backend change")


def request_sync(_owner: Any = None, _context: Any = None) -> None:
    """Queue a single undoable transition outside preference/property draw callbacks."""
    if not _transitioning and not bpy.app.timers.is_registered(_apply_requested):
        bpy.app.timers.register(_apply_requested, first_interval=0.0)


def auto_evaluation_changed(instance: Any, _context: Any) -> None:
    """Release or reacquire output ownership when the user changes auto evaluation."""
    if enabled() and engine.instance_id(instance) not in _suspended:
        request_sync()


def after_load() -> None:
    """Recover saved carrier ownership or remove it when this install is unsupported."""
    ui_refresh.clear()
    if enabled() and engine.capability()[0]:
        engine.restore()
        bpy.ops.character_dna.sync_native_runtime()
    else:
        engine.discard()
        engine.invalidate()


def before_undo() -> None:
    """Drop native sessions before Blender frees evaluated IDs."""
    global _undoing
    _undoing = True
    ui_refresh.clear()
    engine.invalidate()


def after_undo() -> None:
    """Rehydrate the bindings restored by Blender's undo, without rewriting its stack."""
    global _undoing
    _undoing = False
    if enabled():
        engine.restore()
    elif engine.carriers():
        request_sync()


def suspend(instance: Any) -> bool:
    """Give an editor or animation baker exclusive access to the legacy SDK state."""
    was_active = engine.active(instance)
    identity = engine.instance_id(instance)
    if was_active:
        _suspended.add(identity)
        engine.discard(instance)
        instance.evaluate()
    return was_active


def resume(instance: Any, was_active: bool) -> None:
    """Rebind only after the owning operation has released its mutable SDK state."""
    if was_active:
        _suspended.discard(engine.instance_id(instance))
        if enabled():
            instance.evaluate()
            try:
                engine.install(instance)
            except ValueError as error:
                logger.warning("Native runtime remains suspended after operation: %s", error)


@contextmanager
def legacy_operation(instance: Any):
    """Scope regular animation baking/export, not simulation pre-baking."""
    was_active = suspend(instance)
    try:
        yield
    finally:
        resume(instance, was_active)


def with_legacy_runtime(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run an existing SDK-based operation with native output ownership suspended."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        instance = kwargs.get("instance", args[0] if args else None)
        with legacy_operation(instance):
            return function(*args, **kwargs)

    return wrapped


def native_scene_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Suspend owned drivers while an operator duplicates or remaps scene IDs."""

    @functools.wraps(function)
    def wrapped(operator: Any, context: Any) -> Any:
        had_bindings = bool(engine.carriers())
        if had_bindings:
            engine.discard()
            engine.invalidate()
        try:
            return function(operator, context)
        finally:
            if had_bindings and enabled():
                reconcile()

    return wrapped


def register() -> None:
    """Register the operator and native driver namespace independently of opt-in."""
    bpy.utils.register_class(CHARACTER_DNA_OT_sync_native_runtime)
    bpy.app.driver_namespace[engine.NAMESPACE] = engine.solve
    bpy.app.timers.register(_startup, first_interval=0.0)


def _startup() -> None:
    if enabled() or engine.carriers():
        _apply_requested()


def unregister() -> None:
    """Leave no owned output drivers, native sessions or scheduled transitions behind."""
    ui_refresh.clear()
    if bpy.app.timers.is_registered(_apply_requested):
        bpy.app.timers.unregister(_apply_requested)
    if bpy.app.timers.is_registered(_startup):
        bpy.app.timers.unregister(_startup)
    engine.discard()
    engine.invalidate()
    _suspended.clear()
    bpy.app.driver_namespace.pop(engine.NAMESPACE, None)
    bpy.utils.unregister_class(CHARACTER_DNA_OT_sync_native_runtime)
