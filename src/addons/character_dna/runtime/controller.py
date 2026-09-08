"""Rig runtime ownership and Blender lifecycle coordination."""

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
_status = "Runtime ready"


def is_suspended(instance: Any) -> bool:
    """Whether an editor currently owns the character's output channels."""
    return engine.instance_id(instance) in _suspended or bool(instance.get("native_editor_resume", False))


def status() -> str:
    """Return current runtime setup or evaluation failures."""
    failures = engine.errors()
    return "Native evaluation error: " + "; ".join(failures) if failures else _status


def release(instance: Any) -> None:
    """Release native ownership before DNA replacement or component reinitialization."""
    if not _undoing and not _transitioning and engine.carriers(instance):
        engine.discard(instance)
        if not is_suspended(instance):
            request_sync()


def reconcile() -> None:
    """Rebuild rig bindings in a write-safe operator context."""
    global _transitioning, _status
    from .. import rig_instance

    if _transitioning:
        return
    if rig_instance.is_rendering() or bpy.app.is_job_running("RENDER"):
        raise RuntimeError("Rebuild the runtime after rendering finishes")
    _transitioning = True
    failures = []
    bound = 0
    identities = set()
    try:
        engine.discard()
        available, reason = engine.capability()
        if not available:
            raise RuntimeError(reason)
        for scene in bpy.data.scenes:
            properties = getattr(scene, ToolInfo.NAME, None)
            for instance in getattr(properties, "rig_instance_list", []):
                identity = engine.instance_id(instance)
                if identity and identity in identities:
                    instance["native_runtime_id"] = uuid.uuid4().hex
                identities.add(engine.instance_id(instance))
                if is_suspended(instance):
                    continue
                try:
                    instance.initialize()
                    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
                        engine.install(instance)
                        bpy.context.view_layer.update()
                        bound += int(engine.active(instance))
                except Exception as error:
                    engine.discard(instance)
                    failures.append(f"{instance.name}: {error}")
                    logger.exception("Native runtime transition failed for %s", instance.name)
        _status = "; ".join(failures) if failures else f"Native: {bound} rig(s)"
        if failures:
            raise RuntimeError(_status)
    finally:
        _transitioning = False


class CHARACTER_DNA_OT_sync_native_runtime(bpy.types.Operator):
    """Rebuild the runtime bindings after structural changes."""

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
    """Queue a single undoable rebuild outside property callbacks."""
    if not _transitioning and not bpy.app.timers.is_registered(_apply_requested):
        bpy.app.timers.register(_apply_requested, first_interval=0.0)


def auto_evaluation_changed(instance: Any, _context: Any) -> None:
    """Release or reacquire output ownership when the user changes auto evaluation."""
    if not is_suspended(instance):
        request_sync()


def after_load() -> None:
    """Rebuild saved bindings once after the scene's data has been loaded."""
    ui_refresh.clear()
    bpy.ops.character_dna.sync_native_runtime()


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
    engine.restore()


def suspend(instance: Any) -> bool:
    """Give an editor or animation baker exclusive access to output channels."""
    was_active = engine.active(instance)
    identity = engine.instance_id(instance)
    if was_active:
        for component in ("head", "body"):
            engine.synchronize_authoring(
                instance, component, instance.data.get(instance.cache_key(component, "instance"))
            )
        _suspended.add(identity)
        engine.discard(instance)
    return was_active


def resume(instance: Any, was_active: bool) -> None:
    """Rebind only after the owning operation has released its mutable SDK state."""
    if was_active:
        _suspended.discard(engine.instance_id(instance))
        instance.evaluate()


@contextmanager
def authoring_operation(instance: Any):
    """Temporarily release output drivers for editing or animation baking."""
    was_active = suspend(instance)
    try:
        yield
    finally:
        resume(instance, was_active)


def with_authoring_output(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run a tool with exclusive output ownership and native sampling."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        instance = kwargs.get("instance", args[0] if args else None)
        with authoring_operation(instance):
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
            if had_bindings:
                reconcile()

    return wrapped


def register() -> None:
    """Register the runtime's rebuild operator, driver callback and deferred startup."""
    bpy.utils.register_class(CHARACTER_DNA_OT_sync_native_runtime)
    bpy.app.driver_namespace[engine.NAMESPACE] = engine.solve
    bpy.app.timers.register(_startup, first_interval=0.0)


def _startup() -> None:
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
