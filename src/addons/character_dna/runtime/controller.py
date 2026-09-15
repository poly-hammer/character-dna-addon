"""Rig runtime ownership and Blender lifecycle coordination."""

from __future__ import annotations

import functools
import logging

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
_warning = ""


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


def rebuild(instance: Any) -> None:
    """Explicitly rebuild one writable character without affecting other instances."""
    global _transitioning
    if any(carrier.library or carrier.override_library for carrier in engine.carriers(instance)):
        raise ValueError("Linked character bindings must be migrated in their source file")
    previous = _transitioning
    _transitioning = True
    try:
        engine.discard(instance)
        instance.initialize()
        engine.install(instance)
        bpy.context.view_layer.update()
    finally:
        _transitioning = previous


def ensure_bindings(instance: Any) -> None:
    """Bind newly enabled local components before validating the saved runtime."""
    if engine.missing_components(instance) and not any(
        carrier.library or carrier.override_library for carrier in engine.carriers(instance)
    ):
        rebuild(instance)
    engine.adopt(instance)


@contextmanager
def preserve_bindings():
    """Keep portable drivers while display-name changes release authoring caches."""
    global _transitioning
    previous = _transitioning
    _transitioning = True
    try:
        yield
    finally:
        _transitioning = previous


def reconcile() -> None:  # noqa: PLR0912
    """Rebuild rig bindings in a write-safe operator context."""
    global _transitioning, _status, _warning
    from .. import rig_instance

    if _transitioning:
        return
    if rig_instance.is_rendering() or bpy.app.is_job_running("RENDER"):
        raise RuntimeError("Rebuild the runtime after rendering finishes")
    _transitioning = True
    _warning = ""
    failures = []
    bound = 0
    try:
        available, reason = engine.capability()
        if not available:
            raise RuntimeError(reason)
        for scene in bpy.data.scenes:
            if scene.library:
                continue
            properties = getattr(scene, ToolInfo.NAME, None)
            for instance in getattr(properties, "rig_instance_list", []):
                if is_suspended(instance):
                    continue
                try:
                    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
                        if not instance.auto_evaluate and not any(
                            carrier.library or carrier.override_library for carrier in engine.carriers(instance)
                        ):
                            engine.discard(instance)
                            continue
                        if engine.carriers(instance):
                            outdated = [
                                carrier for carrier in engine.carriers(instance) if engine.requires_migration(carrier)
                            ]
                            if outdated:
                                _warning = engine.MIGRATION_WARNING
                                for carrier in outdated:
                                    engine.warn_unavailable(carrier, _warning)
                                continue
                            ensure_bindings(instance)
                            if not any(
                                carrier.library or carrier.override_library for carrier in engine.carriers(instance)
                            ) and (
                                (instance.head_rig and not instance.head_initialized)
                                or (instance.body_rig and not instance.body_initialized)
                            ):
                                instance.initialize()
                            engine.sync_settings(instance)
                        elif instance.get("reference_mode") in {"LINK", "EDITABLE_LINK"}:
                            _warning = "Legacy data detected. Open the source file, run Migrate Legacy Data, then save."
                            engine.warn_unavailable(instance, _warning)
                            continue
                        elif not instance.head_initialized and not instance.body_initialized:
                            if instance.head_rig or instance.body_rig:
                                _warning = engine.MIGRATION_WARNING
                                engine.warn_unavailable(instance, _warning)
                            continue
                        else:
                            rebuild(instance)
                        bpy.context.view_layer.update()
                        bound += int(engine.active(instance))
                except FileNotFoundError:
                    _warning = "DNA file not found. Update the DNA file path, then rebuild evaluation."
                    engine.warn_unavailable(instance, _warning)
                except Exception as error:
                    failures.append(f"{instance.name}: {error}")
                    logger.exception("Native runtime transition failed for %s", instance.name)
        _status = "; ".join(failures) if failures else (_warning or f"Native: {bound} rig(s)")
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
        if _warning:
            self.report({"WARNING"}, _warning)
        return {"FINISHED"}


def _apply_requested() -> None:
    try:
        bpy.ops.character_dna.sync_native_runtime()
    except Exception:
        logger.exception("Could not apply the requested native backend change")


def request_sync(_owner: Any = None, _context: Any = None) -> None:
    """Queue a single undoable rebuild outside property callbacks."""
    ui_refresh.invalidate_migration()
    if not _transitioning and not bpy.app.timers.is_registered(_apply_requested):
        bpy.app.timers.register(_apply_requested, first_interval=0.0)


def auto_evaluation_changed(instance: Any, _context: Any) -> None:
    """Release or reacquire output ownership when the user changes auto evaluation."""
    if not is_suspended(instance):
        engine.sync_settings(instance)
        request_sync()


def after_load() -> None:
    """Rebuild saved bindings once after the scene's data has been loaded."""
    ui_refresh.clear()
    engine.restore()


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
    if any(carrier.library or carrier.override_library for carrier in engine.carriers(instance)):
        raise ValueError("Linked character data is read-only; edit the source file or append the character")
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
    """Release the active source's drivers and rebuild only it and its duplicates."""

    @functools.wraps(function)
    def wrapped(operator: Any, context: Any) -> Any:
        from ..utilities import get_active_rig_instance, get_addon_window_manager_properties

        scene = context.scene
        source = get_active_rig_instance()
        if source is None:
            return function(operator, context)
        source_name = source.name
        instances = getattr(scene, ToolInfo.NAME).rig_instance_list
        existing_names = {instance.name for instance in instances}
        had_bindings = bool(engine.carriers(instances[source_name]))
        window_properties = get_addon_window_manager_properties(context)
        previous_evaluation = window_properties.evaluate_dependency_graph
        try:
            with preserve_bindings():
                window_properties.evaluate_dependency_graph = False
                try:
                    engine.discard(instances[source_name])
                    return function(operator, context)
                finally:
                    instances = getattr(scene, ToolInfo.NAME).rig_instance_list
                    rebuild_names = ([source_name] if had_bindings else []) + [
                        instance.name for instance in instances if instance.name not in existing_names
                    ]
                    for name in rebuild_names:
                        instance = instances.get(name)
                        if instance is not None:
                            rebuild(instance)
        finally:
            window_properties.evaluate_dependency_graph = previous_evaluation

    return wrapped


def register() -> None:
    """Register the runtime's rebuild operator, driver callback and deferred startup."""
    operator = bpy.types.Operator.bl_rna_get_subclass_py(CHARACTER_DNA_OT_sync_native_runtime.__name__)
    if operator is not None and operator is not CHARACTER_DNA_OT_sync_native_runtime:
        bpy.utils.unregister_class(operator)
    if operator is not CHARACTER_DNA_OT_sync_native_runtime:
        bpy.utils.register_class(CHARACTER_DNA_OT_sync_native_runtime)
    bpy.app.driver_namespace[engine.NAMESPACE] = engine.solve
    if _after_import not in bpy.app.handlers.blend_import_post:
        bpy.app.handlers.blend_import_post.append(_after_import)
    if not bpy.app.timers.is_registered(_startup):
        bpy.app.timers.register(_startup, first_interval=0.0)


@bpy.app.handlers.persistent
def _after_import(_context: Any) -> None:
    """Hydrate library-imported sessions without changing saved scene data."""
    if engine.capability()[0]:
        engine.hydrate()


def _startup() -> None:
    engine.hydrate()


def unregister() -> None:
    """Release registered runtime services while preserving saved driver bindings."""
    ui_refresh.clear()
    if bpy.app.timers.is_registered(_apply_requested):
        bpy.app.timers.unregister(_apply_requested)
    if bpy.app.timers.is_registered(_startup):
        bpy.app.timers.unregister(_startup)
    if _after_import in bpy.app.handlers.blend_import_post:
        bpy.app.handlers.blend_import_post.remove(_after_import)
    engine.invalidate()
    _suspended.clear()
    bpy.app.driver_namespace.pop(engine.NAMESPACE, None)
    operator = bpy.types.Operator.bl_rna_get_subclass_py(CHARACTER_DNA_OT_sync_native_runtime.__name__)
    if operator is not None:
        bpy.utils.unregister_class(operator)
