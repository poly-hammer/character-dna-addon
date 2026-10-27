"""Reuse a shape editor's native models and drivers after checked geometry edits.

Checks happen at editor boundaries, never during dependency-graph evaluation.
Undo/load drops every borrowed Blender wrapper; missing state forces rebuilding.
"""

# Runtime retention and the engine share private lifecycle state.
# ruff: noqa: SLF001

import hashlib
import logging

from array import array
from pathlib import Path
from typing import Any

import bpy

from . import engine
from .bindings import owned_curve, resolve_output_target


logger = logging.getLogger(__name__)
_sessions: dict[str, dict] = {}


def clear() -> None:
    _sessions.clear()


def retained(instance: Any) -> bool:
    return engine.instance_id(instance) in _sessions


def forget(instance: Any) -> None:
    _sessions.pop(engine.instance_id(instance), None)


def _read(path: Path) -> tuple[bytes, tuple[int, ...], str]:
    signature = engine._file_signature(path)
    data = path.read_bytes()
    if engine._file_signature(path) != signature:
        raise RuntimeError("DNA changed during runtime compatibility validation")
    return data, signature, hashlib.sha256(data).hexdigest()


def _curves(carriers: list) -> list:
    owners, result = {}, []
    for carrier in carriers:
        for target in carrier.get("targets", []):
            owner, path = resolve_output_target(carrier, target)
            if owner is None or owner.animation_data is None:
                raise ValueError("A native output owner was removed")
            identity = owner.as_pointer()
            if identity not in owners:
                owners[identity] = {(item.data_path, item.array_index): item for item in owner.animation_data.drivers}
            curve = owners[identity].get((path, max(0, target["index"])))
            if (
                curve is None
                or not owned_curve(curve, carrier)
                or curve.driver.variables["output"].targets[0].data_path != f'["outputs"][{target["channel"]}]'
            ):
                raise ValueError("A native output driver changed")
            result.append((owner, curve, curve.mute))
    return result


def _scene_signature(instance: Any, carriers: list, curves: list) -> tuple:
    """Copy identities/structure only; sculpted coordinates and pose values may change."""
    rigs = [
        (
            rig.as_pointer(),
            rig.data.as_pointer(),
            tuple(
                (
                    bone.name,
                    bone.parent.name if bone.parent else None,
                    bone.rotation_mode,
                    tuple(value for row in bone.bone.matrix_local for value in row),
                )
                for bone in rig.pose.bones
            ),
        )
        for rig in (instance.head_rig, instance.body_rig, instance.face_board)
        if rig
    ]
    # Include merged visible meshes and hidden source meshes. An added key or
    # replaced Key ID must acquire new drivers, even if existing ones are valid.
    meshes = tuple(
        (
            obj.as_pointer(),
            obj.data.as_pointer(),
            obj.name,
            len(obj.data.vertices),
            tuple((key.as_pointer(), key.name, len(key.data)) for key in obj.data.shape_keys.key_blocks)
            if obj.data.shape_keys
            else (),
        )
        for obj in bpy.data.objects
        if obj.type == "MESH"
    )
    return (
        tuple(rigs),
        meshes,
        tuple((carrier.as_pointer(), carrier.get("token")) for carrier in carriers),
        tuple(
            (owner.as_pointer(), curve.as_pointer(), curve.data_path, curve.array_index) for owner, curve, _ in curves
        ),
    )


def pause(instance: Any) -> bool:
    """Opt in only if live models still describe the files and bindings on disk."""
    carriers = engine.carriers(instance)
    sdk = engine.native_module()
    if not callable(getattr(sdk, "runtime_definition", None)):
        return False
    try:
        files = {}
        for carrier in carriers:
            component = carrier["component"]
            if component == "switches":
                continue
            path = Path(bpy.path.abspath(getattr(instance, f"{component}_dna_file_path"))).resolve()
            source = (str(path), component == "body")
            entry = engine._model_files.get(source)
            data, _signature, digest = _read(path)
            record = engine._records.get(carrier.as_pointer())
            if (
                entry is None
                or entry.key[0] != digest
                or entry.module is not sdk
                or record is None
                or record["model"] is not entry.model
                or not data.startswith(b"DNA")
            ):
                return False
            files[source] = (entry, hashlib.sha256(sdk.runtime_definition(data)).digest())
        curves = _curves(carriers)
        snapshot = _scene_signature(instance, carriers, curves)
    except (OSError, ValueError, RuntimeError, ReferenceError):
        logger.debug("Runtime retention unavailable", exc_info=True)
        return False
    _sessions[engine.instance_id(instance)] = {
        "files": files,
        "signature": snapshot,
        "curves": curves,
        "records": {carrier.as_pointer(): id(engine._records[carrier.as_pointer()]) for carrier in carriers},
    }
    try:
        for carrier in carriers:
            carrier["authoring_paused"] = True
            engine._records[carrier.as_pointer()]["paused"] = True
        _set_mutes(curves, True)
    except Exception:
        forget(instance)
        engine.discard(instance)
        raise
    return True


def resume(instance: Any) -> list:
    """Return prior mute states on reuse, or discard and let the caller rebuild."""
    snapshot = _sessions.pop(engine.instance_id(instance), None)
    carriers = engine.carriers(instance)
    if snapshot is None:
        if any(carrier.get("authoring_paused") for carrier in carriers):
            engine.discard(instance)
            instance.destroy()
        return []
    try:
        curves = _curves(carriers)
        if _scene_signature(instance, carriers, curves) != snapshot["signature"]:
            raise ValueError("Scene binding structure changed")
        if {carrier.as_pointer(): id(engine._records.get(carrier.as_pointer())) for carrier in carriers} != snapshot[
            "records"
        ]:
            raise ValueError("Native sessions changed")
        updates = {}
        for source, (entry, expected) in snapshot["files"].items():
            path, body = source
            component = "body" if body else "head"
            if str(Path(bpy.path.abspath(getattr(instance, f"{component}_dna_file_path"))).resolve()) != path:
                raise ValueError("Component file changed")
            data, signature, digest = _read(Path(path))
            sdk = engine.native_module()
            if entry.module is not sdk or hashlib.sha256(sdk.runtime_definition(data)).digest() != expected:
                raise ValueError("DNA runtime definition changed")
            updates[source] = engine._ModelFile(signature, (digest, body), entry.model, sdk)
        for source, entry in updates.items():
            engine._model_files[source] = entry
            engine._models[entry.key] = entry.model
        referenced = {entry.key for entry in (*engine._model_files.values(), *engine._undo_models.values())}
        for entry, _expected in snapshot["files"].values():
            if entry.key not in referenced:
                engine._models.pop(entry.key, None)
        for carrier in carriers:
            record = engine._records[carrier.as_pointer()]
            record.pop("paused", None)
            record["contexts"].clear()
            record.pop("last_context", None)
            record.pop("authoring_revision", None)
            record.pop("preview", None)
            record.pop("preview_revision", None)
            carrier.pop("preview", None)
            carrier["preview_revision"] = 0
            driver = engine._solve_driver(carrier)
            if driver is not None and driver.expression != engine.SOLVE_EXPRESSION:
                driver.expression = engine.SOLVE_EXPRESSION
            del carrier["authoring_paused"]
            carrier.update_tag()
        restore_mutes(snapshot["curves"])
        return snapshot["curves"]
    except (OSError, ValueError, RuntimeError, ReferenceError):
        logger.debug("Rebuilding runtime after editor changes", exc_info=True)
        engine.discard(instance)
        instance.destroy()
        return []


def restore_mutes(curves: list) -> None:
    _set_mutes(curves, None)


def _set_mutes(curves: list, override: bool | None) -> None:
    """Batch public RNA writes per owner, preserving unrelated drivers' states."""
    owners = {}
    for owner, curve, muted in curves:
        _owner, states = owners.setdefault(owner.as_pointer(), (owner, {}))
        states[curve.as_pointer()] = muted if override is None else override
    for owner, states in owners.values():
        drivers = owner.animation_data.drivers
        current = array("b", [0]) * len(drivers)
        drivers.foreach_get("mute", current)
        values = array(
            "b", (int(states.get(curve.as_pointer(), current[index])) for index, curve in enumerate(drivers))
        )
        if values != current:
            drivers.foreach_set("mute", values)
            owner.update_tag()
