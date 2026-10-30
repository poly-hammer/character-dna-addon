"""Transactional ownership of native runtime output drivers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import bpy


CARRIER_MARKER = "character_dna_native_carrier"
DRIVER_EXPRESSION = "output + 0 * epoch"


def resolve_output_target(carrier: Any, target: Any) -> tuple[Any, str]:
    # Utilities also use the runtime controller. Resolve only when needed so
    # either package can be imported first, without an eager addon bootstrap.
    from ..utilities.reference import resolve_output_target as resolve

    return resolve(carrier, target)


@dataclass(frozen=True)
class Target:
    """An original ID property channel written by one native output slot."""

    owner: Any
    path: str
    index: int
    channel: int


def add_variable(driver: Any, name: str, owner: Any, path: str) -> None:
    """Declare a driver dependency using public RNA paths."""
    variable = driver.variables.new()
    variable.name = name
    # Even assigning the existing RNA value invalidates Blender's dependencies.
    # Most carriers already match the defaults; avoid two updates per output.
    if variable.type != "SINGLE_PROP":
        variable.type = "SINGLE_PROP"
    target = variable.targets[0]
    if target.id_type != owner.id_type:
        target.id_type = owner.id_type
    target.id = owner
    target.data_path = path


def owned_curve(curve: Any, carrier: Any) -> bool:
    """Recognize only the exact runtime expression and carrier dependencies."""
    driver = curve.driver
    if driver.expression != DRIVER_EXPRESSION or len(driver.variables) != 2:
        return False
    variables = {variable.name: variable for variable in driver.variables}
    if set(variables) != {"output", "epoch"}:
        return False
    for variable in variables.values():
        if variable.type != "SINGLE_PROP" or variable.targets[0].id != carrier:
            return False
    return variables["epoch"].targets[0].data_path == '["epoch"]' and variables["output"].targets[
        0
    ].data_path.startswith('["outputs"][')


def validate_targets(targets: list[Target]) -> None:
    """Reject ambiguous writers and invalid or read-only targets before installation."""
    seen = set()
    animated = {}
    for target in targets:
        key = (target.owner.as_pointer(), target.path, target.index)
        if key in seen:
            raise ValueError(f"Duplicate native output target: {target.owner.name}:{target.path}[{target.index}]")
        seen.add(key)
        if target.owner.library or not target.owner.is_editable:
            raise ValueError(f"Native output is not editable: {target.owner.name}")
        target.owner.path_resolve(target.path)
        animation = target.owner.animation_data
        if animation and animation.drivers.find(target.path, index=max(0, target.index)):
            raise ValueError(f"Existing driver on native output: {target.owner.name}:{target.path}[{target.index}]")
        owner_key = target.owner.as_pointer()
        if owner_key not in animated:
            animated[owner_key] = _animated_channels(target.owner)
        if (target.path, max(0, target.index)) in animated[owner_key]:
            raise ValueError(f"Keyframed native output requires Python evaluation: {target.owner.name}:{target.path}")


def install_targets(carrier: Any, targets: list[Target]) -> None:
    """Install owned identity drivers with rollback on any failed write."""
    for _curve in install_targets_iter(carrier, targets):
        pass


def _set_targets(carrier: Any, targets: list) -> None:
    # Blender's array-of-properties IDProperty cannot be resized by assignment.
    if "targets" in carrier:
        del carrier["targets"]
    carrier["targets"] = targets
    carrier.property_overridable_library_set('["targets"]', True)


def install_targets_iter(carrier: Any, targets: list[Target], *, append: bool = False) -> Iterator:
    """Install targets in resumable steps, retaining existing drivers when appending."""
    validate_targets(targets)
    installed = []
    previous = [dict(item) for item in carrier.get("targets", [])] if append else []
    records = previous + [
        {
            "owner": _persistent_owner(target.owner),
            "embedded": bool(target.owner.is_embedded_data),
            "path": target.path,
            "index": target.index,
            "channel": target.channel,
        }
        for target in targets
    ]
    _set_targets(carrier, records)
    template_owner = template = None
    try:
        for target in targets:
            curve = (
                target.owner.driver_add(target.path)
                if target.index < 0
                else target.owner.driver_add(target.path, target.index)
            )
            installed.append((target, curve))
            if template is None:
                curve.keyframe_points.clear()
                for modifier in tuple(curve.modifiers):
                    curve.modifiers.remove(modifier)
                add_variable(curve.driver, "output", carrier, f'["outputs"][{target.channel}]')
                add_variable(curve.driver, "epoch", carrier, '["epoch"]')
                curve.driver.expression = DRIVER_EXPRESSION
                if len(targets) > 1:
                    # Keep a private template: callers may mute or edit the
                    # yielded curves before the next resumable step.
                    template_owner = bpy.data.objects.new("NativeOutputTemplate", None)
                    template = template_owner.animation_data_create().drivers.from_existing(src_driver=curve)
            else:
                # driver_add above still performs Blender's property/index
                # validation. Copy the immutable variables in C, then replace
                # that empty curve without repeating their RNA assignments.
                drivers = target.owner.animation_data.drivers
                copied = drivers.from_existing(src_driver=template)
                installed.append((target, copied))
                copied.data_path = target.path
                copied.array_index = max(0, target.index)
                copied.driver.variables["output"].targets[0].data_path = f'["outputs"][{target.channel}]'
                drivers.remove(curve)
                installed[-2:] = [(target, copied)]
                curve = copied
            yield target.owner, curve
    except BaseException:
        remaining = {}
        for target, curve in reversed(installed):
            try:
                animation = target.owner.animation_data
                if animation is None:
                    continue
                owner = target.owner.as_pointer()
                if owner not in remaining:
                    remaining[owner] = {item.as_pointer() for item in animation.drivers}
                pointer = curve.as_pointer()
                if pointer in remaining[owner]:
                    animation.drivers.remove(curve)
                    remaining[owner].remove(pointer)
            except ReferenceError:
                # A resumable caller may already have removed its yielded curve.
                pass
        _set_targets(carrier, previous)
        raise
    finally:
        if template_owner is not None:
            bpy.data.objects.remove(template_owner, do_unlink=True)


def remove_shape_targets(carrier: Any) -> set[int]:
    """Detach only Key outputs before an importer replaces their datablocks."""
    remaining, owners = [], set()
    for item in carrier.get("targets", []):
        owner, path = resolve_output_target(carrier, item)
        if owner is not None and owner.id_type == "KEY":
            owners.add(owner.as_pointer())
            curve = (
                owner.animation_data.drivers.find(path, index=max(0, item["index"])) if owner.animation_data else None
            )
            if curve is not None and owned_curve(curve, carrier):
                _remove_driver(owner, path, item["index"])
        else:
            remaining.append(dict(item))
    if owners:
        _set_targets(carrier, remaining)
    return owners


def mute_targets(carrier: Any, muted: bool) -> None:
    """Release output channels immediately without deleting their portable bindings."""
    owners = {}
    for item in carrier.get("targets", []):
        owner, path = resolve_output_target(carrier, item)
        if owner is None or owner.animation_data is None:
            continue
        identity = owner.as_pointer()
        if identity not in owners:
            owners[identity] = {(curve.data_path, curve.array_index): curve for curve in owner.animation_data.drivers}
        curve = owners[identity].get((path, max(0, item["index"])))
        if curve and curve.mute != muted and owned_curve(curve, carrier):
            curve.mute = muted
            owner.update_tag()


def remove_targets(carrier: Any) -> None:
    """Remove only recorded, still-owned drivers; preserve user replacements."""
    for item in carrier.get("targets", []):
        owner, path = resolve_output_target(carrier, item)
        if owner is None or owner.animation_data is None:
            continue
        curve = owner.animation_data.drivers.find(path, index=max(0, item["index"]))
        if curve and owned_curve(curve, carrier):
            _remove_driver(owner, path, item["index"])


def _remove_driver(owner: Any, path: str, index: int) -> None:
    if index < 0:
        owner.driver_remove(path)
    else:
        owner.driver_remove(path, index)


def _persistent_owner(owner: Any) -> Any:
    if not owner.is_embedded_data:
        return owner
    for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights, bpy.data.scenes):
        for candidate in collection:
            if getattr(candidate, "node_tree", None) == owner:
                return candidate
    raise ValueError(f"Cannot resolve embedded native output owner: {owner.name}")


def _animated_channels(owner: Any) -> set[tuple[str, int]]:
    animation = owner.animation_data
    if not animation:
        return set()
    assignments = [(animation.action, getattr(animation, "action_slot", None))]
    for track in animation.nla_tracks:
        assignments.extend((strip.action, getattr(strip, "action_slot", None)) for strip in track.strips)
    channels = set()
    for action, slot in assignments:
        if not action:
            continue
        if hasattr(action, "layers") and action.is_action_layered:
            for layer in action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        if slot is None or bag.slot_handle == slot.handle:
                            channels.update((curve.data_path, curve.array_index) for curve in bag.fcurves)
        else:
            channels.update((curve.data_path, curve.array_index) for curve in action.fcurves)
    return channels
