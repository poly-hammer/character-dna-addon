"""Transactional ownership of native runtime output drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bpy


CARRIER_MARKER = "character_dna_native_carrier"
DRIVER_EXPRESSION = "output + 0 * epoch"


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
    variable.type = "SINGLE_PROP"
    variable.targets[0].id_type = owner.id_type
    variable.targets[0].id = owner
    variable.targets[0].data_path = path


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
    validate_targets(targets)
    installed = []
    carrier["targets"] = [
        {
            "owner": _persistent_owner(target.owner),
            "embedded": bool(target.owner.is_embedded_data),
            "path": target.path,
            "index": target.index,
            "channel": target.channel,
        }
        for target in targets
    ]
    try:
        for target in targets:
            curve = (
                target.owner.driver_add(target.path)
                if target.index < 0
                else target.owner.driver_add(target.path, target.index)
            )
            installed.append((target, curve))
            curve.keyframe_points.clear()
            for modifier in tuple(curve.modifiers):
                curve.modifiers.remove(modifier)
            add_variable(curve.driver, "output", carrier, f'["outputs"][{target.channel}]')
            add_variable(curve.driver, "epoch", carrier, '["epoch"]')
            curve.driver.expression = DRIVER_EXPRESSION
    except Exception:
        for target, _curve in reversed(installed):
            _remove_driver(target.owner, target.path, target.index)
        del carrier["targets"]
        raise


def remove_targets(carrier: Any) -> None:
    """Remove only recorded, still-owned drivers; preserve user replacements."""
    for item in carrier.get("targets", []):
        owner = item.get("owner")
        if owner and item.get("embedded", False):
            owner = owner.node_tree
        if owner is None or owner.animation_data is None:
            continue
        curve = owner.animation_data.drivers.find(item["path"], index=max(0, item["index"]))
        if curve and owned_curve(curve, carrier):
            _remove_driver(owner, item["path"], item["index"])


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
