"""Dependency-ordered eye convergence using Blender's native C++ transform blend."""

from __future__ import annotations

import bpy

from .bindings import add_variable


CONSTRAINT_NAME = "Character DNA Eye Convergence"
SWITCH_NAME = "CTRL_convergenceSwitch"
CENTER_NAME = "CTRL_C_eyesAim"


def install(face: bpy.types.Object | None) -> None:
    """Blend the eye spaces, preserving the independent controls below them.

    Unreal's Face_ControlBoard_CtrlRig lerps each initial eye space (projected
    under the current central aim) to the central aim transform. Copy Transforms
    performs that blend in Blender's C++ evaluator before the RigLogic eye solve.
    Its influence uses a simple expression, also evaluated without Python.
    """
    if not face or not face.pose or face.library or face.override_library or not face.is_editable:
        return
    switch = face.pose.bones.get(SWITCH_NAME)
    center = face.pose.bones.get(CENTER_NAME)
    if not switch or not center:
        return
    limit = next((c for c in switch.constraints if c.type == "LIMIT_LOCATION" and c.use_max_x), None)
    if not limit or limit.max_x <= limit.min_x:
        return
    # The supplied face board uses a 2 cm horizontal slider, not the unit-length
    # vertical channel used by the other face switches. Read its authored range.
    expression = f"min(max((convergence - {limit.min_x!r}) / {limit.max_x - limit.min_x!r}, 0.0), 1.0)"
    for side in ("L", "R"):
        group = face.pose.bones.get(f"GRP_{side}_eyeAim")
        if not group or group.parent != center:
            continue
        constraint = group.constraints.get(CONSTRAINT_NAME)
        if constraint:
            if constraint.type != "COPY_TRANSFORMS":
                raise ValueError(f"{group.name}: incompatible {CONSTRAINT_NAME} constraint")
            path = constraint.path_from_id("influence")
            curve = face.animation_data.drivers.find(path) if face.animation_data else None
            if curve and constraint.target == face:
                continue
        else:
            constraint = group.constraints.new("COPY_TRANSFORMS")
            constraint.name = CONSTRAINT_NAME
        constraint.target = face
        constraint.subtarget = CENTER_NAME
        constraint.owner_space = "POSE"
        constraint.target_space = "POSE"
        constraint.mix_mode = "REPLACE"
        curve = constraint.driver_add("influence")
        for variable in tuple(curve.driver.variables):
            curve.driver.variables.remove(variable)
        for modifier in tuple(curve.modifiers):
            curve.modifiers.remove(modifier)
        add_variable(curve.driver, "convergence", face, switch.path_from_id("location") + "[0]")
        curve.driver.expression = expression


def upgrade_loaded() -> None:
    """Add convergence to writable, current-runtime files at load/startup only."""
    from . import engine

    for carrier in engine.carriers():
        if carrier.get("component") == "head" and not engine.requires_migration(carrier):
            install(carrier.get("face"))
