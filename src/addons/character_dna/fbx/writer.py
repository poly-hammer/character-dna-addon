"""Fast Action writing for animation loaded by :mod:`.reader`.

Values are converted straight into Blender pose-bone basis space and written
with ``foreach_set``, so an entire channel becomes a single bulk call instead of
one ``keyframe_insert`` per frame.

Because pose basis values are rest-relative, any constant change of basis
between the FBX file and Blender cancels out of both the rotation and the
translation, so no axis conversion is applied here.
"""

import logging

import bpy
import numpy as np

from ..constants import HAS_ACTION_SLOTS, IS_BLENDER_5
from ..validators.animation import is_face_board_animation_control
from .maths import (
    ensure_continuity,
    quat_conjugate,
    quat_multiply,
    quat_rotate_vector,
    quat_to_euler,
)
from .reader import FbxAnimationClip


if HAS_ACTION_SLOTS:
    from bpy_extras import anim_utils
else:
    anim_utils = None

logger = logging.getLogger(__name__)
# Blender scenes are 1 based, and Blender's own FBX importer starts takes on frame 1.
FIRST_FRAME = 1

EULER_ORDERS = frozenset({"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"})

_ZERO_TOLERANCE = 1e-9


def ensure_action(name: str, replace: bool = True, id_root: str = "OBJECT") -> bpy.types.Action:
    """Create an Action, optionally replacing an existing one with the same name.

    Args:
        name: Name for the Action.
        replace: Remove an existing Action of this name first.
        id_root: Data-block type the Action animates. Only used on Blender 4.5,
            where assigning an Action does not set it.

    Returns:
        The new Action.
    """
    existing = bpy.data.actions.get(name)
    if existing and replace:
        bpy.data.actions.remove(existing)

    action = bpy.data.actions.new(name=name)
    if not IS_BLENDER_5:
        action.id_root = id_root  # pyright: ignore[reportAttributeAccessIssue]
    return action


def ensure_action_channelbag(action: bpy.types.Action, slot: object) -> object:
    """Return the channelbag for a slot, creating the layer and strip if needed.

    Blender 5.x exposes ``action_ensure_channelbag_for_slot``; 4.5 ships only the
    non-creating ``action_get_channelbag_for_slot``, so the layer, strip and
    channelbag have to be built by hand there.

    Args:
        action: The action that owns the slot.
        slot: The slot whose channelbag is needed.

    Returns:
        The slot's channelbag.
    """
    ensure = getattr(anim_utils, "action_ensure_channelbag_for_slot", None)
    if ensure is not None:
        return ensure(action, slot)

    channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)  # pyright: ignore[reportOptionalMemberAccess]
    if channelbag is not None:
        return channelbag

    layer = action.layers[0] if action.layers else action.layers.new("Layer")
    strip = layer.strips[0] if layer.strips else layer.strips.new(type="KEYFRAME")
    return strip.channelbags.new(slot)  # pyright: ignore[reportAttributeAccessIssue]


def ensure_action_slot(action: bpy.types.Action, name: str, slot_type: str = "OBJECT") -> object | None:
    """Return the action's first slot, creating one when it has none.

    Args:
        action: The action to inspect.
        name: Name for a newly created slot.
        slot_type: Slot type identifier.

    Returns:
        The slot, or ``None`` on Blender versions without slotted actions.
    """
    if not HAS_ACTION_SLOTS:
        return None
    if len(action.slots) == 0:
        action.slots.new(slot_type, name=name)
    return action.slots[0]


def get_channel_container(
    action: bpy.types.Action,
    id_owner: bpy.types.Object,
    slot_type: str = "OBJECT",
) -> object | None:
    """Return the object that owns the Action's fcurves.

    Blender 4.5 stores fcurves on the Action, Blender 5.x stores them on a
    channel bag belonging to a slot.

    Args:
        action: The Action to write into.
        id_owner: The data-block the Action will be assigned to.
        slot_type: Slot type identifier used on Blender 5.x.

    Returns:
        The Action itself on Blender 4.5, its channel bag on Blender 5.x, or
        ``None`` if the channel bag could not be created.
    """
    slot = ensure_action_slot(action, id_owner.name, slot_type)
    if slot is None:
        return action
    return ensure_action_channelbag(action, slot)


def assign_action(id_owner: bpy.types.Object, action: bpy.types.Action) -> None:
    """Assign an Action, and its first slot, to a data-block.

    Args:
        id_owner: The object to assign the Action to.
        action: The Action to assign.

    Raises:
        RuntimeError: If animation data could not be created.
    """
    if not id_owner.animation_data:
        id_owner.animation_data_create()
    if not id_owner.animation_data:
        raise RuntimeError(f"Failed to create animation data for {id_owner.name}.")

    id_owner.animation_data.action = action
    slot = ensure_action_slot(action, id_owner.name)
    # Re-assigning the slot Blender already picked crashes 5.2, so only set a new one.
    if slot is not None and id_owner.animation_data.action_slot != slot:
        id_owner.animation_data.action_slot = slot


def write_bulk_fcurves(
    container: object,
    data_path: str,
    values: np.ndarray,
    frames: np.ndarray,
) -> None:
    """Create one FCurve per channel and bulk insert every keyframe.

    Args:
        container: An Action or channel bag from :func:`get_channel_container`.
        data_path: The RNA path to animate, without the array index.
        values: ``(F, C)`` values, one column per channel.
        frames: ``(F,)`` frame numbers.
    """
    num_frames, num_channels = values.shape
    coordinates = np.empty(num_frames * 2, dtype=np.float32)
    coordinates[0::2] = frames

    for channel in range(num_channels):
        fcurve = container.fcurves.new(data_path=data_path, index=channel)  # type: ignore[attr-defined]
        fcurve.keyframe_points.add(count=num_frames)
        coordinates[1::2] = values[:, channel]
        fcurve.keyframe_points.foreach_set("co", coordinates)
        fcurve.update()


def _write_rotation(
    container: object,
    pose_bone: bpy.types.PoseBone,
    rotations: np.ndarray,
    frames: np.ndarray,
) -> None:
    """Write a rotation track in whatever rotation mode the pose bone uses.

    Args:
        container: An Action or channel bag.
        pose_bone: The pose bone being animated.
        rotations: ``(F, 4)`` w-first basis quaternions.
        frames: ``(F,)`` frame numbers.
    """
    data_path = f'pose.bones["{pose_bone.name}"]'
    mode = pose_bone.rotation_mode

    if mode in EULER_ORDERS:
        euler = quat_to_euler(rotations, mode)
        # Keep angles continuous so the curve does not jump by a full turn.
        euler = np.unwrap(euler, axis=0)
        write_bulk_fcurves(container, f"{data_path}.rotation_euler", euler.astype(np.float32), frames)
        return

    if mode == "AXIS_ANGLE":
        pose_bone.rotation_mode = "QUATERNION"
    write_bulk_fcurves(container, f"{data_path}.rotation_quaternion", rotations.astype(np.float32), frames)


def _is_effectively_zero(values: np.ndarray) -> bool:
    """Return ``True`` when every value is within floating point noise of zero."""
    return bool(np.all(np.abs(values) <= _ZERO_TOLERANCE))


def _basis_rotations(rest_rotation: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Convert local rotations into rest-relative basis quaternions.

    Args:
        rest_rotation: ``(4,)`` rest rotation of the node.
        rotations: ``(F, 4)`` animated local rotations.

    Returns:
        ``(F, 4)`` continuous basis quaternions, starting in the ``w >= 0``
        hemisphere so the track reads as a deviation from rest rather than a
        full turn away from it.
    """
    basis = ensure_continuity(quat_multiply(quat_conjugate(rest_rotation), rotations))
    if basis[0, 0] < 0.0:
        basis = -basis
    return basis


def frame_range(clip: FbxAnimationClip) -> np.ndarray:
    """Return the Blender frame numbers for a clip's samples.

    Args:
        clip: The clip being written.

    Returns:
        ``(F,)`` frame numbers starting at :data:`FIRST_FRAME`.
    """
    return np.arange(clip.num_frames, dtype=np.float64) + FIRST_FRAME


def write_skeleton_animation(
    clip: FbxAnimationClip,
    armature: bpy.types.Object,
    action: bpy.types.Action,
    include_bones: set[str] | None = None,
) -> list[str]:
    """Write a skeletal clip onto an armature's pose bones.

    Node transforms are converted from FBX parent-relative space into Blender
    pose basis space: ``rotation = rest⁻¹ ∘ local`` and
    ``location = rest⁻¹ · (local - rest)`` scaled to meters. Location channels
    that never leave the rest pose are skipped.

    Args:
        clip: The loaded animation.
        armature: The target armature object.
        action: The Action to write into.
        include_bones: Only write these bone names when given.

    Returns:
        The names of the bones that were written.
    """
    if not armature.pose:
        return []

    container = get_channel_container(action, armature)
    if container is None:
        return []

    frames = frame_range(clip)
    node_indices = clip.node_indices
    written: list[str] = []

    for pose_bone in armature.pose.bones:
        name = pose_bone.name
        if include_bones is not None and name not in include_bones:
            continue

        node_index = node_indices.get(name)
        if node_index is None:
            continue

        rest_rotation = clip.rest_rotations[node_index]
        rest_translation = clip.rest_translations[node_index]
        inverse_rest = quat_conjugate(rest_rotation)

        _write_rotation(container, pose_bone, _basis_rotations(rest_rotation, clip.rotations[:, node_index]), frames)

        offsets = clip.translations[:, node_index] - rest_translation
        basis_locations = quat_rotate_vector(inverse_rest, offsets) * clip.unit_meters
        if not _is_effectively_zero(basis_locations):
            write_bulk_fcurves(
                container,
                f'pose.bones["{name}"].location',
                basis_locations.astype(np.float32),
                frames,
            )

        written.append(name)

    return written


def write_face_board_animation(
    clip: FbxAnimationClip,
    armature: bpy.types.Object,
    action: bpy.types.Action,
    exclude_bones: frozenset[str] = frozenset(),
) -> list[str]:
    """Write a face board clip onto the face board armature's control bones.

    Face board controls are driven by raw translation in the board's own space,
    so values are copied without unit conversion. Rotation channels are written
    only when a control actually rotates.

    Args:
        clip: The loaded animation.
        armature: The face board armature object.
        action: The Action to write into.
        exclude_bones: Control names that must not be animated.

    Returns:
        The names of the controls that were written.
    """
    if not armature.pose:
        return []

    container = get_channel_container(action, armature)
    if container is None:
        return []

    frames = frame_range(clip)
    node_indices = clip.node_indices
    written: list[str] = []

    for pose_bone in armature.pose.bones:
        name = pose_bone.name
        if not is_face_board_animation_control(name, exclude_bones):
            continue

        node_index = node_indices.get(name)
        if node_index is None:
            continue

        locations = clip.translations[:, node_index]
        write_bulk_fcurves(
            container,
            f'pose.bones["{name}"].location',
            locations.astype(np.float32),
            frames,
        )

        rest_rotation = clip.rest_rotations[node_index]
        basis_rotations = _basis_rotations(rest_rotation, clip.rotations[:, node_index])
        if not _is_effectively_zero(basis_rotations[:, 1:]):
            _write_rotation(container, pose_bone, basis_rotations, frames)

        written.append(name)

    return written
