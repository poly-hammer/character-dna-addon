"""FBX animation ingestion through the ufbx bindings.

Only the node hierarchy and its animation are read; geometry and embedded media
are skipped, which is what makes this dramatically faster than round-tripping
through ``bpy.ops.import_scene.fbx``.

The data is returned in the file's own coordinate space and units. Converting it
onto a Blender armature is the writer's job, because that conversion needs the
target rig's rest pose.

Requires the ``pyufbx`` package (imported as ``ufbx``). The import is deferred so
this module can be imported without the bindings present.
"""

import logging

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .maths import decompose_matrix, ensure_continuity, quat_normalize


logger = logging.getLogger(__name__)

DEFAULT_FRAME_RATE = 30.0
MAX_CLIP_BUFFER_BYTES = 1024**3

# ufbx CoordinateAxis values.
AXIS_NAMES = {
    0: "X",
    1: "-X",
    2: "Y",
    3: "-Y",
    4: "Z",
    5: "-Z",
}


@dataclass(frozen=True)
class FbxAnimationClip:
    """A node hierarchy and its baked animation, in the FBX file's own space.

    Rotations are w-first ``(w, x, y, z)`` and every transform is
    parent-relative unless the name says otherwise. Animation arrays are
    frame-major so a whole channel can be sliced without a copy.

    Attributes:
        node_names: Node names in hierarchy order (breadth first from the root).
        parent_indices: ``(N,)`` index of each node's parent, ``-1`` for roots.
        rest_rotations: ``(N, 4)`` parent-relative rest rotations.
        rest_translations: ``(N, 3)`` parent-relative rest translations.
        rest_world_rotations: ``(N, 4)`` rest rotations in file world space.
        rest_world_translations: ``(N, 3)`` rest translations in file world space.
        rotations: ``(F, N, 4)`` parent-relative animated rotations.
        translations: ``(F, N, 3)`` parent-relative animated translations.
        animated_node_names: Names of nodes that actually carried keys.
        frame_rate: Rate the animation was sampled at, in frames per second.
        file_frame_rate: The frame rate declared by the file.
        unit_meters: Size of one file unit in meters, e.g. ``0.01`` for centimeters.
        up_axis: The file's up axis, such as ``"Y"`` or ``"-Z"``.
        take_name: Name of the animation stack that was baked.
    """

    node_names: tuple[str, ...]
    parent_indices: np.ndarray
    rest_rotations: np.ndarray
    rest_translations: np.ndarray
    rest_world_rotations: np.ndarray
    rest_world_translations: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    animated_node_names: frozenset[str]
    frame_rate: float
    file_frame_rate: float
    unit_meters: float
    up_axis: str
    take_name: str

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the hierarchy."""
        return len(self.node_names)

    @property
    def num_frames(self) -> int:
        """Number of sampled frames."""
        return int(self.rotations.shape[0])

    @cached_property
    def node_indices(self) -> dict[str, int]:
        """Map of node name to its index, keeping the first of any duplicates."""
        indices: dict[str, int] = {}
        for index, name in enumerate(self.node_names):
            indices.setdefault(name, index)
        return indices


def _require_ufbx() -> ModuleType:
    """Import and return the ufbx module.

    Returns:
        The imported ``ufbx`` module.

    Raises:
        ImportError: If the bindings are not installed.
    """
    try:
        import ufbx
    except ImportError as error:
        raise ImportError(
            "The 'pyufbx' package is required for FBX animation import. It ships with the "
            "Character DNA extension; if you are running from source, install it with "
            "'uv sync' from a checkout that has ../ufbx-python next to it."
        ) from error
    return ufbx


def _interpolate_track(
    key_times: np.ndarray,
    key_values: np.ndarray,
    sample_times: np.ndarray,
    rest_value: np.ndarray,
) -> np.ndarray:
    """Sample a keyed track at uniform times, one component at a time.

    Args:
        key_times: ``(K,)`` strictly increasing key times in seconds.
        key_values: ``(K, C)`` key values.
        sample_times: ``(F,)`` output sample times in seconds.
        rest_value: ``(C,)`` fallback used when the track has no keys.

    Returns:
        ``(F, C)`` values. Samples outside the keyed range clamp to the first
        or last key.
    """
    num_frames = sample_times.shape[0]
    num_components = rest_value.shape[0]
    if key_times.shape[0] == 0:
        return np.broadcast_to(rest_value, (num_frames, num_components)).copy()

    out = np.empty((num_frames, num_components), dtype=np.float64)
    for component in range(num_components):
        out[:, component] = np.interp(sample_times, key_times, key_values[:, component])
    return out


def _order_nodes(scene: Any) -> list[Any]:
    """Return every non-root node in breadth-first hierarchy order.

    Args:
        scene: A loaded ufbx scene.

    Returns:
        The ordered node list, guaranteeing parents precede their children.
    """
    ordered: list[Any] = []
    queue = [node for node in scene.nodes if node.parent is not None and node.parent.is_root]
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        queue.extend(node.children)
    return ordered


def _select_baked_anim(scene: Any, frame_rate: float) -> tuple[Any, str]:
    """Bake the animation stack that carries the most animation.

    Files routinely contain empty leftover stacks next to the real take, so the
    stack that animates the most nodes wins, with duration breaking ties.

    Args:
        scene: A loaded ufbx scene.
        frame_rate: Resample rate in frames per second.

    Returns:
        Tuple of the baked animation and the name of the stack it came from.
    """
    baked = scene.bake_anim(resample_rate=frame_rate, trim_start_time=True)
    stack_name = ""

    anim_stacks = scene.anim_stacks
    if anim_stacks:
        stack_name = anim_stacks[0].name
        for stack in anim_stacks:
            candidate = scene.bake_anim(anim=stack.anim, resample_rate=frame_rate, trim_start_time=True)
            better_coverage = len(candidate.nodes) > len(baked.nodes)
            longer_tie_break = (
                len(candidate.nodes) == len(baked.nodes) > 0 and candidate.playback_duration > baked.playback_duration
            )
            if better_coverage or longer_tie_break:
                baked = candidate
                stack_name = stack.name

    return baked, stack_name


def load_fbx_animation(file_path: str | Path, frame_rate: float | None = None) -> FbxAnimationClip:
    """Load a node hierarchy and its animation from an FBX file.

    Args:
        file_path: Path to the FBX file.
        frame_rate: Rate to resample the animation at. Defaults to the file's
            own rate, falling back to 30 fps when the file does not declare one.

    Returns:
        The loaded clip, in the file's native space and units.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no node hierarchy or the sampled clip is too large.
        ImportError: If the ufbx bindings are not installed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FBX file not found: {file_path}")

    ufbx = _require_ufbx()
    scene = ufbx.load_file(str(file_path), ignore_geometry=True, ignore_embedded=True)
    try:
        return _build_clip(scene, frame_rate)
    finally:
        scene.close()


def load_fbx_animation_buffer(
    data: bytes | bytearray | memoryview,
    frame_rate: float | None = None,
) -> FbxAnimationClip:
    """Load a node hierarchy and its animation from an in-memory FBX buffer.

    Args:
        data: The raw contents of an FBX file.
        frame_rate: Rate to resample the animation at, as in
            :func:`load_fbx_animation`.

    Returns:
        The loaded clip, in the file's native space and units.

    Raises:
        ValueError: If the buffer contains no node hierarchy or the sampled clip is too large.
        ImportError: If the ufbx bindings are not installed.
    """
    ufbx = _require_ufbx()
    scene = ufbx.load_memory(data, ignore_geometry=True, ignore_embedded=True)
    try:
        return _build_clip(scene, frame_rate)
    finally:
        scene.close()


def _build_clip(scene: Any, frame_rate: float | None) -> FbxAnimationClip:
    """Convert a loaded ufbx scene into an :class:`FbxAnimationClip`.

    Args:
        scene: A loaded ufbx scene.
        frame_rate: Requested resample rate, or ``None`` to use the file's rate.

    Returns:
        The converted clip.

    Raises:
        ValueError: If the scene has no node hierarchy or the sampled clip is too large.
    """
    file_frame_rate = float(scene.settings.frames_per_second or 0.0)
    if frame_rate is None:
        frame_rate = file_frame_rate if file_frame_rate > 0.0 else DEFAULT_FRAME_RATE

    ordered = _order_nodes(scene)
    if not ordered:
        raise ValueError("FBX file contains no node hierarchy to import.")

    num_nodes = len(ordered)
    typed_id_to_index = {node.typed_id: index for index, node in enumerate(ordered)}

    parent_indices = np.empty(num_nodes, dtype=np.int64)
    local_matrices = np.empty((num_nodes, 4, 4), dtype=np.float64)
    world_matrices = np.empty((num_nodes, 4, 4), dtype=np.float64)

    for index, node in enumerate(ordered):
        parent = node.parent
        parent_indices[index] = (
            typed_id_to_index.get(parent.typed_id, -1) if parent is not None and not parent.is_root else -1
        )
        # The binding fills column-major data into a row-major buffer, so what
        # comes back is the transpose of the mathematical matrix.
        local_matrices[index] = np.asarray(node.local_transform, dtype=np.float64).T
        world_matrices[index] = np.asarray(node.world_transform, dtype=np.float64).T

    rest_rotations, rest_translations = decompose_matrix(local_matrices)
    rest_world_rotations, rest_world_translations = decompose_matrix(world_matrices)

    baked, take_name = _select_baked_anim(scene, frame_rate)

    duration = max(baked.playback_duration, baked.key_time_max, 0.0)
    num_frames = max(round(duration * frame_rate) + 1, 1)
    buffer_bytes = num_frames * (num_nodes * 7 + 1) * np.dtype(np.float64).itemsize
    if buffer_bytes > MAX_CLIP_BUFFER_BYTES:
        raise ValueError(
            f"FBX take {take_name!r} requires {num_frames:,} frames across {num_nodes:,} nodes "
            f"at {frame_rate:g} fps ({buffer_bytes / 1024**3:.2f} GiB of animation buffers; "
            f"limit {MAX_CLIP_BUFFER_BYTES / 1024**3:g} GiB). "
            "Check the FBX take's start/end times and re-export a shorter range, or import at a lower frame rate."
        )
    sample_times = np.arange(num_frames, dtype=np.float64) / frame_rate

    rotations = np.broadcast_to(rest_rotations[None], (num_frames, num_nodes, 4)).copy()
    translations = np.broadcast_to(rest_translations[None], (num_frames, num_nodes, 3)).copy()

    animated_node_names: set[str] = set()
    for baked_node in baked.nodes:
        node_index = typed_id_to_index.get(baked_node.typed_id)
        if node_index is None:
            continue

        has_rotation = len(baked_node.rotation_times) > 0
        has_translation = len(baked_node.translation_times) > 0
        if has_rotation or has_translation:
            animated_node_names.add(ordered[node_index].name)

        if has_rotation:
            # ufbx stores (x, y, z, w); roll it to the w-first convention, then
            # remove hemisphere flips before interpolating componentwise.
            keys = np.roll(np.asarray(baked_node.rotation_values, dtype=np.float64), 1, axis=1)
            sampled = _interpolate_track(
                np.asarray(baked_node.rotation_times, dtype=np.float64),
                ensure_continuity(keys),
                sample_times,
                rest_rotations[node_index],
            )
            rotations[:, node_index] = quat_normalize(sampled)

        if has_translation:
            translations[:, node_index] = _interpolate_track(
                np.asarray(baked_node.translation_times, dtype=np.float64),
                np.asarray(baked_node.translation_values, dtype=np.float64),
                sample_times,
                rest_translations[node_index],
            )

    return FbxAnimationClip(
        node_names=tuple(node.name for node in ordered),
        parent_indices=parent_indices,
        rest_rotations=rest_rotations,
        rest_translations=rest_translations,
        rest_world_rotations=rest_world_rotations,
        rest_world_translations=rest_world_translations,
        rotations=rotations,
        translations=translations,
        animated_node_names=frozenset(animated_node_names),
        frame_rate=float(frame_rate),
        file_frame_rate=file_frame_rate,
        unit_meters=float(scene.settings.unit_meters or 1.0),
        up_axis=AXIS_NAMES.get(int(scene.settings.axes.up), "Y"),
        take_name=take_name,
    )
