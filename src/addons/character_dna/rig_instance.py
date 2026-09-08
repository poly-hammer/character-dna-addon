# standard library imports
import logging  # noqa: I001
import math
import threading

from collections.abc import Callable
from pathlib import Path
from pprint import pformat

# third party imports
import bpy
import numpy as np

from mathutils import Euler, Matrix, Quaternion, Vector

# local imports
from . import utilities
from .constants import FLOATING_POINT_PRECISION, IS_BLENDER_5, SCALE_FACTOR, SHAPE_KEY_NAME_MAX_LENGTH
from .ui import callbacks
from .typing import *  # noqa: F403


MEMORY_RESOURCE_SIZE = 1024 * 1024 * 4  # 4MB
MEMORY_RESOURCE_ALIGNMENT = 16
ATTR_COUNT_PER_QUATERNION_JOINT = 10
ATTR_COUNT_PER_EULER_JOINT = 9

logger = logging.getLogger(__name__)

# Blender runs an animation render as a job on its own thread and fires every app handler
# (render_init, frame_change_post, render_write, render_complete) from that thread. Writing
# pose bones, shape keys or materials from any thread but the main one races Blender's
# notifier queue and crashes it, so off-thread evaluations are handed to the main thread and
# waited on. Blocking is what keeps the rendered frame in step with the face board.
_MAIN_THREAD_IDENT = threading.main_thread().ident
MAIN_THREAD_TIMEOUT_SECONDS = 10.0
# Short enough to be irrelevant next to a render frame, long enough that the main loop still sleeps.
_RENDER_POLL_SECONDS = 0.001
_IDLE_POLL_SECONDS = 0.25

_main_thread_lock = threading.Lock()
_main_thread_queue: list[tuple[str, "ComponentType", bpy.types.Depsgraph | None, threading.Event]] = []
_rendering = False
_post_render_pending = False
_suppress_evaluation = False
_logged_main_thread_timeout = False


def is_main_thread() -> bool:
    return threading.current_thread().ident == _MAIN_THREAD_IDENT


def is_id_write_locked(error: BaseException) -> bool:
    """Whether the error is Blender refusing a write to ID data rather than a real coding fault.

    Depsgraph and render contexts reject the write attempt itself, and raise an ``AttributeError``
    that is indistinguishable from a typo except by its message.
    """
    return isinstance(error, AttributeError) and "Writing to ID classes" in str(error)


def apply_id_writes(description: str, apply: Callable[[], None]) -> bool:
    """Run writes to ID data, reporting rather than raising when Blender has them locked."""
    try:
        apply()
    except AttributeError as error:
        if not is_id_write_locked(error):
            raise
        logger.debug(f"Deferred {description}, Blender is not accepting writes to ID data: {error}")
        return False
    return True


def is_rendering() -> bool:
    return _rendering


def begin_render() -> None:
    """Called from ``render_init``, which Blender fires on the render job thread."""
    global _rendering
    _rendering = True


def end_render() -> None:
    """Called from ``render_complete``/``render_cancel`` on the render job thread.

    Only plain Python state is touched here; the Blender-side cleanup is queued for
    :func:`run_main_thread_evaluations` to perform on the main thread.
    """
    global _post_render_pending, _suppress_evaluation
    _suppress_evaluation = True
    _post_render_pending = True
    from .runtime.engine import clear_contexts

    clear_contexts()


def _apply_post_render_cleanup() -> None:
    global _rendering, _post_render_pending, _suppress_evaluation

    scene_properties = utilities.get_addon_scene_properties()
    if scene_properties:
        for instance in scene_properties.rig_instance_list:
            try:
                instance.clear_evaluated_references()
            except ReferenceError:
                continue

    window_manager_properties = utilities.get_addon_window_manager_properties()
    if window_manager_properties:
        window_manager_properties.is_rendering = False
        window_manager_properties.evaluate_dependency_graph = True

    _rendering = False
    _post_render_pending = False
    _suppress_evaluation = False


def run_main_thread_evaluations() -> float:
    """Timer that performs queued rig evaluations on the main thread.

    Registered by :func:`ensure_main_thread_timer` so nothing ever has to call into
    ``bpy.app.timers`` from the render job thread.
    """
    try:
        while True:
            with _main_thread_lock:
                if not _main_thread_queue:
                    break
                name, component, dependency_graph, done = _main_thread_queue.pop(0)

            try:
                scene_properties = utilities.get_addon_scene_properties()
                # Re-resolve by name: the queued instance could have been freed by an undo.
                instance = scene_properties.rig_instance_list.get(name) if scene_properties else None
                if instance:
                    instance.evaluate(component=component, dependency_graph=dependency_graph)
            except ReferenceError:
                pass
            except Exception as error:
                if is_id_write_locked(error):
                    # The next evaluation retries once Blender accepts writes again.
                    logger.debug(f"Deferred evaluation of rig instance '{name}': {error}")
                else:
                    logger.exception(f"Error evaluating rig instance '{name}': {error}")
            finally:
                done.set()

        if _post_render_pending:
            _apply_post_render_cleanup()
    except Exception as error:
        # Blender drops a timer whose callback raises, and losing this one makes renders
        # silently produce stale frames, so never let anything escape.
        logger.exception(f"Error draining main thread evaluations: {error}")

    # Poll hard while a render is waiting on us, otherwise stay out of the way.
    return _RENDER_POLL_SECONDS if _rendering else _IDLE_POLL_SECONDS


def ensure_main_thread_timer() -> None:
    """Arm the drain timer if it is not already running.

    Without it a render blocks on evaluations nothing ever performs, which renders the
    character frozen at whatever the viewport last evaluated. Cheap enough to re-check
    from the listener so the timer re-arms itself if it is ever lost.
    """
    if not bpy.app.timers.is_registered(run_main_thread_evaluations):
        bpy.app.timers.register(run_main_thread_evaluations, first_interval=0.0, persistent=True)


def _compute_input_signature(
    instance: "RigInstance", component: str, dependency_graph: bpy.types.Depsgraph | None
) -> tuple | None:
    """Build a comparable signature of a component's driver (input) bone rotations.

    A rig's driver bones are the only inputs to RigLogic's evaluation of that component. The
    bones it writes back are a disjoint set (``head_bone_transform_plan`` skips driver bones
    outright, and the body's driven/twist/swing bones are likewise separate), so when
    ``evaluate`` writes those outputs Blender re-tags the armature's transform and the listener
    fires again for an update it caused itself. Comparing this input signature lets the listener
    tell a genuine user change from that self-induced echo and break the feedback loop.

    Returns ``None`` when the signature can't be determined yet (e.g. the instance has never
    been evaluated, so the driver bone names aren't cached), in which case the caller should
    evaluate rather than risk skipping a real update.
    """
    driver_bone_names = instance.data.get(instance.cache_key(component, "driver_bone_names"))
    rig = instance.head_rig if component == "head" else instance.body_rig
    if not driver_bone_names or not rig:
        return None

    evaluated = rig.evaluated_get(dependency_graph) if dependency_graph else rig
    if not evaluated or not evaluated.pose:
        return None

    signature = []
    for name in driver_bone_names:
        pose_bone = evaluated.pose.bones.get(name)
        if not pose_bone:
            continue
        quaternion = utilities.get_pose_bone_local_quaternion(pose_bone)
        signature.append(
            (name, round(quaternion.w, 6), round(quaternion.x, 6), round(quaternion.y, 6), round(quaternion.z, 6))
        )
    return tuple(signature)


def _uses_action(animated_object: bpy.types.Object | None, action_name: str) -> bool:
    """Whether the object is driven by the named action, directly or through an NLA strip."""
    animation_data = animated_object.animation_data if animated_object else None
    if not animation_data:
        return False
    if animation_data.action and animation_data.action.name == action_name:
        return True
    return any(
        strip.action and strip.action.name == action_name
        for track in animation_data.nla_tracks
        for strip in track.strips
    )


def _is_self_induced_echo(
    instance: "RigInstance", component: str, dependency_graph: bpy.types.Depsgraph | None
) -> bool:
    """Whether an armature update is the echo of RigLogic's own write rather than a user change.

    ``evaluate`` writes the driven bones, which re-tags the armature's transform and fires the
    listener again. Driver bones are a disjoint set, so an unchanged input signature means
    nothing the rig actually reads has moved. Records the signature when it does change, so the
    next echo has something to compare against.
    """
    input_signature = _compute_input_signature(instance, component, dependency_graph)
    # A None signature means we can't tell yet (never evaluated), so evaluate rather than risk
    # skipping a real update.
    if input_signature is None:
        return False
    if input_signature == instance.data.get(instance.cache_key(component, "input_signature")):
        return True
    instance.data[instance.cache_key(component, "input_signature")] = input_signature
    return False


def _get_action_update_component(instance: "RigInstance", action_name: str) -> "ComponentType | None":
    """Which component of this instance the named action drives, if any."""
    from .runtime.engine import active as native_active

    if native_active(instance) or not instance.auto_evaluate:
        return None

    if instance.auto_evaluate_head and _uses_action(instance.face_board, action_name):
        return "head"

    if instance.auto_evaluate_body and (
        _uses_action(instance.body_rig, action_name) or _uses_action(instance.control_rig, action_name)
    ):
        # heads have rbf driven bones that move based on neck quaternions, so if head rig is present,
        # evaluate all
        if instance.head_rig and instance.auto_evaluate_head and instance.evaluate_rbfs:
            return "all"
        return "body"

    # A head imported without a body is animated on the head rig itself; with a body it follows
    # the body rig's action instead.
    if instance.auto_evaluate_head and _uses_action(instance.head_rig, action_name):
        return "head"

    return None


def _get_armature_update_component(
    instance: "RigInstance", armature_name: str, dependency_graph: bpy.types.Depsgraph | None
) -> "ComponentType | None":
    """Which component of this instance the named armature datablock drives, if any."""
    from .runtime.engine import active as native_active

    if native_active(instance) or not instance.auto_evaluate:
        return None

    if (
        instance.auto_evaluate_head
        and instance.face_board
        and instance.face_board.data
        and instance.face_board.data.name == armature_name
    ):
        return "head"

    if instance.auto_evaluate_body and (
        (instance.body_rig and instance.body_rig.data and instance.body_rig.data.name == armature_name)
        or (instance.control_rig and instance.control_rig.data and instance.control_rig.data.name == armature_name)
    ):
        # The body armature is both driven and written by RigLogic, so filter out its own echo.
        if _is_self_induced_echo(instance, "body", dependency_graph):
            return None
        # heads have rbf driven bones that move based on neck quaternions, so if head rig is present,
        # evaluate all
        if instance.head_rig and instance.auto_evaluate_head and instance.evaluate_rbfs:
            return "all"
        return "body"

    # With a full character the head rig's neck bones are copy-transform driven by the body, so the
    # body branch above already covers them. Imported on its own the head rig has no body to follow
    # and is posed directly, and it is the only thing that feeds the neck quaternions to the head
    # RBFs and re-solves the eye aim against the new head orientation.
    if (
        instance.auto_evaluate_head
        and instance.head_rig
        and instance.head_rig.data
        and instance.head_rig.data.name == armature_name
    ):
        if _is_self_induced_echo(instance, "head", dependency_graph):
            return None
        return "head"

    return None


def rig_instance_listener(_: "Scene", dependency_graph: bpy.types.Depsgraph, is_frame_change: bool = False):  # noqa: PLR0912
    addon_window_manager = utilities.get_addon_window_manager_properties()
    if not addon_window_manager:
        return

    # Safety net for a post-render cleanup that the drain timer never got to run, which would
    # otherwise leave evaluation suppressed for the rest of the session.
    if is_main_thread():
        ensure_main_thread_timer()
        if _post_render_pending:
            _apply_post_render_cleanup()

    # this condition prevents constant evaluation
    if not addon_window_manager.evaluate_dependency_graph or _suppress_evaluation:
        return

    # this condition prevents 2 evaluations per frame change, causes issues with
    # render threads accessing data while it's being updated, and causing a crash.
    if is_rendering() and not is_frame_change:
        return

    # this condition prevents evaluation after an undo operation
    if addon_window_manager.is_undoing:
        addon_window_manager.is_undoing = False
        return

    scene_properties = utilities.get_addon_scene_properties()
    if not scene_properties:
        return

    from .runtime.engine import active as native_active

    legacy_instances = [instance for instance in scene_properties.rig_instance_list if not native_active(instance)]
    if not legacy_instances:
        return

    # track the minimal set of instances that need to be updated and their components
    instance_updates = set()

    # only evaluate if in pose mode or if animation is
    if is_frame_change or bpy.context.mode == "POSE":
        for update in dependency_graph.updates:
            if not update.id:
                continue

            data_type = update.id.bl_rna.name  # type: ignore[attr-defined]
            if data_type == "Action":
                for instance in legacy_instances:
                    component = _get_action_update_component(instance, update.id.name)
                    if component:
                        instance_updates.add((instance, component))

            elif data_type == "Armature" and update.is_updated_transform:
                for instance in legacy_instances:
                    component = _get_armature_update_component(instance, update.id.name, dependency_graph)
                    if component:
                        instance_updates.add((instance, component))

    # reduce redundant updates if 'all' components are being updated anyway, no need to
    # update head/body again separately
    final_instance_updates = set()
    for instance, component in instance_updates:
        if (instance, "all") in instance_updates:
            final_instance_updates.add((instance, "all"))
        else:
            final_instance_updates.add((instance, component))

    if not final_instance_updates:
        return

    for instance, component in final_instance_updates:
        try:
            instance.evaluate(component=component, dependency_graph=dependency_graph)
        except ReferenceError:
            # The underlying data was freed out from under us; skip it.
            continue
        except Exception as error:
            if is_id_write_locked(error):
                # The next evaluation retries once Blender accepts writes again.
                logger.debug(f"Deferred evaluation of rig instance '{instance.name}': {error}")
                continue
            logger.exception(f"Error evaluating rig instance '{instance.name}': {error}")


def frame_change_handler(scene: "Scene", dependency_graph: bpy.types.Depsgraph):
    rig_instance_listener(scene, dependency_graph, is_frame_change=True)


def stop_listening():
    if bpy.app.timers.is_registered(run_main_thread_evaluations):
        bpy.app.timers.unregister(run_main_thread_evaluations)

    for handler in bpy.app.handlers.depsgraph_update_post:
        if handler.__name__ == rig_instance_listener.__name__:
            bpy.app.handlers.depsgraph_update_post.remove(handler)

    for handler in bpy.app.handlers.frame_change_post:
        if handler.__name__ == frame_change_handler.__name__:
            bpy.app.handlers.frame_change_post.remove(handler)  # pyright: ignore[reportArgumentType]


def start_listening():
    stop_listening()
    logger.info("Listening for Rig Logic...")
    # Register before anything that can fail, so a bad scene cannot leave the session deaf.
    bpy.app.handlers.depsgraph_update_post.append(rig_instance_listener)  # type: ignore[call-arg]
    bpy.app.handlers.frame_change_post.append(frame_change_handler)  # type: ignore[call-arg]
    ensure_main_thread_timer()

    context: "Context" = bpy.context  # pyright: ignore[reportAssignmentType]  # noqa: UP037
    try:
        callbacks.update_head_output_items(None, context)
        callbacks.update_body_output_items(None, context)
    except Exception as error:
        logger.exception(f"Failed to refresh the output items: {error}")


class RigInstance(bpy.types.PropertyGroup):
    from .runtime.controller import auto_evaluation_changed

    name: bpy.props.StringProperty(
        default="my_metahuman",
        description=(
            "The name associated with this Rig Instance. This is also the unique identifier "
            "for all data associated with the MetaHuman"
        ),
        update=callbacks.update_instance_name,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    auto_evaluate: bpy.props.BoolProperty(
        default=True,
        name="Auto Evaluate",
        description="Whether to automatically evaluate this rig instance when the scene is updated",
        update=auto_evaluation_changed,
    )  # pyright: ignore[reportInvalidTypeForm]
    auto_evaluate_head: bpy.props.BoolProperty(
        update=auto_evaluation_changed,
        default=True,
        name="Auto Evaluate Head",
        description=(
            "Whether to automatically evaluate the head components on this rig instance when the scene is updated"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    auto_evaluate_body: bpy.props.BoolProperty(
        update=auto_evaluation_changed,
        default=True,
        name="Auto Evaluate Body",
        description=(
            "Whether to automatically evaluate the body components on this rig instance when the scene is updated"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    evaluate_bones: bpy.props.BoolProperty(
        default=True,
        name="Evaluate Bones",
        description="Whether to evaluate bone positions based on the face board controls",
    )  # pyright: ignore[reportInvalidTypeForm]
    evaluate_shape_keys: bpy.props.BoolProperty(
        default=True,
        name="Evaluate Shape Keys",
        description="Whether to evaluate shape keys based on the face board controls",
    )  # pyright: ignore[reportInvalidTypeForm]
    evaluate_texture_masks: bpy.props.BoolProperty(
        default=True,
        name="Evaluate Texture Masks",
        description="Whether to evaluate texture masks based on the face board controls",
    )  # pyright: ignore[reportInvalidTypeForm]
    evaluate_rbfs: bpy.props.BoolProperty(
        default=True,
        name="Evaluate RBFs",
        description="Whether to evaluate RBFs based on the driver bones quaternion rotations",
        update=callbacks.update_evaluate_rbfs_value,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    face_board: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Face Board",
        description="The face board that rig logic reads control positions from",
        poll=callbacks.poll_face_boards,
    )  # pyright: ignore[reportInvalidTypeForm]
    control_rig: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Control Rig",
        description="The control rig that drives the body rig",
        poll=callbacks.poll_control_rig,
    )  # pyright: ignore[reportInvalidTypeForm]
    head_dna_file_path: bpy.props.StringProperty(
        name="Head DNA File",
        description="The path to the head DNA file that rig logic reads from when evaluating the face board controls",
        subtype="FILE_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"},
    )  # pyright: ignore[reportInvalidTypeForm]
    head_mesh: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Head Mesh",
        description="The head mesh with the shape keys that rig logic will evaluate",
        poll=callbacks.poll_head_mesh,
        update=callbacks.update_head_output_items,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    head_rig: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Head Rig",
        description="The armature object that rig logic will evaluate",
        poll=callbacks.poll_head_rig,
        update=callbacks.update_head_output_items,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    head_material: bpy.props.PointerProperty(
        type=bpy.types.Material,
        name="Head Material",
        description="The head material that has a node with wrinkle map sliders that rig logic will evaluate",
        poll=callbacks.poll_head_materials,
        update=callbacks.update_head_output_items,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    body_dna_file_path: bpy.props.StringProperty(
        name="Body DNA File",
        description="The path to the body DNA file",
        subtype="FILE_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"},
    )  # pyright: ignore[reportInvalidTypeForm]
    body_mesh: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Body Mesh",
        description="The body mesh",
        poll=callbacks.poll_body_mesh,
        update=callbacks.update_body_output_items,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    body_rig: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Body Rig",
        description="The armature object for the body that RBF will evaluate",
        poll=callbacks.poll_body_rig,
        update=callbacks.update_body_output_items,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    body_material: bpy.props.PointerProperty(
        type=bpy.types.Material,
        name="Body Material",
        description="The body material",
        poll=callbacks.poll_body_materials,
        update=callbacks.update_body_output_items,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]

    # ----- Internal Properties -----
    head_to_body_constraint_influence: bpy.props.FloatProperty(
        name="Constrain Head to Body",
        default=1.0,
        description="The influence of the head to body constraint",
        update=callbacks.update_head_to_body_constraint_influence,  # type: ignore[call-arg]
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )  # pyright: ignore[reportInvalidTypeForm]

    old_name: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]

    # this holds the rig logic references
    data = {}

    warning_messages = []

    def cache_key(self, component: str, descriptor: str) -> str:
        return f"{self.name}_{component}_{descriptor}"

    def get_shape_key(self, mesh_index: int) -> bpy.types.Key | None:
        shape_key = self.data.get(self.cache_key("head", "shape_key"), {}).get(mesh_index)
        try:
            if shape_key:
                return shape_key
        except ReferenceError:
            return None

    def get_shape_key_block(self, mesh_index: int, name: str) -> bpy.types.ShapeKey | None:
        cached_shape_key = self.get_shape_key(mesh_index)
        try:
            if cached_shape_key and cached_shape_key.key_blocks:
                return cached_shape_key.key_blocks.get(name)
        except ReferenceError:
            pass

        mesh_object = self.head_mesh_index_lookup.get(mesh_index)
        if mesh_object:
            self.data[self.cache_key("head", "shape_key")] = self.data.get(self.cache_key("head", "shape_key"), {})
            for shape_key in bpy.data.shape_keys:
                if shape_key.user == mesh_object.data:
                    key_block = shape_key.key_blocks.get(name)
                    if key_block:
                        # store the shape key in the shape key property so we don't have to search for it again
                        self.data[self.cache_key("head", "shape_key")][mesh_index] = shape_key
                        return key_block
        return None

    def apply_dependency_graph_update(self, dependency_graph: bpy.types.Depsgraph | None = None):
        if not dependency_graph:
            dependency_graph = bpy.context.evaluated_depsgraph_get()

        if self.head_rig:
            self.data[self.cache_key("head", "rig_evaluated")] = self.head_rig.evaluated_get(dependency_graph)
        if self.body_rig:
            self.data[self.cache_key("body", "rig_evaluated")] = self.body_rig.evaluated_get(dependency_graph)
        if self.face_board:
            self.data[self.cache_key("head", "face_board_evaluated")] = self.face_board.evaluated_get(dependency_graph)

    @property
    def head_rig_evaluated(self) -> bpy.types.Object | None:
        result = self.data.get(self.cache_key("head", "rig_evaluated"))
        if result is not None:
            return result
        # Lazy fallback: only call evaluated_depsgraph_get() when the cached value is missing.
        # Temporarily disable the dependency graph flag to prevent re-entrant handler execution,
        # since evaluated_depsgraph_get() can trigger depsgraph_update_post handlers.
        if self.head_rig:
            window_manager_properties = utilities.get_addon_window_manager_properties()
            prev_flag = window_manager_properties.evaluate_dependency_graph if window_manager_properties else True
            if window_manager_properties:
                window_manager_properties.evaluate_dependency_graph = False
            try:
                result = self.head_rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
            finally:
                if window_manager_properties:
                    window_manager_properties.evaluate_dependency_graph = prev_flag
        return result

    @property
    def body_rig_evaluated(self) -> bpy.types.Object | None:
        result = self.data.get(self.cache_key("body", "rig_evaluated"))
        if result is not None:
            return result
        # Lazy fallback: only call evaluated_depsgraph_get() when the cached value is missing.
        # Temporarily disable the dependency graph flag to prevent re-entrant handler execution,
        # since evaluated_depsgraph_get() can trigger depsgraph_update_post handlers.
        if self.body_rig:
            window_manager_properties = utilities.get_addon_window_manager_properties()
            prev_flag = window_manager_properties.evaluate_dependency_graph if window_manager_properties else True
            if window_manager_properties:
                window_manager_properties.evaluate_dependency_graph = False
            try:
                result = self.body_rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
            finally:
                if window_manager_properties:
                    window_manager_properties.evaluate_dependency_graph = prev_flag
        return result

    @property
    def face_board_evaluated(self) -> bpy.types.Object | None:
        """The face board as the active dependency graph evaluated it.

        Rig logic inputs must be read from here rather than ``self.face_board``. Blender
        renders through a separate dependency graph and never flushes the animated pose back
        to the original datablock, so during a render the original face board still holds the
        frame the viewport last evaluated -- one frame behind what is being rendered.
        """
        result = self.data.get(self.cache_key("head", "face_board_evaluated"))
        if result is not None:
            return result
        # Lazy fallback: only call evaluated_depsgraph_get() when the cached value is missing.
        # Temporarily disable the dependency graph flag to prevent re-entrant handler execution,
        # since evaluated_depsgraph_get() can trigger depsgraph_update_post handlers.
        if self.face_board:
            window_manager_properties = utilities.get_addon_window_manager_properties()
            prev_flag = window_manager_properties.evaluate_dependency_graph if window_manager_properties else True
            if window_manager_properties:
                window_manager_properties.evaluate_dependency_graph = False
            try:
                result = self.face_board.evaluated_get(bpy.context.evaluated_depsgraph_get())
            finally:
                if window_manager_properties:
                    window_manager_properties.evaluate_dependency_graph = prev_flag
        return result

    @property
    def is_pro(self) -> bool:
        return utilities.pro_features_visible()

    @property
    def head_valid(self) -> bool:
        logged_warning = self.data.get(self.cache_key("head", "logged_validation_warning"), False)

        if not self.head_dna_file_path:
            if not logged_warning:
                logger.warning(
                    f"The Head DNA file path is not set. The Rig Instance {self.name} will not be initialized."
                )
                self.data[self.cache_key("head", "logged_validation_warning")] = True
            return False
        dna_file_path = Path(bpy.path.abspath(self.head_dna_file_path))
        if not dna_file_path.is_file():
            if not logged_warning:
                logger.warning(
                    f'The Head DNA file path "{dna_file_path}" is not a file. The Rig Instance {self.name} '
                    "will not be initialized."
                )
                self.data[self.cache_key("head", "logged_validation_warning")] = True
            return False

        if not dna_file_path.exists():
            if not logged_warning:
                logger.warning(
                    f'The Head DNA file path "{dna_file_path}" does not exist. The Rig Instance {self.name} '
                    "will not be initialized."
                )
                self.data[self.cache_key("head", "logged_validation_warning")] = True
            return False
        if not self.face_board:
            if not logged_warning:
                logger.warning(f"The Face board is not set. The Rig Instance {self.name} will not be initialized.")
                self.data[self.cache_key("head", "logged_validation_warning")] = True
            return False
        return True

    @property
    def body_valid(self) -> bool:
        logged_warning = self.data.get(self.cache_key("body", "logged_validation_warning"), False)
        if not self.body_dna_file_path:
            if not logged_warning:
                logger.warning(
                    f"The Body DNA file path is not set. The Rig Instance {self.name} will not be initialized."
                )
                self.data[self.cache_key("body", "logged_validation_warning")] = True
            return False

        dna_file_path = Path(bpy.path.abspath(self.body_dna_file_path))
        if not dna_file_path.is_file():
            if not logged_warning:
                logger.warning(
                    f'The Body DNA file path "{dna_file_path}" is not a file. The Rig Instance {self.name}'
                    " will not be initialized."
                )
                self.data[self.cache_key("body", "logged_validation_warning")] = True
            return False

        if not dna_file_path.exists():
            if not logged_warning:
                logger.warning(
                    f'The Body DNA file path "{dna_file_path}" does not exist. The Rig Instance {self.name} '
                    "will not be initialized."
                )
                self.data[self.cache_key("body", "logged_validation_warning")] = True
            return False
        return True

    @property
    def head_texture_masks_node(self) -> bpy.types.ShaderNodeGroup | None:
        # first check if the texture masks node is set
        if not self.head_material:
            return None

        return callbacks.get_head_texture_logic_node(self.head_material)

    @property
    def head_initialized(self) -> bool:
        return bool(self.data.get(self.cache_key("head", "initialized")))

    @property
    def body_initialized(self) -> bool:
        return bool(self.data.get(self.cache_key("body", "initialized")))

    @property
    def head_constrained_to_body(self) -> bool:
        return bool(self.body_rig) and round(self.head_to_body_constraint_influence, 4) > 0.0

    @property
    def head_use_eye_aim(self) -> bool:
        face_board = self.face_board_evaluated
        if not face_board or not face_board.pose:
            return False
        look_at_switch = face_board.pose.bones.get("CTRL_lookAtSwitch")
        return bool(look_at_switch and look_at_switch.location.y >= 0.99)

    @property
    def head_mesh_index_lookup(self) -> dict[int, bpy.types.Object]:
        if not self.head_dna_reader:
            return {}

        mesh_index_lookup = self.data.get(self.cache_key("head", "mesh_index_lookup"), {})
        if mesh_index_lookup:
            return mesh_index_lookup

        for mesh_index in range(self.head_dna_reader.getMeshCount()):
            dna_mesh_name = self.head_dna_reader.getMeshName(mesh_index)
            mesh_object = bpy.data.objects.get(f"{self.name}_{dna_mesh_name}")
            if mesh_object:
                mesh_index_lookup[mesh_index] = mesh_object

        self.data[self.cache_key("head", "mesh_index_lookup")] = mesh_index_lookup
        return self.data[self.cache_key("head", "mesh_index_lookup")]

    @property
    def head_channel_name_to_index_lookup(self) -> dict[str, int]:
        if not self.head_dna_reader:
            return {}

        channel_name_to_index_lookup = self.data.get(self.cache_key("head", "channel_name_to_index_lookup"), {})
        if channel_name_to_index_lookup:
            return channel_name_to_index_lookup

        for mesh_index in self.head_dna_reader.getMeshIndicesForLOD(0):
            mesh_name = self.head_dna_reader.getMeshName(mesh_index)
            for index in range(self.head_dna_reader.getBlendShapeTargetCount(mesh_index)):
                channel_index = self.head_dna_reader.getBlendShapeChannelIndex(mesh_index, index)
                shape_key_name = self.head_dna_reader.getBlendShapeChannelName(channel_index)
                channel_name_to_index_lookup[f"{mesh_name}__{shape_key_name}"] = channel_index

        self.data[self.cache_key("head", "channel_name_to_index_lookup")] = channel_name_to_index_lookup
        return self.data[self.cache_key("head", "channel_name_to_index_lookup")]

    @property
    def head_channel_index_to_mesh_index_lookup(self) -> dict[int, int]:
        if not self.head_dna_reader:
            return {}

        mesh_shape_key_index_lookup = self.data.get(self.cache_key("head", "mesh_shape_key_index_lookup"), {})
        if mesh_shape_key_index_lookup:
            return mesh_shape_key_index_lookup

        # build a lookup dictionary of shape key index to mesh index
        for mesh_index in self.head_dna_reader.getMeshIndicesForLOD(0):
            for index in range(self.head_dna_reader.getBlendShapeTargetCount(mesh_index)):
                channel_index = self.head_dna_reader.getBlendShapeChannelIndex(mesh_index, index)
                mesh_shape_key_index_lookup[channel_index] = mesh_index
        self.data[self.cache_key("head", "mesh_shape_key_index_lookup")] = mesh_shape_key_index_lookup
        return mesh_shape_key_index_lookup

    @property
    def head_manager(self) -> "riglogic.RigLogic":
        return self.data.get(self.cache_key("head", "manager"))  # pyright: ignore[reportReturnType]

    @property
    def head_instance(self) -> "riglogic.RigInstance":
        from .runtime.engine import synchronize_legacy

        return synchronize_legacy(self, "head", self.data.get(self.cache_key("head", "instance")))

    @property
    def head_dna_reader(self) -> "dna.BinaryStreamReader":
        return self.data.get(self.cache_key("head", "dna_reader"))  # pyright: ignore[reportReturnType]

    @property
    def body_manager(self) -> "riglogic.RigLogic":
        return self.data.get(self.cache_key("body", "manager"))  # pyright: ignore[reportReturnType]

    @property
    def body_instance(self) -> "riglogic.RigInstance":
        from .runtime.engine import synchronize_legacy

        return synchronize_legacy(self, "body", self.data.get(self.cache_key("body", "instance")))

    @property
    def body_dna_reader(self) -> "dna.BinaryStreamReader":
        return self.data.get(self.cache_key("body", "dna_reader"))  # pyright: ignore[reportReturnType]

    @property
    def head_shape_key_blocks(self) -> dict[int, list[bpy.types.ShapeKey]]:
        if not self.head_dna_reader:
            return {}

        shape_key_blocks = self.data.get(self.cache_key("head", "shape_key_blocks"))
        if shape_key_blocks is None:
            mesh_index = 0  # this is the head lod 0 mesh index
            shape_key_blocks = {}
            # Ordered, namespaced block names backing the UI shape key list. Kept as plain
            # strings (undo-safe) and written to the scene-side `shape_key_list` collection
            # separately by `sync_shape_key_list`. This getter only writes to `self.data`
            # (never ID data), so it is safe to call from a property getter / UI draw -- e.g.
            # when undo clears the volatile block cache via `destroy_references`.
            shape_key_block_names: list[str] = []

            # Note: That lod 0 is the only lod that has shape keys
            failed_to_cache_count = 0
            for mesh_index in self.head_dna_reader.getMeshIndicesForLOD(0):
                mesh_object = self.head_mesh_index_lookup.get(mesh_index)
                if not mesh_object:
                    logger.warning(f'The mesh object for mesh index "{mesh_index}" was not found')
                    continue

                for target_index in range(self.head_dna_reader.getBlendShapeTargetCount(mesh_index)):
                    channel_index = self.head_dna_reader.getBlendShapeChannelIndex(mesh_index, target_index)
                    name = self.head_dna_reader.getBlendShapeChannelName(channel_index)
                    dna_mesh_name = utilities.remove_instance_prefix(mesh_object.name, self.name)
                    shape_key_block_name = f"{dna_mesh_name}__{name}"
                    shape_key_block = self.get_shape_key_block(mesh_index=mesh_index, name=shape_key_block_name)
                    if shape_key_block:
                        # remember the block name for the UI list (built in a write-safe context)
                        shape_key_block_names.append(shape_key_block_name)

                        # store the shape key block in a list on the dictionary
                        key_block_list = shape_key_blocks.get(channel_index, [])
                        key_block_list.append(shape_key_block)
                        shape_key_blocks[channel_index] = key_block_list

                    elif len(shape_key_block_name) <= SHAPE_KEY_NAME_MAX_LENGTH:
                        failed_to_cache_count += 1

            if failed_to_cache_count > 0:
                logger.warning(
                    f"Rig Instance {self.name} did not cache {failed_to_cache_count} shape key blocks, "
                    "because they are not in the scene. However they are in the DNA file. Import all shape "
                    "keys to cache them."
                )

            self.data[self.cache_key("head", "shape_key_block_names")] = shape_key_block_names
            self.data[self.cache_key("head", "shape_key_blocks")] = shape_key_blocks

        return self.data[self.cache_key("head", "shape_key_blocks")]

    @property
    def head_shape_key_apply_plan(
        self,
    ) -> list[tuple["bpy.types.bpy_prop_collection", np.ndarray, np.ndarray, list[bpy.types.ShapeKey], np.ndarray]]:
        """Precomputed per-mesh scatter plan for bulk shape-key value writes.

        Each entry is ``(key_blocks, positions, channels, blocks, buffer)`` for one LOD0
        head mesh, where ``positions`` are the collection indices of the RigLogic-driven
        blocks inside that mesh's ``key_blocks``, ``channels`` are the matching blend-shape
        channel indices into ``getBlendShapeOutputs()``, ``blocks`` are the parallel block
        references (only used when collecting values for baking), and ``buffer`` is a
        preallocated ``float32`` scratch array sized to the whole collection.

        This lets ``update_head_shape_keys`` replace hundreds of per-block ``.value =`` RNA
        writes with one ``foreach_get`` / scatter / ``foreach_set`` per mesh. It holds live
        ``bpy`` collection wrappers, so it is registered in ``destroy_references`` and rebuilt
        lazily after an undo.
        """
        plan = self.data.get(self.cache_key("head", "shape_key_apply_plan"))
        if plan is not None:
            return plan

        plan = []
        if self.head_dna_reader:
            for mesh_index in self.head_dna_reader.getMeshIndicesForLOD(0):
                mesh_object = self.head_mesh_index_lookup.get(mesh_index)
                if (
                    not mesh_object
                    or not isinstance(mesh_object.data, bpy.types.Mesh)
                    or not mesh_object.data.shape_keys
                ):
                    continue

                key_blocks = mesh_object.data.shape_keys.key_blocks
                # Resolve each namespaced block name to its collection index once.
                name_to_position = {block.name: position for position, block in enumerate(key_blocks)}
                dna_mesh_name = utilities.remove_instance_prefix(mesh_object.name, self.name)

                positions: list[int] = []
                channels: list[int] = []
                blocks: list[bpy.types.ShapeKey] = []
                for target_index in range(self.head_dna_reader.getBlendShapeTargetCount(mesh_index)):
                    channel_index = self.head_dna_reader.getBlendShapeChannelIndex(mesh_index, target_index)
                    name = self.head_dna_reader.getBlendShapeChannelName(channel_index)
                    position = name_to_position.get(f"{dna_mesh_name}__{name}")
                    if position is None:
                        continue
                    positions.append(position)
                    channels.append(channel_index)
                    blocks.append(key_blocks[position])

                if not positions:
                    continue

                plan.append(
                    (
                        key_blocks,
                        np.asarray(positions, dtype=np.intp),
                        np.asarray(channels, dtype=np.intp),
                        blocks,
                        np.empty(len(key_blocks), dtype=np.float32),
                    )
                )

        # An empty plan means no mesh resolved yet (e.g. a renamed or merged head mesh).
        # Caching it would shadow a later correction for the rest of the session.
        if not plan:
            return plan

        self.data[self.cache_key("head", "shape_key_apply_plan")] = plan
        return plan

    def sync_shape_key_list(self) -> None:
        """Rebuild the UI shape-key list -- a ``CollectionProperty`` on the scene-stored
        ``ShapeKeyEditorProperties`` -- from the cached block names.

        This writes ID data, so it must only be called from a write-safe context such as
        ``head_initialize`` or an operator, never from a property getter or UI draw. The
        list is undo-tracked by Blender, so it does not need rebuilding on undo; only the
        volatile ``shape_key_blocks`` wrapper cache does (see ``destroy_references``)."""
        shape_key_editor: ShapeKeyEditorProperties | None = getattr(self, "shape_key_editor", None)
        if not shape_key_editor:
            return

        # Ensure the block cache (and its ordered names) exist before mirroring them.
        self.head_shape_key_blocks  # noqa: B018
        shape_key_block_names = self.data.get(self.cache_key("head", "shape_key_block_names"), [])

        # The has-deltas map is derived from the live block coords; drop it so it
        # recomputes lazily against the freshly synced blocks.
        self.data.pop(self.cache_key("head", "shape_key_has_deltas"), None)

        shape_key_editor.shape_key_list.clear()
        for shape_key_block_name in shape_key_block_names:
            shape_key_item = shape_key_editor.shape_key_list.add()
            shape_key_item.name = shape_key_block_name

    @property
    def head_rest_pose(self) -> dict[str, tuple[Vector, Euler, Vector, Matrix]]:
        rest_pose = self.data.get(self.cache_key("head", "rest_pose"), {})
        if rest_pose:
            return rest_pose

        # make sure the rig bone are using the correct rotation mode
        if self.head_rig_evaluated and self.head_rig_evaluated.pose:
            for pose_bone in self.head_rig_evaluated.pose.bones:
                mode = "QUATERNION" if pose_bone.name in self.head_driver_bone_names else "XYZ"
                if pose_bone.rotation_mode != mode:
                    pose_bone.rotation_mode = mode
                # save the rest pose and their parent space matrix so we don't have to calculate it again
                rest_pose[pose_bone.name] = utilities.get_bone_rest_transformations(pose_bone.bone)

        # save the rest pose so we don't have to calculate it again
        self.data[self.cache_key("head", "rest_pose")] = rest_pose
        # return a copy so the original rest position is not modified
        return self.data[self.cache_key("head", "rest_pose")]

    @property
    def head_driven_bone_names(self) -> list[str]:
        driven_bone_names = self.data.get(self.cache_key("head", "driven_bone_names"), [])
        if driven_bone_names:
            return driven_bone_names

        # get the head rbf driven bone names
        for solver_index in range(self.head_dna_reader.getRBFSolverCount()):
            for pose_index in self.head_dna_reader.getRBFSolverPoseIndices(solver_index):
                for attr_index in self.head_dna_reader.getRBFPoseJointOutputIndices(pose_index):
                    joint_index = attr_index // ATTR_COUNT_PER_EULER_JOINT
                    driven_bone_names.append(self.head_dna_reader.getJointName(joint_index))

        # save the driven bone names so we don't have to query them again
        self.data[self.cache_key("head", "driven_bone_names")] = list(set(driven_bone_names))
        return self.data[self.cache_key("head", "driven_bone_names")]

    @property
    def head_driver_bone_names(self) -> list[str]:
        driver_bone_names = self.data.get(self.cache_key("head", "driver_bone_names"), [])
        if driver_bone_names:
            return driver_bone_names

        driver_bone_names = set()
        for index in range(self.head_dna_reader.getRawControlCount()):
            full_name = self.head_dna_reader.getRawControlName(index)
            control_name, axis = full_name.split(".")
            if axis.startswith("q"):
                driver_bone_names.add(control_name)

        # save the raw control bone names so we don't have to query them again
        self.data[self.cache_key("head", "driver_bone_names")] = list(driver_bone_names)
        # return a copy so the original raw control bone names are not modified
        return self.data[self.cache_key("head", "driver_bone_names")]

    @property
    def head_gui_control_plan(self) -> list[tuple[int, str, str]]:
        """Precomputed ``(index, control_name, axis)`` for every head GUI control.

        The control names and axes come from the DNA and never change, so we parse
        them once at initialization instead of calling ``getGUIControlName`` and
        splitting strings for every control on every evaluation.
        """
        plan = self.data.get(self.cache_key("head", "gui_control_plan"))
        if plan is not None:
            return plan

        plan = []
        if self.head_dna_reader:
            for index in range(self.head_dna_reader.getGUIControlCount()):
                full_name = self.head_dna_reader.getGUIControlName(index)
                control_name, axis = full_name.split(".")
                axis = axis.rsplit("t", -1)[-1].lower()
                plan.append((index, control_name, axis))

        self.data[self.cache_key("head", "gui_control_plan")] = plan
        return plan

    @property
    def head_raw_quat_plan(self) -> list[tuple[int, str, str]]:
        """Precomputed ``(index, control_name, axis)`` for the head quaternion raw controls.

        Only ``.q*`` raw controls are driven by bone rotations, so we precompute just
        those (with their parsed axis) once instead of scanning and parsing every raw
        control name on every evaluation.
        """
        plan = self.data.get(self.cache_key("head", "raw_quat_plan"))
        if plan is not None:
            return plan

        plan = []
        if self.head_dna_reader:
            for index in range(self.head_dna_reader.getRawControlCount()):
                full_name = self.head_dna_reader.getRawControlName(index)
                control_name, axis = full_name.split(".")
                if not axis.startswith("q"):
                    continue
                axis = axis.rsplit("q", -1)[-1].lower()
                plan.append((index, control_name, axis))

        self.data[self.cache_key("head", "raw_quat_plan")] = plan
        return plan

    @property
    def head_animated_map_plan(self) -> list[tuple[int, str]]:
        """Precomputed ``(index, slider_name)`` for every head animated (texture) map.

        The slider name string is derived purely from the DNA animated-map name, so we
        build it once at initialization instead of rebuilding it for every map on every
        evaluation.
        """
        plan = self.data.get(self.cache_key("head", "animated_map_plan"))
        if plan is not None:
            return plan

        plan = []
        if self.head_dna_reader:
            for index in range(self.head_dna_reader.getAnimatedMapCount()):
                name = self.head_dna_reader.getAnimatedMapName(index)
                slider_name = (
                    f"{name.split('.')[0].split('_')[1].lower().replace('cm', 'wm')}.{name.split('.')[-1]}_msk"
                )
                plan.append((index, slider_name))

        self.data[self.cache_key("head", "animated_map_plan")] = plan
        return plan

    @property
    def head_bone_transform_plan(self) -> list[tuple[int, str, Vector, Euler, Vector, Matrix, bool]]:
        """Precomputed per-joint transform plan for the head rig (written joints only).

        Each entry is ``(joint_index, bone_name, rest_location, rest_rotation, rest_scale,
        rest_to_parent_inverse, has_children)``. Driver bones and bones missing from the rig are
        excluded once here instead of being filtered every frame, and the rest-to-parent matrix
        inverse (constant for the lifetime of the rig) is precomputed instead of being recomputed
        for all 870 joints on every evaluation.
        """
        plan = self.data.get(self.cache_key("head", "bone_transform_plan"))
        if plan is not None:
            return plan

        # don't cache an empty plan until the rig and rest pose are available
        if not (self.head_rig and self.head_dna_reader and self.head_rest_pose):
            return []

        plan = []
        rest_pose = self.head_rest_pose
        driver_bone_names = frozenset(self.head_driver_bone_names)
        pose_bones = self.head_rig.pose.bones
        for index in range(self.head_dna_reader.getJointCount()):
            name = self.head_dna_reader.getJointName(index)
            # only update the facial bones or non-driver bones
            if name in driver_bone_names:
                continue
            pose_bone = pose_bones.get(name)
            if not pose_bone:
                logger.warning(
                    f'The bone "{name}" was not found on "{self.head_rig.name}". Rig Logic will not update the bone.'
                )
                continue
            rest_transformations = rest_pose.get(name)
            if rest_transformations is None:
                continue
            rest_location, rest_rotation, rest_scale, rest_to_parent_matrix = rest_transformations
            plan.append(
                (
                    index,
                    name,
                    rest_location,
                    rest_rotation,
                    rest_scale,
                    rest_to_parent_matrix.inverted_safe(),
                    bool(pose_bone.children),
                )
            )

        self.data[self.cache_key("head", "bone_transform_plan")] = plan
        return plan

    @property
    def body_rest_pose(self) -> dict[str, tuple[Vector, Euler, Vector, Matrix]]:
        rest_pose = self.data.get(self.cache_key("body", "rest_pose"), {})
        if rest_pose:
            return rest_pose

        # make sure the rig bone are using the correct rotation mode
        if self.body_rig_evaluated and self.body_rig_evaluated.pose:
            for pose_bone in self.body_rig_evaluated.pose.bones:
                # make sure the body bones are using the correct rotation mode
                mode = "QUATERNION" if pose_bone.name in self.body_driver_bone_names else "XYZ"
                if pose_bone.rotation_mode != mode:
                    pose_bone.rotation_mode = mode

                # save the rest pose and their parent space matrix so we don't have to calculate it again
                rest_pose[pose_bone.name] = utilities.get_bone_rest_transformations(pose_bone.bone, rotation_mode="XYZ")

        # save the rest pose so we don't have to calculate it again
        self.data[self.cache_key("body", "rest_pose")] = rest_pose
        # return a copy so the original rest position is not modified
        return self.data[self.cache_key("body", "rest_pose")]

    @property
    def body_twist_bone_names(self) -> list[str]:
        twist_bone_names = self.data.get(self.cache_key("body", "twist_bone_names"), [])
        if twist_bone_names:
            return twist_bone_names

        # get the updated twist bone names
        for twist_index in range(self.body_dna_reader.getTwistCount()):
            for output_index in self.body_dna_reader.getTwistOutputJointIndices(twist_index):
                twist_bone_names.append(self.body_dna_reader.getJointName(output_index))

        # save the updated bone names so we don't have to query them again
        self.data[self.cache_key("body", "twist_bone_names")] = list(set(twist_bone_names))
        return self.data[self.cache_key("body", "twist_bone_names")]

    @property
    def body_swing_bone_names(self) -> list[str]:
        swing_bone_names = self.data.get(self.cache_key("body", "swing_bone_names"), [])
        if swing_bone_names:
            return swing_bone_names

        # get the body swing bone names
        for swing_index in range(self.body_dna_reader.getSwingCount()):
            for output_index in self.body_dna_reader.getSwingOutputJointIndices(swing_index):
                swing_bone_names.append(self.body_dna_reader.getJointName(output_index))

        # save the updated bone names so we don't have to query them again
        self.data[self.cache_key("body", "swing_bone_names")] = list(set(swing_bone_names))
        return self.data[self.cache_key("body", "swing_bone_names")]

    @property
    def body_driven_bone_names(self) -> list[str]:
        driven_bone_names = self.data.get(self.cache_key("body", "driven_bone_names"), [])
        if driven_bone_names:
            return driven_bone_names

        # get the body rbf driven bone names
        for solver_index in range(self.body_dna_reader.getRBFSolverCount()):
            for pose_index in self.body_dna_reader.getRBFSolverPoseIndices(solver_index):
                for attr_index in self.body_dna_reader.getRBFPoseJointOutputIndices(pose_index):
                    joint_index = attr_index // ATTR_COUNT_PER_EULER_JOINT
                    driven_bone_names.append(self.body_dna_reader.getJointName(joint_index))

        # save the driven bone names so we don't have to query them again
        self.data[self.cache_key("body", "driven_bone_names")] = list(set(driven_bone_names))
        return self.data[self.cache_key("body", "driven_bone_names")]

    @property
    def body_driver_bone_names(self) -> list[str]:
        if not self.body_rig:
            return []

        # check if we have already cached the driver bone names
        driver_bone_names = self.data.get(self.cache_key("body", "driver_bone_names"), [])
        if driver_bone_names:
            return driver_bone_names

        # get the rbf driver bone names
        driver_bone_names = {
            self.body_dna_reader.getRawControlName(i).split(".")[0]
            for i in range(self.body_dna_reader.getRawControlCount())
        }
        # also include the head driver bone names since they are stored in the head DNA, but the
        # body rig uses those same bones (neck_01, neck_02, head)
        if self.head_dna_reader:
            for bone_name in self.head_driver_bone_names:
                # only add the driver bone if it exists in the body rig
                if self.body_rig.pose.bones.get(bone_name):
                    driver_bone_names.add(bone_name)

        # save the driver bone names so we don't have to query them again
        self.data[self.cache_key("body", "driver_bone_names")] = list(driver_bone_names)
        return self.data[self.cache_key("body", "driver_bone_names")]

    @property
    def body_raw_plan(self) -> list[tuple[int, str, str]]:
        """Precomputed ``(index, control_name, axis)`` for every body raw control.

        Body raw controls are all quaternion channels driven by bone rotations. Their
        names/axes come from the DNA and never change, so we parse them once instead of
        calling ``getRawControlName`` and splitting strings for every control every frame.
        """
        plan = self.data.get(self.cache_key("body", "raw_plan"))
        if plan is not None:
            return plan

        plan = []
        if self.body_dna_reader:
            for index in range(self.body_dna_reader.getRawControlCount()):
                full_name = self.body_dna_reader.getRawControlName(index)
                control_name, axis = full_name.split(".")
                axis = axis.rsplit("q", -1)[-1].lower()
                plan.append((index, control_name, axis))

        self.data[self.cache_key("body", "raw_plan")] = plan
        return plan

    @property
    def body_bone_transform_plan(self) -> list[tuple[int, str, Vector, Euler, Vector, Matrix]]:
        """Precomputed per-joint transform plan for the body rig (written joints only).

        Each entry is ``(joint_index, bone_name, rest_location, rest_rotation, rest_scale,
        rest_to_parent_inverse)``. Only bones updated via RBFs, twists, or swings are included, so
        the per-frame ``driven + swing + twist`` list concatenation and membership test (run for
        all 342 joints every frame) is collapsed into a single precomputed list, and the
        rest-to-parent inverse is precomputed once instead of every evaluation.
        """
        plan = self.data.get(self.cache_key("body", "bone_transform_plan"))
        if plan is not None:
            return plan

        # don't cache an empty plan until the rig and rest pose are available
        if not (self.body_rig and self.body_dna_reader and self.body_rest_pose):
            return []

        plan = []
        rest_pose = self.body_rest_pose
        # bones that are updated via RBFs, twists, or swings
        updatable_bone_names = (
            frozenset(self.body_driven_bone_names)
            | frozenset(self.body_swing_bone_names)
            | frozenset(self.body_twist_bone_names)
        )
        pose_bones = self.body_rig.pose.bones
        for joint_index in range(self.body_dna_reader.getJointCount()):
            # skip the root joint
            if joint_index == 0:
                continue
            name = self.body_dna_reader.getJointName(joint_index)
            if name not in updatable_bone_names:
                continue
            pose_bone = pose_bones.get(name)
            if not pose_bone:
                logger.warning(
                    f'The bone "{name}" was not found on "{self.body_rig.name}". Rig Logic will not update the bone.'
                )
                continue
            rest_transformations = rest_pose.get(name)
            if rest_transformations is None:
                continue
            rest_location, rest_rotation, rest_scale, rest_to_parent_matrix = rest_transformations
            plan.append(
                (
                    joint_index,
                    name,
                    rest_location,
                    rest_rotation,
                    rest_scale,
                    rest_to_parent_matrix.inverted_safe(),
                )
            )

        self.data[self.cache_key("body", "bone_transform_plan")] = plan
        return plan

    def _apply_head_rotation_modes(self):
        # Only assigned when it differs: Blender rejects the write attempt itself in a locked
        # context, so a rig that is already correct must not attempt one at all.
        if self.head_rig and self.head_rig.pose:
            for pose_bone in self.head_rig.pose.bones:
                mode = "XYZ" if pose_bone.name.startswith("FACIAL_") else "QUATERNION"
                if pose_bone.rotation_mode != mode:
                    pose_bone.rotation_mode = mode

    def head_initialize(self, update_raw_control_list: bool = True):
        from .bindings import riglogic  # pyright: ignore[reportAttributeAccessIssue]
        from .dna_io import get_dna_reader

        if not self.head_valid:
            return

        # Done before destroy_head so a context that rejects ID writes leaves the previous state
        # intact and this simply runs again on the next evaluation.
        if not apply_id_writes(f"head rotation modes for '{self.name}'", self._apply_head_rotation_modes):
            return

        # Release any previous head state first: re-initializing without this leaks the old
        # RigLogic/reader and leaves the derived caches pointing at the previous DNA.
        self.destroy_head()

        # ---- Initialize the Head Rig Instance ---
        # set the dna reader
        dna_reader = get_dna_reader(
            file_path=Path(bpy.path.abspath(self.head_dna_file_path)).absolute(), memory_resource=None
        )
        if not dna_reader:
            logger.warning(f"Failed to read the head DNA for Rig Instance {self.name}.")
            return
        self.data[self.cache_key("head", "dna_reader")] = dna_reader

        # set the rig logic manager and instance
        self.data[self.cache_key("head", "manager")] = riglogic.RigLogic(
            self.head_dna_reader, riglogic.Configuration(), None
        )
        self.data[self.cache_key("head", "instance")] = riglogic.RigInstance(rigLogic=self.head_manager, memRes=None)

        # populate the body rbf solver list
        if update_raw_control_list:
            self.update_head_raw_control_list()

        # calling theses properties will cache their values
        self.head_texture_masks_node  # noqa: B018
        self.head_mesh_index_lookup  # noqa: B018
        self.head_channel_name_to_index_lookup  # noqa: B018
        self.head_channel_index_to_mesh_index_lookup  # noqa: B018
        self.head_shape_key_blocks  # noqa: B018
        self.head_shape_key_apply_plan  # noqa: B018
        # Mirror the cached blocks into the scene-side UI list now (write-safe context).
        self.sync_shape_key_list()
        self.head_driven_bone_names  # noqa: B018
        self.head_driver_bone_names  # noqa: B018
        self.head_rest_pose  # noqa: B018
        # precompute the per-frame evaluation plans (index/name/axis parsed once)
        self.head_gui_control_plan  # noqa: B018
        self.head_raw_quat_plan  # noqa: B018
        self.head_animated_map_plan  # noqa: B018
        self.head_bone_transform_plan  # noqa: B018

        self.data[self.cache_key("head", "initialized")] = True

    def body_initialize(self, update_rbf_solver_list: bool = True):
        from .bindings import riglogic  # pyright: ignore[reportAttributeAccessIssue]
        from .dna_io import get_dna_reader

        if not self.body_valid:
            return

        # Release any previous body state first: re-initializing without this leaks the old
        # RigLogic/reader and leaves the derived caches pointing at the previous DNA.
        self.destroy_body()

        # ---- Initialize the Body Rig Instance ---
        # set the body dna reader
        dna_reader = get_dna_reader(
            file_path=Path(bpy.path.abspath(self.body_dna_file_path)).absolute(), memory_resource=None
        )
        if not dna_reader:
            logger.warning(f"Failed to read the body DNA for Rig Instance {self.name}.")
            return
        self.data[self.cache_key("body", "dna_reader")] = dna_reader

        # make sure the body bones are using the correct rotation mode
        if self.body_rig and self.body_rig.pose:
            for pose_bone in self.body_rig.pose.bones:
                if pose_bone.name in self.body_driver_bone_names:
                    pose_bone.rotation_mode = "QUATERNION"
                else:
                    pose_bone.rotation_mode = "XYZ"

        # set the rig logic manager and instance
        body_config = riglogic.Configuration()
        body_config.calculationType = riglogic.CalculationType_AnyVector
        body_config.loadJoints = True
        body_config.loadBlendShapes = True
        body_config.loadAnimatedMaps = True
        body_config.loadMachineLearnedBehavior = True
        body_config.loadRBFBehavior = True
        body_config.loadTwistSwingBehavior = True
        body_config.translationType = riglogic.TranslationType_Vector
        body_config.rotationType = riglogic.RotationType_Quaternions
        body_config.scaleType = riglogic.ScaleType_Vector
        self.data[self.cache_key("body", "manager")] = riglogic.RigLogic(self.body_dna_reader, body_config, None)
        self.data[self.cache_key("body", "instance")] = riglogic.RigInstance(rigLogic=self.body_manager, memRes=None)

        # populate the body rbf solver list
        if update_rbf_solver_list:
            self.update_body_rbf_solver_list()

        # calling theses properties will cache their values
        self.body_rest_pose  # noqa: B018
        self.body_twist_bone_names  # noqa: B018
        self.body_swing_bone_names  # noqa: B018
        self.body_driven_bone_names  # noqa: B018
        self.body_driver_bone_names  # noqa: B018
        # precompute the per-frame evaluation plan (index/name/axis parsed once)
        self.body_raw_plan  # noqa: B018
        self.body_bone_transform_plan  # noqa: B018

        self.data[self.cache_key("body", "initialized")] = True

    def initialize(self):
        self.head_initialize()
        self.body_initialize()
        if self.is_pro:
            from .editors.backup_manager.core import sync_backup_list_with_disk as _sync_backup_list_with_disk

            _sync_backup_list_with_disk(instance=self)  # pyright: ignore[reportArgumentType]

    def _release_rig_logic(self, component: str):
        """Free a component's RigLogic handles in the order OpenRigLogic requires.

        A RigInstance holds a raw pointer to its RigLogic, so it must be destroyed first;
        the DNA reader (which owns its file stream) can go last. Relying on Python
        garbage collection here would leave the destruction order undefined.
        """
        from .dna_io import release_dna_handle
        from .runtime.controller import release

        release(self)

        for descriptor in ("instance", "manager", "dna_reader"):
            release_dna_handle(self.data.get(self.cache_key(component, descriptor)))

    def destroy_head(self):
        self._release_rig_logic("head")
        # clear the head rig logic data, this frees them up to be garbage collected
        for key in list(self.data.keys()):
            if key.startswith(f"{self.name}_head_"):
                del self.data[key]
        self.data[self.cache_key("head", "initialized")] = False

    def destroy_body(self):
        self._release_rig_logic("body")
        # clear the body rig logic data, this frees them up to be garbage collected
        for key in list(self.data.keys()):
            if key.startswith(f"{self.name}_body_"):
                del self.data[key]
        self.data[self.cache_key("body", "initialized")] = False

    def destroy_references(self):
        # The `data` cache dict survives an undo/redo, but the live `bpy` RNA wrappers it
        # holds do not: undo can free and reallocate the underlying objects, leaving these
        # entries pointing at removed StructRNA. Any later access then raises
        # `ReferenceError: StructRNA of type Object has been removed`. Drop only the
        # wrapper-holding caches so they lazily rebuild from fresh wrappers on next access.
        # The RigLogic C++ instances and the plain value-copy caches (rest pose, bone-name
        # lists, channel lookups) are undo-safe, so the component stays initialized.
        reference_descriptors = (
            ("head", "mesh_index_lookup"),
            ("head", "shape_key"),
            ("head", "shape_key_blocks"),
            ("head", "shape_key_apply_plan"),
            ("head", "shape_key_has_deltas"),
            ("head", "body_constraints"),
            ("head", "rig_evaluated"),
            ("head", "face_board_evaluated"),
            ("body", "rig_evaluated"),
        )
        for component, descriptor in reference_descriptors:
            self.data.pop(self.cache_key(component, descriptor), None)

    def clear_evaluated_references(self):
        """Drop the cached evaluated objects without touching the rest of the cache.

        A render caches wrappers into the render dependency graph, which Blender frees once
        the render finishes. Clearing them makes the next read resolve against a live graph.
        """
        for component, descriptor in (
            ("head", "rig_evaluated"),
            ("head", "face_board_evaluated"),
            ("body", "rig_evaluated"),
        ):
            self.data.pop(self.cache_key(component, descriptor), None)

    def destroy(self):
        self.destroy_head()
        self.destroy_body()

    def update_head_switch_values(self):
        if not self.face_board:
            return

        # Switch values are read from the evaluated face board (see `face_board_evaluated`),
        # but the constraints and visibility they drive are written to the original datablock.
        evaluated_face_board = self.face_board_evaluated
        evaluated_pose_bones = (
            evaluated_face_board.pose.bones if evaluated_face_board and evaluated_face_board.pose else None
        )
        if evaluated_pose_bones is None:
            return

        # update the head follow body switch constraint influence
        face_gui_control = self.face_board.pose.bones.get("CTRL_faceGUI")
        face_follow_head_switch = evaluated_pose_bones.get("CTRL_faceGUIfollowHead")
        if face_follow_head_switch and face_gui_control:
            constraint = None
            for existing_constraint in face_gui_control.constraints:
                if existing_constraint.type == "CHILD_OF":
                    constraint = existing_constraint
                    break
            if constraint and round(constraint.influence, 3) != round(face_follow_head_switch.location.y, 3):
                constraint.influence = face_follow_head_switch.location.y

        # update the eye aim follow head switch constraint influence
        eye_aim_control = self.face_board.pose.bones.get("CTRL_C_eyesAim")
        eye_aim_follow_head_switch = evaluated_pose_bones.get("CTRL_eyesAimFollowHead")
        if eye_aim_follow_head_switch and eye_aim_control:
            constraint = None
            for existing_constraint in eye_aim_control.constraints:
                if existing_constraint.type == "CHILD_OF":
                    constraint = existing_constraint
                    break
            if constraint and round(constraint.influence, 3) != round(eye_aim_follow_head_switch.location.y, 3):
                constraint.influence = eye_aim_follow_head_switch.location.y

        # update the eye aim control visibility, but only when the eye-aim mode actually changes.
        # The visibility sweep walks children_recursive, so running it on every evaluation is pure
        # overhead once the hide states already match the current mode.
        # Note: In Blender 5.0+, the hide property moved from Bone to PoseBone
        if eye_aim_control:
            use_eye_aim = self.head_use_eye_aim
            if self.data.get(self.cache_key("head", "eye_aim_visibility")) != use_eye_aim:
                hidden = not use_eye_aim

                def apply_eye_aim_visibility():
                    for pose_bone in [eye_aim_control, *eye_aim_control.children_recursive]:
                        if pose_bone != eye_aim_control and pose_bone.name.startswith(("GRP_", "LOC_")):
                            continue
                        target = pose_bone if IS_BLENDER_5 else pose_bone.bone
                        if target.hide != hidden:
                            target.hide = hidden

                # Only cache once the writes land, otherwise a locked context is never retried.
                if apply_id_writes(f"eye aim visibility for '{self.name}'", apply_eye_aim_visibility):
                    self.data[self.cache_key("head", "eye_aim_visibility")] = use_eye_aim

    def get_head_gui_control_values_from_eye_aim(
        self, dependency_graph: bpy.types.Depsgraph | None = None
    ) -> dict[str, dict[str, float]]:
        values = {}
        if not self.face_board or not self.head_rig:
            return values

        # Read the *evaluated* objects so the posed bone matrices reflect the current head-bone
        # rotation and the eye aim follow-head constraint. The caller's graph is used when given
        # (during a render that is the only graph holding this frame's pose); otherwise this is
        # taken fresh from the current dependency graph so it is also correct per-frame during
        # baking, where the scene frame is stepped just before this is called.
        if dependency_graph is None:
            dependency_graph = bpy.context.evaluated_depsgraph_get()
        head_rig = self.head_rig.evaluated_get(dependency_graph)
        face_board = self.face_board.evaluated_get(dependency_graph)
        if not head_rig.pose or not face_board.pose:
            return values

        for target_name, eye_bone_name, control_name in [
            ("CTRL_L_eyeAim", "FACIAL_L_Eye", "CTRL_L_eye"),
            ("CTRL_R_eyeAim", "FACIAL_R_Eye", "CTRL_R_eye"),
        ]:
            target = face_board.pose.bones.get(target_name)
            eye = head_rig.pose.bones.get(eye_bone_name)
            if not (target and eye):
                continue

            # The eye control values are expressed relative to the eye's neutral (control = 0)
            # orientation. That neutral orientation is posed by the eye's parent chain, so it
            # rotates with the head. Building the reference frame from the parent's *current* pose
            # matrix (instead of the static rest matrix) is what lets the eyes keep aiming at a
            # world-fixed target when the head bone is turned (the eyes-follow-head-off case).
            eye_parent = eye.parent
            if eye_parent:
                rest_relative_to_parent = eye_parent.bone.matrix_local.inverted_safe() @ eye.bone.matrix_local
                eye_reference_matrix = head_rig.matrix_world @ eye_parent.matrix @ rest_relative_to_parent
            else:
                eye_reference_matrix = head_rig.matrix_world @ eye.bone.matrix_local

            # The eye and the aim target live on different objects, so map each into true world
            # space with its own object matrix (head rig for the eye, face board for the target).
            eye_pos = head_rig.matrix_world @ eye.head
            target_pos = face_board.matrix_world @ target.head
            look_direction = target_pos - eye_pos

            if look_direction.length < FLOATING_POINT_PRECISION:
                continue

            look_direction.normalize()

            # Convert the world look direction into the eye's reference (local) space.
            local_look_direction = (eye_reference_matrix.to_3x3().inverted_safe() @ look_direction).normalized()

            # Calculate horizontal distance (projection onto XZ plane, forward is local -Z)
            horizontal_dist = math.sqrt(local_look_direction.x**2 + local_look_direction.z**2)

            if horizontal_dist > FLOATING_POINT_PRECISION:
                # Remap yaw to continuous range centered on forward direction (-Z)
                # Instead of atan2(x, -z), we use the normalized x component directly
                # This gives us a smooth -1 to 1 range for horizontal movement
                x_normalized = local_look_direction.x / horizontal_dist

                # For better control, we can use asin which gives -90° to 90° range
                yaw = math.asin(max(-1.0, min(1.0, x_normalized)))
            else:
                # Looking straight up/down, yaw is undefined
                yaw = 0.0

            # Pitch is the angle from the horizontal plane
            pitch = math.atan2(local_look_direction.y, horizontal_dist)

            # Map angles to -1..1 range based on max rotation
            x_max_rad = math.radians(60.0)
            y_max_rad = math.radians(30.0)

            x_control = max(-1.0, min(1.0, yaw / x_max_rad))
            y_control = max(-1.0, min(1.0, pitch / y_max_rad))

            values[control_name] = {"x": x_control, "y": y_control}

        return values

    def update_head_raw_control_values(self, override_values: dict[str, dict[str, float]] | None = None):
        # skip if the body rig is not set
        if not self.head_rig or not self.head_rig_evaluated or not self.head_dna_reader:
            return

        # skip if the rest pose is not initialized
        if not self.head_rest_pose:
            return

        if not self.head_rig_evaluated.pose:
            return

        missing_raw_controls = []
        converted_quaternions = {}

        # convert the quaternion values to the correct coordinate system
        driver_bone_names = frozenset(self.head_driver_bone_names)
        for pose_bone in self.head_rig_evaluated.pose.bones:
            if pose_bone.name in driver_bone_names:
                # get the local quaternion, but from the world matrix to account for constraints, since we
                # can't always assume the local quaternion value is what is driving the bone rotation. For
                # example, if the body is driving the head bone transforms via constraints.
                # TODO: This math might have performance implications, so we might want review this later.
                quaternion = utilities.get_pose_bone_local_quaternion(pose_bone)
                converted_quaternions[pose_bone.name] = quaternion

        head_instance = self.head_instance
        for index, control_name, axis in self.head_raw_quat_plan:
            # override the values can be provided to update values based on them vs current head rig bone locations
            # This can be used for baking the values to an action
            if override_values:
                value = override_values.get(control_name, {}).get(axis)
                if value is not None:
                    head_instance.setRawControl(index, value)
            else:
                quaternion = converted_quaternions.get(control_name)
                if quaternion:
                    value = getattr(quaternion, axis)
                    head_instance.setRawControl(index, value)
                else:
                    missing_raw_controls.append(control_name)

        if missing_raw_controls and not self.data.get(self.cache_key("head", "logged_missing_raw_controls")):
            logger.warning(
                f'The following raw controls are missing on "{self.head_rig.name}":\n{pformat(missing_raw_controls)}.'
            )
            logger.warning(f"You are not listening to {len(missing_raw_controls)} raw controls")
            logger.warning(
                f"This is most likely due to the these bones being missing from the rig {self.head_rig.name}."
            )
            self.data[self.cache_key("head", "logged_missing_raw_controls")] = True

    def _resolve_eye_control_value(
        self,
        value: float | None,
        control_name: str,
        axis: str,
        eye_aim_override_values: dict[str, dict[str, float]],
        center_value: float | None,
    ) -> float | None:
        """Apply the eye aim / master center eye override to an individual L/R eye control value.

        The eye aim (when active) takes priority; otherwise the master ``CTRL_C_eye`` value
        (``center_value``, read from either the live face board bone or the baked override
        values) overrides the individual ``CTRL_L_eye`` / ``CTRL_R_eye`` value. Any control that
        is not an eye control is returned unchanged. This is shared by the live and bake paths.
        """
        if control_name not in ("CTRL_L_eye", "CTRL_R_eye"):
            return value

        eye_aim_value = eye_aim_override_values.get(control_name, {}).get(axis)
        if eye_aim_value is not None:
            if abs(eye_aim_value) > FLOATING_POINT_PRECISION:
                return eye_aim_value
        elif center_value is not None and abs(center_value) > FLOATING_POINT_PRECISION:
            return center_value

        return value

    def update_head_gui_control_values(
        self,
        override_values: dict[str, dict[str, float]] | None = None,
        dependency_graph: bpy.types.Depsgraph | None = None,
    ):
        # The face board only supplies the GUI control positions. Without one the head still has
        # to solve its raw controls below, so a missing face board skips the loop, not the whole
        # evaluation.
        if not self.head_dna_reader:
            return

        missing_gui_controls = []

        # Control positions are inputs, so they come from the evaluated face board rather than
        # the original datablock, which is stale while rendering (see `face_board_evaluated`).
        evaluated_face_board = self.face_board_evaluated
        face_pose_bones = (
            evaluated_face_board.pose.bones if evaluated_face_board and evaluated_face_board.pose else None
        )
        center_eye_control = face_pose_bones.get("CTRL_C_eye") if face_pose_bones else None

        eye_aim_override_values = {}
        if self.head_use_eye_aim:
            eye_aim_override_values = self.get_head_gui_control_values_from_eye_aim(dependency_graph)

        head_instance = self.head_instance
        for index, control_name, axis in self.head_gui_control_plan:
            # Override values can be provided to update values based on them vs current face board
            # bone locations. This can be used for baking the values to an action.
            if override_values:
                value = override_values.get(control_name, {}).get(axis)
                # Mirror the live path's eye handling so baking respects the eye aim and the
                # master center eye control (read from the baked CTRL_C_eye override values).
                value = self._resolve_eye_control_value(
                    value, control_name, axis, eye_aim_override_values, override_values.get("CTRL_C_eye", {}).get(axis)
                )
                if value is not None:
                    head_instance.setGUIControl(index, value)
            elif face_pose_bones is not None:
                pose_bone = face_pose_bones.get(control_name)
                if pose_bone:
                    value = getattr(pose_bone.location, axis)
                    # special case for the eye controls: the eye aim and the master center eye
                    # control override the individual L/R eye controls.
                    center_value = getattr(center_eye_control.location, axis) if center_eye_control else None
                    value = self._resolve_eye_control_value(
                        value, control_name, axis, eye_aim_override_values, center_value
                    )
                    head_instance.setGUIControl(index, value)
                else:
                    missing_gui_controls.append(control_name)

        if missing_gui_controls and not self.data.get(self.cache_key("head", "logged_missing_gui_controls")):
            logger.warning(
                f'The following GUI controls are missing on "{self.face_board.name}":\n{pformat(missing_gui_controls)}.'
            )
            logger.warning(f"You are not listening to {len(missing_gui_controls)} GUI controls")
            logger.warning(
                "This is most likely due to the DNA file being an older version then what "
                "the face board currently supports."
            )
            logger.warning(
                "Using a new .dna file created from the latest version of MetaHuman Creator will probably resolve this."
            )
            self.data[self.cache_key("head", "logged_missing_gui_controls")] = True

        # set the active LOD level for the head instance to optimize performance
        self.head_instance.setLOD(level=int(self.view_options.active_lod[-1]))  # pyright: ignore[reportAttributeAccessIssue]
        # map the GUI changes to the raw controls
        self.head_manager.mapGUIToRawControls(self.head_instance)

        if self.evaluate_rbfs:
            self.update_head_raw_control_values()

        # calculate the controls
        self.head_manager.calculate(self.head_instance)

    def apply_gui_controls_to_face_board(self):
        if not self.face_board or not self.head_dna_reader or not self.head_instance:
            return

        for index in range(self.head_dna_reader.getGUIControlCount()):
            full_name = self.head_dna_reader.getGUIControlName(index)
            control_name, axis = full_name.split(".")
            axis = axis.rsplit("t", -1)[-1].lower()
            pose_bone = self.face_board.pose.bones.get(control_name)
            if pose_bone:
                setattr(pose_bone.location, axis, self.head_instance.getGUIControl(index))

    def solo_head_shape_key_value(self, shape_key: bpy.types.ShapeKey):
        # skip if the head mesh is not set
        if not self.head_mesh or not self.head_dna_reader:
            return

        # skip if there are no shape keys
        if len(bpy.data.shape_keys) == 0:
            return

        # make all other shape keys 0.0
        for index, _ in enumerate(self.head_instance.getBlendShapeOutputs()):
            for _shape_key in self.head_shape_key_blocks.get(index, []):
                if _shape_key and _shape_key != shape_key:
                    _shape_key.value = 0.0

        # set the provided shape key value to 1.0
        shape_key.value = 1.0

    def update_head_shape_keys(self, collect_values: bool = False) -> list[tuple[bpy.types.ShapeKey, float]]:
        """Push the RigLogic blend-shape outputs onto the head shape-key blocks.

        Values are written per mesh with a single ``foreach_get`` / scatter /
        ``foreach_set`` instead of one RNA assignment per block, which avoids firing a
        per-property update for every one of the hundreds of driven blocks. ``foreach_set``
        does not tag the data for refresh, so each touched shape-key datablock is tagged
        once afterwards.

        When ``collect_values`` is ``True`` (animation baking) the driven
        ``(shape_key, value)`` pairs are returned for keyframing; the real-time path leaves
        it ``False`` and skips building that list.
        """
        # skip if the head mesh is not set
        if not self.head_mesh or not self.head_dna_reader:
            return []

        # skip if there are no shape keys
        if len(bpy.data.shape_keys) == 0:
            return []

        outputs = np.asarray(self.head_instance.getBlendShapeOutputs(), dtype=np.float32)
        output_count = outputs.shape[0]

        shape_key_values: list[tuple[bpy.types.ShapeKey, float]] = []
        for key_blocks, positions, channels, blocks, buffer in self.head_shape_key_apply_plan:
            # A lower LOD shrinks getBlendShapeOutputs(); drop channels beyond the active
            # range so the gather never indexes past the array end (higher channels stay
            # untouched, matching the original enumerate() that stopped at the array end).
            mask = channels < output_count
            if mask.all():
                driven_positions, driven_channels, driven_blocks = positions, channels, blocks
            else:
                driven_positions = positions[mask]
                driven_channels = channels[mask]
                driven_blocks = [block for block, keep in zip(blocks, mask, strict=True) if keep]

            try:
                key_blocks.foreach_get("value", buffer)
                buffer[driven_positions] = outputs[driven_channels]
                key_blocks.foreach_set("value", buffer)
            except (AttributeError, RuntimeError, ReferenceError) as error:
                logger.error(f'Failed to update the shape keys on "{self.head_mesh.name}": {error}')
                return []

            # foreach_set bypasses the per-property update, so tag the shape-key datablock
            # for the dependency graph to re-evaluate the deformed mesh.
            if key_blocks.id_data:
                key_blocks.id_data.update_tag()

            if collect_values:
                driven_values = outputs[driven_channels]
                shape_key_values.extend(
                    (block, float(value)) for block, value in zip(driven_blocks, driven_values, strict=True)
                )

        return shape_key_values

    def zero_head_shape_keys(self) -> None:
        """Set every RigLogic-driven head shape-key block value to 0.0 so the
        LOD0 head meshes show the basis (bone-deformed) shape with no
        blend-shape contribution.
        """
        # skip if the head mesh is not set
        if not self.head_mesh or not self.head_dna_reader:
            return

        # skip if there are no shape keys
        if len(bpy.data.shape_keys) == 0:
            return

        for key_blocks, positions, _channels, _blocks, buffer in self.head_shape_key_apply_plan:
            try:
                # Read current values, zero only the RigLogic-driven blocks
                # (leaving any non-driven blocks untouched), write back.
                key_blocks.foreach_get("value", buffer)
                buffer[positions] = 0.0
                key_blocks.foreach_set("value", buffer)
            except (AttributeError, RuntimeError, ReferenceError) as error:
                logger.error(f'Failed to zero the shape keys on "{self.head_mesh.name}": {error}')
                continue

            # foreach_set bypasses the per-property update, so tag the shape-key
            # datablock for the dependency graph to re-evaluate the deformed mesh.
            if key_blocks.id_data:
                key_blocks.id_data.update_tag()

    def update_head_texture_masks(self) -> list[tuple[str, float]]:
        # skip if the material is not set
        if not self.head_material or not self.head_dna_reader:
            return []

        head_texture_masks_node = self.head_texture_masks_node
        # if the texture masks node is not set, we can't update the texture masks
        if not head_texture_masks_node:
            logger.warning(f'The texture masks node was not found on the material "{self.head_material.name}"')
            return []

        texture_mask_values = []

        # update texture masks values
        node_inputs = head_texture_masks_node.inputs
        animated_map_outputs = self.head_instance.getAnimatedMapOutputs()
        for index, slider_name in self.head_animated_map_plan:
            value = animated_map_outputs[index]
            mask_slider = node_inputs.get(slider_name)
            if mask_slider:
                try:
                    mask_slider.default_value = value  # type: ignore[attr-defined]
                except AttributeError as error:
                    logger.error(
                        f'Failed to update the texture mask slider "{slider_name}" on '
                        f'"{self.head_material.name}": {error}'
                    )
                    return []
                texture_mask_values.append((slider_name, value))
            else:
                logger.warning(
                    f'The texture mask slider "{slider_name}" was not found on the material "{self.head_material.name}"'
                )

        return texture_mask_values

    def update_head_bone_transforms(self, collect_transforms: bool = False) -> list[tuple[str, Vector, Euler, Vector]]:
        """Update head bone transforms from RigLogic joint outputs.

        Args:
            collect_transforms: When True, build and return the decomposed
                (bone_name, location, rotation_euler, scale) tuples used by action baking. The
                interactive evaluation path leaves this False to skip the per-bone matrix
                decomposition, which is otherwise pure overhead.

        Returns:
            A list of (bone_name, location, rotation_euler, scale) tuples for each updated bone,
            or an empty list when ``collect_transforms`` is False.
        """
        # skip if the head rig is not set
        if not self.head_rig or not self.head_dna_reader:
            return []

        # skip if the rest pose is not initialized
        # https://github.com/poly-hammer/character-dna-addon/issues/58
        if not self.head_rest_pose:
            return []

        bone_transforms: list[tuple[str, Vector, Euler, Vector]] = []

        raw_joint_output = self.head_instance.getJointOutputs()
        pose_bones = self.head_rig.pose.bones
        # update joint transforms using the precomputed plan (index/name/rest/inverse parsed once)
        for (
            index,
            name,
            rest_location,
            rest_rotation,
            rest_scale,
            rest_to_parent_inverse,
            has_children,
        ) in self.head_bone_transform_plan:
            pose_bone = pose_bones.get(name)
            if not pose_bone:
                continue

            # get the values
            matrix_index = (index + 1) * 9
            values = raw_joint_output[(index * 9) : matrix_index]

            # extract the delta values
            location_delta = Vector([values[0] / SCALE_FACTOR, values[1] / SCALE_FACTOR, values[2] / SCALE_FACTOR])
            rotation_delta = Euler([math.radians(values[3]), math.radians(values[4]), math.radians(values[5])])
            scale_delta = Vector(values[6:9])

            # update the transformations using the rest pose and the delta values
            # we need to copy the vectors so we don't modify the original rest pose
            location = Vector(
                [
                    rest_location.x + location_delta.x,
                    rest_location.y + location_delta.y,
                    rest_location.z + location_delta.z,
                ]
            )
            rotation = Euler(
                [
                    rest_rotation.x + rotation_delta.x,
                    rest_rotation.y + rotation_delta.y,
                    rest_rotation.z + rotation_delta.z,
                ],
                "XYZ",
            )
            scale = Vector([rest_scale.x + scale_delta.x, rest_scale.y + scale_delta.y, rest_scale.z + scale_delta.z])

            # update the bone matrix (rest-to-parent inverse is precomputed in the plan)
            modified_matrix = Matrix.LocRotScale(location[:], rotation, scale[:])
            try:
                pose_bone.matrix_basis = rest_to_parent_inverse @ modified_matrix
            except AttributeError as error:
                logger.error(f'Failed to update the bone "{name}" on "{self.head_rig.name}": {error}')
                continue

            # if the bone is not a leaf bone, we need to update the rotation again
            if has_children:
                pose_bone.rotation_euler = rotation_delta

            if collect_transforms:
                # for non-leaf bones use the rotation_delta as the final euler, for leaf bones decompose
                # from the matrix_basis
                final_rotation = rotation_delta if has_children else pose_bone.matrix_basis.to_euler("XYZ")
                final_location = pose_bone.matrix_basis.to_translation()
                final_scale = pose_bone.matrix_basis.to_scale()
                bone_transforms.append((name, final_location, final_rotation, final_scale))

        return bone_transforms

    def reset_body_raw_control_values(self):
        from .runtime.engine import active as native_active

        if native_active(self):
            self.evaluate()
            return
        # skip if the body rig is not set
        if not self.body_initialized:
            self.body_initialize()

        if not self.body_dna_reader:
            logger.warning("The body DNA reader is not set. The body raw control values will not be reset.")
            return

        if not self.evaluate_rbfs:
            # reset all raw controls to 0.0
            for index in range(self.body_dna_reader.getRawControlCount()):
                full_name = self.body_dna_reader.getRawControlName(index)
                _, axis = full_name.split(".")
                axis = axis.rsplit("q", -1)[-1].lower()
                if axis == "w":
                    self.body_instance.setRawControl(index, 1.0)
                else:
                    self.body_instance.setRawControl(index, 0.0)

            self.body_instance.setLOD(level=int(self.view_options.active_lod[-1]))  # pyright: ignore[reportAttributeAccessIssue]
            self.body_manager.calculate(self.body_instance)
        else:
            self.update_body_raw_control_values()

        self.update_body_bone_transforms()

    def reset_head_raw_control_values(self):
        from .runtime.engine import active as native_active

        if native_active(self):
            self.evaluate()
            return
        # skip if the head rig is not set
        if not self.head_initialized:
            self.head_initialize()

        if not self.head_dna_reader:
            logger.warning("The head DNA reader is not set. The head raw control values will not be reset.")
            return

        if not self.evaluate_rbfs:
            # reset all raw controls to 0.0
            for index in range(self.head_dna_reader.getRawControlCount()):
                full_name = self.head_dna_reader.getRawControlName(index)
                control_name, axis = full_name.split(".")
                if control_name in self.head_driver_bone_names:
                    axis = axis.rsplit("q", -1)[-1].lower()
                    if axis == "w":
                        self.head_instance.setRawControl(index, 1.0)
                    else:
                        self.head_instance.setRawControl(index, 0.0)

            self.head_instance.setLOD(level=int(self.view_options.active_lod[-1]))  # pyright: ignore[reportAttributeAccessIssue]
            self.head_manager.calculate(self.head_instance)
        else:
            self.update_head_raw_control_values()
            self.head_instance.setLOD(level=int(self.view_options.active_lod[-1]))  # pyright: ignore[reportAttributeAccessIssue]
            self.head_manager.calculate(self.head_instance)

        self.update_head_bone_transforms()

    def update_body_raw_control_values(self, override_values: dict[str, dict[str, float]] | None = None):
        # skip if the body rig is not set
        if not self.body_rig or not self.body_rig_evaluated or not self.body_dna_reader:
            return

        # skip if the rest pose is not initialized
        if not self.body_rest_pose:
            return

        if not self.body_rig_evaluated.pose:
            return

        missing_raw_controls = []
        converted_quaternions = {}

        # convert the quaternion values to the correct coordinate system
        driver_bone_names = frozenset(self.body_driver_bone_names)
        for pose_bone in self.body_rig_evaluated.pose.bones:
            if pose_bone.name in driver_bone_names:
                # get the local quaternion, but from the world matrix to account for constraints, since we
                # can't always assume the local quaternion value is what is driving the bone rotation. For
                # example, a control rig might be driving the body bone rotation via constraints.
                # TODO: This math might have performance implications, so we might want review this later.
                quaternion = utilities.get_pose_bone_local_quaternion(pose_bone)
                converted_quaternions[pose_bone.name] = quaternion

        body_instance = self.body_instance
        for index, control_name, axis in self.body_raw_plan:
            # override the values can be provided to update values based on them vs current body rig bone locations
            # This can be used for baking the values to an action
            if override_values:
                value = override_values.get(control_name, {}).get(axis)
                if value is not None:
                    body_instance.setRawControl(index, value)
            else:
                quaternion = converted_quaternions.get(control_name)
                if quaternion:
                    value = getattr(quaternion, axis)
                    body_instance.setRawControl(index, value)
                else:
                    missing_raw_controls.append(control_name)

        if missing_raw_controls and not self.data.get(self.cache_key("body", "logged_missing_raw_controls")):
            logger.warning(
                f'The following raw controls are missing on "{self.body_rig.name}":\n{pformat(missing_raw_controls)}.'
            )
            logger.warning(f"You are not listening to {len(missing_raw_controls)} raw controls")
            logger.warning(
                f"This is most likely due to the these bones being missing from the rig {self.body_rig.name}."
            )
            self.data[self.cache_key("body", "logged_missing_raw_controls")] = True

        # set the active LOD level for the body instance to optimize performance
        self.body_instance.setLOD(level=int(self.view_options.active_lod[-1]))  # pyright: ignore[reportAttributeAccessIssue]

        # calculate the changes
        self.body_manager.calculate(self.body_instance)

    def update_body_bone_transforms(self, collect_transforms: bool = False) -> list[tuple[str, Vector, Euler, Vector]]:
        """Update body bone transforms from RigLogic joint outputs.

        Args:
            collect_transforms: When True, build and return the decomposed
                (bone_name, location, rotation_euler, scale) tuples used by action baking. The
                interactive evaluation path leaves this False to skip the per-bone matrix
                decomposition, which is otherwise pure overhead.

        Returns:
            A list of (bone_name, location, rotation_euler, scale) tuples for each updated bone,
            or an empty list when ``collect_transforms`` is False.
        """
        # skip if the body rig is not set
        if not self.body_rig or not self.body_dna_reader:
            return []

        # skip if the rest pose is not initialized
        if not self.body_rest_pose:
            return []

        bone_transforms: list[tuple[str, Vector, Euler, Vector]] = []

        # get the delta values
        D = self.body_instance.getJointOutputs()
        pose_bones = self.body_rig.pose.bones

        # update joint transforms using the precomputed plan (only RBF/twist/swing-updated bones)
        for (
            joint_index,
            name,
            rest_location,
            rest_rotation,
            rest_scale,
            rest_to_parent_inverse,
        ) in self.body_bone_transform_plan:
            pose_bone = pose_bones.get(name)
            if not pose_bone:
                continue

            # get the values
            attr_index = joint_index * ATTR_COUNT_PER_QUATERNION_JOINT
            # extract the delta values
            location_delta = Vector(
                [D[attr_index] / SCALE_FACTOR, D[attr_index + 1] / SCALE_FACTOR, D[attr_index + 2] / SCALE_FACTOR]
            )
            rotation_delta = Quaternion([D[attr_index + 6], D[attr_index + 3], D[attr_index + 4], D[attr_index + 5]])
            scale_delta = Vector([D[attr_index + 7], D[attr_index + 8], D[attr_index + 9]])

            # update the transformations using the rest pose and the delta values
            # we need to copy the vectors so we don't modify the original rest pose
            location = Vector(
                [
                    rest_location.x + location_delta.x,
                    rest_location.y + location_delta.y,
                    rest_location.z + location_delta.z,
                ]
            )

            rotation = rest_rotation.to_quaternion() @ rotation_delta

            scale = Vector([rest_scale.x + scale_delta.x, rest_scale.y + scale_delta.y, rest_scale.z + scale_delta.z])

            # update the bone matrix (rest-to-parent inverse is precomputed in the plan)
            modified_matrix = Matrix.LocRotScale(location[:], rotation, scale[:])
            try:
                pose_bone.matrix_basis = rest_to_parent_inverse @ modified_matrix
            except AttributeError as error:
                logger.error(f'Failed to update the bone "{name}" on "{self.body_rig.name}": {error}')
                continue

            if collect_transforms:
                # decompose the final matrix_basis for baking output
                final_location = pose_bone.matrix_basis.to_translation()
                final_rotation = pose_bone.matrix_basis.to_euler("XYZ")
                final_scale = pose_bone.matrix_basis.to_scale()
                bone_transforms.append((name, final_location, final_rotation, final_scale))

        return bone_transforms

    def update_body_rbf_solver_list(self):
        try:
            from .editors.rbf_editor.callbacks import update_body_rbf_solver_list as _update

            _update(self)  # pyright: ignore[reportArgumentType]
        except ImportError:
            logger.debug("Could not import the RBF editor module to update the body RBF solver list.")

    def update_head_raw_control_list(self):
        try:
            from .editors.raw_control_editor.callbacks import update_head_raw_control_list as _update

            _update(self)  # pyright: ignore[reportArgumentType]
        except ImportError:
            logger.debug("Could not import the raw control editor module to update the head raw control list.")

    def _evaluate_on_main_thread(
        self, component: "ComponentType", dependency_graph: bpy.types.Depsgraph | None
    ) -> None:
        """Queue an evaluation for the main thread and block until it has run.

        Waiting is deliberate: the render job thread must not produce the frame until the pose
        bones and shape keys for it have been written.
        """
        global _logged_main_thread_timeout

        done = threading.Event()
        with _main_thread_lock:
            _main_thread_queue.append((self.name, component, dependency_graph, done))

        if not done.wait(MAIN_THREAD_TIMEOUT_SECONDS) and not _logged_main_thread_timeout:
            _logged_main_thread_timeout = True
            logger.error(
                f"Timed out waiting for the main thread to evaluate rig instance '{self.name}'. "
                "Bake the animation before rendering to avoid relying on live evaluation."
            )

    def evaluate(self, component: "ComponentType" = "all", dependency_graph: bpy.types.Depsgraph | None = None):
        from .runtime.engine import active as native_active

        if native_active(self):
            if dependency_graph is None and is_main_thread():
                bpy.context.view_layer.update()
            return

        # Only a real render (F12 / Render Animation) hands its handlers to a job thread that the
        # main thread is not servicing, so that is the only case worth blocking for. An OpenGL
        # playblast also runs off the main thread but drives the viewport from it, so waiting
        # there starves both threads; it never fires render_init, which is what is_rendering()
        # keys off. Viewport and timeline evaluation stay on the main thread and go straight through.
        if is_rendering() and not is_main_thread():
            self._evaluate_on_main_thread(component, dependency_graph)
            return

        window_manager_properties = utilities.get_addon_window_manager_properties()
        # this condition prevents constant evaluation
        if window_manager_properties.evaluate_dependency_graph:
            # turn off the dependency graph evaluation so we can update the controls without triggering an update
            window_manager_properties.evaluate_dependency_graph = False

            try:
                if not self.head_initialized:
                    self.head_initialize()

                if not self.body_initialized:
                    self.body_initialize()

                # apply the dependency graph update so we have the latest evaluated bone transforms
                self.apply_dependency_graph_update(dependency_graph)

                if component in ("body", "all") and self.body_initialized:
                    if self.evaluate_rbfs:
                        self.update_body_raw_control_values()

                    # apply the changes
                    if self.evaluate_bones:
                        self.update_body_bone_transforms()

                if component in ("head", "all") and self.head_initialized:
                    # update the gui controls
                    self.update_head_switch_values()
                    self.update_head_gui_control_values(dependency_graph=dependency_graph)

                    # apply the changes
                    if self.evaluate_bones:
                        self.update_head_bone_transforms()
                    if self.evaluate_shape_keys:
                        self.update_head_shape_keys()
                    if self.evaluate_texture_masks:
                        self.update_head_texture_masks()
            finally:
                # always restore the flag so evaluation isn't permanently disabled by an exception
                window_manager_properties.evaluate_dependency_graph = True
