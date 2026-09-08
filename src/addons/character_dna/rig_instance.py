# standard library imports
import logging  # noqa: I001
import threading

from collections.abc import Callable
from pathlib import Path

# third party imports
import bpy
import numpy as np

from mathutils import Euler, Matrix, Vector

# local imports
from . import utilities
from .constants import SHAPE_KEY_NAME_MAX_LENGTH
from .ui import callbacks
from .typing import *  # noqa: F403


ATTR_COUNT_PER_QUATERNION_JOINT = 10
ATTR_COUNT_PER_EULER_JOINT = 9

logger = logging.getLogger(__name__)

_MAIN_THREAD_IDENT = threading.main_thread().ident
_rendering = False


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
    """Retire render graph sessions without writing Blender data from a job thread."""
    global _rendering
    _rendering = False
    from .runtime.engine import clear_contexts

    clear_contexts()


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
        from .runtime.engine import synchronize_authoring

        return synchronize_authoring(self, "head", self.data.get(self.cache_key("head", "instance")))

    @property
    def head_dna_reader(self) -> "dna.BinaryStreamReader":
        return self.data.get(self.cache_key("head", "dna_reader"))  # pyright: ignore[reportReturnType]

    @property
    def body_manager(self) -> "riglogic.RigLogic":
        return self.data.get(self.cache_key("body", "manager"))  # pyright: ignore[reportReturnType]

    @property
    def body_instance(self) -> "riglogic.RigInstance":
        from .runtime.engine import synchronize_authoring

        return synchronize_authoring(self, "body", self.data.get(self.cache_key("body", "instance")))

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
            ("head", "runtime_plan"),
            ("body", "runtime_plan"),
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

    def evaluate(self, component: "ComponentType" = "all", dependency_graph: bpy.types.Depsgraph | None = None):
        """Ensure the character is bound; Blender's graph owns all live evaluation."""
        from .runtime import controller, engine

        if dependency_graph is not None or not is_main_thread() or controller.is_suspended(self):
            return
        if not self.head_initialized and self.head_rig:
            self.head_initialize()
        if not self.body_initialized and self.body_rig:
            self.body_initialize()
        if not engine.active(self) and self.auto_evaluate:
            engine.install(self)
        bpy.context.view_layer.update()

    def update_head_switch_values(self):
        """Switches are outputs of the dependency-ordered runtime carrier."""
        self.evaluate(component="head")

    def get_head_gui_control_values_from_eye_aim(
        self, dependency_graph: bpy.types.Depsgraph | None = None
    ) -> dict[str, dict[str, float]]:
        """Read eye controls solved by the native frame evaluator."""
        from .runtime.authoring import sample

        state = sample(self, "head", graph=dependency_graph)
        values = {}
        for index, name, axis in self.head_gui_control_plan:
            if name in ("CTRL_L_eye", "CTRL_R_eye"):
                values.setdefault(name, {})[axis] = state.getGUIControl(index)
        return values

    def update_head_gui_control_values(
        self,
        override_values: dict[str, dict[str, float]] | None = None,
        dependency_graph: bpy.types.Depsgraph | None = None,
    ) -> None:
        """Sample face-board controls using the runtime's C++ input and eye solve."""
        from .runtime.authoring import sample

        if self.head_dna_reader and self.head_rig:
            sample(self, "head", override_values, dependency_graph)

    def update_head_raw_control_values(self, override_values: dict[str, dict[str, float]] | None = None) -> None:
        """Sample joint-driving raw controls through the runtime."""
        from .runtime.authoring import raw_inputs

        if self.head_dna_reader and self.head_rig:
            raw_inputs(self, "head", override_values)

    def update_body_raw_control_values(self, override_values: dict[str, dict[str, float]] | None = None) -> None:
        """Sample body quaternion inputs through the runtime."""
        from .runtime.authoring import raw_inputs

        if self.body_dna_reader and self.body_rig:
            raw_inputs(self, "body", override_values)

    def update_head_bone_transforms(self, collect_transforms: bool = False) -> list[tuple[str, Vector, Euler, Vector]]:
        """Apply or collect the native-converted head authoring pose."""
        from .runtime.authoring import bone_transforms

        return bone_transforms(self, "head", collect_transforms)

    def update_body_bone_transforms(self, collect_transforms: bool = False) -> list[tuple[str, Vector, Euler, Vector]]:
        """Apply or collect the native-converted body authoring pose."""
        from .runtime.authoring import bone_transforms

        return bone_transforms(self, "body", collect_transforms)

    def reset_head_raw_control_values(self):
        """Re-evaluate the native head inputs after changing evaluation options."""
        self.evaluate(component="head")

    def reset_body_raw_control_values(self):
        """Re-evaluate the native body inputs after changing evaluation options."""
        self.evaluate(component="body")
