# standard library imports
import logging
import queue
import shutil

from datetime import UTC, datetime, timedelta
from pathlib import Path

# third party imports
import bpy

# local imports
from . import constants, utilities
from .components import CharacterComponentBody, CharacterComponentHead, get_meta_human_component
from .constants import (
    HEAD_TEXTURE_LOGIC_NODE_LABEL,
    HEAD_TEXTURE_LOGIC_NODE_NAME,
    NUMBER_OF_HEAD_LODS,
    ToolInfo,
)
from .dna_io import DNACalibrator, DNAExporter
from .fbx.reader import FbxAnimationClip
from .properties import BlendFileCharacterCollection, CharacterImportProperties
from .typing import *  # noqa: F403
from .ui import callbacks, importer
from .utilities import reference
from .validators import ValidationReport


logger = logging.getLogger(__name__)


class GenericUIListOperator:
    """Mix-in class containing functionality shared by operators
    that deal with managing Blender list entries."""

    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    active_index: bpy.props.IntProperty()  # pyright: ignore[reportInvalidTypeForm]

    def _resolve_active_index(self, collection: "bpy.types.bpy_prop_collection") -> int | None:
        """Clamp ``active_index`` into ``collection``, or ``None`` when it is empty.

        The index is passed in as an operator property, so it can be stale or bogus when the
        operator is run from a keymap, the search menu, or a script rather than the list's buttons.
        """
        count = len(collection)
        if count == 0:
            return None
        return max(0, min(self.active_index, count - 1))


class GenericProgressQueueOperator(bpy.types.Operator):
    """
    Mix-in class containing functionality shared by operators that have a progress queue.
    """

    _timer: bpy.types.Timer | None = None
    _commands_queue = queue.Queue()
    _commands_queue_size = 0
    # Number of queued commands to process per timer tick. Processing in batches
    # dramatically reduces the per-tick timer cadence and full-UI redraw overhead
    # when importing hundreds of shape keys, while still keeping the UI responsive
    # and the progress bar updating smoothly.
    _batch_size = 25

    def modal(self, context: "Context", event: bpy.types.Event) -> set[str]:
        addon_window_manager_properties = utilities.get_addon_window_manager_properties(context)

        if event.type == "ESC":
            return self.finish(context)

        if event.type == "TIMER" and context.screen:
            if self._commands_queue.empty():
                return self.finish(context)

            # Process a batch of commands per tick instead of one, so the timer
            # cadence and redraw cost don't dominate the total import time.
            description = ""
            for _ in range(self._batch_size):
                if self._commands_queue.empty():
                    break
                index, mesh_index, description, kwargs_callback, callback = self._commands_queue.get()
                # calculate the kwargs
                kwargs = kwargs_callback(index, mesh_index)
                # inject the kwargs into the description
                description = description.format(**kwargs)
                callback(**kwargs)

            remaining = self._commands_queue.qsize()
            addon_window_manager_properties.progress = (
                self._commands_queue_size - remaining
            ) / self._commands_queue_size
            addon_window_manager_properties.progress_description = description

            # redraw once per batch so the progress bar updates without paying the
            # full-redraw cost for every individual command
            for area in context.screen.areas:
                area.tag_redraw()

        return {"PASS_THROUGH"}

    def execute(self, context: "Context") -> set[str]:
        addon_window_manager_properties = utilities.get_addon_window_manager_properties(context)

        if not self.validate(context):
            return {"CANCELLED"}

        self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        head = utilities.get_active_head()
        if head:
            addon_window_manager_properties.progress = 0
            addon_window_manager_properties.progress_description = ""
            self._commands_queue = queue.Queue()
            self.set_commands_queue(context, head, self._commands_queue)
            self._commands_queue_size = self._commands_queue.qsize()
            return {"RUNNING_MODAL"}
        return {"CANCELLED"}

    def finish(self, context: "Context") -> set[str]:
        addon_window_manager_properties = utilities.get_addon_window_manager_properties(context)

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        addon_window_manager_properties.progress = 1
        # re-initialize the rig instance so the shape key blocks collection is updated for the UI
        instance = callbacks.get_active_rig_instance()
        if instance:
            instance.data.clear()
            instance.initialize()
        return {"FINISHED"}

    def validate(self, context: "Context") -> bool:
        return True

    def set_commands_queue(
        self,
        context: "Context",
        component: CharacterComponentHead | CharacterComponentBody,
        commands_queue: queue.Queue,
    ):
        pass


class AppendOrLinkCharacter(bpy.types.Operator, importer.LinkAppendCharacterImportHelper):
    """Append or link a MetaHuman from a .blend file. The .blend file must contain a collection with all data related to the MetaHuman asset."""  # noqa: E501

    bl_idname = f"{ToolInfo.NAME}.append_or_link_metahuman"
    bl_label = "Import"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".blend"

    filter_glob: bpy.props.StringProperty(
        default="*.blend",
        options={"HIDDEN"},
        subtype="FILE_PATH",
    )  # pyright: ignore[reportInvalidTypeForm]

    relative_path: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    previous_file_path: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    operation_type: bpy.props.EnumProperty(
        items=[
            ("APPEND", "Append", "Append the selected MetaHuman"),
            ("LINK", "Link", "Link the selected MetaHuman"),
        ],
        default="APPEND",
    )  # pyright: ignore[reportInvalidTypeForm]
    editable_rig: bpy.props.BoolProperty(
        name="Editable Rig",
        description="Make the source face board and control rig local and editable while linking character assets",
        default=False,
    )  # pyright: ignore[reportInvalidTypeForm]
    meta_human_list: bpy.props.CollectionProperty(type=BlendFileCharacterCollection)  # pyright: ignore[reportInvalidTypeForm]
    meta_human_names: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        file_path = self.filepath  # type: ignore[attr-defined]
        try:
            selected_names = reference.validate_names(
                self.meta_human_names.split(",")
                if self.meta_human_names
                else [item.name for item in self.meta_human_list if item.include],
                reference.scene_names(context.scene)
                | {instance.name for instance in utilities.get_addon_scene_properties(context).rig_instance_list},
            )
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if not file_path:
            self.report({"ERROR"}, "You must select a .blend file")
            return {"CANCELLED"}

        if not Path(bpy.path.abspath(file_path)).exists():
            self.report({"ERROR"}, f"File not found: {file_path}")
            return {"CANCELLED"}

        if bpy.data.filepath and Path(bpy.data.filepath).resolve() == Path(bpy.path.abspath(file_path)).resolve():
            self.report({"ERROR"}, "You cannot import a MetaHuman from the current .blend file")
            return {"CANCELLED"}

        # warn if the blend file was saved with a different major Blender version
        abs_file_path = Path(bpy.path.abspath(file_path))
        blend_version = utilities.read_blend_file_version(abs_file_path)
        if blend_version and blend_version[0] != bpy.app.version[0]:
            self.report(
                {"WARNING"},
                f"File was saved in Blender {utilities.blend_file_version_string(blend_version)}, "
                f"running Blender {bpy.app.version_string}. This may cause import issues.",
            )

        data, error = utilities.extract_rig_instance_data_from_blend_file(abs_file_path)
        if error:
            logger.error(error)
            self.report({"ERROR"}, f"Failed to extract rig instance data from blend file: {error}")
            return {"CANCELLED"}
        if not data:
            self.report({"ERROR"}, "Failed to read rig instance data in blend file")
            return {"CANCELLED"}

        try:
            descriptors = reference.validate_descriptors(data, selected_names)
            mode = "EDITABLE_LINK" if self.operation_type == "LINK" and self.editable_rig else self.operation_type
            reference.import_characters(context, str(abs_file_path), descriptors, mode, self.relative_path)
        except Exception as error:
            if isinstance(error, reference.LegacyDataError):
                logger.warning("%s", error)
                self.report({"WARNING"}, reference.MIGRATION_MESSAGE)
            else:
                logger.exception("Failed to import character objects")
                self.report({"ERROR"}, f"Failed to import character: {error}")
            return {"CANCELLED"}
        return {"FINISHED"}


class ImportAnimationBase(bpy.types.Operator):
    filename_ext = ".fbx"

    # Set by each subclass: the python operator name under ``bpy.ops.character_dna``.
    # ``bl_idname`` cannot be used because Blender replaces it with the RNA
    # identifier once the class is registered.
    import_operator_id = ""

    filter_glob: bpy.props.StringProperty(
        default="*.fbx",
        options={"HIDDEN"},
        subtype="FILE_PATH",
    )  # pyright: ignore[reportInvalidTypeForm]

    round_sub_frames: bpy.props.BoolProperty(
        name="Round Sub Frames",
        default=True,
        description=(
            "Whether to round sub frames when importing the animation. This "
            "ensure all keyframes are on whole frames with integer values"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]

    match_frame_rate: bpy.props.BoolProperty(
        name="Match Frame Rate",
        default=True,
        description=(
            "Whether to match the frame rate when importing the animation. This "
            "will scale the animation curves to match the current scene frame rate"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]

    prefix_instance_name: bpy.props.BoolProperty(
        name="Prefix Instance Name",
        default=True,
        description=(
            "Prefixes the baked action name with the rig instance name. This helps avoid name "
            "collisions with other action names when multiple are in the same scene."
        ),
    )  # pyright: ignore[reportInvalidTypeForm]

    prefix_component_name: bpy.props.BoolProperty(
        name="Prefix Component Name",
        default=True,
        description=(
            "Prefixes the baked action name with the component name. This helps avoid name collisions "
            "with other components that might have the same action names."
        ),
    )  # pyright: ignore[reportInvalidTypeForm]

    ignore_validation: bpy.props.BoolProperty(
        default=False,
        options={"HIDDEN", "SKIP_SAVE"},
        description="Import even when the file does not look like it belongs to this rig",
    )  # pyright: ignore[reportInvalidTypeForm]

    @property
    def settings_title(self) -> str:
        return "Animation Import Settings:"

    def load_clip(self, file_path: Path) -> FbxAnimationClip | None:
        """Read the FBX file, reporting any failure to the user.

        Args:
            file_path: Path to the FBX file.

        Returns:
            The loaded clip, or ``None`` when the file could not be read.
        """
        try:
            return utilities.load_animation_clip(file_path, match_frame_rate=self.match_frame_rate)
        except (FileNotFoundError, ImportError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return None

    def confirm_or_cancel(self, report: ValidationReport, file_path: Path, component_type: str) -> bool:
        """Surface a validation report, asking the user whether to import anyway.

        Args:
            report: The validation report for the loaded clip.
            file_path: Path to the FBX file.
            component_type: Component the animation is being imported onto.

        Returns:
            ``True`` when the import should continue.
        """
        for warning in report.warnings:
            logger.warning(warning.message)

        if report.is_valid or self.ignore_validation:
            return True

        if bpy.app.background:
            # There is nobody to answer a dialog, so refuse rather than silently
            # importing the wrong file. Scripts can pass ignore_validation=True.
            self.report({"ERROR"}, report.summary())
            return False

        getattr(bpy.ops, ToolInfo.NAME).confirm_animation_import(
            "INVOKE_DEFAULT",
            operator_name=self.import_operator_id,
            filepath=str(file_path),
            component_type=component_type,
            message="Bone names do not match the target rig.",
            round_sub_frames=self.round_sub_frames,
            match_frame_rate=self.match_frame_rate,
            prefix_instance_name=self.prefix_instance_name,
            prefix_component_name=self.prefix_component_name,
        )
        return False


class ConfirmAnimationImport(bpy.types.Operator):
    """Ask whether to import an animation file that does not appear to match the target rig"""

    bl_idname = f"{ToolInfo.NAME}.confirm_animation_import"
    bl_label = "This Might Be the Wrong Animation File"

    operator_name: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    filepath: bpy.props.StringProperty(default="", subtype="FILE_PATH")  # pyright: ignore[reportInvalidTypeForm]
    component_type: bpy.props.StringProperty(default="body")  # pyright: ignore[reportInvalidTypeForm]
    message: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    round_sub_frames: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    match_frame_rate: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    prefix_instance_name: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    prefix_component_name: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        operator = getattr(getattr(bpy.ops, ToolInfo.NAME), self.operator_name, None)
        if not operator:
            self.report({"ERROR"}, f"Unknown import operator: {self.operator_name}")
            return {"CANCELLED"}

        keywords = {
            "filepath": self.filepath,
            "round_sub_frames": self.round_sub_frames,
            "match_frame_rate": self.match_frame_rate,
            "prefix_instance_name": self.prefix_instance_name,
            "prefix_component_name": self.prefix_component_name,
            "ignore_validation": True,
        }
        if self.operator_name == "import_component_animation":
            keywords["component_type"] = self.component_type

        return operator("EXEC_DEFAULT", **keywords)

    def invoke(self, context: "Context", event: bpy.types.Event) -> set[str] | None:
        wm = context.window_manager
        if not wm:
            return None
        return wm.invoke_props_dialog(self, confirm_text="Import Anyway", cancel_default=True, width=500)  # type: ignore[return-value]

    def draw(self, context: "Context"):
        if not self.layout:
            return

        row = self.layout.row()
        row.scale_y = 1.5
        row.label(text=f"{Path(self.filepath).name} might not be {self.component_type} animation.")
        for line in self.message.split("\n"):
            row = self.layout.row()
            row.alert = True
            row.label(text=line)


class ImportFaceBoardAnimation(ImportAnimationBase, importer.ImportAnimation):
    """Import an animation for the metahuman face board exported from an Unreal Engine Level Sequence"""

    bl_idname = f"{ToolInfo.NAME}.import_face_board_animation"
    bl_label = "Import"
    import_operator_id = "import_face_board_animation"

    @property
    def settings_title(self) -> str:
        return "Face Board Animation Import Settings:"

    def execute(self, context: "Context") -> set[str]:
        file_path = Path(bpy.path.abspath(self.filepath))  # type: ignore[attr-defined]
        logger.info(f"Importing animation {file_path}")
        head = utilities.get_active_head()
        if not head:
            self.report({"ERROR"}, "No active head found")
            return {"CANCELLED"}

        if not head.face_board_object:
            self.report({"ERROR"}, "No face board object found for the active head")
            return {"CANCELLED"}

        clip = self.load_clip(file_path)
        if clip is None:
            return {"CANCELLED"}

        report = utilities.validate_animation_clip(clip, head.face_board_object, "face board", is_face_board=True)
        if not self.confirm_or_cancel(report, file_path, "face board"):
            return {"CANCELLED"}

        head.import_action(
            file_path,
            is_face_board=True,
            round_sub_frames=self.round_sub_frames,
            match_frame_rate=self.match_frame_rate,
            prefix_instance_name=self.prefix_instance_name,
            prefix_component_name=self.prefix_component_name,
            clip=clip,
        )
        utilities.switch_to_pose_mode(head.face_board_object)
        self.report({"INFO"}, f"Imported face board animation from {file_path}")
        return {"FINISHED"}


class ImportComponentAnimation(ImportAnimationBase, importer.ImportAnimation):
    """Import an animation for the selected metahuman component that has been exported from an Unreal Engine"""

    bl_idname = f"{ToolInfo.NAME}.import_component_animation"
    bl_label = "Import"
    import_operator_id = "import_component_animation"

    component_type: bpy.props.StringProperty(default="body")  # pyright: ignore[reportInvalidTypeForm]

    @property
    def settings_title(self) -> str:
        return f"{self.component_type.capitalize()} Animation Import Settings:"

    def execute(self, context: "Context") -> set[str]:
        file_path = Path(bpy.path.abspath(self.filepath))  # type: ignore[attr-defined]
        logger.info(f"Importing animation {file_path}")

        if self.component_type == "head":
            result = self._import_head(file_path)
        elif self.component_type == "body":
            result = self._import_body(file_path)
        else:
            self.report({"ERROR"}, f"Unknown component type: {self.component_type}")
            return {"CANCELLED"}

        if result == {"FINISHED"}:
            self.report({"INFO"}, f"Imported {self.component_type} animation from {file_path}")
        return result

    def _import_head(self, file_path: Path) -> set[str]:
        head = utilities.get_active_head()
        if not head:
            self.report({"ERROR"}, "No active head found")
            return {"CANCELLED"}
        if not head.head_rig_object:
            self.report({"ERROR"}, "No head rig object found for the active head")
            return {"CANCELLED"}

        clip = self.load_clip(file_path)
        if clip is None:
            return {"CANCELLED"}

        report = utilities.validate_animation_clip(clip, head.head_rig_object, "head")
        if not self.confirm_or_cancel(report, file_path, "head"):
            return {"CANCELLED"}

        head.import_action(
            file_path,
            is_face_board=False,
            round_sub_frames=self.round_sub_frames,
            match_frame_rate=self.match_frame_rate,
            prefix_instance_name=self.prefix_instance_name,
            prefix_component_name=self.prefix_component_name,
            clip=clip,
        )
        return {"FINISHED"}

    def _import_body(self, file_path: Path) -> set[str]:
        body = utilities.get_active_body()
        if not body:
            self.report({"ERROR"}, "No active body found")
            return {"CANCELLED"}
        if not body.body_rig_object:
            self.report({"ERROR"}, "No body rig object found for the active body")
            return {"CANCELLED"}

        clip = self.load_clip(file_path)
        if clip is None:
            return {"CANCELLED"}

        report = utilities.validate_animation_clip(clip, body.body_rig_object, "body")
        if not self.confirm_or_cancel(report, file_path, "body"):
            return {"CANCELLED"}

        body.import_action(
            file_path,
            round_sub_frames=self.round_sub_frames,
            match_frame_rate=self.match_frame_rate,
            prefix_instance_name=self.prefix_instance_name,
            prefix_component_name=self.prefix_component_name,
            clip=clip,
        )
        utilities.switch_to_pose_mode(body.body_rig_object)
        return {"FINISHED"}


class BakeAnimationBase(bpy.types.Operator):
    action_name: bpy.props.StringProperty(
        name="Action Name",
        default="baked_action",
        description="The name of the action that will be created to store the baked animation data",
    )  # pyright: ignore[reportInvalidTypeForm]

    prefix_instance_name: bpy.props.BoolProperty(
        name="Prefix Instance Name",
        default=True,
        description=(
            "Prefixes the baked action name with the rig instance name. This helps avoid name collisions "
            "with other action names when multiple are in the same scene."
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    prefix_component_name: bpy.props.BoolProperty(
        name="Prefix Component Name",
        default=True,
        description=(
            "Prefixes the baked action name with the component name. This helps avoid name collisions "
            "with other components that might have the same action names."
        ),
    )  # pyright: ignore[reportInvalidTypeForm]

    replace_action: bpy.props.BoolProperty(
        name="Replace Action",
        default=True,
        description="If there is an existing action with the same name, replaces it with the baked action",
    )  # pyright: ignore[reportInvalidTypeForm]

    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        default=1,
        min=1,
        get=callbacks.get_bake_start_frame,
        set=callbacks.set_bake_start_frame,
        description="The frame to start baking the animation on",
    )  # pyright: ignore[reportInvalidTypeForm]

    end_frame: bpy.props.IntProperty(
        name="End Frame",
        default=250,
        min=1,
        get=callbacks.get_bake_end_frame,
        set=callbacks.set_bake_end_frame,
        description="The frame to end baking the animation on",
    )  # pyright: ignore[reportInvalidTypeForm]

    step: bpy.props.IntProperty(
        name="Step",
        default=1,
        min=1,
        description="The frame step to bake the animation on. Essentially add a keyframe every nth frame",
    )  # pyright: ignore[reportInvalidTypeForm]

    masks: bpy.props.BoolProperty(
        name="Masks", default=True, description="Bakes the values of the wrinkle map masks over time"
    )  # pyright: ignore[reportInvalidTypeForm]

    shape_keys: bpy.props.BoolProperty(
        name="Shape Keys", default=True, description="Bakes the values of the shape keys over time"
    )  # pyright: ignore[reportInvalidTypeForm]

    clean_curves: bpy.props.BoolProperty(
        name="Clean Curves", default=False, description="Clean Curves, After baking curves, remove redundant keys"
    )  # pyright: ignore[reportInvalidTypeForm]

    bone_location: bpy.props.BoolProperty(
        name="Bone Location", default=True, description="Bakes the location of the bones"
    )  # pyright: ignore[reportInvalidTypeForm]
    bone_rotation: bpy.props.BoolProperty(
        name="Bone Rotation", default=True, description="Bakes the rotation of the bones"
    )  # pyright: ignore[reportInvalidTypeForm]
    bone_scale: bpy.props.BoolProperty(name="Bone Scale", default=True, description="Bakes the scale of the bones")  # pyright: ignore[reportInvalidTypeForm]

    def invoke(self, context: "Context", event: bpy.types.Event) -> set[str]:
        return context.window_manager.invoke_props_dialog(  # type: ignore[return-value]
            self, title=self.dialog_title, width=250
        )

    @property
    def dialog_title(self) -> str:
        return self.bl_label

    def draw_extra_settings(self, layout: bpy.types.UILayout, context: "Context") -> None:
        pass

    def draw(self, context: "Context") -> None:
        if not self.layout:
            return

        row = self.layout.row()
        row.label(text="Naming:")
        row = self.layout.row()
        row.prop(self, "action_name", text="")
        row = self.layout.row()
        row.prop(self, "prefix_instance_name")
        row = self.layout.row()
        row.prop(self, "prefix_component_name")
        row = self.layout.row()
        row.label(text="Settings:")
        row = self.layout.row()
        row.prop(self, "start_frame")
        row.prop(self, "end_frame")
        row = self.layout.row()
        row.prop(self, "step")
        row = self.layout.row()
        row.prop(self, "replace_action")
        row = self.layout.row()
        row.prop(self, "clean_curves")
        row = self.layout.row()
        row.prop(self, "shape_keys")
        row = self.layout.row()
        row.prop(self, "masks")
        row = self.layout.row()
        self.draw_extra_settings(self.layout, context)
        row = self.layout.row()
        row.label(text="Bone Transforms:")
        row = self.layout.row()
        row.prop(self, "bone_location", text="Location")
        row = self.layout.row()
        row.prop(self, "bone_rotation", text="Rotation")
        row = self.layout.row()
        row.prop(self, "bone_scale", text="Scale")


class BakeFaceBoardAnimation(BakeAnimationBase):
    """Bakes the active face board action to the pose bones, shape key values, and texture logic mask values. Useful for rendering, simulations, etc. where rig logic evaluation is not available"""  # noqa: E501

    bl_idname = f"{ToolInfo.NAME}.bake_face_board_animation"
    bl_label = "Bake Face Board Animation"

    def execute(self, context: "Context") -> set[str]:
        if self.start_frame > self.end_frame:
            self.report({"ERROR"}, "The start frame must be less than the end frame")
            return {"CANCELLED"}

        instance = callbacks.get_active_rig_instance()
        if instance and instance.head_rig:
            channel_types = set()
            if self.bone_location:
                channel_types.add("LOCATION")
            if self.bone_rotation:
                channel_types.add("ROTATION")
            if self.bone_scale:
                channel_types.add("SCALE")

            action_name = utilities.get_action_name(
                instance=instance,
                action_name=self.action_name,
                component="head",
                prefix_component_name=self.prefix_component_name,
                prefix_instance_name=self.prefix_instance_name,
            )

            utilities.bake_face_board_to_action(
                instance=instance,
                armature_object=instance.head_rig,
                action_name=action_name,
                replace_action=self.replace_action,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                step=self.step,
                channel_types=channel_types,
                clean_curves=self.clean_curves,
                masks=self.masks,
                shape_keys=self.shape_keys,
            )
            instance.head_rig.hide_set(False)
            utilities.switch_to_pose_mode(instance.head_rig)
            instance.auto_evaluate_head = False
        return {"FINISHED"}

    @classmethod
    def poll(cls, _: "Context") -> bool:
        instance = callbacks.get_active_rig_instance()
        if not instance:
            return False
        if not instance.head_rig:
            return False
        if not instance.face_board:
            return False
        if not instance.face_board.animation_data:
            return False
        return instance.face_board.animation_data.action


class BakeComponentAnimation(BakeAnimationBase):
    """Bakes the active component action. This takes into account how the driver pose bones effect the rbf driven bones, shape key values, and texture logic mask values. Useful for rendering, simulations, etc. where rig logic evaluation is not available"""  # noqa: E501

    bl_idname = f"{ToolInfo.NAME}.bake_component_animation"
    bl_label = "Bake Component Animation"

    component_type: bpy.props.StringProperty(
        default="body",
        options={"HIDDEN"},
        subtype="FILE_PATH",
    )  # pyright: ignore[reportInvalidTypeForm]

    driver_bones: bpy.props.BoolProperty(
        name="Driver Bones", default=True, description="Bakes the values of the driver bones over time"
    )  # pyright: ignore[reportInvalidTypeForm]
    driven_bones: bpy.props.BoolProperty(
        name="Driven Bones", default=True, description="Bakes the values of the RBF driven bones over time"
    )  # pyright: ignore[reportInvalidTypeForm]
    twist_bones: bpy.props.BoolProperty(
        name="Twist Bones", default=True, description="Bakes the values of the twist bones over time"
    )  # pyright: ignore[reportInvalidTypeForm]
    swing_bones: bpy.props.BoolProperty(
        name="Swing Bones", default=True, description="Bakes the values of the swing bones over time"
    )  # pyright: ignore[reportInvalidTypeForm]
    other_bones: bpy.props.BoolProperty(
        name="Other Bones",
        default=True,
        description=(
            "Bakes the values of other bones on the rig that are not explicitly categorized as "
            "driver, driven, twist, or swing bones"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        if self.start_frame > self.end_frame:
            self.report({"ERROR"}, "The start frame must be less than the end frame")
            return {"CANCELLED"}

        instance = callbacks.get_active_rig_instance()
        if instance and instance.body_rig and self.component_type == "body":
            channel_types = set()
            if self.bone_location:
                channel_types.add("LOCATION")
            if self.bone_rotation:
                channel_types.add("ROTATION")
            if self.bone_scale:
                channel_types.add("SCALE")

            action_name = utilities.get_action_name(
                instance=instance,
                action_name=self.action_name,
                component=self.component_type,
                prefix_component_name=self.prefix_component_name,
                prefix_instance_name=self.prefix_instance_name,
            )

            utilities.bake_body_to_action(
                instance=instance,
                armature_object=instance.body_rig,
                action_name=action_name,
                replace_action=self.replace_action,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                step=self.step,
                channel_types=channel_types,
                clean_curves=self.clean_curves,
                masks=self.masks,
                shape_keys=self.shape_keys,
                driver_bones=self.driver_bones,
                driven_bones=self.driven_bones,
                twist_bones=self.twist_bones,
                swing_bones=self.swing_bones,
                other_bones=self.other_bones,
            )
            instance.body_rig.hide_set(False)
            utilities.switch_to_pose_mode(instance.body_rig)
            instance.auto_evaluate_body = False
        return {"FINISHED"}

    @property
    def dialog_title(self) -> str:
        return f"Bake {self.component_type.capitalize()} Animation"

    def draw_extra_settings(self, layout: bpy.types.UILayout, context: "Context") -> None:
        if self.component_type == "body":
            row = layout.row()
            row.label(text="Bone Types:")
            row = layout.row()
            row.prop(self, "driver_bones")
            row = layout.row()
            row.prop(self, "driven_bones")
            row = layout.row()
            row.prop(self, "twist_bones")
            row = layout.row()
            row.prop(self, "swing_bones")
            row = layout.row()
            row.prop(self, "other_bones")

    @classmethod
    def poll(cls, _: "Context") -> bool:
        instance = callbacks.get_active_rig_instance()

        if not instance:
            return False
        if not instance.body_rig:
            return False
        # the animation may live on the body rig directly, or on a control rig (e.g. a Rigify
        # rig) that drives the body rig's bones through constraints
        has_body_action = bool(instance.body_rig.animation_data and instance.body_rig.animation_data.action)
        has_control_action = bool(
            instance.control_rig and instance.control_rig.animation_data and instance.control_rig.animation_data.action
        )
        return has_body_action or has_control_action


class ImportCharacterDna(bpy.types.Operator, importer.ImportAsset, CharacterImportProperties):
    """Import a metahuman head from a DNA file"""

    bl_idname = f"{ToolInfo.NAME}.import_dna"
    bl_label = "Import DNA"
    filename_ext = ".dna"

    filter_glob: bpy.props.StringProperty(
        default="*.dna",
        options={"HIDDEN"},
        subtype="FILE_PATH",
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        window_manager_properties = utilities.get_addon_window_manager_properties(context)

        file_path = Path(bpy.path.abspath(self.filepath))
        if not file_path.exists():
            self.report({"ERROR"}, f"File not found: {file_path}")
            return {"CANCELLED"}
        if not file_path.is_file():
            self.report({"ERROR"}, f'"{file_path}" is a folder. Please select a DNA file.')
            return {"CANCELLED"}
        if file_path.suffix.lower() != ".dna":
            self.report({"ERROR"}, f'The file "{file_path}" is not a DNA file')
            return {"CANCELLED"}
        if round(context.scene.unit_settings.scale_length, 2) != 1.0:
            self.report({"ERROR"}, "The scene unit scale must be set to 1.0")
            return {"CANCELLED"}

        # we don't want to evaluate the dependency graph while importing the DNA
        window_manager_properties.evaluate_dependency_graph = False
        component = get_meta_human_component(
            file_path=file_path,
            properties=self.properties,  # type: ignore[arg-type]
        )
        # if the component is a head, we import the body first if the user has selected the option
        body_file = file_path.parent / "body.dna"
        if self.properties.include_body and component.component_type == "head" and body_file.exists():
            body_component = get_meta_human_component(
                file_path=body_file,
                properties=self.properties,  # type: ignore[arg-type]
                rig_instance=component.rig_instance,
            )
            valid, message = body_component.ingest()
            logger.info(f'Finished importing "{body_file}"')
            if not valid:
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
            self.report({"INFO"}, message)

        # now we can import the chosen .dna file
        valid, message = component.ingest()
        logger.info(f'Finished importing "{self.filepath}"')
        if not valid:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        self.report({"INFO"}, message)

        # populate the output items based on what was imported
        callbacks.update_head_output_items(None, bpy.context)  # type: ignore[arg-type]
        # now we can evaluate the dependency graph again
        window_manager_properties.evaluate_dependency_graph = True
        ops = utilities.get_addon_ops_module()
        ops.force_evaluate()
        ops.metrics_collection_consent("INVOKE_DEFAULT")

        return {"FINISHED"}

    @classmethod
    def poll(cls, _: "Context") -> bool:
        return utilities.dependencies_are_valid()


class DNA_FH_import_dna(bpy.types.FileHandler):
    bl_idname = "DNA_FH_import_dna"
    bl_label = "File handler for .dna files"
    bl_import_operator = f"{ToolInfo.NAME}.import_dna"
    bl_file_extensions = ".dna"

    @classmethod
    def poll_drop(cls, context: "Context") -> bool:
        if not utilities.dependencies_are_valid():
            return False

        return (
            context.region is not None
            and context.region.type == "WINDOW"
            and context.area is not None
            and context.area.ui_type == "VIEW_3D"
        )


class GenerateMaterial(bpy.types.Operator):
    """Generates a material for the head mesh object that you can then customize"""

    bl_idname = f"{ToolInfo.NAME}.generate_material"
    bl_label = "Generate Material"

    def execute(self, context: "Context") -> set[str]:
        head = utilities.get_active_head()
        if head and head.head_mesh_object:
            head.import_materials()
        return {"FINISHED"}

    @classmethod
    def poll(cls, _: "Context") -> bool:
        instance = callbacks.get_active_rig_instance()
        return (
            instance is not None
            and instance.head_mesh is not None
            and instance.head_material is None
            and bpy.context.mode == "OBJECT"
        )


class ForceEvaluate(bpy.types.Operator):
    """Force the rig logic to evaluate on the active rig instance"""

    bl_idname = f"{ToolInfo.NAME}.force_evaluate"
    bl_label = "Force Evaluate"

    def execute(self, context: "Context") -> set[str]:
        instance = callbacks.get_active_rig_instance()
        if callbacks.is_reference_readonly(instance):
            from .runtime import engine

            try:
                engine.release_records(instance)
                engine.adopt(instance)
                context.view_layer.update()
            except (RuntimeError, ValueError, OSError) as error:
                self.report({"ERROR"}, str(error))
                return {"CANCELLED"}
            return {"FINISHED"}

        utilities.teardown_scene()
        utilities.setup_scene()
        instance = callbacks.get_active_rig_instance()
        if instance:
            instance.evaluate()
            # NOTE: Some dependency graph weirdness here. This is necessary to ensure that the rig logic
            # evaluates the pose bones, otherwise bone transform updates won't be applied when the face
            # board updates.
            current_context = utilities.get_current_context()
            if instance.head_rig:
                instance.head_rig.hide_set(False)
                instance.head_rig.hide_viewport = False
                utilities.switch_to_pose_mode(instance.head_rig)

            utilities.set_context(current_context)

        window_manager_properties = utilities.get_addon_window_manager_properties(context)
        window_manager_properties.evaluate_dependency_graph = True
        return {"FINISHED"}


class RefreshOutputItems(bpy.types.Operator):
    """Refresh the Output items list so it reflects the current scene meshes,
    textures, and rig for the active rig instance"""

    bl_idname = f"{ToolInfo.NAME}.refresh_output_items"
    bl_label = "Refresh Output Items"

    @classmethod
    def poll(cls, _: "Context") -> bool:
        return callbacks.get_active_rig_instance() is not None

    def execute(self, context: "Context") -> set[str]:
        callbacks.update_head_output_items(None, context)  # type: ignore[arg-type]
        callbacks.update_body_output_items(None, context)  # type: ignore[arg-type]
        return {"FINISHED"}


class MapRawToGuiControls(bpy.types.Operator):
    """Backward-solve the face board from the head rig's raw controls. Maps the
    current raw control values to GUI controls via rig logic and applies the result
    to the face board control bone positions. Useful for debugging"""

    bl_idname = f"{ToolInfo.NAME}.map_raw_to_gui_controls"
    bl_label = "Map Raw to GUI Controls"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: "Context") -> set[str]:
        instance = callbacks.get_active_rig_instance()
        if not instance:
            return {"CANCELLED"}

        if callbacks.is_face_board_readonly(instance):
            self.report({"ERROR"}, "The linked face board is read-only. Use Editable Link or Append to edit controls.")
            return {"CANCELLED"}

        if callbacks.is_reference_readonly(instance, "head"):
            self.report(
                {"ERROR"},
                "Raw-to-GUI mapping is unavailable for a linked head. Edit the source .blend file or use Append.",
            )
            return {"CANCELLED"}

        if not instance.head_initialized:
            instance.head_initialize()

        if not instance.head_instance or not instance.head_manager:
            return {"CANCELLED"}

        # map the current raw controls back to the GUI controls
        instance.head_manager.mapRawToGUIControls(instance.head_instance)

        # apply the solved GUI control values to the face board bones without
        # triggering a dependency graph re-evaluation mid-write
        window_manager_properties = utilities.get_addon_window_manager_properties(context)
        window_manager_properties.evaluate_dependency_graph = False
        instance.apply_gui_controls_to_face_board()
        window_manager_properties.evaluate_dependency_graph = True

        if instance.face_board and not instance.face_board.hide_get():
            utilities.switch_to_pose_mode(instance.face_board)

        # re-evaluate so the head mesh matches the newly posed face board
        instance.evaluate()
        return {"FINISHED"}


class TestSentry(bpy.types.Operator):
    """Test the Sentry error reporting system"""

    bl_idname = f"{ToolInfo.NAME}.test_sentry"
    bl_label = "Test Sentry"

    def execute(self, context: "Context") -> set[str]:
        division_by_zero = 1 / 0  # pyright: ignore[reportUnusedVariable] # noqa: F841
        return {"FINISHED"}


class MigrateLegacyData(bpy.types.Operator):
    """Migrate legacy data to the latest format"""

    bl_idname = f"{ToolInfo.NAME}.migrate_legacy_data"
    bl_label = "Migrate Legacy Data"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: "Context") -> set[str]:
        try:
            migrate_type, upgraded, unchanged = utilities.migrate_runtime_data(context)
        except (RuntimeError, ValueError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if migrate_type != "default" or upgraded:
            self.report(
                {"INFO"},
                f"Migrated rig data; upgraded {upgraded} runtime(s), {unchanged} unchanged. "
                "Save the .blend file before appending or linking it.",
            )
        else:
            self.report({"INFO"}, f"No migration needed; {unchanged} instance(s) unchanged.")
        return {"FINISHED"}


class SendToMetaHumanCreator(bpy.types.Operator):
    """Exports the DNA head and body components, as well as, textures in a format supported by MetaHuman Creator."""

    bl_idname = f"{ToolInfo.NAME}.send_to_meta_human_creator"
    bl_label = "Send to MetaHuman Creator"

    def execute(self, context: "Context") -> set[str]:
        instance = callbacks.get_active_rig_instance()
        if instance:
            # store the current auto evaluate settings
            auto_evaluate_head = instance.auto_evaluate_head
            auto_evaluate_body = instance.auto_evaluate_body

            # disable auto evaluate while we are exporting
            instance.auto_evaluate_head = False
            instance.auto_evaluate_body = False

            current_context = utilities.get_current_context()

            for attribute_name in ["head_mesh", "head_rig", "body_mesh", "body_rig"]:
                if not getattr(instance, attribute_name):
                    self.report(
                        {"ERROR"},
                        (
                            f"No {attribute_name} set on the active instance. Please ensure you have a "
                            "head and body mesh and rig set before sending to MetaHuman Creator."
                        ),
                    )
                    return {"CANCELLED"}

            if not bpy.path.abspath(instance.output.folder_path) and not bpy.data.filepath:
                self.report({"ERROR"}, "File must be saved to use a relative path")
                return {"CANCELLED"}

            head = utilities.get_active_head()
            body = utilities.get_active_body()
            if not head or not body:
                self.report(
                    {"ERROR"},
                    "No active instance found. Please select an instance from the list under the RigLogic panel.",
                )
                return {"CANCELLED"}

            last_component: CharacterComponentBase = None  # pyright: ignore[reportAssignmentType]
            # Export the body first so the head can conform its neck edge loop onto
            # the freshly-written body DNA. Auto LOD propagation regenerates each
            # component's lower-LOD meshes independently, so the head must snap to the
            # body that is actually written (not the imported template) for the neck
            # seam to line up across every LOD in the exported files.
            output_folder = Path(bpy.path.abspath(instance.output.folder_path))

            # Build every component's exporter up front so all validations can run
            # before any DNA is written -- otherwise a later component's validation
            # (and its fix dialog) would only surface after an earlier component had
            # already been exported to disk.
            component_exporters: list[tuple[object, DNAExporter]] = []
            for component in [body, head]:
                seam_reference_dna_path = None
                if component.component_type == "head":
                    seam_reference_dna_path = str(output_folder / "body.dna")

                dna_io_instance: DNAExporter = None  # type: ignore[assignment]
                if instance.output.method == "calibrate":
                    dna_io_instance = DNACalibrator(
                        instance=instance,
                        linear_modifier=component.linear_modifier,
                        file_name=f"{component.component_type}.dna",
                        component_type=component.component_type,
                        normals=instance.output.export_normals,
                        seam_reference_dna_path=seam_reference_dna_path,
                    )
                elif instance.output.method == "overwrite":
                    dna_io_instance = DNAExporter(
                        instance=instance,
                        linear_modifier=component.linear_modifier,
                        file_name=f"{component.component_type}.dna",
                        component_type=component.component_type,
                        normals=instance.output.export_normals,
                    )
                component_exporters.append((component, dna_io_instance))

            # Phase 1: validate every component before writing anything to disk.
            for _component, dna_io_instance in component_exporters:
                valid, title, message, fix = dna_io_instance.validate_scene()
                if not valid:
                    utilities.report_error_panel(title=title, message=message, fix=fix, width=500)
                    return {"CANCELLED"}

            # Phase 2: all validations passed -- export each component.
            for component, dna_io_instance in component_exporters:
                valid, title, message, fix = dna_io_instance.run()
                if not valid:
                    utilities.report_error_panel(title=title, message=message, fix=fix, width=500)
                    return {"CANCELLED"}
                self.report({"INFO"}, message)

                last_component = component  # pyright: ignore[reportAssignmentType]

            # write a manifest file to the output folder similar to the MetaHuman Creator DCC export
            if last_component:
                last_component.write_export_manifest()
                ops = utilities.get_addon_ops_module()
                ops.force_evaluate()

            utilities.set_context(current_context)

            # restore the auto evaluate settings
            instance.auto_evaluate_head = auto_evaluate_head
            instance.auto_evaluate_body = auto_evaluate_body

        return {"FINISHED"}


class ExportSelectedComponent(bpy.types.Operator):
    """Export only the selected component to a single DNA file. No textures or supporting files will be exported."""

    bl_idname = f"{ToolInfo.NAME}.export_selected_component"
    bl_label = "Export Selected Component"

    def execute(self, context: "Context") -> set[str]:
        instance = callbacks.get_active_rig_instance()
        if not instance:
            self.report(
                {"ERROR"},
                "No active rig instance found. Please select an instance from the list under the Rig Instance panel.",
            )
            return {"CANCELLED"}

        current_context = utilities.get_current_context()
        component = None
        if instance.output.component == "head":
            component = utilities.get_active_head()
        elif instance.output.component == "body":
            component = utilities.get_active_body()

        if component:
            if not bpy.path.abspath(instance.output.folder_path) and not bpy.data.filepath:
                self.report({"ERROR"}, "File must be saved to use a relative path")
                return {"CANCELLED"}

            if not component.dna_reader:
                self.report(
                    {"ERROR"},
                    f'The {component.component_type} DNA file "{component.dna_file_path}" could not be read. '
                    "Please re-select it under the Rig Instance panel.",
                )
                return {"CANCELLED"}

            dna_io_instance: DNAExporter = None  # type: ignore[assignment]
            if instance.output.method == "calibrate":
                # When re-exporting only the head, conform its neck seam onto a body
                # DNA already present in the output folder (so it matches the body the
                # user previously exported); otherwise fall back to the imported body.
                seam_reference_dna_path = None
                if component.component_type == "head":
                    body_dna_path = Path(bpy.path.abspath(instance.output.folder_path)) / "body.dna"
                    if body_dna_path.exists():
                        seam_reference_dna_path = str(body_dna_path)

                dna_io_instance = DNACalibrator(
                    instance=instance,
                    linear_modifier=component.linear_modifier,
                    file_name=f"{component.component_type}.dna",
                    component_type=component.component_type,
                    textures=False,
                    normals=instance.output.export_normals,
                    seam_reference_dna_path=seam_reference_dna_path,
                )
            elif instance.output.method == "overwrite":
                dna_io_instance = DNAExporter(
                    instance=instance,
                    linear_modifier=component.linear_modifier,
                    file_name=f"{component.component_type}.dna",
                    component_type=component.component_type,
                    textures=False,
                    normals=instance.output.export_normals,
                )

            valid, title, message, fix = dna_io_instance.run()
            ops = utilities.get_addon_ops_module()
            ops.force_evaluate()

            if not valid:
                utilities.report_error_panel(title=title, message=message, fix=fix, width=300)
                return {"CANCELLED"}
            self.report({"INFO"}, message)

        utilities.set_context(current_context)

        return {"FINISHED"}


class ReportError(bpy.types.Operator):
    bl_idname = f"{ToolInfo.NAME}.report_error"
    bl_label = "Error"

    message: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        self.report({"ERROR"}, self.message)
        return {"CANCELLED"}


class ReportErrorWithFix(bpy.types.Operator):
    """Reports and error message to the user with a optional fix"""

    bl_idname = f"{ToolInfo.NAME}.report_error_with_fix"
    bl_label = "Error"

    title: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    message: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    width: bpy.props.IntProperty(default=300)  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        addon_window_manager_properties = utilities.get_addon_window_manager_properties(context)
        fix = addon_window_manager_properties.errors.get(self.title, {}).get("fix", None)
        if fix:
            fix()
        return {"FINISHED"}

    def invoke(self, context: "Context", event: bpy.types.Event) -> set[str] | None:
        wm = context.window_manager
        if not wm:
            return None

        addon_window_manager_properties = utilities.get_addon_window_manager_properties(context)
        fix = addon_window_manager_properties.errors.get(self.title, {}).get("fix", None)
        return wm.invoke_props_dialog(self, confirm_text="Fix" if fix else "OK", cancel_default=False, width=self.width)  # type: ignore[return-value]

    def draw(self, context: "Context"):
        if not self.layout:
            return

        for line in self.title.split("\n"):
            row = self.layout.row()
            row.scale_y = 1.5
            row.label(text=line)
        for line in self.message.split("\n"):
            row = self.layout.row()
            row.alert = True
            row.label(text=line)


class MetricsCollectionConsent(bpy.types.Operator):
    """Tell the user that we collect metrics and ask for their consent"""

    bl_idname = f"{ToolInfo.NAME}.metrics_collection_consent"
    bl_label = "Character DNA Addon Metrics"

    def execute(self, context: "Context") -> set[str]:
        addon_preferences = utilities.get_addon_preferences()
        if not addon_preferences:
            return {"CANCELLED"}
        addon_preferences.metrics_collection = True
        utilities.init_sentry()
        ops = utilities.get_addon_ops_module()
        ops.force_evaluate()
        return {"FINISHED"}

    def invoke(self, context: "Context", event: bpy.types.Event) -> set[str] | None:
        wm = context.window_manager
        if not wm:
            return None

        addon_preferences = utilities.get_addon_preferences()
        if not addon_preferences:
            return {"CANCELLED"}

        current_timestamp = datetime.now(UTC).timestamp()

        if addon_preferences.metrics_collection:
            utilities.init_sentry()
            return {"FINISHED"}

        if bpy.app.online_access and addon_preferences.next_metrics_consent_timestamp < current_timestamp:
            return wm.invoke_props_dialog(self, confirm_text="Allow", cancel_default=False, width=500)  # type: ignore[return-value]
        if bpy.app.online_access and addon_preferences.metrics_collection:
            utilities.init_sentry()

        return {"FINISHED"}

    def cancel(self, context: "Context") -> None:
        addon_preferences = utilities.get_addon_preferences()
        if not addon_preferences:
            return
        # wait 30 days before asking again
        addon_preferences.next_metrics_consent_timestamp = (datetime.now(UTC) + timedelta(days=30)).timestamp()
        addon_preferences.metrics_collection = False
        ops = utilities.get_addon_ops_module()
        ops.force_evaluate()

    def draw(self, context: "Context"):
        if not self.layout:
            return

        row = self.layout.row()
        row.label(text="We collect anonymous metrics and bug reports to help improve the Character DNA addon.")
        row = self.layout.row()
        row.label(text="No personal data is collected.")
        row = self.layout.row()
        row.label(text="Will you allow us to collect bug reports?")
        row.operator("wm.url_open", text="", icon="URL").url = ToolInfo.METRICS_COLLECTION_AGREEMENT


class DuplicateRigInstance(bpy.types.Operator):
    """Duplicate the active Rig Instance. This copies all it's associated data and offsets it to the right"""

    bl_idname = f"{ToolInfo.NAME}.duplicate_rig_instance"
    bl_label = "Duplicate Rig Instance"
    bl_options = {"REGISTER", "UNDO"}

    new_name: bpy.props.StringProperty(
        name="New Name",
        default="",
        get=callbacks.get_copied_rig_instance_name,
        set=callbacks.set_copied_rig_instance_name,
    )  # pyright: ignore[reportInvalidTypeForm]
    new_folder: bpy.props.StringProperty(
        name="New Output Folder", default="", subtype="DIR_PATH", options={"PATH_SUPPORTS_BLEND_RELATIVE"}
    )  # pyright: ignore[reportInvalidTypeForm]
    copy_face_board: bpy.props.BoolProperty(
        name="Copy Face Board",
        description=(
            "Whether to copy the face board to the new rig instance. Otherwise, the new rig instance will "
            "reference the original rig instance's face board"
        ),
        default=True,
    )  # pyright: ignore[reportInvalidTypeForm]

    from .runtime.controller import native_scene_operation

    def execute(self, context: "Context") -> set[str]:
        instance = callbacks.get_active_rig_instance()
        if instance is None:
            self.report({"ERROR"}, "Select a rig instance to duplicate.")
            return {"CANCELLED"}
        if instance.get("reference_mode") in {"LINK", "EDITABLE_LINK"}:
            self.report(
                {"ERROR"}, "Linked characters cannot be duplicated. Append the character from its source file first."
            )
            return {"CANCELLED"}

        new_folder = Path(bpy.path.abspath(self.new_folder))
        if not bpy.path.abspath(self.new_folder) and not bpy.data.filepath:
            self.report({"ERROR"}, "File must be saved to use a relative path")
            return {"CANCELLED"}
        if not self.new_name:
            self.report({"ERROR"}, "You must set a new name.")
            return {"CANCELLED"}
        if not self.new_folder:
            self.report({"ERROR"}, "You must set an output folder.")
            return {"CANCELLED"}
        if not new_folder.exists():
            self.report({"ERROR"}, f"Folder not found: {new_folder}")
            return {"CANCELLED"}

        return self._duplicate(context)

    @native_scene_operation
    def _duplicate(self, context: "Context") -> set[str]:  # noqa: PLR0912, PLR0915
        new_folder = Path(bpy.path.abspath(self.new_folder))
        instance = callbacks.get_active_rig_instance()
        addon_scene_properties = utilities.get_addon_scene_properties(context)
        if instance:
            # Cache instance name before the loop because CollectionProperty.add() can
            # invalidate all existing Python wrappers to items in the collection, causing
            # access violations when reading properties from the stale reference.
            instance_name = instance.name
            for component_type, mesh_object, rig_object in [
                ("body", instance.body_mesh, instance.body_rig),
                ("head", instance.head_mesh, instance.head_rig),
            ]:
                if mesh_object and rig_object:
                    # Re-fetch instance at the start of each iteration since a previous
                    # iteration's .add() may have invalidated the Python wrapper.
                    instance = addon_scene_properties.rig_instance_list.get(instance_name)
                    if not instance:
                        break
                    new_mesh_object = utilities.copy_mesh(
                        mesh_object=mesh_object,
                        new_mesh_name=utilities.replace_instance_prefix(mesh_object.name, instance.name, self.new_name),
                        modifiers=False,
                        materials=True,
                    )
                    new_rig_object = utilities.copy_armature(
                        armature_object=rig_object,
                        new_armature_name=utilities.replace_instance_prefix(
                            rig_object.name, instance.name, self.new_name
                        ),
                    )
                    # move the new rig to the right collection
                    utilities.move_to_collection(
                        scene_objects=[new_mesh_object], collection_name=f"{self.new_name}_lod0", exclusively=True
                    )
                    # move the new rig to the right collection
                    utilities.move_to_collection(
                        scene_objects=[new_rig_object], collection_name=self.new_name, exclusively=True
                    )

                    # duplicate the mesh materials
                    new_mesh_material = utilities.copy_materials(
                        mesh_object=new_mesh_object,
                        old_prefix=instance.name,
                        new_prefix=self.new_name,
                        new_folder=new_folder / self.new_name,
                    )
                    # duplicate the texture logic node
                    if new_mesh_material:
                        texture_logic_node = getattr(callbacks, f"get_{component_type}_texture_logic_node")(
                            new_mesh_material
                        )
                        if texture_logic_node and texture_logic_node.node_tree:
                            node_name_constant = getattr(constants, f"{component_type.upper()}_TEXTURE_LOGIC_NODE_NAME")
                            new_name = f"{self.new_name}_{node_name_constant}"
                            texture_logic_node.label = new_name
                            texture_logic_node_tree_copy = texture_logic_node.node_tree.copy()
                            texture_logic_node_tree_copy.name = new_name
                            texture_logic_node.node_tree = texture_logic_node_tree_copy

                    # match the hide state of the original
                    new_mesh_object.hide_set(mesh_object.hide_get())
                    new_rig_object.hide_set(rig_object.hide_get())

                    # assign the rig to the duplicated mesh
                    modifier = new_mesh_object.modifiers.new(name="Armature", type="ARMATURE")
                    modifier.object = new_rig_object
                    new_mesh_object.parent = new_rig_object

                    # now we need to duplicate the output items
                    for item in getattr(instance.output, f"{component_type}_item_list"):
                        if item.scene_object and item.scene_object.type == "MESH":
                            if item.scene_object == mesh_object:
                                continue

                            new_extra_mesh_object = utilities.copy_mesh(
                                mesh_object=item.scene_object,
                                new_mesh_name=utilities.replace_instance_prefix(
                                    item.scene_object.name, instance.name, self.new_name
                                ),
                                modifiers=False,
                                materials=True,
                            )

                            lod_index = utilities.get_lod_index(new_extra_mesh_object.name)
                            if lod_index == -1:
                                lod_index = 0

                            # move the new mesh to the right collection
                            utilities.move_to_collection(
                                scene_objects=[new_extra_mesh_object],
                                collection_name=f"{self.new_name}_lod{lod_index}",
                                exclusively=True,
                            )
                            main_collection = bpy.data.collections.get(self.new_name)
                            lod_collection = bpy.data.collections.get(f"{self.new_name}_lod{lod_index}")
                            if main_collection and lod_collection and bpy.context.scene:
                                # unlink the lod collection from the scene collection if it exists
                                if lod_collection in bpy.context.scene.collection.children.values():
                                    bpy.context.scene.collection.children.unlink(lod_collection)
                                # link the lod collection to the main collection if it is not already linked
                                if lod_collection not in main_collection.children.values():
                                    main_collection.children.link(lod_collection)

                            # assign the rig to the duplicated extra mesh
                            modifier = new_extra_mesh_object.modifiers.new(name="Armature", type="ARMATURE")
                            modifier.object = new_rig_object
                            new_extra_mesh_object.parent = new_rig_object

                            # match the hide state of the original
                            new_extra_mesh_object.hide_set(item.scene_object.hide_get())
                            new_extra_mesh_object.hide_viewport = item.scene_object.hide_viewport

                            # duplicate the extra mesh's materials
                            utilities.copy_materials(
                                mesh_object=new_extra_mesh_object,
                                old_prefix=instance.name,
                                new_prefix=self.new_name,
                                new_folder=new_folder / self.new_name,
                            )

                    # move the duplicated rig to the right of the last mesh
                    last_instance = addon_scene_properties.rig_instance_list[-1]

                    if component_type == "body":
                        new_rig_object.location.x = utilities.get_bounding_box_left_x(last_instance.body_mesh) - (
                            utilities.get_bounding_box_width(last_instance.body_mesh) / 2
                        )

                    if component_type == "head" and last_instance.body_rig:
                        # Align the head rig with the body rig if it exists
                        body_object_head_bone = last_instance.body_rig.pose.bones.get("head")
                        head_object_head_bone = new_rig_object.pose.bones.get("head")
                        if body_object_head_bone and head_object_head_bone:
                            # get the location of the body head bone and the head head bone in world space
                            body_head_location = (
                                body_object_head_bone.id_data.matrix_world @ body_object_head_bone.matrix
                            ).to_translation()
                            head_head_location = (
                                head_object_head_bone.id_data.matrix_world @ head_object_head_bone.matrix
                            ).to_translation()
                            delta = body_head_location - head_head_location
                            # move the head rig object to align with the body rig head bone
                            new_rig_object.location += delta

                    # otherwise move it to the right of the last instance's head mesh
                    elif component_type == "head":
                        new_rig_object.location.x = utilities.get_bounding_box_left_x(last_instance.head_mesh) - (
                            utilities.get_bounding_box_width(last_instance.head_mesh) / 2
                        )

                    new_dna_file_path = new_folder / self.new_name / f"{component_type}.dna"
                    new_dna_file_path.parent.mkdir(parents=True, exist_ok=True)
                    if (
                        component_type == "head"
                        and instance.head_dna_file_path
                        and Path(bpy.path.abspath(instance.head_dna_file_path)).exists()
                    ):
                        shutil.copy(instance.head_dna_file_path, new_dna_file_path)
                    if (
                        component_type == "body"
                        and instance.body_dna_file_path
                        and Path(bpy.path.abspath(instance.body_dna_file_path)).exists()
                    ):
                        shutil.copy(instance.body_dna_file_path, new_dna_file_path)

                    # add the duplicated instance to the list if it doesn't already exist
                    for _rig_instance in addon_scene_properties.rig_instance_list:
                        if _rig_instance.name == self.new_name:
                            new_instance: "RigInstance" = _rig_instance  # noqa: UP037
                            break
                    else:
                        new_instance: "RigInstance" = addon_scene_properties.rig_instance_list.add()  # noqa: UP037

                    instance = addon_scene_properties.rig_instance_list[instance_name]

                    # now set the values on the instance
                    new_instance.name = self.new_name
                    setattr(new_instance, f"{component_type}_dna_file_path", str(new_dna_file_path))
                    new_instance.view_options.active_lod = instance.view_options.active_lod
                    new_instance.view_options.active_material_preview = instance.view_options.active_material_preview
                    new_instance.face_board = instance.face_board
                    setattr(new_instance, f"{component_type}_mesh", new_mesh_object)
                    setattr(new_instance, f"{component_type}_rig", new_rig_object)
                    setattr(new_instance, f"{component_type}_material", new_mesh_material)
                    new_instance.output.folder_path = self.new_folder

                    # set the new instance as the active one
                    addon_scene_properties.rig_instance_list_active_index = (
                        len(addon_scene_properties.rig_instance_list) - 1
                    )

                    # Duplicate the face board if copy_face_board is enabled
                    if component_type == "head" and self.copy_face_board:
                        new_face_board = utilities.duplicate_face_board(name=self.new_name)
                        # switch to pose mode on the face gui object
                        if new_face_board and bpy.context.view_layer:
                            bpy.context.view_layer.objects.active = new_face_board
                            utilities.position_face_board(
                                head_mesh_object=new_instance.head_mesh,
                                head_rig_object=new_instance.head_rig,
                                face_board_object=new_face_board,
                            )
                            utilities.move_to_collection(
                                scene_objects=[new_face_board], collection_name=self.new_name, exclusively=True
                            )
                            utilities.switch_to_pose_mode(new_face_board)
                            # constrain the face board to the head rig
                            utilities.constrain_face_board_to_head(
                                face_board_object=new_face_board,
                                head_rig_object=new_instance.head_rig,
                                body_rig_object=new_instance.body_rig,
                                bone_name="CTRL_faceGUI",
                            )
                            utilities.constrain_face_board_to_head(
                                face_board_object=new_face_board,
                                head_rig_object=new_instance.head_rig,
                                body_rig_object=new_instance.body_rig,
                                bone_name="CTRL_C_eyesAim",
                            )
                            # Assign the new face board to the new instance
                            new_instance.face_board = new_face_board

                    if component_type == "head":
                        # constrain the head rig to the body rig with a copy transforms constraint
                        utilities.constrain_head_to_body(new_instance)

        # notify all handlers that the rig instance list has been updated
        utilities.notify_rig_instances_changed(addon_scene_properties.rig_instance_list.get(self.new_name))
        return {"FINISHED"}

    def invoke(self, context: "Context", event: bpy.types.Event) -> set[str] | None:
        return context.window_manager.invoke_props_dialog(self, width=450)  # type: ignore[return-value]

    @classmethod
    def poll(cls, _: "Context") -> bool:
        return callbacks.get_active_rig_instance() is not None

    def draw(self, context: "Context"):
        if not self.layout:
            return

        self.layout.prop(self, "new_name")
        self.layout.prop(self, "new_folder")
        self.layout.prop(self, "copy_face_board")


class AddHeadTextureLogicNode(bpy.types.Operator):
    """Add a new Rig Logic Texture Node to the active material. This is used to control the wrinkle map blending on Metahuman faces"""  # noqa: E501

    bl_idname = f"{ToolInfo.NAME}.add_head_texture_logic_node"
    bl_label = "Add Rig Logic Texture Node"

    @classmethod
    def get_active_material(cls, context: "Context") -> bpy.types.Material | None:
        space = context.space_data
        if space and space.type == "NODE_EDITOR":
            node_tree = space.node_tree  # type: ignore[attr-defined]
            for material in bpy.data.materials:
                if material.node_tree == node_tree:
                    return material
        return None

    @classmethod
    def poll(cls, context: "Context") -> bool | None:
        space = context.space_data
        node_tree = getattr(space, "node_tree", None)
        if node_tree and node_tree.type == "SHADER":
            active_material = cls.get_active_material(context)
            if not active_material:
                return False

            return bool(not callbacks.get_head_texture_logic_node(active_material))
        return None

    def execute(self, context: "Context") -> set[str]:
        space = context.space_data
        node_tree = space.node_tree  # type: ignore[attr-defined]
        cursor_location = space.cursor_location  # type: ignore[attr-defined]

        active_material = self.get_active_material(context)
        if not active_material:
            self.report({"ERROR"}, "Could not find the active material")
            return {"CANCELLED"}

        texture_logic_node = utilities.import_head_texture_logic_node()
        if not texture_logic_node:
            self.report({"ERROR"}, "Could not import the Texture Logic Node")
            return {"CANCELLED"}

        node = node_tree.nodes.new(type="ShaderNodeGroup")
        node.name = f"{active_material.name}_{HEAD_TEXTURE_LOGIC_NODE_NAME}"
        node.label = f"{active_material.name} {HEAD_TEXTURE_LOGIC_NODE_LABEL}"
        node.node_tree = texture_logic_node
        node.location = cursor_location

        # apply the same masks, placeholder textures, and node settings corrections
        # that a material import performs so the manually added node evaluates correctly
        utilities.setup_texture_logic_node_groups("head")
        return {"FINISHED"}


class UILIST_ADDON_PREFERENCES_OT_extra_dna_entry_remove(GenericUIListOperator, bpy.types.Operator):
    """Remove the selected entry from the list"""

    bl_idname = f"{ToolInfo.NAME}.addon_preferences_extra_dna_entry_remove"
    bl_label = "Remove Selected Entry"

    def execute(self, context: "Context") -> set[str]:
        addon_preferences = utilities.get_addon_preferences()
        if not addon_preferences:
            return {"CANCELLED"}
        my_list = addon_preferences.extra_dna_folder_list
        active_index = addon_preferences.extra_dna_folder_list_active_index
        my_list.remove(active_index)
        to_index = min(active_index, len(my_list) - 1)
        addon_preferences.extra_dna_folder_list_active_index = to_index
        return {"FINISHED"}


class UILIST_ADDON_PREFERENCES_OT_extra_dna_entry_add(GenericUIListOperator, bpy.types.Operator):
    """Add an entry to the list after the current active item"""

    bl_idname = f"{ToolInfo.NAME}.addon_preferences_extra_dna_entry_add"
    bl_label = "Add Entry"

    def execute(self, context: "Context") -> set[str]:
        addon_preferences = utilities.get_addon_preferences()
        if not addon_preferences:
            return {"CANCELLED"}
        my_list = addon_preferences.extra_dna_folder_list
        active_index = addon_preferences.extra_dna_folder_list_active_index
        to_index = min(len(my_list), active_index + 1)
        my_list.add()
        my_list.move(len(my_list) - 1, to_index)
        addon_preferences.extra_dna_folder_list_active_index = to_index
        return {"FINISHED"}


class FaceBoardSearchPose(bpy.types.Operator):
    """Search face poses by name and set the current pose"""

    bl_idname = f"{ToolInfo.NAME}.face_board_search_pose"
    bl_label = "Search Poses"
    bl_property = "pose_enum"
    bl_options = {"INTERNAL"}

    pose_enum: bpy.props.EnumProperty(
        name="Pose",
        description="The face pose to set",
        items=callbacks.get_face_pose_search_items,  # type: ignore[arg-type]
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        addon_scene_properties = utilities.get_addon_scene_properties(context)
        addon_scene_properties.face_board.face_pose_previews = self.pose_enum
        return {"FINISHED"}

    def invoke(self, context: "Context", event: "bpy.types.Event") -> set[str]:
        context.window_manager.invoke_search_popup(self)
        return {"FINISHED"}


class UILIST_RIG_INSTANCE_OT_entry_remove(GenericUIListOperator, bpy.types.Operator):
    """Remove the selected entry from the list"""

    bl_idname = f"{ToolInfo.NAME}.rig_instance_entry_remove"
    bl_label = "Remove Selected Entry"

    delete_associated_data: bpy.props.BoolProperty(
        name="Delete associated data",
        description="Delete all associated objects and collections linked to this rig instance",
        default=True,
    )  # pyright: ignore[reportInvalidTypeForm]

    @staticmethod
    def _associated_data(
        instance: "RigInstance",
    ) -> tuple[list[bpy.types.Collection], set[bpy.types.Object], set[bpy.types.Image]]:
        objects = set()
        images = set()
        if "reference_root" in instance:
            root = instance.get("reference_root")
            collections = [root] if root else []
        else:
            collections = []
            for component_type in ("body", "head"):
                for item in getattr(instance.output, f"{component_type}_item_list"):
                    if item.scene_object:
                        objects.add(item.scene_object)
                    if item.image_object:
                        images.add(item.image_object)
                for collection_name in [instance.name] + [
                    f"{instance.name}_{component_type}_lod{index}" for index in range(NUMBER_OF_HEAD_LODS)
                ]:
                    collection = bpy.data.collections.get((collection_name, None))
                    if collection and collection not in collections:
                        collections.append(collection)
        for collection in list(collections):
            for child in collection.children_recursive:
                if child not in collections:
                    collections.append(child)
        return collections, objects, images

    def execute(self, context: "Context") -> set[str]:
        from .runtime import engine

        addon_scene_properties = utilities.get_addon_scene_properties(context)
        my_list = addon_scene_properties.rig_instance_list
        active_index = self._resolve_active_index(my_list)
        if active_index is None:
            self.report({"WARNING"}, "There is no rig instance to remove")
            return {"CANCELLED"}

        instance = my_list[active_index]
        engine.release_records(instance)
        objects = set()
        images = set()
        if self.delete_associated_data:
            collections, objects, images = self._associated_data(instance)
            for collection in reversed(collections):
                if collection.library or collection.override_library:
                    continue
                for owner in list(collection.objects):
                    objects.add(owner)
                    collection.objects.unlink(owner)
                bpy.data.collections.remove(collection, do_unlink=True)

        my_list.remove(active_index)
        unused = {owner for owner in objects if owner.library is None and owner.users == 0}
        while unused:
            objects.difference_update(unused)
            bpy.data.batch_remove(ids=unused)
            unused = {owner for owner in objects if owner.library is None and owner.users == 0}
        for image in images:
            if image.library is None and image.users == 0:
                bpy.data.images.remove(image)
        to_index = min(active_index, len(my_list) - 1)
        addon_scene_properties.rig_instance_list_active_index = to_index
        # notify registered callbacks that a rig instance was removed from the list
        utilities.notify_rig_instances_changed()
        return {"FINISHED"}

    def invoke(self, context: "Context", event: bpy.types.Event) -> set[str] | None:
        addon_scene_properties = utilities.get_addon_scene_properties(context)
        my_list = addon_scene_properties.rig_instance_list
        active_index = self._resolve_active_index(my_list)
        if active_index is None:
            self.report({"WARNING"}, "There is no rig instance to remove")
            return {"CANCELLED"}

        self.instance_name = my_list[active_index].name
        return context.window_manager.invoke_props_dialog(  # type: ignore[return-value]
            self, title=f"Remove: {self.instance_name}", confirm_text="Remove", width=400
        )

    def draw(self, context: "Context"):
        if not self.layout:
            return

        row = self.layout.row()
        row.label(text=f"Are you sure you want to remove the '{self.instance_name}' rig instance?", icon="ERROR")
        row = self.layout.row()
        row.prop(self, "delete_associated_data")


class UILIST_RIG_INSTANCE_OT_entry_add(GenericUIListOperator, bpy.types.Operator):
    """Add an entry to the list after the current active item"""

    bl_idname = f"{ToolInfo.NAME}.rig_instance_entry_add"
    bl_label = "Add Entry"

    def execute(self, context: "Context") -> set[str]:
        utilities.add_rig_instance()
        return {"FINISHED"}

    @classmethod
    def poll(cls, _: "Context") -> bool:
        return utilities.dependencies_are_valid()


class UILIST_RIG_INSTANCE_OT_entry_move(GenericUIListOperator, bpy.types.Operator):
    """Move an entry in the list up or down"""

    bl_idname = f"{ToolInfo.NAME}.rig_instance_entry_move"
    bl_label = "Move Entry"

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=(
            ("UP", "UP", "UP"),
            ("DOWN", "DOWN", "DOWN"),
        ),
        default="UP",
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: "Context") -> set[str]:
        addon_scene_properties = utilities.get_addon_scene_properties(context)
        my_list = addon_scene_properties.rig_instance_list
        active_index = self._resolve_active_index(my_list)
        if active_index is None:
            self.report({"WARNING"}, "There is no rig instance to move")
            return {"CANCELLED"}

        delta = {
            "DOWN": 1,
            "UP": -1,
        }[self.direction]

        to_index = (active_index + delta) % len(my_list)

        from_instance = my_list[active_index]
        to_instance = my_list[to_index]

        if from_instance.body_rig and to_instance.body_rig:
            to_x = to_instance.body_rig.location.x
            from_x = from_instance.body_rig.location.x

            # swap the x locations of the body rigs
            to_instance.body_rig.location.x = from_x
            from_instance.body_rig.location.x = to_x
            # swap the x locations of the head rigs
            to_instance.head_rig.location.x = from_x
            from_instance.head_rig.location.x = to_x
            # swap the x locations of the face boards
            to_instance.face_board.location.x += from_x - to_x
            from_instance.face_board.location.x += to_x - from_x

        elif from_instance.head_rig and to_instance.head_rig:
            to_x = to_instance.head_rig.location.x
            from_x = from_instance.head_rig.location.x

            # swap the x locations of the head rigs
            to_instance.head_rig.location.x = from_x
            from_instance.head_rig.location.x = to_x
            # swap the x locations of the face boards
            to_instance.face_board.location.x += from_x - to_x
            from_instance.face_board.location.x += to_x - from_x

        my_list.move(active_index, to_index)
        addon_scene_properties.rig_instance_list_active_index = to_index
        return {"FINISHED"}
