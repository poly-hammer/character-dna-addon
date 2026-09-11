# standard library imports
from pathlib import Path

# third party imports
import bpy

from bl_ui.generic_ui_list import draw_ui_list

# local imports
from ..constants import PRO_EDITORS, PanelOrder, ToolInfo
from ..typing import *  # noqa: F403
from .callbacks import is_face_board_readonly, is_readonly_id, is_reference_readonly


def get_active_rig_instance() -> "RigInstance | None":
    # Avoid circular import
    from ..ui.callbacks import get_active_rig_instance as _get_active_rig_instance

    return _get_active_rig_instance()


def dependencies_are_valid() -> bool:
    # Avoid circular import
    from ..utilities import dependencies_are_valid as _dependencies_are_valid

    return _dependencies_are_valid()


def valid_rig_instance_exists(_: "Context", ignore_face_board: bool = False) -> str:  # noqa: PLR0911
    instance = get_active_rig_instance()
    if instance:
        if not instance.face_board and not ignore_face_board:
            return f'"{instance.name}" Has No Face Board set.'

        if instance.head_mesh or instance.head_rig:
            if not instance.head_dna_file_path:
                return f'"{instance.name}" Has no head DNA file set.'
            if not Path(bpy.path.abspath(instance.head_dna_file_path)).exists():
                return f'"{instance.name}" head DNA file is not found on disk.'
            if Path(bpy.path.abspath(instance.head_dna_file_path)).suffix.lower() != ".dna":
                return f'"{instance.name}" head DNA file must be a binary .dna file.'
        elif instance.body_mesh or instance.body_rig:
            if not instance.body_dna_file_path:
                return f'"{instance.name}" Has no body DNA file set.'
            if not Path(bpy.path.abspath(instance.body_dna_file_path)).exists():
                return f'"{instance.name}" body DNA file is not found on disk.'
            if Path(bpy.path.abspath(instance.body_dna_file_path)).suffix.lower() != ".dna":
                return f'"{instance.name}" body DNA file must be a binary .dna file.'

        return ""
    return "Missing data. Create/Import DNA data."


def draw_rig_instance_error(layout: bpy.types.UILayout, error: str):
    # Validate installed dependencies.
    if not dependencies_are_valid():
        row = layout.row()
        row.alert = True
        row.label(text="Dependencies are missing.", icon="ERROR")
        row = layout.row()
        row.operator("wm.url_open", text="Show Me How to Fix This?", icon="URL").url = ToolInfo.HOW_TO_INSTALL
        return

    row = layout.row()
    row.label(text="Rig Instance Error:", icon="ERROR")
    row = layout.row()
    row.alignment = "CENTER"
    row.alert = True
    row.label(text=error)


class RigInstanceDependentPanel(bpy.types.Panel):
    @classmethod
    def poll(cls, context: "Context") -> bool:
        error = valid_rig_instance_exists(context, ignore_face_board=True)
        return bool(not error)


class CHARACTER_DNA_UL_output_items(bpy.types.UIList):
    def draw_item(
        self,
        context: "Context",
        layout: bpy.types.UILayout,
        data: "CharacterOutputProperties",
        item: "OutputData",
        icon: int | None,
        active_data: "CharacterOutputProperties",
        active_prop_name: str,
    ):
        layout.separator(factor=0.1)
        layout.prop(item, "include", text="")

        item_icon = "MESH_DATA"
        prop_name = "scene_object"
        if item.scene_object and item.scene_object.type == "ARMATURE":
            item_icon = "ARMATURE_DATA"
        elif item.image_object:
            item_icon = "IMAGE_DATA"
            prop_name = "image_object"

        if item.editable_name:
            layout.prop(item, "name", text="", emboss=False, icon=item_icon)
        else:
            layout.label(text=item.name, icon=item_icon)

        row = layout.row()
        row.prop(item, prop_name, text="", emboss=False)


class CHARACTER_DNA_UL_rig_instances(bpy.types.UIList):
    def draw_item(
        self,
        context: "Context",
        layout: bpy.types.UILayout,
        data: "CharacterSceneProperties",
        item: "RigInstance",
        icon: int | None,
        active_data: "CharacterSceneProperties",
        active_prop_name: str,
    ):
        layout.enabled = not is_reference_readonly(item)
        layout.prop(item, "auto_evaluate", text="")

        row = layout.row()
        row.enabled = True

        row.prop(item, "name", text="", emboss=False, icon="NETWORK_DRIVE")
        row.alignment = "RIGHT"

        col = row.column(align=True)
        col.enabled = item.auto_evaluate and (item.auto_evaluate_head or item.auto_evaluate_body)
        col.alert = not item.evaluate_bones
        col.prop(item, "evaluate_bones", text="", icon="BONE_DATA", emboss=False)

        col = row.column(align=True)
        col.enabled = item.auto_evaluate and (item.auto_evaluate_head or item.auto_evaluate_body)
        col.alert = not item.evaluate_shape_keys
        col.prop(item, "evaluate_shape_keys", text="", icon="SHAPEKEY_DATA", emboss=False)

        col = row.column(align=True)
        col.enabled = item.auto_evaluate and (item.auto_evaluate_head or item.auto_evaluate_body)
        col.alert = not item.evaluate_texture_masks
        col.prop(item, "evaluate_texture_masks", text="", icon="NODE_TEXTURE", emboss=False)

        col = row.column(align=True)
        col.enabled = item.auto_evaluate and (item.auto_evaluate_head or item.auto_evaluate_body)
        col.alert = not item.evaluate_rbfs
        col.prop(item, "evaluate_rbfs", text="", icon="DRIVER_ROTATIONAL_DIFFERENCE", emboss=False)


class CHARACTER_DNA_PT_face_pose_tags(bpy.types.Panel):
    bl_label = "Tags"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_options = {"INSTANCED"}

    def draw(self, context: "Context"):
        if not self.layout:
            return

        from ..ui.callbacks import get_face_pose_tag_property_map, get_tags_for_category

        scene_properties = getattr(context.scene, ToolInfo.NAME)
        face_board = scene_properties.face_board

        layout = self.layout
        layout.enabled = not is_face_board_readonly(get_active_rig_instance())
        row = layout.row(align=True)
        row.label(text="Match")
        row.prop(face_board, "tag_match_mode", expand=True)

        tag_property_map = get_face_pose_tag_property_map()
        # Only show tags that belong to the actively selected category so out-of-category
        # tags can't be toggled (which would filter the pose list to nothing).
        allowed_tags = set(get_tags_for_category(face_board.category))
        visible_properties = [
            property_name for property_name, tag_name in tag_property_map.items() if tag_name in allowed_tags
        ]
        if not visible_properties:
            layout.label(text="No tags found", icon="INFO")
            return

        column = layout.column(align=True)
        for property_name in visible_properties:
            column.prop(face_board, property_name)


class CHARACTER_DNA_PT_face_board(RigInstanceDependentPanel):
    bl_label = "Face Board"
    bl_category = "Character DNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = PanelOrder.FACE_BOARD.value

    def draw(self, context: "Context"):
        if not self.layout:
            return

        instance = get_active_rig_instance()
        self.layout.enabled = not is_face_board_readonly(instance)
        if instance and instance.is_pro and instance.raw_control_editor.is_editing:
            self.layout.enabled = False

        error = valid_rig_instance_exists(context)
        if not error:
            scene_properties = getattr(context.scene, ToolInfo.NAME)
            face_board = scene_properties.face_board

            row = self.layout.row(align=True)
            row.prop(face_board, "category", text="")
            row.popover(panel="CHARACTER_DNA_PT_face_pose_tags", text="", icon="FILTER")

            self.layout.label(text="Poses:")
            self.layout.template_icon_view(face_board, "face_pose_previews", show_labels=True, scale_popup=5.0)
            row = self.layout.row(align=True)
            row.prop(face_board, "face_pose_previews", text="")
            row.operator(f"{ToolInfo.NAME}.face_board_search_pose", text="", icon="VIEWZOOM")

            column = self.layout.column(align=True)
            column.prop(face_board, "use_eye_aim")
            column.prop(face_board, "eyes_follow_head")
            column.prop(face_board, "face_board_follow_head")
        else:
            draw_rig_instance_error(self.layout, error)


class CHARACTER_DNA_PT_face_board_footer(bpy.types.Panel):
    """Footer subpanel pinned to the bottom of the Face Board panel, holding the
    Map Raw to GUI Controls action button.

    ``HIDE_HEADER`` keeps the button drawn inline (no collapsible header)."""

    bl_label = "(Not Shown)"
    bl_idname = "CHARACTER_DNA_PT_face_board_footer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_parent_id = "CHARACTER_DNA_PT_face_board"
    bl_options = {"HIDE_HEADER"}

    def draw(self, context: "Context") -> None:
        layout = self.layout
        if layout is None:
            return
        # Only show the action once a valid rig instance exists (matches the
        # parent panel, which otherwise draws an error message).
        if valid_rig_instance_exists(context):
            return

        # Greyed out while the Raw Control Editor holds an edit session, exactly
        # as the button behaved when it lived inside the parent panel's draw.
        instance = get_active_rig_instance()
        layout.enabled = not is_reference_readonly(instance) and not is_face_board_readonly(instance)
        if instance and instance.is_pro and instance.raw_control_editor.is_editing:
            layout.enabled = False

        if instance and instance.is_pro:
            row = layout.row()
            row.scale_y = 1.5
            row.operator(f"{ToolInfo.NAME}.map_raw_to_gui_controls", icon="UV_SYNC_SELECT")


class CHARACTER_DNA_PT_animation_panel(bpy.types.Panel):
    bl_label = "Animation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = PanelOrder.ANIMATION.value

    @classmethod
    def poll(cls, context: "Context") -> bool:
        error = valid_rig_instance_exists(context, ignore_face_board=True)
        if error:
            return False

        instance = get_active_rig_instance()
        if not instance:
            return False

        return bool(instance.face_board or instance.body_rig)

    def draw(self, context: "Context"):
        error = valid_rig_instance_exists(context, ignore_face_board=True)

        if not self.layout:
            return

        if not error:
            instance = get_active_rig_instance()
            box = self.layout.box()
            box.enabled = not is_face_board_readonly(instance)
            box.label(text="Face Board:")
            split = box.split(factor=0.5)
            split.scale_y = 1.5
            split.operator(f"{ToolInfo.NAME}.import_face_board_animation", icon="IMPORT", text="Import")
            column = split.column()
            column.enabled = not is_reference_readonly(instance)
            column.operator(f"{ToolInfo.NAME}.bake_face_board_animation", icon="ACTION", text="Bake")

            box = self.layout.box()
            box.enabled = not is_reference_readonly(instance)
            box.label(text="Body Animation:")
            component_type = "body"
            row = box.row()
            row.scale_y = 1.5
            split = row.split(factor=0.5)
            split.operator(
                f"{ToolInfo.NAME}.import_component_animation",
                icon="IMPORT",
                text="Import",
            ).component_type = component_type
            split.operator(
                f"{ToolInfo.NAME}.bake_component_animation",
                icon="ACTION",
                text="Bake",
            ).component_type = component_type


class CHARACTER_DNA_PT_view_options(RigInstanceDependentPanel):
    bl_label = "View Options"
    bl_category = "Character DNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = PanelOrder.VIEW_OPTIONS.value

    def draw(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        error = valid_rig_instance_exists(context, ignore_face_board=True)
        if not error:
            active_index = properties.rig_instance_list_active_index
            instance = properties.rig_instance_list[active_index]
            readonly = is_reference_readonly(instance)
            self.layout.enabled = instance.get("reference_mode") != "LINK"
            # Resolve the linked objects once. The visibility property getters each do a
            # relatively expensive RNA path round-trip, so we read the icon state directly
            # from the objects here instead of re-invoking the getters in the icon ternary.
            view_options = instance.view_options
            head_rig = instance.head_rig
            body_rig = instance.body_rig
            face_board = instance.face_board
            control_rig = instance.control_rig
            grid = self.layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
            col = grid.column()
            col.enabled = bool(instance.head_material)
            col.label(text="Head Material Color:")
            row = col.row()
            row.enabled = not readonly and not is_readonly_id(instance.head_material)
            row.prop(view_options, "active_material_preview", text="")
            row = col.row()
            row.label(text="Bone Visibility:")
            row = col.row()
            row.enabled = bool(head_rig) and not readonly
            row.prop(
                view_options,
                "show_head_bones",
                text="Head Bones",
                icon="HIDE_OFF" if head_rig and not head_rig.hide_get() else "HIDE_ON",
            )
            row = col.row()
            row.enabled = bool(body_rig) and not readonly
            row.prop(
                view_options,
                "show_body_bones",
                text="Body Bones",
                icon="HIDE_OFF" if body_rig and not body_rig.hide_get() else "HIDE_ON",
            )

            # Bone Visibility toggles for matching, volume, and internal bones.
            row = col.row()
            row.enabled = view_options.show_head_bones and not readonly
            row.prop(instance.view_options, "hide_volume_bones", text="Hide Volume")

            row = col.row()
            row.enabled = view_options.show_head_bones and not readonly
            row.prop(instance.view_options, "solo_internal_bones", text="Solo Internal")

            row = col.row()
            row.prop(properties, "highlight_matching_active_bone", text="Show Matching")

            col = grid.column()
            col.enabled = bool(instance.head_mesh)
            col.label(text="Active LOD:")
            row = col.row()
            row.enabled = not readonly
            row.prop(view_options, "active_lod", text="")
            row = col.row()
            row.label(text="Control Visibility:")
            row = col.row()
            row.enabled = bool(face_board) and not is_face_board_readonly(instance)
            row.prop(
                view_options,
                "show_face_board",
                text="Face Board",
                icon="HIDE_OFF" if face_board and not face_board.hide_get() else "HIDE_ON",
            )
            row = col.row()
            row.enabled = bool(control_rig) and not is_readonly_id(control_rig)
            row.prop(
                view_options,
                "show_control_rig",
                text="Control Rig",
                icon="HIDE_OFF" if control_rig and not control_rig.hide_get() else "HIDE_ON",
            )
        else:
            draw_rig_instance_error(self.layout, error)


class CHARACTER_DNA_PT_rig_instance(bpy.types.Panel):
    bl_label = "Rig Instances"
    bl_category = "Character DNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = PanelOrder.RIG_INSTANCES.value

    def draw(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        row = self.layout.row()
        col = draw_ui_list(
            row,
            context,  # type: ignore[arg-type]
            class_name="CHARACTER_DNA_UL_rig_instances",
            list_path=f"scene.{ToolInfo.NAME}.rig_instance_list",
            active_index_path=f"scene.{ToolInfo.NAME}.rig_instance_list_active_index",
            unique_id="rig_instance_list_id",
            insertion_operators=False,
            move_operators=False,  # type: ignore[arg-type]
        )

        enabled = len(properties.rig_instance_list) > 0

        # plus and minus buttons
        row = col.row()
        props = row.operator(f"{ToolInfo.NAME}.rig_instance_entry_add", text="", icon="ADD")
        props.active_index = properties.rig_instance_list_active_index

        row = col.row()
        row.enabled = enabled
        props = row.operator(f"{ToolInfo.NAME}.rig_instance_entry_remove", text="", icon="REMOVE")
        props.active_index = properties.rig_instance_list_active_index

        if enabled:
            row = col.row()
            row.enabled = not is_reference_readonly(
                properties.rig_instance_list[properties.rig_instance_list_active_index]
            )
            row.operator(f"{ToolInfo.NAME}.duplicate_rig_instance", icon="DUPLICATE", text="")

            row = col.row()
            props = row.operator(f"{ToolInfo.NAME}.rig_instance_entry_move", text="", icon="TRIA_UP")
            props.direction = "UP"
            props.active_index = properties.rig_instance_list_active_index

            row = col.row()
            props = row.operator(f"{ToolInfo.NAME}.rig_instance_entry_move", text="", icon="TRIA_DOWN")
            props.direction = "DOWN"
            props.active_index = properties.rig_instance_list_active_index

            row = self.layout.row()
            row.label(text="Rig Logic Linked Data:")


class CHARACTER_DNA_PT_rig_instance_head_sub_panel(bpy.types.Panel):
    bl_parent_id = "CHARACTER_DNA_PT_rig_instance"
    bl_label = ""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        addon_scene_properties = getattr(context.scene, ToolInfo.NAME)
        return len(addon_scene_properties.rig_instance_list) > 0

    def draw_header(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        active_index = properties.rig_instance_list_active_index
        if len(properties.rig_instance_list) > 0:
            instance = properties.rig_instance_list[active_index]
            row = self.layout.row()
            row.enabled = instance.auto_evaluate and not is_reference_readonly(instance)
            row.prop(instance, "auto_evaluate_head", text="Head")

    def draw(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        active_index = properties.rig_instance_list_active_index
        if len(properties.rig_instance_list) > 0:
            instance = properties.rig_instance_list[active_index]

            box = self.layout.box()
            box.enabled = not is_reference_readonly(instance)
            row = box.row()
            row.alert = False
            bad_path = instance.head_dna_file_path and not Path(bpy.path.abspath(instance.head_dna_file_path)).exists()
            if not instance.head_dna_file_path or bad_path:
                row.alert = True
            row.prop(instance, "head_dna_file_path", icon="RNA")
            if bad_path:
                row = box.row()
                row.alert = True
                row.label(text="DNA File not found on disk.", icon="ERROR")
            row = box.row()
            row.alert = False
            if not instance.face_board:
                row.alert = True
            row.prop(instance, "face_board", icon="PIVOT_BOUNDBOX")
            row = box.row()
            row.prop(instance, "head_mesh", icon="OUTLINER_OB_MESH")
            row = box.row()
            row.prop(instance, "head_rig", icon="OUTLINER_OB_ARMATURE")
            row = box.row()
            row.prop(instance, "head_material", icon="MATERIAL")


class CHARACTER_DNA_PT_rig_instance_body_sub_panel(bpy.types.Panel):
    bl_parent_id = "CHARACTER_DNA_PT_rig_instance"
    bl_label = ""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        addon_scene_properties = getattr(context.scene, ToolInfo.NAME)
        return len(addon_scene_properties.rig_instance_list) > 0

    def draw_header(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        active_index = properties.rig_instance_list_active_index
        if len(properties.rig_instance_list) > 0:
            instance = properties.rig_instance_list[active_index]
            row = self.layout.row()
            row.enabled = instance.auto_evaluate and not is_reference_readonly(instance)
            row.prop(instance, "auto_evaluate_body", text="Body")

    def draw(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        active_index = properties.rig_instance_list_active_index
        if len(properties.rig_instance_list) > 0:
            instance = properties.rig_instance_list[active_index]

            box = self.layout.box()
            box.enabled = not is_reference_readonly(instance)
            row = box.row()
            row.alert = False
            bad_path = instance.body_dna_file_path and not Path(bpy.path.abspath(instance.body_dna_file_path)).exists()
            if not instance.body_dna_file_path or bad_path:
                row.alert = True
            row.prop(instance, "body_dna_file_path", icon="RNA")
            if bad_path:
                row = box.row()
                row.alert = True
                row.label(text="DNA File not found on disk.", icon="ERROR")
            row = box.row()
            row.prop(instance, "control_rig", icon="CON_ARMATURE")
            row = box.row()
            row.prop(instance, "body_mesh", icon="OUTLINER_OB_MESH")
            row = box.row()
            row.prop(instance, "body_rig", icon="OUTLINER_OB_ARMATURE")
            row = box.row()
            row.prop(instance, "body_material", icon="MATERIAL")


class CHARACTER_DNA_PT_rig_instance_footer_sub_panel(RigInstanceDependentPanel):
    bl_parent_id = "CHARACTER_DNA_PT_rig_instance"
    bl_label = "(Not Shown)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_options = {"HIDE_HEADER"}

    def draw(self, context: "Context"):
        if not self.layout:
            return

        instance = get_active_rig_instance()
        if not instance:
            return

        row = self.layout.row()
        row.enabled = not is_reference_readonly(instance)
        row.prop(instance, "head_to_body_constraint_influence", text="Head to Body Constraint Influence")
        self.layout.separator()
        row = self.layout.row()
        row.scale_y = 1.5
        row.operator(f"{ToolInfo.NAME}.force_evaluate", icon="FILE_REFRESH")


class CHARACTER_DNA_PT_output_panel(RigInstanceDependentPanel):
    """
    This class defines the user interface for the panel in the tab in the 3d view
    """

    bl_label = "Output"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_order = PanelOrder.OUTPUT.value

    def draw(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        error = valid_rig_instance_exists(context, ignore_face_board=True)
        if not error:
            active_index = properties.rig_instance_list_active_index
            instance = properties.rig_instance_list[active_index]
            self.layout.enabled = not is_reference_readonly(instance)
            grid = self.layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
            col = grid.column()
            col.label(text="Component:")
            row = col.row()
            row.prop(instance.output, "component", text="")
            col = grid.column()
            col.label(text="Method:")
            row = col.row()
            row.prop(instance.output, "method", text="")

            row = self.layout.row()
            column = None
            if instance.output.component == "head":
                column = draw_ui_list(
                    row,
                    context,  # type: ignore[arg-type]
                    class_name="CHARACTER_DNA_UL_output_items",
                    list_path=f"scene.{ToolInfo.NAME}.rig_instance_list[{active_index}].output.head_item_list",
                    active_index_path=f"scene.{ToolInfo.NAME}.rig_instance_list[{active_index}].output.head_item_active_index",
                    unique_id="output_head_item_list_id",
                    move_operators=False,  # type: ignore[arg-type]
                    insertion_operators=False,
                )
            elif instance.output.component == "body":
                column = draw_ui_list(
                    row,
                    context,  # type: ignore[arg-type]
                    class_name="CHARACTER_DNA_UL_output_items",
                    list_path=f"scene.{ToolInfo.NAME}.rig_instance_list[{active_index}].output.body_item_list",
                    active_index_path=f"scene.{ToolInfo.NAME}.rig_instance_list[{active_index}].output.body_item_active_index",
                    unique_id="output_body_item_list_id",
                    move_operators=False,  # type: ignore[arg-type]
                    insertion_operators=False,
                )
            # Refresh button aligned in the list's side column
            if column is not None:
                column.operator(f"{ToolInfo.NAME}.refresh_output_items", text="", icon="FILE_REFRESH")
            row = self.layout.row()
            row.label(text="Output Folder:")
            row = self.layout.row()
            if not instance.output.folder_path:
                row.alert = True
            row.prop(instance.output, "folder_path", text="", icon="RNA")
            if not instance.output.folder_path:
                row = self.layout.row()
                row.alert = True
                row.label(text="Must set an output folder.", icon="ERROR")
        else:
            draw_rig_instance_error(self.layout, error)


class CHARACTER_DNA_PT_output_buttons_sub_panel(bpy.types.Panel):
    bl_parent_id = "CHARACTER_DNA_PT_output_panel"
    bl_label = "(Not Shown)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Character DNA"
    bl_options = {"HIDE_HEADER"}

    def draw(self, context: "Context"):
        if not self.layout:
            return

        properties = getattr(context.scene, ToolInfo.NAME)
        error = valid_rig_instance_exists(context, ignore_face_board=True)
        row = self.layout.row()
        if not error:
            row.label(text="Export:")
            row = self.layout.row()
            active_index = properties.rig_instance_list_active_index
            instance = properties.rig_instance_list[active_index]
            self.layout.enabled = not is_reference_readonly(instance)
            row.prop(instance.output, "run_validations")
            if instance.is_pro:
                row.prop(instance.output, "auto_update_lods", text="Update LODs")
            row = self.layout.row()

            if instance.output.method == "calibrate":
                row.prop(instance.output, "align_head_and_body")
                row = self.layout.row()

            if not instance.output.folder_path:
                row.enabled = False
            row.scale_y = 2.0
            row.operator(f"{ToolInfo.NAME}.export_selected_component", icon="EXPORT", text="Only Component")
            row.operator(f"{ToolInfo.NAME}.send_to_meta_human_creator", icon="UV_SYNC_SELECT", text="MetaHuman Creator")


class CHARACTER_DNA_PT_migrate_legacy_data(bpy.types.Panel):
    bl_label = "Migrate Legacy Data"
    bl_category = "Character DNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"HEADER_LAYOUT_EXPAND"}
    bl_order = PanelOrder.MIGRATE_LEGACY_DATA.value

    @classmethod
    def poll(cls, context: "Context") -> bool:
        from ..runtime.ui_refresh import migration_needed

        return migration_needed(context.scene)

    def draw(self, context: "Context"):
        if not self.layout:
            return

        row = self.layout.row()
        row.alert = True
        row.label(text="Legacy data detected", icon="ERROR")
        row = self.layout.row()
        row.label(text="You must migrate your .blend file then save it.")
        row = self.layout.row()
        row.scale_y = 1.5
        row.operator(f"{ToolInfo.NAME}.migrate_legacy_data", icon="FILE_NEW", text="Migrate Now")


class CHARACTER_DNA_PT_pro_upsell(bpy.types.Panel):
    """Advertise the Pro editor tools.

    Always registered, but only shown when the Pro editor UI is *not* visible:
    either this is the free edition (no ``editors`` submodule) or a Pro user has
    toggled ``show_pro_features`` off to preview the free edition's UI.
    """

    bl_label = "Upgrade to Pro"
    bl_category = "Character DNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = PanelOrder.PRO_UPSELL.value

    @classmethod
    def poll(cls, _: "Context") -> bool:
        from ..utilities import pro_features_visible

        return not pro_features_visible()

    def draw_header(self, _: "Context"):
        self.layout.label(text="", icon="FUND")

    def draw(self, context: "Context"):
        if not self.layout:
            return

        box = self.layout.box()
        col = box.column(align=True)
        col.label(text="Please consider supporting this project", icon="INFO")
        col.label(text="by upgrading to Character DNA Pro to unlock")
        col.label(text="all features and help fund future development.")
        row = self.layout.row()
        row.label(text="Character DNA Pro unlocks:")

        row = self.layout.row()
        box = row.box()
        for editor_name, icon in PRO_EDITORS:
            box.label(text=editor_name, icon=icon)  # pyright: ignore[reportArgumentType]

        row = self.layout.row()
        row.operator("wm.url_open", text="Upgrade to Pro", icon="URL").url = ToolInfo.GET_PRO
