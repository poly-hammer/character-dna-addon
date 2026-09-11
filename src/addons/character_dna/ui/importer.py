# standard library imports
from pathlib import Path

# third party imports
import bpy

from bpy_extras.io_utils import ImportHelper  # type: ignore

# local imports
from .. import utilities
from ..constants import NUMBER_OF_HEAD_LODS, ToolInfo
from ..dna_io import get_dna_reader
from ..typing import *  # noqa: F403


class CHARACTER_DNA_FILE_DATA_PT_panel(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "File Data"
    bl_parent_id = "FILE_PT_operator"
    bl_options = {"HEADER_LAYOUT_EXPAND"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        return context.space_data.active_operator.bl_idname == f"{ToolInfo.NAME.upper()}_OT_import_dna"  # type: ignore[attr-defined]

    def draw(self, context: "Context"):
        if not self.layout:
            return

        operator = context.space_data.active_operator  # type: ignore
        stem = Path(operator.filepath).stem.lower()
        layout = self.layout
        row = layout.row()
        row.prop(operator, "import_mesh")
        # TODO: Support importing normals again, only when Blender supports normals on
        # shape keys. Otherwise, we hit the closed eyelid artifact problem.
        # row = layout.row()  # noqa: ERA001
        # row.prop(operator, "import_normals")  # noqa: ERA001
        row = layout.row()
        row.prop(operator, "import_bones")
        row = layout.row()
        row.prop(operator, "import_vertex_groups")
        # TODO: Maybe support importing regions as vertex groups on the head
        # if stem != "body":
        #     row = layout.row() # noqa: ERA001
        #     row.prop(operator, "import_region_vertex_groups") # noqa: ERA001
        row = layout.row()
        row.prop(operator, "import_bone_collections")
        if stem != "body":
            row = layout.row()
            row.prop(operator, "import_vertex_colors")
        row = layout.row()
        row.prop(operator, "import_materials")
        if stem != "body":
            row = layout.row()
            row.prop(operator, "import_face_board")


class CHARACTER_DNA_LODS_PT_panel(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "LODs"
    bl_parent_id = "FILE_PT_operator"
    bl_options = {"HEADER_LAYOUT_EXPAND"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        return context.space_data.active_operator.bl_idname == f"{ToolInfo.NAME.upper()}_OT_import_dna"  # type: ignore[attr-defined]

    def draw(self, context: "Context"):
        if not self.layout:
            return

        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        stem = Path(operator.filepath).stem.lower()
        layout = self.layout
        row = layout.row()
        for i in range(NUMBER_OF_HEAD_LODS):
            if i == 0:
                row.enabled = False
            # bodies only have one LOD, so we don't need to show the LODs for them
            if stem == "body" and i > 3:
                return
            row.prop(operator, f"import_lod{i}")
            row = layout.row()


class CHARACTER_DNA_EXTRAS_PT_panel(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Extras"
    bl_parent_id = "FILE_PT_operator"
    bl_options = {"HEADER_LAYOUT_EXPAND"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        return context.space_data.active_operator.bl_idname == f"{ToolInfo.NAME.upper()}_OT_import_dna"  # type: ignore[attr-defined]

    def _get_path_error(self, folder_path: str) -> str:
        if not folder_path:
            return ""

        path = Path(folder_path)
        if not path.exists():
            return "Folder does not exist"
        if not path.is_dir():
            return "Path is not a folder"
        return ""

    def draw(self, context: "Context"):
        if not self.layout:
            return

        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        stem = Path(operator.filepath).stem.lower()
        body_file = Path(operator.filepath).parent / "body.dna"
        layout = self.layout
        if stem == "head":
            row = layout.row()
            row.enabled = body_file.exists()
            row.prop(operator, "include_body")
            row = layout.row()
            row.enabled = operator.import_face_board
            row.prop(operator, "reuse_face_board")
        row = layout.row()
        row.label(text="Alternate Maps Folder:")
        row = layout.row()
        path_error = self._get_path_error(operator.alternate_maps_folder)

        if path_error:
            row.alert = True

        row.prop(operator, "alternate_maps_folder", text="")

        if path_error:
            row = layout.row()
            row.alert = True
            row.label(text=path_error, icon="ERROR")


class CHARACTER_DNA_FILE_INFO_PT_panel(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "DNA File Info"
    bl_parent_id = "FILE_PT_operator"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        is_dna_importer = context.space_data.active_operator.bl_idname == f"{ToolInfo.NAME.upper()}_OT_import_dna"  # type: ignore[attr-defined]
        if not hasattr(operator, "filepath"):
            return False

        is_dna_file = operator.filepath.lower().endswith(".dna") and Path(operator.filepath).exists()
        return is_dna_importer and is_dna_file

    def draw(self, context: "Context"):
        if not self.layout:
            return

        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        addon_window_manager_properties = utilities.get_addon_window_manager_properties(context)
        dna_info = addon_window_manager_properties.dna_info

        if operator.filepath.lower().endswith(".dna") and Path(operator.filepath).exists():
            if not dna_info["_dna_reader"] or operator.filepath != dna_info["_previous_file_path"]:
                dna_info["_previous_file_path"] = operator.filepath
                reader = get_dna_reader(
                    file_path=Path(operator.filepath), file_format="binary", data_layer="Descriptor"
                )
                if not reader:
                    return

                dna_info["_dna_reader"] = reader  # pyright: ignore[reportArgumentType]

            dna_reader = dna_info["_dna_reader"]
            from ..bindings import enums  # type: ignore[reportAttributeAccessIssue]

            row = self.layout.row()
            row.label(text="Name: ")
            row.label(text=str(dna_reader.getName()))
            row = self.layout.row()
            row.label(text="Archetype: ")
            row.label(text=str(enums.Archetype(dna_reader.getArchetype()).name))
            row = self.layout.row()
            row.label(text="Gender: ")
            row.label(text=str(enums.Gender(dna_reader.getGender()).name))
            row = self.layout.row()
            row.label(text="Age: ")
            row.label(text=str(dna_reader.getAge()))
            row = self.layout.row()
            row.label(text="LOD Count: ")
            row.label(text=str(dna_reader.getLODCount()))
            row = self.layout.row()
            row.label(text="Max LOD: ")
            row.label(text=str(dna_reader.getDBMaxLOD()))
            row = self.layout.row()
            row.label(text="Complexity: ")
            row.label(text=str(dna_reader.getDBComplexity()))
            row = self.layout.row()
            row.label(text="Database Name: ")
            row.label(text=str(dna_reader.getDBName()))
            row = self.layout.row()
            row.label(text="Translation Units: ")
            row.label(text=str(enums.TranslationUnit(dna_reader.getTranslationUnit()).name))
            row = self.layout.row()
            row.label(text="Rotation Units: ")
            row.label(text=str(enums.RotationUnit(dna_reader.getRotationUnit()).name))
            row = self.layout.row()
            row.label(text="X Axis: ")
            row.label(text=str(enums.Direction(dna_reader.getCoordinateSystem().x).name))
            row = self.layout.row()
            row.label(text="Y Axis: ")
            row.label(text=str(enums.Direction(dna_reader.getCoordinateSystem().y).name))
            row = self.layout.row()
            row.label(text="Z Axis: ")
            row.label(text=str(enums.Direction(dna_reader.getCoordinateSystem().z).name))


class ImportAsset(ImportHelper):
    """
    This class subclasses the import helper to define a custom file browser
    """

    bl_options = {"UNDO", "PRESET"}

    def draw(self, context: "Context"):
        pass

    @property
    def settings_title(self) -> str:
        return ""


class ImportAnimation(ImportAsset):
    def draw(self, context: "Context"):
        layout = getattr(self, "layout", None)
        if not layout:
            return

        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        if not operator:
            return

        row = layout.row()
        row.label(text=self.settings_title)
        row = layout.row()
        row.prop(operator, "round_sub_frames")
        row = layout.row()
        row.prop(operator, "match_frame_rate")
        row = layout.row()
        row.prop(operator, "prefix_instance_name")
        row = layout.row()
        row.prop(operator, "prefix_component_name")


class CHARACTER_DNA_LINK_OPTIONS_PT_panel(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Options"
    bl_parent_id = "FILE_PT_operator"
    bl_options = {"HEADER_LAYOUT_EXPAND"}

    @classmethod
    def poll(cls, context: "Context") -> bool:
        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        return (
            operator is not None
            and operator.bl_idname == f"{ToolInfo.NAME.upper()}_OT_append_or_link_metahuman"
            and operator.operation_type == "LINK"
        )

    def draw(self, context: "Context"):
        if not self.layout:
            return

        operator = context.space_data.active_operator  # type: ignore[attr-defined]
        self.layout.prop(operator, "editable_rig")


class LinkAppendCharacterImportHelper(ImportHelper):
    """
    This class subclasses the import helper to define a custom file browser
    """

    bl_options = {"UNDO"}

    @staticmethod
    def _source_stamp(file_path: str) -> str:
        path = Path(bpy.path.abspath(file_path)).resolve()
        stat = path.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"

    def refresh_meta_human_list(self, operator: bpy.types.Operator):
        self.meta_human_list.clear()  # type: ignore[attr-defined]
        scene_properties = utilities.get_addon_scene_properties()
        rig_instance_names = [i.name for i in scene_properties.rig_instance_list]

        data, error = utilities.extract_rig_instance_data_from_blend_file(Path(bpy.path.abspath(operator.filepath)))
        if error:
            operator.report({"WARNING"}, error)
        for name in data:
            item = operator.meta_human_list.add()  # type: ignore[attr-defined]
            item.name = name
            item.include = False
            item.enabled = name not in rig_instance_names

        # save the current filepath to detect changes
        operator.previous_file_path = self._source_stamp(operator.filepath)  # type: ignore[attr-defined]

    def draw(self, context: "Context"):
        layout = self.layout  # type: ignore
        if not layout:
            return

        operator = context.space_data.active_operator  # type: ignore[attr-defined]

        row = layout.row()
        row.prop(operator, "operation_type", expand=True)
        file_path = Path(bpy.path.abspath(operator.filepath))

        if not operator.filepath or not file_path.is_file():
            row = layout.row()
            row.alert = True
            row.label(text="Select a .blend file to see MetaHuman(s)")
            return

        if Path(bpy.data.filepath) == file_path:
            row = layout.row()
            row.alert = True
            row.label(text="Select different .blend than current")
            return

        if not file_path.exists() or not file_path.is_file() or file_path.suffix.lower() != ".blend":
            return

        # check the blend file version against the running Blender version
        blend_version = utilities.read_blend_file_version(file_path)
        if blend_version and (blend_version[0], blend_version[1]) != (bpy.app.version[0], bpy.app.version[1]):
            box = layout.box()
            col = box.column(align=True)
            col.alert = True
            col.label(text="Version Mismatch", icon="ERROR")
            col.label(text=f"File saved in Blender {utilities.blend_file_version_string(blend_version)}")
            col.label(text=f"Running Blender {bpy.app.version_string}")
            col.separator()
            col.label(text="This may cause import issues.")

        row = layout.row()
        row.label(text=f"Choose MetaHuman(s) to {operator.operation_type.lower()}:")

        # only refresh the list if the selected file path has changed
        if operator.previous_file_path != self._source_stamp(operator.filepath):
            self.refresh_meta_human_list(operator)

        for item in operator.meta_human_list:
            row = layout.row()
            icon = "NONE"
            text = item.name
            if not item.enabled:
                row.enabled = False
                row.alert = True
                icon = "ERROR"
                text = f"{item.name} (Exists in current scene)"

            row.prop(item, "include", text="")
            row.label(text=text, icon=icon)
