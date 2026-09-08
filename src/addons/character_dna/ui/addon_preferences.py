# standard library imports

from pathlib import Path

# third party imports
import bpy

# local imports
from .. import __package__ as package_name
from ..constants import ToolInfo
from ..properties import CharacterAddonProperties, ExtraDnaFolder
from ..typing import *  # noqa: F403
from ..utilities import editors_available, get_editors


class FOLDER_UL_extra_dna_path(bpy.types.UIList):
    def draw_item(
        self,
        context: "Context",
        layout: bpy.types.UILayout,
        data: "CharacterDnaPreferences",
        item: "ExtraDnaFolder",
        icon: int | None,
        active_data: "CharacterDnaPreferences",
        active_prop_name: str,
    ):
        row = layout.row()
        row.alert = False
        if item.folder_path and not Path(item.folder_path).exists():
            row.alert = True
        row.prop(item, "folder_path", text="", emboss=False)


class CharacterDnaPreferences(CharacterAddonProperties, bpy.types.AddonPreferences):
    bl_idname = str(package_name)

    def draw(self, context: "Context"):
        layout = self.layout
        # General Settings
        row = layout.row()
        row.prop(self, "metrics_collection", text="Allow Metrics Collection")

        # Editor Settings (Pro only). The ``show_pro_features`` toggle lets Pro
        # users preview what the free edition's UI looks like. When the editors
        # submodule is absent (free edition), show a note advertising Pro instead.
        if editors_available():
            row = layout.row()
            row.prop(self, "show_pro_features")
            editors = get_editors()
            if self.show_pro_features and editors is not None:
                editors.draw_preferences(self, layout, context)
        else:
            layout.separator()
            box = layout.box()
            box.label(text="Editor tools are available in Character DNA Pro.", icon="FUND")
            box.operator(
                "wm.url_open",
                text="Upgrade to Pro",
                icon="URL",
            ).url = ToolInfo.GET_PRO

        # Extra DNA Folder Paths
        layout.separator()
        row = layout.row()

        row.label(text="Extra DNA Folder Paths:")
        row = self.layout.row()
        row.template_list(
            "FOLDER_UL_extra_dna_path",
            "extra_dna_folder_list_id",
            self,
            "extra_dna_folder_list",
            self,
            "extra_dna_folder_list_active_index",
            rows=4 if self.extra_dna_folder_list else 1,
        )

        col = row.column()
        col.operator(f"{ToolInfo.NAME}.addon_preferences_extra_dna_entry_add", text="", icon="ADD")
        row = col.row()
        row.enabled = len(self.extra_dna_folder_list) > 0
        row.operator(
            f"{ToolInfo.NAME}.addon_preferences_extra_dna_entry_remove",
            text="",
            icon="REMOVE",
        )


def register():
    bpy.utils.register_class(ExtraDnaFolder)
    bpy.utils.register_class(FOLDER_UL_extra_dna_path)

    # Register the nested editor preference groups and attach them to the addon
    # preferences before the preferences class itself is registered. When the
    # editors submodule is absent (free edition), this is skipped.
    editors = get_editors()
    if editors is not None:
        editors.register_preferences(CharacterDnaPreferences)

    bpy.utils.register_class(CharacterDnaPreferences)


def unregister():
    bpy.utils.unregister_class(CharacterDnaPreferences)

    editors = get_editors()
    if editors is not None:
        editors.unregister_preferences(CharacterDnaPreferences)

    bpy.utils.unregister_class(FOLDER_UL_extra_dna_path)
    bpy.utils.unregister_class(ExtraDnaFolder)
