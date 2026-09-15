import logging
import os

from collections.abc import Callable

import bpy

# This import is necessary to register custom icons
import bpy.utils.previews  # pyright: ignore[reportMissingModuleSource, reportUnusedImport]

from . import constants, manual_map, operators, properties, rig_instance, utilities
from .runtime import controller as native_runtime
from .ui import addon_preferences, importer, menus, view_3d


logger = logging.getLogger(constants.ToolInfo.NAME)

bl_info = {
    "name": "Character DNA",
    "author": "Poly Hammer",
    "version": (0, 13, 3),
    "blender": (4, 5, 0),
    "location": "File > Import > MetaHuman DNA",
    "description": (
        "Imports MetaHuman head and body components from a their DNA files, "
        "lets you customize them, then send them back to MetaHuman Creator."
    ),
    "warning": "",
    "wiki_url": "https://docs.polyhammer.com/character-dna-addon/",
    "category": "Poly Hammer",
}

# Callbacks that other addons can register to be notified when the rig instance list changes
# (a rig instance is set up on load, added, or removed). The argument is the affected rig
# instance for additions/setup, or ``None`` for removals.
post_setup_scene_callbacks: list[Callable[[rig_instance.RigInstance | None], None]] = []

# Main Addon. Editor operators/panels are registered separately via the optional
# ``editors`` submodule registry (see ``utilities.get_editors``), so this list
# contains only the core classes. The upsell panel is always registered; its
# ``poll`` hides it when the Pro editor UI is visible.
classes = [
    operators.ImportCharacterDna,
    operators.DNA_FH_import_dna,
    operators.AppendOrLinkCharacter,
    operators.ImportFaceBoardAnimation,
    operators.ImportComponentAnimation,
    operators.ConfirmAnimationImport,
    operators.BakeFaceBoardAnimation,
    operators.BakeComponentAnimation,
    operators.TestSentry,
    operators.MigrateLegacyData,
    operators.MetricsCollectionConsent,
    operators.ForceEvaluate,
    operators.RefreshOutputItems,
    operators.MapRawToGuiControls,
    operators.SendToMetaHumanCreator,
    operators.ExportSelectedComponent,
    operators.GenerateMaterial,
    operators.DuplicateRigInstance,
    operators.AddHeadTextureLogicNode,
    operators.ReportError,
    operators.ReportErrorWithFix,
    operators.UILIST_RIG_INSTANCE_OT_entry_move,
    operators.UILIST_RIG_INSTANCE_OT_entry_add,
    operators.UILIST_RIG_INSTANCE_OT_entry_remove,
    operators.UILIST_ADDON_PREFERENCES_OT_extra_dna_entry_add,
    operators.UILIST_ADDON_PREFERENCES_OT_extra_dna_entry_remove,
    operators.FaceBoardSearchPose,
    importer.CHARACTER_DNA_FILE_DATA_PT_panel,
    importer.CHARACTER_DNA_LODS_PT_panel,
    importer.CHARACTER_DNA_EXTRAS_PT_panel,
    importer.CHARACTER_DNA_FILE_INFO_PT_panel,
    importer.CHARACTER_DNA_LINK_OPTIONS_PT_panel,
    view_3d.CHARACTER_DNA_PT_face_pose_tags,
    view_3d.CHARACTER_DNA_PT_face_board,
    view_3d.CHARACTER_DNA_PT_face_board_footer,
    view_3d.CHARACTER_DNA_PT_view_options,
    view_3d.CHARACTER_DNA_PT_rig_instance,
    view_3d.CHARACTER_DNA_PT_rig_instance_head_sub_panel,
    view_3d.CHARACTER_DNA_PT_rig_instance_body_sub_panel,
    view_3d.CHARACTER_DNA_PT_rig_instance_footer_sub_panel,
    view_3d.CHARACTER_DNA_PT_animation_panel,
    view_3d.CHARACTER_DNA_UL_output_items,
    view_3d.CHARACTER_DNA_UL_rig_instances,
    view_3d.CHARACTER_DNA_PT_output_panel,
    view_3d.CHARACTER_DNA_PT_output_buttons_sub_panel,
    view_3d.CHARACTER_DNA_PT_migrate_legacy_data,
    view_3d.CHARACTER_DNA_PT_pro_upsell,
]

app_handlers = {
    "load_pre": bpy.app.handlers.persistent(utilities.teardown_scene),
    "load_post": bpy.app.handlers.persistent(utilities.setup_scene),
    "undo_pre": bpy.app.handlers.persistent(utilities.pre_undo),
    "undo_post": bpy.app.handlers.persistent(utilities.post_undo),
    "redo_pre": bpy.app.handlers.persistent(utilities.pre_redo),
    "redo_post": bpy.app.handlers.persistent(utilities.post_redo),
    "render_init": bpy.app.handlers.persistent(utilities.pre_render),
    "render_complete": bpy.app.handlers.persistent(utilities.post_render),
    "render_cancel": bpy.app.handlers.persistent(utilities.post_render),
    "save_post": bpy.app.handlers.persistent(utilities.post_save),
}


def register():
    """
    Registers the addon classes when the addon is enabled.
    """
    if os.environ.get("CHARACTER_DNA_DEV"):
        logging.basicConfig(level=logging.DEBUG)

    try:
        # register the manual map
        bpy.utils.register_manual_map(manual_map.manual_map)

        # register the properties
        addon_preferences.register()
        properties.register()

        # register the core classes
        for cls in classes:
            bpy.utils.register_class(cls)

        # register the optional Pro editor classes (no-op in the free edition)
        editors = utilities.get_editors()
        if editors is not None:
            editors.register_classes()

        # add menu items
        menus.add_dna_import_menu()
        menus.add_rig_logic_texture_node_menu()

        # start the optional Pro editor runtime services (toast overlay, worker cleanup)
        if editors is not None:
            editors.register_runtime()

    except Exception as error:
        logger.error(error)

    utilities.init_sentry()

    native_runtime.register()

    # add event handlers
    for handler_name, handler_function in app_handlers.items():
        getattr(bpy.app.handlers, handler_name).append(handler_function)


def unregister():
    """
    Un-registers the addon classes when the addon is disabled.
    """
    utilities.disable_duplicate_addons()

    native_runtime.unregister()

    utilities.teardown_scene()

    editors = utilities.get_editors()

    # Stop the optional Pro editor runtime services (solver workers, toast overlay).
    if editors is not None:
        editors.unregister_runtime()

    # remove event handlers
    for handler_name, handler_function in app_handlers.items():
        handler_list = getattr(bpy.app.handlers, handler_name)
        if handler_function in handler_list:
            handler_list.remove(handler_function)

    try:
        # unregister the manual map
        bpy.utils.unregister_manual_map(manual_map.manual_map)

        # remove menu items
        menus.remove_dna_import_menu()
        menus.remove_rig_logic_texture_node_menu()

        # unregister the optional Pro editor classes (no-op in the free edition)
        if editors is not None:
            editors.unregister_classes()

        # unregister the core classes
        for cls in reversed(classes):
            if hasattr(cls, "bl_rna"):
                bpy.utils.unregister_class(cls)

        # unregister the properties
        properties.unregister()
        addon_preferences.unregister()
    except Exception as error:
        logger.error(error)
