# standard library imports
import logging

# third party imports
import bpy

# local imports
from .constants import NUMBER_OF_HEAD_LODS, SIBLING_EDITIONS, ToolInfo
from .rig_instance import (
    RigInstance,
)
from .typing import *  # noqa: F403
from .ui import callbacks


logger = logging.getLogger(__name__)

face_pose_preview_collections = {}


def get_dna_import_property_group_base_class() -> type:
    """
    Dynamically generates the number of LOD import properties
    """
    _properties = {}

    for i in range(NUMBER_OF_HEAD_LODS):
        # add in import options for lods
        _properties[f"import_lod{i}"] = bpy.props.BoolProperty(
            default=i == 0, name=f"LOD{i}", description=f"Whether to import LOD{i} for the face mesh"
        )

    return type(
        "DnaImportPropertiesBase",
        (object,),
        {
            "__annotations__": _properties,
        },
    )


class BlendFileCharacterCollection(bpy.types.PropertyGroup):
    include: bpy.props.BoolProperty(
        default=True,
        description=(
            "Whether to include this rig instance data in the append or link operation. Note: you can not "
            "append or link rig instances that have the same name as another in the current scene. "
            "Names must be unique"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    name: bpy.props.StringProperty(
        default="",
        description="The name of the rig instance",
    )  # pyright: ignore[reportInvalidTypeForm]
    enabled: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]


class ExtraDnaFolder(bpy.types.PropertyGroup):
    folder_path: bpy.props.StringProperty(
        default="", description="The folder location of the extension repo.", subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]


class CharacterFaceBoardProperties(bpy.types.PropertyGroup):
    """
    Defines the per-scene state for the Face Board pose panel, including the
    selected pose preview and the category filter. One boolean tag filter property
    per unique pose tag is injected dynamically at registration time (see
    :func:`register`).
    """

    face_pose_previews: bpy.props.EnumProperty(
        name="Face Poses",
        items=callbacks.get_face_pose_previews_items,  # type: ignore[arg-type]
        update=callbacks.update_face_pose,  # type: ignore[arg-type]
    )  # pyright: ignore[reportInvalidTypeForm]
    category: bpy.props.EnumProperty(
        name="Category",
        description="Limit the displayed poses to the selected category",
        items=[
            ("ALL", "All", "Show poses from all categories"),
            ("visemes", "Visemes", "Show mouth shapes used for speech"),
            ("emotions", "Emotions", "Show emotional expression poses"),
            ("wrinkle_maps", "Wrinkle Maps", "Show wrinkle map poses"),
            ("scan_reference", "Scan Reference", "Show scan reference poses"),
        ],
        default="ALL",
        update=callbacks.update_face_pose_category,  # type: ignore[arg-type]
    )  # pyright: ignore[reportInvalidTypeForm]
    use_eye_aim: bpy.props.BoolProperty(
        name="Use Eye Aim",
        description="Drive the eyes with the look-at aim target instead of the GUI eye controls",
        get=callbacks.get_use_eye_aim,
        set=callbacks.set_use_eye_aim,
    )  # pyright: ignore[reportInvalidTypeForm]
    eyes_follow_head: bpy.props.BoolProperty(
        name="Eyes Follow Head",
        description="Make the eye aim target follow the head as it moves",
        get=callbacks.get_eyes_follow_head,
        set=callbacks.set_eyes_follow_head,
    )  # pyright: ignore[reportInvalidTypeForm]
    face_board_follow_head: bpy.props.BoolProperty(
        name="Face Board Follow Head",
        description="Make the face board GUI controls follow the head as it moves",
        get=callbacks.get_face_board_follow_head,
        set=callbacks.set_face_board_follow_head,
    )  # pyright: ignore[reportInvalidTypeForm]
    tag_match_mode: bpy.props.EnumProperty(
        name="Tag Match",
        description="How to combine the selected tag filters",
        items=[
            ("ALL", "All", "Only show poses that have all of the selected tags (intersection)"),
            ("ANY", "Any", "Show poses that have any of the selected tags (union)"),
        ],
        default="ANY",
        update=callbacks.update_face_pose_filter,  # type: ignore[arg-type]
    )  # pyright: ignore[reportInvalidTypeForm]


class OutputData(bpy.types.PropertyGroup):
    include: bpy.props.BoolProperty(default=True, description="Whether to include this data in the output")  # pyright: ignore[reportInvalidTypeForm]
    name: bpy.props.StringProperty(default="", description="The name of the shape key")  # pyright: ignore[reportInvalidTypeForm]
    scene_object: bpy.props.PointerProperty(
        type=bpy.types.Object,
        description=(
            "A object that is associated with the dna data. This automatically "
            "gets set based on what is linked in the Rig Instance data"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    image_object: bpy.props.PointerProperty(
        type=bpy.types.Image,
        description=(
            "A object that is associated with the dna data. This automatically "
            "gets set based on what is linked in the Rig Instance data"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    relative_file_path: bpy.props.StringProperty(
        default="", description="The relative file path from the output folder"
    )  # pyright: ignore[reportInvalidTypeForm]
    editable_name: bpy.props.BoolProperty(default=True, description="Whether to include this data in the output")  # pyright: ignore[reportInvalidTypeForm]


class CharacterOutputProperties(bpy.types.PropertyGroup):
    """
    Defines the per-rig-instance state for the Output panel, including the export
    method/component/format, output folder, and the collections of head and body
    output items. This is assigned onto :class:`RigInstance` as the ``output``
    pointer property at registration time (see :func:`register`).
    """

    run_validations: bpy.props.BoolProperty(
        name="Validate", description="Whether to run validations before exporting", default=True
    )  # pyright: ignore[reportInvalidTypeForm]
    folder_path: bpy.props.StringProperty(
        name="Output Folder",
        description="The root folder where the output files will be saved",
        subtype="DIR_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"},
    )  # pyright: ignore[reportInvalidTypeForm]
    method: bpy.props.EnumProperty(
        name="DNA Output Method",
        description="The output method to use when creating the dna file",
        default="calibrate",
        items=[
            (
                "calibrate",
                "Calibrate",
                (
                    "Uses the original dna file and calibrates the included bones and mesh changes into a new dna"
                    " file. Use this method if your vert indices and bone names are the same as the original DNA."
                    " This is the recommended method"
                ),
                "NONE",
                0,
            ),
            (
                "overwrite",
                "Overwrite",
                (
                    "(Experimental) Uses the original dna file and rewrites the mesh and armature data from scratch "
                    "based on the current scene, while preserving the underlying rig behavior controls, and "
                    "blend-shape channel wiring. Use this method if your vert indices, mesh set, or bones differ "
                    "from the original DNA. Only use this method when the calibration method is not possible"
                ),
                "ERROR",
                1,
            ),
        ],
    )  # pyright: ignore[reportInvalidTypeForm]
    component: bpy.props.EnumProperty(
        name="DNA Output Component",
        description="Which component to output use when creating the dna file",
        default="head",
        items=[("head", "Head", "The head component of the DNA"), ("body", "Body", "The body component of the DNA")],
        update=callbacks.update_output_component,  # type: ignore[call-arg]
    )  # pyright: ignore[reportInvalidTypeForm]
    format: bpy.props.EnumProperty(
        name="File Format",
        description="The file format to use when output the dna file. Either binary or json",
        default="binary",
        items=[
            (
                "json",
                "JSON",
                (
                    "Writes the dna file in a human readable json format. Use this method if you want to manually "
                    "edit the dna file"
                ),
            ),
            (
                "binary",
                "Binary",
                (
                    "Writes the dna file in a binary format. Use this method if you want to use the dna file with the"
                    " rig logic system"
                ),
            ),
        ],
    )  # pyright: ignore[reportInvalidTypeForm]
    align_head_and_body: bpy.props.BoolProperty(
        name="Align Head and Body",
        description=(
            "Whether to align the overlapping head and body bones, as well as, aligning the vertices "
            "in the edge loop around the neck during the calibration process"
        ),
        default=True,
    )  # pyright: ignore[reportInvalidTypeForm]
    auto_update_lods: bpy.props.BoolProperty(
        name="Auto Update LODs",
        description=(
            "After calibrating the LOD0 mesh vertex positions, propagate the changes to every lower-LOD "
            "mesh that is not present in the scene, using UV-space matching against the new LOD0 shape"
        ),
        default=True,
    )  # pyright: ignore[reportInvalidTypeForm]
    export_normals: bpy.props.BoolProperty(
        name="Normals",
        description=(
            "Whether to write the scene's normals over the DNA's own. Off by default because the DNA's "
            "authored normals are what let the head sit invisibly on the body, and Blender's are not "
            "imported by default"
        ),
        default=False,
    )  # pyright: ignore[reportInvalidTypeForm]

    head_item_list: bpy.props.CollectionProperty(type=OutputData)  # pyright: ignore[reportInvalidTypeForm]
    head_item_active_index: bpy.props.IntProperty()  # pyright: ignore[reportInvalidTypeForm]
    body_item_list: bpy.props.CollectionProperty(type=OutputData)  # pyright: ignore[reportInvalidTypeForm]
    body_item_active_index: bpy.props.IntProperty()  # pyright: ignore[reportInvalidTypeForm]
    calibrate_bones: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    calibrate_meshes: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    calibrate_shape_keys: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]


class CharacterViewOptionsProperties(bpy.types.PropertyGroup):
    """
    Defines the per-rig-instance state for the View Options panel, including the
    active LOD, material color preview, and the visibility toggles for the face
    board, control rig, head/body bones, and deformer soloing. This is assigned
    onto :class:`RigInstance` as the ``view_options`` pointer property at
    registration time (see :func:`register`).
    """

    active_lod: bpy.props.EnumProperty(
        name="Active LOD",
        items=callbacks.get_head_mesh_lod_items,  # type: ignore[call-arg]
        description="Choose what Level of Detail should be displayed from the face",
        options={"ANIMATABLE"},
        set=callbacks.set_active_lod,
        get=callbacks.get_active_lod,
    )  # pyright: ignore[reportInvalidTypeForm]
    active_material_preview: bpy.props.EnumProperty(
        name="Material Color",
        items=callbacks.get_active_material_preview_items,  # type: ignore[arg-type]
        description="Choose what color should be shown by the material",
        set=callbacks.set_active_material_preview,
        get=callbacks.get_active_material_preview,
    )  # pyright: ignore[reportInvalidTypeForm]
    show_face_board: bpy.props.BoolProperty(
        name="Show Face Board",
        default=False,
        description="Whether to show or hide the face board that belongs to this rig instance in the 3D view",
        set=callbacks.set_show_face_board,
        get=callbacks.get_show_face_board,
    )  # pyright: ignore[reportInvalidTypeForm]
    show_control_rig: bpy.props.BoolProperty(
        name="Show Control Rig",
        default=False,
        description="Whether to show or hide the control rig that belongs to this rig instance in the 3D view",
        set=callbacks.set_show_control_rig,
        get=callbacks.get_show_control_rig,
    )  # pyright: ignore[reportInvalidTypeForm]
    show_head_bones: bpy.props.BoolProperty(
        name="Show Head Bones",
        default=False,
        description="Whether to show or hide the head bones that belong to this rig instance in the 3D view",
        set=callbacks.set_show_head_bones,
        get=callbacks.get_show_head_bones,
    )  # pyright: ignore[reportInvalidTypeForm]
    show_body_bones: bpy.props.BoolProperty(
        name="Show Body Bones",
        default=False,
        description="Whether to show or hide the body bones that belong to this MetaHuman instance in the 3D view",
        set=callbacks.set_show_body_bones,
        get=callbacks.get_show_body_bones,
    )  # pyright: ignore[reportInvalidTypeForm]
    hide_volume_bones: bpy.props.BoolProperty(
        name="Hide Volume Bones",
        default=False,
        description=("Hide the volume bones (the non-skinned leaf bones) by hiding the Volume bone collection"),
        set=callbacks.set_hide_volume_bones,
        get=callbacks.get_hide_volume_bones,
    )  # pyright: ignore[reportInvalidTypeForm]
    solo_internal_bones: bpy.props.BoolProperty(
        name="Solo Internal Bones",
        default=False,
        description=(
            "Solo the internal bones (every non-leaf joint) by soloing the Internal bone collection on the head "
            "rig, hiding the volume and skinned surface leaf bones for a clear view of the internal skeleton"
        ),
        set=callbacks.set_solo_internal_bones,
        get=callbacks.get_solo_internal_bones,
    )  # pyright: ignore[reportInvalidTypeForm]


class CharacterAddonProperties:
    """
    This class holds the properties for the addon.
    """

    metrics_collection: bpy.props.BoolProperty(
        name="Collect Metrics",
        default=False,
        description="This will send anonymous usage data to Poly Hammer to help improve the addon and help catch bugs",
    )  # pyright: ignore[reportInvalidTypeForm]

    next_metrics_consent_timestamp: bpy.props.FloatProperty(default=0.0)  # pyright: ignore[reportInvalidTypeForm]
    extra_dna_folder_list: bpy.props.CollectionProperty(type=ExtraDnaFolder)  # pyright: ignore[reportInvalidTypeForm]
    extra_dna_folder_list_active_index: bpy.props.IntProperty()  # pyright: ignore[reportInvalidTypeForm]
    show_pro_features: bpy.props.BoolProperty(
        name="Show Pro Features",
        default=True,
        description=(
            "Show the Pro editor tools and panels. Disable this to preview what the free edition's UI looks like"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]


class CharacterImportProperties(get_dna_import_property_group_base_class()):
    import_mesh: bpy.props.BoolProperty(default=True, name="Mesh", description="Whether to import the head meshes")  # pyright: ignore[reportInvalidTypeForm]
    import_normals: bpy.props.BoolProperty(
        default=False,
        name="Normals",
        description="Whether to import the DNA's own custom split normals",
    )  # pyright: ignore[reportInvalidTypeForm]
    import_bones: bpy.props.BoolProperty(
        default=True, name="Bones", description="Whether to import the bones for the head"
    )  # pyright: ignore[reportInvalidTypeForm]
    import_shape_keys: bpy.props.BoolProperty(
        default=False,
        name="Shape Keys",
        description="Whether to import the shapes key for the head. You can also import these later",
    )  # pyright: ignore[reportInvalidTypeForm]
    import_vertex_groups: bpy.props.BoolProperty(
        default=True,
        name="Vertex Groups",
        description="Whether to import the vertex groups that skin the bones to the head mesh",
    )  # pyright: ignore[reportInvalidTypeForm]
    import_bone_collections: bpy.props.BoolProperty(
        default=True,
        name="Bone Collections",
        description=(
            "Whether to author a bone collection for each rig-definition joint group and color "
            "each bone with its rig-definition color"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    import_region_vertex_groups: bpy.props.BoolProperty(
        default=False,
        name="Region Vertex Groups",
        description=(
            "Whether to author a REGION_ prefixed vertex group for each rig-definition mesh region "
            "on the LOD0 head meshes"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    import_vertex_colors: bpy.props.BoolProperty(
        default=True,
        name="Vertex Colors",
        description=(
            "Whether to import the vertex colors for the head mesh. Note this will first look "
            "for a vertex_colors.json in the same folder as the .dna file. Otherwise it will use the "
            "default vertex_colors.json in the addon resources"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    import_materials: bpy.props.BoolProperty(
        default=True, name="Materials", description="Whether to import the materials for the head mesh"
    )  # pyright: ignore[reportInvalidTypeForm]
    import_face_board: bpy.props.BoolProperty(
        default=True, name="Face Board", description="Whether to import the face board that drives the rig logic"
    )  # pyright: ignore[reportInvalidTypeForm]
    reuse_face_board: bpy.props.BoolProperty(
        default=False,
        name="Reuse Face Board",
        description=(
            "Whether to reuse or import a unique face board that drives the rig logic instead of a shared one. "
            "This is useful if you want to have multiple rigs in the same scene that drive different face meshes"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    include_body: bpy.props.BoolProperty(
        default=True,
        name="Include Body",
        description=(
            "If true, this will try to find a body.dna file in the same folder as this .dna file. "
            "If the body.dna file is found, it will be imported as well"
        ),
    )  # pyright: ignore[reportInvalidTypeForm]
    alternate_maps_folder: bpy.props.StringProperty(
        default="",
        name="Maps Folder",
        description=(
            "This can be set to an alternate folder location for the face wrinkle maps. "
            'If no folder is set, the importer looks for a "Maps" folder next to the .dna file'
        ),
    )  # pyright: ignore[reportInvalidTypeForm]


class CharacterWindowManagerProperties(bpy.types.PropertyGroup, CharacterImportProperties):
    """
    Defines a property group that stores constants in the window manager context.
    """

    errors = {}
    dna_info = {"_previous_file_path": None, "_dna_reader": None}

    # Global, per-session cache shared across the whole addon.
    data = {}

    error_message: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    progress: bpy.props.FloatProperty(default=1.0)  # pyright: ignore[reportInvalidTypeForm]
    progress_description: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    progress_mesh_name: bpy.props.StringProperty(default="")  # pyright: ignore[reportInvalidTypeForm]
    evaluate_dependency_graph: bpy.props.BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    is_undoing: bpy.props.BoolProperty(default=False)  # pyright: ignore[reportInvalidTypeForm]
    is_rendering: bpy.props.BoolProperty(default=False)  # pyright: ignore[reportInvalidTypeForm]


class CharacterSceneProperties(bpy.types.PropertyGroup):
    """
    Defines a property group that lives in the scene.
    """

    # --------------------- read/write properties ------------------
    context = {}

    # --------------------- user interface properties ------------------
    highlight_matching_active_bone: bpy.props.BoolProperty(
        name="Highlight Matching Active Bone",
        description="Highlights bones that match the name of the active pose bone across all rig instances",
        default=False,
        set=callbacks.set_highlight_matching_active_bone,
        get=callbacks.get_highlight_matching_active_bone,
    )  # pyright: ignore[reportInvalidTypeForm]
    # --------------------- riglogic properties ------------------
    rig_instance_list: bpy.props.CollectionProperty(type=RigInstance)  # pyright: ignore[reportInvalidTypeForm]
    rig_instance_list_active_index: bpy.props.IntProperty(
        update=callbacks.update_head_output_items  # type: ignore[arg-type]
    )  # pyright: ignore[reportInvalidTypeForm]
    # --------------------- face board properties ------------------
    face_board: bpy.props.PointerProperty(type=CharacterFaceBoardProperties)  # pyright: ignore[reportInvalidTypeForm]


def register():
    """
    Registers the addon's property group classes when the addon is enabled.
    """
    # register the list data classes first, since the scene property groups depends on them
    bpy.utils.register_class(OutputData)

    # Note: All editors that add properties to RigInstance must be imported and
    # registered and dynamically assigned to the RigInstance before it is registered.
    # When the optional Pro editors submodule is absent, this is a no-op.
    from .utilities import get_editors

    editors = get_editors()
    if editors is not None:
        editors.register_property_groups()

    # ----------------- Output Properties -----------------
    # Register the output property group and assign it onto the RigInstance before
    # it is registered, mirroring the editor property-group pattern above.
    bpy.utils.register_class(CharacterOutputProperties)
    RigInstance.__annotations__["output"] = bpy.props.PointerProperty(type=CharacterOutputProperties)

    # ----------------- View Options Properties -----------------
    # Register the view options property group and assign it onto the RigInstance
    # before it is registered, mirroring the output property-group pattern above.
    bpy.utils.register_class(CharacterViewOptionsProperties)
    RigInstance.__annotations__["view_options"] = bpy.props.PointerProperty(type=CharacterViewOptionsProperties)

    # Now register RigLogicInstance
    bpy.utils.register_class(RigInstance)
    bpy.utils.register_class(BlendFileCharacterCollection)

    # add the pose previews collection. This must exist before the dynamic tag
    # properties are built, since collecting the tags reads the pose metadata cache.
    face_pose_previews_collection = bpy.utils.previews.new()
    face_pose_previews_collection.face_pose_previews_root_folder = ""  # type: ignore[attr-defined]
    face_pose_previews_collection.face_pose_previews = ()  # type: ignore[attr-defined]
    face_pose_previews_collection.face_pose_previews_cache_key = None  # type: ignore[attr-defined]
    face_pose_previews_collection.face_pose_metadata = []  # type: ignore[attr-defined]
    face_pose_preview_collections["face_poses"] = face_pose_previews_collection

    try:
        # Dynamically build one boolean property per unique face pose tag and inject
        # them into the face board property group before it is registered. This uses
        # the same dynamic-class technique as the LOD import options above.
        for property_name, property_definition in callbacks.build_face_pose_tag_properties().items():
            CharacterFaceBoardProperties.__annotations__[property_name] = property_definition

        bpy.utils.register_class(CharacterFaceBoardProperties)
        bpy.utils.register_class(CharacterSceneProperties)
        setattr(bpy.types.Scene, ToolInfo.NAME, bpy.props.PointerProperty(type=CharacterSceneProperties))  # type: ignore[attr-defined]
        # Register the sibling edition's scene pointer read-only so rig-instance
        # data saved by the other edition (Free vs Pro) is exposed on load and can
        # be migrated. Saved PointerProperty data is only accessible when a
        # matching property is registered.
        sibling_id = SIBLING_EDITIONS.get(ToolInfo.NAME)
        if sibling_id and not hasattr(bpy.types.Scene, sibling_id):
            setattr(bpy.types.Scene, sibling_id, bpy.props.PointerProperty(type=CharacterSceneProperties))  # type: ignore[attr-defined]
    except ValueError as error:
        logger.debug(error)

    try:
        bpy.utils.register_class(CharacterWindowManagerProperties)
        setattr(
            bpy.types.WindowManager, ToolInfo.NAME, bpy.props.PointerProperty(type=CharacterWindowManagerProperties)
        )  # type: ignore[attr-defined]
    except ValueError as error:
        logger.debug(error)


def unregister():
    """
    Un-registers the addon's property group classes when the addon is disabled.
    """
    # remove the pose previews collections
    for preview_collection in face_pose_preview_collections.values():
        bpy.utils.previews.remove(preview_collection)
    face_pose_preview_collections.clear()

    window_manager_property_class = bpy.types.PropertyGroup.bl_rna_get_subclass_py(
        CharacterWindowManagerProperties.__name__
    )
    if window_manager_property_class:
        bpy.utils.unregister_class(window_manager_property_class)

    scene_property_class = bpy.types.PropertyGroup.bl_rna_get_subclass_py(CharacterSceneProperties.__name__)
    if scene_property_class:
        bpy.utils.unregister_class(scene_property_class)

    face_board_property_class = bpy.types.PropertyGroup.bl_rna_get_subclass_py(CharacterFaceBoardProperties.__name__)
    if face_board_property_class:
        bpy.utils.unregister_class(face_board_property_class)

    # Remove the dynamically injected tag properties so re-registration starts clean.
    for property_name in callbacks.get_face_pose_tag_property_map():
        CharacterFaceBoardProperties.__annotations__.pop(property_name, None)

    # Remove the dynamically injected output pointer so re-registration starts clean.
    RigInstance.__annotations__.pop("output", None)

    # Remove the dynamically injected view options pointer so re-registration starts clean.
    RigInstance.__annotations__.pop("view_options", None)

    # unregister the list data classes
    bpy.utils.unregister_class(RigInstance)

    try:
        # ----------------- Editor Properties -----------------
        # Unregister the optional Pro editor property groups (no-op when absent).
        from .utilities import get_editors

        editors = get_editors()
        if editors is not None:
            editors.unregister_property_groups()

        bpy.utils.unregister_class(CharacterOutputProperties)
        bpy.utils.unregister_class(CharacterViewOptionsProperties)
        bpy.utils.unregister_class(OutputData)
        bpy.utils.unregister_class(BlendFileCharacterCollection)

    except RuntimeError as error:
        logger.debug(error)

    if hasattr(bpy.types.WindowManager, ToolInfo.NAME):
        delattr(bpy.types.WindowManager, ToolInfo.NAME)

    if hasattr(bpy.types.Scene, ToolInfo.NAME):
        delattr(bpy.types.Scene, ToolInfo.NAME)

    sibling_id = SIBLING_EDITIONS.get(ToolInfo.NAME)
    if sibling_id and hasattr(bpy.types.Scene, sibling_id):
        delattr(bpy.types.Scene, sibling_id)
