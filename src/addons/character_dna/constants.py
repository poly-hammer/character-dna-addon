import math
import tempfile

from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import bpy

from mathutils import Euler, Vector


IS_BLENDER_5 = bpy.app.version >= (5, 0, 0)

# Slotted actions landed in Blender 4.4. Writing to Action.fcurves on 4.4/4.5 goes through
# the legacy layer, which creates a "Legacy Slot" but never assigns it to the animated ID,
# so the action looks empty until the user picks the slot by hand.
HAS_ACTION_SLOTS = bpy.app.version >= (4, 4, 0)


class ToolInfo:
    NAME: str = "character_dna"
    EXTENSION_ID: str | None = None
    HOW_TO_INSTALL: str = "https://youtu.be/WvJCRUxT5c0"
    INSTALL_TUTORIAL_VIDEO: str = "https://youtu.be/WvJCRUxT5c0"
    GET_PRO: str = "https://polyhammer.com/character-dna-addon"
    METRICS_COLLECTION_AGREEMENT: str = "https://www.polyhammer.com/dpa"


class PanelOrder(IntEnum):
    """Top-to-bottom ordering of the addon's N-panel UI panels."""

    FACE_BOARD = 10
    VIEW_OPTIONS = 20
    RIG_INSTANCES = 30
    ANIMATION = 40
    CONVERTER = 50
    MESH_EDITOR = 55
    RAW_CONTROL_EDITOR = 60
    SHAPE_KEY_EDITOR = 70
    RBF_EDITOR = 80
    BEHAVIOR_VIEWER = 85
    BACKUP_MANAGER = 90
    OUTPUT = 100
    MIGRATE_LEGACY_DATA = 110
    PRO_UPSELL = 120


Axis = Literal["X", "Y", "Z"]
ComponentType = Literal["head", "body"]

FACE_BOARD_NAME = "face_gui"
HEAD_MATERIAL_NAME = "head_shader"
BODY_MATERIAL_NAME = "body_shader"
MASKS_TEXTURE = "combined_masks.tga"
# material-color preview enum items
MATERIAL_PREVIEW_ITEMS_BASE = (
    ("combined", "Combined", "Displays all combined textures maps", "NONE", 0),
    ("masks", "Masks", "Displays only the color of the mask texture maps", "NONE", 1),
    ("normals", "Normals", "Displays only the color of the normal texture maps", "NONE", 2),
)
NUMBER_OF_HEAD_LODS = 8
SENTRY_DSN = "https://38575ef4609265865b46dcc274249962@sentry.poly-hammer.com/13"

INVALID_NAME_CHARACTERS_REGEX = r"[^-+\w]+"
LOD_REGEX = r"(?i)(_LOD\d).*"

# Prefix of every MetaHuman face expression raw control in the DNA
# (e.g. ``CTRL_expressions.jawOpen``). Lives at the top level so both the
# Raw Control Editor and the shared dependency-chain resolver can use it.
RAW_CONTROL_PREFIX = "CTRL_expressions."

# this is the difference in scale between unreal and blender
SCALE_FACTOR = 100.0
SHAPE_KEY_NAME_MAX_LENGTH = 63
SHAPE_KEY_DELTA_THRESHOLD = 1e-6
BONE_DELTA_THRESHOLD = 1e-3
NORMAL_DELTA_THRESHOLD = 0.1
"""Below this a changed normal cannot be told apart from Blender's own storage noise.

``normals_split_custom_set`` does not hold a normal exactly. Measured over Ada's 96008 head
corners the round trip moves a normal by a mean of 5e-5 and a p99 of 3.2e-4, but by as much as
0.0709 on a handful it cannot represent at all. A tighter threshold rewrites nearly every
normal with its quantized value instead of leaving the untouched ones alone.
"""
SHAPE_KEY_BASIS_NAME = "Basis"
BONE_TAIL_OFFSET = 1 / (SCALE_FACTOR * SCALE_FACTOR * 10)
CUSTOM_BONE_SHAPE_SCALE = Vector([0.15] * 3)
CUSTOM_BONE_SHAPE_NAME = "sphere_control"
HEAD_TEXTURE_LOGIC_NODE_NAME = "head_texture_logic"
HEAD_TEXTURE_LOGIC_NODE_LABEL = "Head Texture Logic"
BODY_TEXTURE_LOGIC_NODE_NAME = "body_texture_logic"
BODY_TEXTURE_LOGIC_NODE_LABEL = "Body Texture Logic"
UV_MAP_NAME = "DiffuseUV"
VERTEX_COLOR_ATTRIBUTE_NAME = "Color"
MESH_VERTEX_COLORS_FILE_NAME = "head_vertex_colors.json"
FLOATING_POINT_PRECISION = 0.0001
DEFAULT_UV_TOLERANCE = 0.001
RBF_SOLVER_POSTFIX = "_UERBFSolver"

HEAD_MESH_SHADER_MAPPING = {
    "head_lod": "head_shader",
    "teeth_lod": "teeth_shader",
    "saliva_lod": "saliva_shader",
    "eyeLeft_lod": "eyeLeft_shader",
    "eyeRight_lod": "eyeRight_shader",
    "eyeshell_lod": "eyeshell_shader",
    "eyelashes_lod": "eyelashes_shader",
    "eyelashesShadow_lod": "eyelashesShadow_shader",
    "eyeEdge_lod": "eyeEdge_shader",
    "cartilage_lod": "cartilage_shader",
}
BODY_MESH_SHADER_MAPPING = {"body_lod": "body_shader"}

TEMP_FOLDER = Path(tempfile.gettempdir()) / f"{ToolInfo.NAME}_addon"


@lru_cache(maxsize=1)
def get_user_data_folder() -> Path:
    """Return this extension's persistent, per-user data folder.

    Resolves to ``bpy.utils.extension_path_user`` — a user-writable
    location that requires no admin rights and, unlike the system temp
    folder, is never swept by the OS (Windows Storage Sense, Linux
    ``/tmp`` clearing on reboot, macOS ``/var/folders`` pruning). Use it
    for expensive-to-regenerate caches that must survive across sessions.

    Falls back to ``TEMP_FOLDER`` when the add-on is loaded as a legacy
    add-on rather than an installed extension (e.g. the test harness),
    where Blender cannot resolve an extension path.
    """
    try:
        return Path(bpy.utils.extension_path_user(__package__, create=True))  # pyright: ignore[reportArgumentType]
    except (ValueError, AttributeError):
        TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
        return TEMP_FOLDER


RESOURCES_FOLDER = Path(__file__).parent / "resources"
BINDINGS_FOLDER = Path(__file__).parent / "bindings"
POSES_FOLDER = RESOURCES_FOLDER / "poses"
BLENDS_FOLDER = RESOURCES_FOLDER / "blends"
SCRIPTS_FOLDER = RESOURCES_FOLDER / "scripts"
IMAGES_FOLDER = RESOURCES_FOLDER / "images"
MAPPINGS_FOLDER = RESOURCES_FOLDER / "mappings"
UI_FOLDER = RESOURCES_FOLDER / "ui"
RIG_DEFINITIONS_FOLDER = RESOURCES_FOLDER / "rig_definitions"

DEFAULT_BACKUPS_FOLDER = TEMP_FOLDER / "backups"

HEAD_TO_BODY_EDGE_LOOP_FILE_PATH = MAPPINGS_FOLDER / "head_to_body_edge_loop.json"

MESH_VERTEX_COLORS_FILE_PATH = MAPPINGS_FOLDER / MESH_VERTEX_COLORS_FILE_NAME

MASKS_TEXTURE_FILE_PATH = IMAGES_FOLDER / MASKS_TEXTURE

MATERIALS_FILE_PATH = BLENDS_FOLDER / "materials.blend"

FACE_BOARD_FILE_PATH = BLENDS_FOLDER / "face_board.blend"

ALTERNATE_TEXTURE_FILE_EXTENSIONS = [".tga", ".png"]

ALTERNATE_HEAD_TEXTURE_FILE_NAMES = {
    "head_color_map.tga": "Head_Basecolor",
    "head_normal_map.tga": "Head_Normal",
    "head_cm1_color_map.tga": "Head_Basecolor_Animated_CM1",
    "head_cm2_color_map.tga": "Head_Basecolor_Animated_CM2",
    "head_cm3_color_map.tga": "Head_Basecolor_Animated_CM3",
    "head_wm1_normal_map.tga": "Head_Normal_Animated_WM1",
    "head_wm2_normal_map.tga": "Head_Normal_Animated_WM2",
    "head_wm3_normal_map.tga": "Head_Normal_Animated_WM3",
    "eyes_color_map.tga": "Eyes_Color",
    "eyes_normal_map.tga": "Eyes_Normal",
    "teeth_color_map.tga": "Teeth_Color",
    "teeth_normal_map.tga": "Teeth_Normal",
    "eyelashes_color_map.tga": "Eyelashes_Color",
    "body_color_map.tga": "Body_Basecolor",
    "body_normal_map.tga": "Body_Normal",
}

LEGACY_ALTERNATE_HEAD_TEXTURE_FILE_NAMES = {
    "head_color_map.tga": "FaceColor_MAIN",
    "head_cm1_color_map.tga": "FaceColor_CM1",
    "head_cm2_color_map.tga": "FaceColor_CM2",
    "head_cm3_color_map.tga": "FaceColor_CM3",
    "head_normal_map.tga": "FaceNormal_MAIN",
    "head_wm1_normal_map.tga": "FaceNormal_WM1",
    "head_wm2_normal_map.tga": "FaceNormal_WM2",
    "head_wm3_normal_map.tga": "FaceNormal_WM3",
    "head_roughness_map.tga": "FaceRoughness_MAIN",
}

HEAD_MAPS = {
    "Color_MAIN": "Head_Basecolor.png",
    "Color_CM1": "Head_Basecolor_Animated_CM1.png",
    "Color_CM2": "Head_Basecolor_Animated_CM2.png",
    "Color_CM3": "Head_Basecolor_Animated_CM3.png",
    "Normal_MAIN": "Head_Normal.png",
    "Normal_WM1": "Head_Normal_Animated_WM1.png",
    "Normal_WM2": "Head_Normal_Animated_WM2.png",
    "Normal_WM3": "Head_Normal_Animated_WM3.png",
}

BODY_MAPS = {"Color_MAIN": "Body_Basecolor.png", "Normal_MAIN": "Body_Normal.png"}

UNREAL_EXPORTED_HEAD_MATERIAL_NAMES = ["MI_HeadSynthesized_Baked"]

PLATFORM_NAMES = {
    "linux": "Linux",
    "linux2": "Linux",
    "win32": "Windows",
    "darwin": "Mac OS X",
}

FACE_GUI_EMPTIES = [
    "GRP_C_eyesAim",
    "GRP_faceGUI",
    "LOC_C_eyeDriver",
    "head_grp",
    "headRig_grp",
    "headGui_grp",
    "headRigging_grp",
    "eyesSetup_grp",
]

EYE_AIM_BONES = [
    "LOC_R_eyeUIDriver",
    "LOC_L_eyeUIDriver",
    "LOC_C_eyeUIDriver",
    "LOC_R_eyeDriver",
    "LOC_L_eyeDriver",
    "LOC_C_eyeDriver",
    "LOC_R_eyeAimDriver",
    "LOC_L_eyeAimDriver",
    "LOC_R_eyeAimUp",
    "LOC_L_eyeAimUp",
    "GRP_convergenceGUI",
    "GRP_L_eyeAim",
    "GRP_R_eyeAim",
    "FRM_convergenceGUI",
    "FRM_convergenceSwitch",
    "TEXT_convergence",
    "CTRL_C_eyesAim",
    "CTRL_L_eyeAim",
    "CTRL_R_eyeAim",
    "CTRL_convergenceSwitch",
]

FACE_BOARD_SWITCHES = ["CTRL_rigLogicSwitch", "CTRL_lookAtSwitch", "CTRL_faceGUIfollowHead", "CTRL_eyesAimFollowHead"]

EXCLUDED_FACE_BOARD_CONTROLS = ["CTRL_rigLogic", "CTRL_expressions", "CTRL_faceGUI"]

HEAD_TO_BODY_LOD_MAPPING = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

# Set to Ada's height, but locations will be scaled proportionally to match spine_04 location from DNA file.
# Also in Y-up coordinate system like the metahuman creator DNA files
FIRST_BONE_Y_LOCATION = 107.86403

EXTRA_BONES = [
    ("root", {"parent": None, "location": Vector((0, 0, 0)), "rotation": Euler((0, 0, 0), "XYZ")}),
    (
        "pelvis",
        {
            "parent": "root",
            "location": Vector((0.0, 0.8707, 0.0209)),
            "rotation": Euler((math.radians(-90.0), math.radians(-2.053), math.radians(90.0)), "XYZ"),
        },
    ),
    (
        "spine_01",
        {
            "parent": "pelvis",
            "location": Vector((0.0, 0.8910, 0.0206)),
            "rotation": Euler((math.radians(-90.0), math.radians(-13.003), math.radians(90.0)), "XYZ"),
        },
    ),
    (
        "spine_02",
        {
            "parent": "spine_01",
            "location": Vector((0.0, 0.9326, 0.0302)),
            "rotation": Euler((math.radians(-90.0), math.radians(-5.68216), math.radians(90.0)), "XYZ"),
        },
    ),
    (
        "spine_03",
        {
            "parent": "spine_02",
            "location": Vector((0.0, 0.9998, 0.0369)),
            "rotation": Euler((math.radians(-90.0), math.radians(3.82404), math.radians(90.0)), "XYZ"),
        },
    ),
]


class BodyBoneCollection:
    DRIVERS = "Drivers"
    DRIVEN = "Driven"
    TWISTS = "Twists"
    SWINGS = "Swings"


# Bone collection that holds the volume bones (the non-skinned leaf joints,
# such as the volumetric face helpers), so they can be hidden out of the way
# and kept out of the per-joint-group collections.
VOLUME_BONE_COLLECTION = "Volume"

# Bone collection that holds the internal bones, so the user can solo it to
# quickly hide the surface-skinning and volume leaf bones and get a clear view
# of the internal skeleton on the head rig. A bone is internal when it has
# children, or it is a leaf joint that skins a non-surface mesh (such as the
# teeth or eyes) rather than the surface (``head_lod0_mesh``) mesh.
INTERNAL_BONE_COLLECTION = "Internal"


# Prefix used for the vertex groups that store the rig-definition mesh regions.
REGION_VERTEX_GROUP_PREFIX = "REGION_"


ADDON_IDS = ["meta_human_dna", "meta_human_dna_pro", "character_dna", "character_dna_pro"]
# Maps each current-generation edition to its sibling edition. Both editions
# store rig instances under the same ``rig_instance_list`` key with an identical
# layout. Saved scene ``PointerProperty`` data is only readable when a matching
# property is registered, so the sibling's scene pointer is registered read-only
# (see ``properties.register``) to expose data saved by the other edition for
# migration.
SIBLING_EDITIONS = {"character_dna": "character_dna_pro", "character_dna_pro": "character_dna"}
# The old prototype (``meta_human_dna``) stored its rig instances under
# ``rig_logic_instance_list``. These keys require the field-by-field legacy
# migration that maps the old field names onto the current property group.
LEGACY_DATA_KEYS = ["rig_logic_instance_list"]
# Every scene-data key that can hold rig instance data, newest first. The
# current format (shared by ``character_dna`` and ``character_dna_pro``) uses
# ``rig_instance_list``; the old prototype used ``rig_logic_instance_list``.
# Used to detect migratable data and to read it regardless of which edition or
# addon version saved the .blend file.
MIGRATABLE_DATA_KEYS = ["rig_instance_list", "rig_logic_instance_list"]

PRO_EDITORS = (
    ("Converter", "RNA"),
    ("Mesh Editor", "MESH_DATA"),
    ("Raw Editor", "DECORATE_DRIVER"),
    ("RBF Editor", "DRIVER_ROTATIONAL_DIFFERENCE"),
    ("Shape Key Editor", "SHAPEKEY_DATA"),
    ("Backup Manager", "FILE_BACKUP"),
)
