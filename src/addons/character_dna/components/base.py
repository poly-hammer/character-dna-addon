# standard library imports
import json
import logging
import math
import re
import sys

from abc import ABCMeta, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# third party imports
import bpy

# local imports
from .. import utilities
from ..constants import (
    ALTERNATE_HEAD_TEXTURE_FILE_NAMES,
    ALTERNATE_TEXTURE_FILE_EXTENSIONS,
    BODY_MATERIAL_NAME,
    BODY_MESH_SHADER_MAPPING,
    BODY_TEXTURE_LOGIC_NODE_NAME,
    DEFAULT_UV_TOLERANCE,
    FACE_BOARD_NAME,
    HEAD_MATERIAL_NAME,
    HEAD_MESH_SHADER_MAPPING,
    HEAD_TEXTURE_LOGIC_NODE_NAME,
    INVALID_NAME_CHARACTERS_REGEX,
    LEGACY_ALTERNATE_HEAD_TEXTURE_FILE_NAMES,
    MASKS_TEXTURE,
    MATERIALS_FILE_PATH,
    NUMBER_OF_HEAD_LODS,
    SCALE_FACTOR,
    UNREAL_EXPORTED_HEAD_MATERIAL_NAMES,
    UV_MAP_NAME,
    ComponentType,
)
from ..dna_io import DNAImporter, get_dna_reader
from ..typing import *  # noqa: F403
from ..utilities import preserve_context


logger = logging.getLogger(__name__)


class CharacterComponentBase(metaclass=ABCMeta):
    def __init__(
        self,
        name: str | None = None,
        rig_instance: "RigInstance | None" = None,
        dna_file_path: Path | None = None,
        dna_import_properties: "CharacterImportProperties | None" = None,
        component_type: ComponentType = "head",
        focus_on_import: bool = True,
    ):
        # make sure dna file path is a Path object
        dna_file_path = Path(bpy.path.abspath(str(dna_file_path))) if dna_file_path else None

        assert rig_instance or dna_file_path, (
            f"Either rig_instance or dna_file_path must be provided to {self.__class__.__name__}!"
        )

        self._linear_modifier = None
        self._angle_modifier = None
        self._component_type = component_type
        self._focus_on_import = focus_on_import

        # determine the asset root folder based on the dna file path
        self.asset_root_folder = None
        if dna_file_path:
            self.asset_root_folder = dna_file_path.parent
        elif rig_instance:
            if rig_instance.head_dna_file_path:
                self.asset_root_folder = Path(bpy.path.abspath(str(rig_instance.head_dna_file_path))).parent
            elif rig_instance.body_dna_file_path:
                self.asset_root_folder = Path(bpy.path.abspath(str(rig_instance.body_dna_file_path))).parent

        self.rig_instance: "RigInstance" = rig_instance  # type: ignore[assignment]  # noqa: UP037
        self.addon_properties: "CharacterAddonPreferences" = utilities.get_addon_preferences()  # pyright: ignore[reportAttributeAccessIssue]  # noqa: UP037
        self.window_manager_properties: CharacterWindowManagerProperties = (
            utilities.get_addon_window_manager_properties()
        )
        self.scene_properties: "CharacterSceneProperties" = utilities.get_addon_scene_properties()  # noqa: UP037
        self.dna_import_properties: "CharacterImportProperties" = dna_import_properties  # noqa: UP037

        # if no rig_instance is provided, create a new one and supply the dna_file_path to it
        if not self.rig_instance and dna_file_path:
            name = self._get_name(name=name, dna_file_path=dna_file_path)
            # find a rig instance with the same name and use it if it exists
            for instance in self.scene_properties.rig_instance_list:
                if instance.name == name:
                    self.rig_instance = instance
                    break
            # otherwise create a new one
            else:
                self.rig_instance = self.scene_properties.rig_instance_list.add()
                self.rig_instance.name = name
                # set the active rig instance
                self.scene_properties.rig_instance_list_active_index = len(self.scene_properties.rig_instance_list) - 1
                # notify registered callbacks that a rig instance was added to the list
                utilities.notify_rig_instances_changed(self.rig_instance)

            if component_type == "head":
                self.rig_instance.head_dna_file_path = str(dna_file_path)
            elif component_type == "body":
                self.rig_instance.body_dna_file_path = str(dna_file_path)

        if (not self.dna_import_properties or not self.dna_import_properties.alternate_maps_folder) and dna_file_path:
            self.maps_folder = dna_file_path.parent / "Maps"
            if not self.maps_folder.exists():
                self.maps_folder = dna_file_path.parent / "maps"
        elif self.dna_import_properties and self.dna_import_properties.alternate_maps_folder:
            self.maps_folder = Path(self.dna_import_properties.alternate_maps_folder)

        file_path = dna_file_path or self.dna_file_path
        # Only set when the DNA is actually on disk, so every reader access has to tolerate None.
        self.dna_reader = None
        self.dna_importer = None
        if file_path.is_file() and file_path.exists():
            self.dna_reader = get_dna_reader(file_path=dna_file_path or self.dna_file_path, file_format="binary")
            self.dna_importer = DNAImporter(
                instance=self.rig_instance,
                import_properties=self.dna_import_properties,
                linear_modifier=self.linear_modifier,
                reader=self.dna_reader,
                component_type=self.component_type,
                dna_file_path=dna_file_path,
            )

    @property
    def component_type(self) -> ComponentType:
        return self._component_type  # type: ignore[return-value]

    @property
    def linear_modifier(self) -> float:
        from ..bindings import enums  # type: ignore[reportAttributeAccessIssue]

        if not self.dna_reader:
            logger.warning(f'No DNA file was read for "{self.name}", assuming meters.')
            return 1

        unit = enums.TranslationUnit(self.dna_reader.getTranslationUnit())  # pyright: ignore[reportCallIssue]
        # is centimeter
        if unit.name.lower() == "cm":
            return 1 / SCALE_FACTOR
        # is meter
        if unit.name.lower() == "m":
            return 1
        return 1

    @property
    def angle_modifier(self) -> float:
        from ..bindings import enums  # type: ignore[reportAttributeAccessIssue]

        if not self.dna_reader:
            logger.warning(f'No DNA file was read for "{self.name}", assuming degrees.')
            return 180 / math.pi

        unit = enums.RotationUnit(self.dna_reader.getRotationUnit())  # pyright: ignore[reportCallIssue]
        # is degree
        if unit.name.lower() == "degrees":
            return 180 / math.pi
        # is radians
        if unit.name.lower() == "radians":
            return math.pi / 180
        return 1

    @property
    def name(self) -> str:
        return self.rig_instance.name

    @property
    def dna_file_path(self) -> Path:
        if self._component_type == "head":
            return Path(bpy.path.abspath(self.rig_instance.head_dna_file_path))
        if self._component_type == "body":
            return Path(bpy.path.abspath(self.rig_instance.body_dna_file_path))
        return None  # type: ignore[return-value]

    @property
    def face_board_object(self) -> bpy.types.Object | None:
        return self.rig_instance.face_board or bpy.data.objects.get(f"{self.name}_{FACE_BOARD_NAME}")

    @property
    def head_mesh_object(self) -> bpy.types.Object | None:
        return self.rig_instance.head_mesh or bpy.data.objects.get(f"{self.name}_head_lod0_mesh")

    @property
    def head_rig_object(self) -> bpy.types.Object | None:
        return self.rig_instance.head_rig or bpy.data.objects.get(f"{self.name}_head_rig")

    @property
    def body_mesh_object(self) -> bpy.types.Object | None:
        return self.rig_instance.body_mesh or bpy.data.objects.get(f"{self.name}_body_lod0_mesh")

    @property
    def body_rig_object(self) -> bpy.types.Object | None:
        return self.rig_instance.body_rig or bpy.data.objects.get(f"{self.name}_body_rig")

    @property
    def metadata(self) -> dict:
        if not self.asset_root_folder:
            return {}

        export_manifest = self.asset_root_folder / "ExportManifest.json"
        if export_manifest.exists():
            with export_manifest.open() as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load metadata from '{export_manifest}'")
                    return {}
        logger.warning("Could not load metahuman metadata file! Must not be in a metahuman directory.")
        return {}

    @property
    def thumbnail(self) -> Path | None:
        if not self.asset_root_folder:
            return None

        name = self.metadata.get("metaHumanName")
        if name:
            thumbnail_path = self.asset_root_folder / f"{name}.png"
            if thumbnail_path.exists():
                return thumbnail_path
        return None

    def _get_name(self, name: str | None = None, dna_file_path: Path | None = None) -> str:
        if name:
            return re.sub(INVALID_NAME_CHARACTERS_REGEX, "_", name)
        if dna_file_path:
            name = re.sub(INVALID_NAME_CHARACTERS_REGEX, "_", name or dna_file_path.stem.strip())
        return self.metadata.get("metaHumanName", name)

    def _get_lods_settings(self) -> list[tuple[int, bool]]:
        return [(i, getattr(self.dna_import_properties, f"import_lod{i}")) for i in range(NUMBER_OF_HEAD_LODS)]

    def _get_rig_definition(self) -> "RigDefinition | None":
        """Return the rig definition for this component, or ``None`` if missing.

        The colors, joint-group names and region memberships used on import live
        only in the rig definition, so callers degrade gracefully (log + skip)
        when it cannot be loaded.
        """
        from ..rig_definition import get_rig_definition

        try:
            return get_rig_definition(self.component_type)
        except (FileNotFoundError, ValueError, KeyError) as error:
            logger.warning(f"No rig definition available for '{self.component_type}': {error}")
            return None

    def _build_joint_group_collections(self):
        """Author one bone collection per rig-definition joint group and color bones.

        Joint-group names and per-joint colors are derived from the rig
        definition (the DNA stores neither), so this is skipped when the rig
        definition is unavailable.
        """
        rig_object = self.head_rig_object if self.component_type == "head" else self.body_rig_object
        if not rig_object:
            return

        # The volume bones are the leaf joints that do not skin the LOD0 mesh
        # (the volumetric helper joints, which only drive the lower LODs). They
        # depend only on skin weights and the bone hierarchy, so they are
        # authored even when no rig definition is available.
        skinned_joint_names = self._get_skinned_joint_names(lod=0)
        volume_joint_names = utilities.get_volume_joint_names(
            rig_object=rig_object,
            skinned_joint_names=skinned_joint_names,
        )
        utilities.assign_volume_bone_collection(
            rig_object=rig_object,
            volume_joint_names=volume_joint_names,
        )
        # The internal bones are the non-leaf joints plus the leaf joints that
        # skin a non-surface mesh (such as the teeth or eyes) instead of the
        # surface (``*_lod0_mesh``) mesh, so the surface skinning is resolved
        # separately from the full LOD0 skin set.
        utilities.assign_internal_bone_collection(
            rig_object=rig_object,
            skinned_joint_names=skinned_joint_names,
            surface_skinned_joint_names=self._get_skinned_joint_names(
                lod=0,
                mesh_names={f"{self.component_type}_lod0_mesh"},
            ),
        )

        rig_definition = self._get_rig_definition()
        if not rig_definition or not rig_definition.joint_groups:
            return

        # Keep the volume bones out of the per-joint-group collections so they
        # are owned solely by the Volume collection (the rig-definition joint
        # groups still legitimately share surface joints with one another).
        utilities.assign_joint_group_bone_collections(
            rig_object=rig_object,
            joint_groups=rig_definition.joint_groups,
            color_by_joint_name=rig_definition.color_by_joint_name,
            exclude_joint_names=volume_joint_names,
        )

    def _get_skinned_joint_names(self, lod: "int | None" = None, mesh_names: "set[str] | None" = None) -> set[str]:
        """Return the names of every joint that carries at least one skin-weight
        influence on the DNA's meshes.

        When ``lod`` is given, only the meshes for that LOD are considered; this
        is how the volume joints are found (a leaf joint that does not skin the
        LOD0 mesh is a volumetric helper, even though it skins a lower LOD).
        When ``mesh_names`` is given, only the meshes with those DNA names are
        considered; this is how the surface-skinning joints are isolated from the
        joints that skin the other meshes (such as the teeth or eyes).
        """
        reader = self.dna_reader
        joint_name_cache: dict[int, str] = {}
        skinned_joint_names: set[str] = set()
        mesh_indices = reader.getMeshIndicesForLOD(lod) if lod is not None else range(reader.getMeshCount())
        for mesh_index in mesh_indices:
            if mesh_names is not None and reader.getMeshName(mesh_index) not in mesh_names:
                continue
            for vertex_index in range(reader.getSkinWeightsCount(mesh_index)):
                for joint_index in reader.getSkinWeightsJointIndices(mesh_index, vertex_index):
                    joint_name = joint_name_cache.get(joint_index)
                    if joint_name is None:
                        joint_name = reader.getJointName(joint_index)
                        joint_name_cache[joint_index] = joint_name
                    skinned_joint_names.add(joint_name)
        return skinned_joint_names

    def _organize_viewport(self):
        if self.head_rig_object:
            for mesh_object in self.head_rig_object.children:
                if mesh_object.type == "MESH" and "lod0" not in mesh_object.name.lower():
                    utilities.set_hidden(mesh_object, True)

            utilities.hide_empties()
            utilities.set_hidden(self.head_rig_object, True)
            utilities.move_to_collection(
                scene_objects=[self.head_rig_object], collection_name=self.name, exclusively=True
            )

        if self.body_rig_object:
            for mesh_object in self.body_rig_object.children:
                if mesh_object.type == "MESH" and "lod0" not in mesh_object.name.lower():
                    utilities.set_hidden(mesh_object, True)

            utilities.set_hidden(self.body_rig_object, True)
            utilities.move_to_collection(
                scene_objects=[self.body_rig_object], collection_name=self.name, exclusively=True
            )

        # move the lod collections under the main asset collection
        asset_collection = bpy.data.collections.get(self.name)
        # Unlinking the lods from the scene while the asset collection sits outside it would drop
        # the whole character out of the view layer.
        if asset_collection and bpy.context.scene and utilities.is_collection_in_scene(asset_collection):
            for lod_index in range(NUMBER_OF_HEAD_LODS):
                lod_collection = bpy.data.collections.get(f"{self.name}_lod{lod_index}")
                # move the lod collection to the asset collection
                if lod_collection and lod_collection not in asset_collection.children.values():
                    asset_collection.children.link(lod_collection)
                # unlink the lod collection from the scene collection
                if lod_collection in bpy.context.scene.collection.children.values():
                    bpy.context.scene.collection.children.unlink(lod_collection)

    def _get_alternate_image_path(self, image_file: Path, mapping: dict) -> Path:
        # Check for alternate image file names
        if not image_file.exists():
            # check for alternate file names with different extensions
            for extension in ALTERNATE_TEXTURE_FILE_EXTENSIONS:
                alternate_file_name = mapping.get(image_file.name)
                if alternate_file_name:
                    # check for lowercase extension
                    alternate_image_path = self.maps_folder / f"{alternate_file_name}{extension.lower()}"
                    if alternate_image_path.exists():
                        return alternate_image_path

                    # check for uppercase extension
                    alternate_image_path = self.maps_folder / f"{alternate_file_name}{extension.upper()}"
                    if alternate_image_path.exists():
                        return alternate_image_path
        return image_file

    def _set_image_textures(self, materials: list[bpy.types.Material]):
        for material in materials:
            if not material.node_tree:
                continue

            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:  # type: ignore[attr-defined]
                    # get the image file name without the postfixes for duplicates i.e. .001
                    image_file = node.image.name  # type: ignore[attr-defined]
                    if image_file.count(".") > 1:
                        image_file = image_file.rsplit(".", 1)[0]

                    # update the texture paths to images in the maps folder
                    new_image_path = self.maps_folder / image_file

                    # Check for alternate image file names
                    new_image_path = self._get_alternate_image_path(
                        new_image_path, mapping=ALTERNATE_HEAD_TEXTURE_FILE_NAMES
                    )
                    if not new_image_path.exists():
                        new_image_path = self._get_alternate_image_path(
                            new_image_path, mapping=LEGACY_ALTERNATE_HEAD_TEXTURE_FILE_NAMES
                        )

                    if new_image_path.exists():
                        node.image = bpy.data.images.load(str(new_image_path))  # type: ignore[attr-defined]

                    # Set the color space for color and normal textures, taking into account alternate
                    # color management workflows like ACES
                    stem = new_image_path.stem.lower()
                    try:
                        if stem.endswith(("color_map", "color")) or "color_animated_" in stem:
                            try:
                                node.image.colorspace_settings.name = "sRGB"  # type: ignore[attr-defined]
                            except TypeError:
                                node.image.colorspace_settings.name = "sRGB - Display"  # type: ignore[attr-defined]

                        if stem.endswith(("normal_map", "normal")) or "normal_animated_" in stem:
                            try:
                                node.image.colorspace_settings.name = "Non-Color"  # type: ignore[attr-defined]
                            except TypeError:
                                node.image.colorspace_settings.name = "Raw"  # type: ignore[attr-defined]

                    except Exception as error:
                        logger.error(f"Failed to set colorspace for {node.image.name}: {error}")  # type: ignore[attr-defined]

        # load the placeholder mask/topology textures and fix up the texture logic node groups
        utilities.setup_texture_logic_node_groups(self.component_type)  # pyright: ignore[reportArgumentType]

    def _purge_existing_materials(self):
        shader_mapping = HEAD_MESH_SHADER_MAPPING if self.component_type == "head" else BODY_MESH_SHADER_MAPPING
        for material_name in shader_mapping.values():
            material = bpy.data.materials.get(f"{self.name}_{material_name}")
            if material:
                bpy.data.materials.remove(material)

        shared_constants = utilities.get_topology_texture_constants()
        if self.component_type == "head":
            masks_image = bpy.data.images.get(MASKS_TEXTURE)
            if masks_image:
                bpy.data.images.remove(masks_image)

            if shared_constants:
                head_topology_image = bpy.data.images.get(shared_constants.HEAD_TOPOLOGY_TEXTURE)
                if head_topology_image:
                    bpy.data.images.remove(head_topology_image)

        elif self.component_type == "body":
            if shared_constants:
                body_topology_image = bpy.data.images.get(shared_constants.BODY_TOPOLOGY_TEXTURE)
                if body_topology_image:
                    bpy.data.images.remove(body_topology_image)

    def _delete_rig_instance(self):
        if (
            not self.rig_instance.head_mesh
            and not self.rig_instance.head_rig
            and not self.rig_instance.body_mesh
            and not self.rig_instance.body_rig
        ):
            my_list = self.scene_properties.rig_instance_list
            active_index = self.scene_properties.rig_instance_list_active_index
            my_list.remove(active_index)
            to_index = min(active_index, len(my_list) - 1)
            self.scene_properties.rig_instance_list_active_index = to_index
            # notify registered callbacks that a rig instance was removed from the list
            utilities.notify_rig_instances_changed()

    def import_materials(self) -> list[bpy.types.Material] | None:  # noqa: PLR0912
        if self.dna_import_properties and not self.dna_import_properties.import_materials:
            return None

        from ..ui import callbacks

        sep = "\\"
        if sys.platform != "win32":
            sep = "/"

        logger.info(f"Importing materials for {self.name}")
        materials = []
        directory_path = f"{MATERIALS_FILE_PATH}{sep}Material{sep}"

        # Set the active collection to the scene collection. This ensures that the materials are appended
        # to the scene collection.
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection  # pyright: ignore[reportOptionalMemberAccess]

        # remove existing matching materials for this face to avoid duplicates being imported
        self._purge_existing_materials()

        shader_mapping = HEAD_MESH_SHADER_MAPPING if self.component_type == "head" else BODY_MESH_SHADER_MAPPING
        for key, material_name in shader_mapping.items():
            material = bpy.data.materials.get(material_name)
            if not material:
                # import the materials
                file_path = f"{MATERIALS_FILE_PATH}{sep}Material{sep}{material_name}"
                bpy.ops.wm.append(filepath=file_path, filename=material_name, directory=directory_path)

                # get the imported material
                material = bpy.data.materials.get(material_name)
                if not material:
                    material = bpy.data.materials.get(f"{self.name}_{material_name}")
                    # create the transparent materials if they don't exist
                    # These are for eyes and saliva
                    if not material:
                        material = utilities.create_new_material(
                            name=f"{self.name}_{material_name}", color=(1.0, 1.0, 1.0, 0.0), alpha=0.0
                        )

                # set the material on the head texture logic instance
                if material.name == HEAD_MATERIAL_NAME:
                    self.rig_instance.head_material = material
                    node = callbacks.get_head_texture_logic_node(material)
                    if node:
                        node.name = f"{self.name}_{HEAD_TEXTURE_LOGIC_NODE_NAME}"
                        node.label = f"{self.name}_{HEAD_TEXTURE_LOGIC_NODE_NAME}"
                        if node.node_tree:
                            node.node_tree.name = f"{self.name}_{HEAD_TEXTURE_LOGIC_NODE_NAME}"

                # set the material on the body texture logic instance
                if material.name == BODY_MATERIAL_NAME:
                    self.rig_instance.body_material = material
                    node = callbacks.get_body_texture_logic_node(material)
                    if node:
                        node.name = f"{self.name}_{BODY_TEXTURE_LOGIC_NODE_NAME}"
                        node.label = f"{self.name}_{BODY_TEXTURE_LOGIC_NODE_NAME}"
                        if node.node_tree:
                            node.node_tree.name = f"{self.name}_{BODY_TEXTURE_LOGIC_NODE_NAME}"

                # rename to match metahuman
                material.name = f"{self.name}_{material_name}"

                # set the uv maps on the material nodes
                if material.node_tree:
                    for node in material.node_tree.nodes:
                        if node.type == "UVMAP":
                            node.uv_map = UV_MAP_NAME  # type: ignore[attr-defined]
                for mesh_object in bpy.data.objects:
                    if mesh_object.name.startswith(f"{self.name}_{key}"):
                        if mesh_object.data.materials:  # type: ignore[attr-defined]
                            mesh_object.data.materials[0] = material  # type: ignore[attr-defined]
                        else:
                            mesh_object.data.materials.append(material)  # type: ignore[attr-defined]

            if material:
                materials.append(material)

        # switch to material view
        utilities.set_viewport_shading("MATERIAL")

        # set the image textures to match
        self._set_image_textures(materials)
        # prefix the material image names with the metahuman name
        for material in materials:
            utilities.prefix_material_image_names(material=material, prefix=self.name)

        return materials

    def validate_conversion(
        self, mesh_object: bpy.types.Object, tolerance: float = DEFAULT_UV_TOLERANCE
    ) -> tuple[bool, str]:
        if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
            return False, f'The mesh "{mesh_object.name}" has no data! Please provide a valid mesh object.'

        uv_layers = mesh_object.data.uv_layers
        if len(uv_layers) != 1:
            return (
                False,
                (
                    f'The mesh "{mesh_object.name}" must have exactly one UV layer! '
                    "Please ensure the mesh has a single UV map."
                ),
            )

        uv_layer = uv_layers.active
        if uv_layer is None:
            return (
                False,
                f'The mesh "{mesh_object.name}" has no active UV layer! Please ensure the mesh has an active UV map.',
            )

        # the first mesh index is always the head mesh or body mesh
        dna_u_values = utilities.reduce_close_floats(
            float_list=[float(i) for i in self.dna_reader.getVertexTextureCoordinateUs(0)], tolerance=tolerance
        )
        dna_v_values = utilities.reduce_close_floats(
            float_list=[float(i) for i in self.dna_reader.getVertexTextureCoordinateVs(0)], tolerance=tolerance
        )

        u_values, v_values = utilities.get_uv_values(mesh_object=mesh_object)
        u_values = utilities.reduce_close_floats(float_list=u_values, tolerance=tolerance)
        v_values = utilities.reduce_close_floats(float_list=v_values, tolerance=tolerance)

        if len(u_values) != len(dna_u_values) or len(v_values) != len(dna_v_values):
            uv_differences = abs(len(u_values) - len(dna_u_values)) + abs(len(v_values) - len(dna_v_values))
            return False, (
                f'UV validation failed! The mesh "{mesh_object.name}" has {uv_differences} UV values '
                "that do not match the layout in the template DNA file. Did you select the correct component? "
                'Right-click the "Convert Selected to DNA" button to see the online manual that shows '
                "the correct UV layout. Otherwise, disable the UV validation or adjust "
                "the tolerance value."
            )

        return True, "Validation successful!"

    @preserve_context
    def pre_convert_mesh_cleanup(self, mesh_object: bpy.types.Object) -> bpy.types.Object | None:
        if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
            return None

        mesh_object_name = mesh_object.name
        mesh_name = mesh_object.data.name
        head_material_name = None
        for material in mesh_object.data.materials:
            if material and material.name in UNREAL_EXPORTED_HEAD_MATERIAL_NAMES:
                head_material_name = material.name

        # separate the head mesh by material if it has the a unreal head material
        if head_material_name:
            new_mesh_object = None
            utilities.switch_to_edit_mode(mesh_object)
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.separate(type="MATERIAL")
            for separated_mesh in bpy.context.selectable_objects or []:
                if head_material_name in [i.name for i in separated_mesh.data.materials]:  # type: ignore[attr-defined]
                    new_mesh_object = separated_mesh
                    new_mesh_object.name = mesh_object_name
                    new_mesh_object.data.name = mesh_name  # type: ignore[attr-defined]
                else:
                    bpy.data.objects.remove(separated_mesh, do_unlink=True)
            return new_mesh_object

        return mesh_object

    def write_export_manifest(self):
        """
        Writes the export manifest to a JSON file like MetaHuman Creator does for a DCC export.
        """
        file_path = Path(bpy.path.abspath(str(self.rig_instance.output.folder_path))) / "ExportManifest.json"
        with file_path.open("w") as file:
            json.dump(
                {
                    "metaHumanName": self.name,
                    "exportBlenderAddonVersion": utilities.get_addon_version(),
                    "exportPluginVersion": self.metadata.get("exportPluginVersion", "1.0.0"),
                    "exportEngineVersion": self.metadata.get("exportEngineVersion", "5.6.0-0+UE5"),
                    "exportedAt": datetime.now(tz=UTC).strftime("%Y.%m.%d-%H.%M.%S"),
                },
                file,
                indent=4,
            )

    @preserve_context
    def constrain_head_to_body(self):
        utilities.constrain_head_to_body(self.rig_instance)

    def _get_head_to_body_constraints(self) -> list[bpy.types.Constraint]:
        """Return the head→body copy-transforms constraints, cached.

        Dragging the influence slider fires the update callback on every
        step, and rescanning every head pose bone (rebuilding the
        per-bone constraint name and doing a name lookup) made the panel
        stutter on dense MetaHuman head rigs. The constraint references
        are stable for the lifetime of the rig, so they are cached on the
        instance and rebuilt lazily; the cache is dropped when the
        constraints are (re)built and on undo via ``destroy_references``.
        """
        instance = self.rig_instance
        head_rig = instance.head_rig
        if not head_rig:
            return []

        key = instance.cache_key("head", "body_constraints")
        constraints = instance.data.get(key)
        if constraints is None:
            constraints = []
            for pose_bone in head_rig.pose.bones:
                constraint = pose_bone.constraints.get(utilities.get_body_constraint_name(pose_bone.name))
                if constraint:
                    constraints.append(constraint)
            instance.data[key] = constraints
        return constraints

    def set_head_to_body_constraint_influence(self, influence: float):
        try:
            for constraint in self._get_head_to_body_constraints():
                constraint.influence = influence
        except ReferenceError:
            # A cached constraint wrapper went stale (e.g. the rig was rebuilt
            # without the cache being invalidated); drop it and rebuild once.
            self.rig_instance.data.pop(self.rig_instance.cache_key("head", "body_constraints"), None)
            for constraint in self._get_head_to_body_constraints():
                constraint.influence = influence

    @abstractmethod
    def ingest(self, align: bool = True, constrain: bool = True) -> tuple[bool, str]:
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def set_pose(self):
        pass

    @abstractmethod
    def reset_poses(self):
        pass

    @abstractmethod
    def import_action(self, file_path: Path, **kwargs: Any) -> bpy.types.Action | None:
        pass
