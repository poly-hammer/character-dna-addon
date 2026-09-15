# standard library imports
import json
import logging
import math

from collections.abc import Callable
from pathlib import Path

# third party imports
import bmesh
import bpy
import numpy as np

from mathutils import Matrix, Vector

# local imports
from .. import utilities
from ..bindings import dna  # pyright: ignore[reportAttributeAccessIssue]
from ..constants import (
    EXTRA_BONES,
    SCALE_FACTOR,
    SHAPE_KEY_BASIS_NAME,
    SHAPE_KEY_DELTA_THRESHOLD,
    SHAPE_KEY_NAME_MAX_LENGTH,
    VERTEX_COLOR_ATTRIBUTE_NAME,
    ComponentType,
)
from ..exceptions import InvalidComponentTypeError
from ..typing import *  # noqa: F403  # noqa: F403
from .misc import get_dna_reader, get_dna_writer


logger = logging.getLogger(__name__)


class DNAExporter:
    def __init__(
        self,
        instance: "RigInstance",
        linear_modifier: float,
        meshes: bool = True,
        shape_keys: bool = True,
        bones: bool = True,
        textures: bool = True,
        vertex_colors: bool = True,
        vertex_groups: bool = True,
        normals: bool = False,
        file_name: str | None = None,
        component_type: ComponentType | None = None,
        reader: "BinaryStreamReader | None" = None,
        progress_callback: "Callable[[str, float | None], None] | None" = None,
        seam_follower: ComponentType | None = "head",
        seam_reference_dna_path: "str | Path | None" = None,
        zero_shape_deltas: bool = False,
    ):
        self._instance = instance
        self._linear_modifier = linear_modifier
        self._prefix = instance.name

        self._include_meshes = meshes
        self._include_shape_keys = shape_keys
        self._include_vertex_groups = vertex_groups
        self._include_bones = bones
        self._include_textures = textures
        self._include_vertex_colors = vertex_colors
        self._include_normals = normals
        self._zero_shape_deltas = zero_shape_deltas
        self._progress_callback = progress_callback
        # Seam alignment between the head and body neck edge loop. ``seam_follower``
        # names which component is snapped onto the other ("head" -> head conforms
        # to the body, the default Output-panel behavior; "body" -> body conforms
        # to the head, used by the converter once the head DNA is already written).
        # ``None`` disables seam alignment entirely. ``seam_reference_dna_path`` is
        # the on-disk DNA to read the reference component's seam from when the
        # cached reader would be stale (the converter's just-written head.dna).
        self._seam_follower = seam_follower
        self._seam_reference_dna_path = Path(seam_reference_dna_path) if seam_reference_dna_path is not None else None
        self._component_type = component_type or instance.output.component

        self._output_folder = Path(bpy.path.abspath(instance.output.folder_path))

        if self._component_type == "head":
            self.source_dna_file = Path(bpy.path.abspath(instance.head_dna_file_path))
        elif self._component_type == "body":
            self.source_dna_file = Path(bpy.path.abspath(instance.body_dna_file_path))
        else:
            raise InvalidComponentTypeError(self._component_type)

        self._target_dna_file = Path(bpy.path.abspath(instance.output.folder_path)) / (
            file_name or f"{instance.name}.dna"
        )

        # Open a read to the source DNA file if an existing reader is not provided
        if not reader:
            self._dna_reader = get_dna_reader(file_path=self.source_dna_file)
        else:
            self._dna_reader = reader

        self._dna_writer = get_dna_writer(file_path=self._target_dna_file, file_format=self._instance.output.format)
        # Populate the writer with the data from the reader
        self._dna_writer.setFrom(self._dna_reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)

        # The head and body mesh are always the first mesh in the DNA file
        if self._component_type == "head":
            self._export_lods = {0: [(instance.head_mesh, 0)]}
            self._extra_bones = EXTRA_BONES
            self._rig_object = instance.head_rig
        elif self._component_type == "body":
            self._export_lods = {0: [(instance.body_mesh, 0)]}
            self._extra_bones = []
            self._rig_object = instance.body_rig
        else:
            raise InvalidComponentTypeError(self._component_type)

        self._mesh_indices = [0]
        self._non_lod_mesh_objects = []
        self._images = []
        self._bone_index_lookup = {}
        self._vertex_color_data = []
        # (export_mesh_index, blend_shape_channel_index) pairs collected during shape
        # key export so the mesh -> blend-shape channel mapping can be rebuilt.
        self._mesh_blend_shape_channel_pairs: list[tuple[int, int]] = []
        # Guards ``initialize_scene_data`` so it can be safely called by both the
        # pre-flight validation pass (``validate_scene``) and ``run`` without
        # doubling the collected mesh/image lists.
        self._scene_initialized = False

    def _report(self, message: str, fraction: float | None = None) -> None:
        """Forward a status update to the optional progress callback.

        ``fraction`` is a ``0.0..1.0`` position within the export/calibration
        phase (or ``None`` for a message-only update). Failures in the callback
        (e.g. a UI that has gone away) are swallowed so they can never break the
        DNA write."""
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(message, fraction)
        except Exception:
            logger.exception("Progress callback failed; continuing export.")

    def initialize_scene_data(self):
        # Idempotent: the collected lists (``_mesh_indices``, ``_images``, ...) are
        # appended to below, so re-running would duplicate their contents.
        if self._scene_initialized:
            return
        self._scene_initialized = True

        mesh_objects = []
        output_items = []
        main_mesh_object = None

        if self._component_type == "head":
            output_items = self._instance.output.head_item_list
            main_mesh_object = self._instance.head_mesh
        elif self._component_type == "body":
            output_items = self._instance.output.body_item_list
            main_mesh_object = self._instance.body_mesh

        for output_item in output_items:
            if output_item.include:
                if output_item.scene_object and output_item.scene_object.type == "ARMATURE":
                    self._rig_object = output_item.scene_object
                elif main_mesh_object and output_item.scene_object == main_mesh_object:
                    continue
                elif output_item.scene_object and output_item.scene_object.type == "MESH":
                    if not self._include_meshes:
                        continue
                    mesh_objects.append(output_item.scene_object)
                elif output_item.image_object:
                    self._images.append((output_item.image_object, output_item.name))

        # Sort the meshes by the order in the ORDER dictionary
        mesh_objects.sort(key=lambda x: utilities.remove_instance_prefix(x.name, self._prefix))

        # Populate the LODs with the mesh objects and their indices
        mesh_index = 1
        for mesh_object in mesh_objects:
            index = utilities.get_lod_index(mesh_object.name)
            if index == -1:
                self._non_lod_mesh_objects.append(mesh_object)
            else:
                self._export_lods[index] = self._export_lods.get(index, [])
                self._export_lods[index].append((mesh_object, mesh_index))
                self._mesh_indices.append(mesh_index)
                mesh_index += 1

        # Also check if the main mesh is not an LOD mesh
        if main_mesh_object and utilities.get_lod_index(main_mesh_object.name) == -1:
            self._non_lod_mesh_objects.append(main_mesh_object)

        # Initialize the vertex color data array. A fresh dict is created per entry
        # (a ``[ {...} ] * n`` expression would alias a single dict across every
        # mesh, so each per-mesh write would clobber the others). The array is
        # indexed by DNA mesh index; ``self._mesh_indices`` is seeded with the main
        # mesh (index 0) in ``__init__`` and the secondary meshes (1..K) are added
        # above, so its length already equals the total mesh count.
        self._vertex_color_data = [{"indices": [], "values": []} for _ in range(len(self._mesh_indices))]

    def validate(self) -> tuple[bool, str, str, Callable | None]:
        if not self._rig_object:
            return (False, "No Rig Object", "No rig object found. Must link a head rig to export DNA.", None)

        if self._include_meshes:
            if self._non_lod_mesh_objects:
                mesh_names = "\n".join([f'"{i.name}"' for i in self._non_lod_mesh_objects])
                return (
                    False,
                    "Invalid LOD names. Fix by renaming to LOD 0 meshes?",
                    mesh_names,
                    lambda: utilities.rename_as_lod0_meshes(self._non_lod_mesh_objects),
                )

            meshes_missing_uvs = []
            for mesh_objects in self._export_lods.values():
                for mesh_object, _ in mesh_objects:
                    if not mesh_object.data.uv_layers.active:
                        meshes_missing_uvs.append(mesh_object)
            if meshes_missing_uvs:
                mesh_names = "\n".join([f'"{i.name}"' for i in meshes_missing_uvs])
                return (
                    False,
                    "Missing UVs. Auto unwrap the following meshes?",
                    mesh_names,
                    lambda: utilities.auto_unwrap_uvs(meshes_missing_uvs),
                )

            # make sure the mesh objects have the same origin as the rig
            meshes_with_mismatched_origins = []
            for mesh_objects in self._export_lods.values():
                for mesh_object, _ in mesh_objects:
                    if (self._rig_object.location.copy() - mesh_object.location).length > 1e-4:
                        meshes_with_mismatched_origins.append(mesh_object)

            if meshes_with_mismatched_origins:
                mesh_names = "\n".join([f'"{i.name}"' for i in meshes_with_mismatched_origins])
                return (
                    False,
                    "Mesh origin mismatch. Fix by matching and applying to the rig's origin?",
                    mesh_names,
                    lambda: utilities.set_objects_origins(
                        meshes_with_mismatched_origins, location=self._rig_object.location.copy()
                    ),
                )

        # TODO: Add more validations
        return (True, "Success", "All validations passed.", None)

    def validate_scene(self) -> tuple[bool, str, str, Callable | None]:
        """Initialize the scene data and run validations without writing any DNA.

        This lets a caller run every component's validations up-front (before any
        file is written), so a validation failure and its fix dialog surface before
        the exporter has already exported another component."""
        self.initialize_scene_data()
        if self._instance.output.run_validations:
            return self.validate()
        return (True, "Success", "All validations passed.", None)

    @staticmethod
    def get_bmesh(mesh_object: bpy.types.Object, rotation: float = -90) -> bmesh.types.BMesh:
        # create an empty BMesh and fill it in from the mesh data
        bmesh_object = bmesh.new()
        bmesh_object.from_mesh(mesh=mesh_object.data)  # type: ignore[arg-type]

        # Rotate the mesh so that it's Y-up before reading the vertex data
        bmesh.ops.rotate(
            bmesh_object,
            cent=Vector((0, 0, 0)),  # pyright: ignore[reportArgumentType]
            matrix=Matrix.Rotation(math.radians(rotation), 4, "X"),  # type: ignore[arg-type]
            verts=list(bmesh_object.verts),
        )
        bmesh_object.verts.index_update()
        bmesh_object.verts.ensure_lookup_table()
        return bmesh_object

    @staticmethod
    def get_mesh_faces(bmesh_object: bmesh.types.BMesh) -> list[tuple[int, list[int]]]:
        bmesh_object.faces.ensure_lookup_table()
        return [(face.index, [vert.index for vert in face.verts]) for face in bmesh_object.faces]

    @staticmethod
    def get_bone_transforms(
        armature_object: bpy.types.Object,
        extra_bones: list[tuple[str, dict]] = EXTRA_BONES,
    ) -> tuple[list[int], list[str], list[int], list[bool], list[list[float]], list[list[float]]]:
        indices = []
        bone_names = []
        hierarchy = []
        is_leaf = []
        translations = []
        rotations = []

        hierarchy_lookup = {}

        # Change the rotation of the bones since DNA expects Y-up
        rotation_x = Matrix.Rotation(math.radians(-90), 4, "X")  # type: ignore[arg-type]
        global_matrix = rotation_x.to_4x4()

        # Read the rest pose directly
        ignored_bone_names = [i for i, _ in extra_bones]
        bones = [i for i in armature_object.data.bones if i.name not in ignored_bone_names]  # type: ignore[attr-defined]
        for index, bone in enumerate(bones):
            if index == 0:
                # get translation and rotation of the bone globally
                translation, rotation, _ = (global_matrix @ bone.matrix_local).decompose()
            elif bone.parent:
                # get translation and rotation relative to its parent
                # Use inverted_safe() to handle singular matrices gracefully
                local_matrix = bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
                translation, rotation, _ = local_matrix.decompose()

            indices.append(index)
            bone_names.append(bone.name)
            is_leaf.append(not bone.children)

            hierarchy_index = index
            # If the bone has a parent, get the index of the parent bone.
            # We don't want to include the extra bones as parents.
            if bone.parent and bone.parent.name not in ignored_bone_names:
                hierarchy_index = hierarchy_lookup[bone.parent.name]

            hierarchy.append(hierarchy_index)
            # Store the index of the bone in the hierarchy lookup so we can find parent indices later
            hierarchy_lookup[bone.name] = index

            # Convert translation from blender meters to centimeters
            translations.append(
                [translation.x * SCALE_FACTOR, translation.y * SCALE_FACTOR, translation.z * SCALE_FACTOR]
            )
            # Convert rotation from quaternion to euler
            euler_rotation = rotation.to_euler("XYZ")
            # Convert rotation from radians to degrees
            rotations.append(
                [math.degrees(euler_rotation.x), math.degrees(euler_rotation.y), math.degrees(euler_rotation.z)]
            )

        return indices, bone_names, hierarchy, is_leaf, translations, rotations

    @staticmethod
    def get_mesh_vertex_positions(
        bmesh_object: bmesh.types.BMesh, duplicate_lookup: dict | None = None
    ) -> tuple[list[int], list[list[float]]]:
        indices = []
        positions = []
        if not duplicate_lookup:
            duplicate_lookup = {}

        for vert in bmesh_object.verts:
            positions.append([vert.co.x * SCALE_FACTOR, vert.co.y * SCALE_FACTOR, vert.co.z * SCALE_FACTOR])
            # Get the original vertex index if the vertex is a duplicate, otherwise use the current index
            vertex_index = duplicate_lookup.get(vert.index, vert.index)
            indices.append(vertex_index)
        return indices, positions

    @staticmethod
    def get_mesh_split_normals(mesh_object: bpy.types.Object) -> dict[tuple[int, int], Vector]:
        """The mesh's custom split normals in DNA space, keyed by ``(face index, vertex index)``.

        The DNA gives a normal per face corner, reached through the vertex layout the way UVs
        are, so a vertex on a hard edge carries a different normal per face. Reading Blender's
        per-vertex normal instead collapses those splits and loses the DNA's own shading.

        Keyed by face corner rather than by loop index because the export splits the mesh along
        its UV islands first, which renumbers loops and vertices but leaves faces alone.
        """
        mesh = mesh_object.data
        if not isinstance(mesh, bpy.types.Mesh) or not mesh.loops:
            return {}

        flat = np.empty(len(mesh.loops) * 3, dtype=np.float32)
        mesh.corner_normals.foreach_get("vector", flat)
        # A rotation needs no inverse transpose, so the normals take the same matrix the
        # vertices do -- but never the linear unit, because a normal is a direction.
        rotation = np.array(
            Matrix.Rotation(math.radians(-90), 3, "X"),  # type: ignore[arg-type]
            dtype=np.float32,
        )
        rotated = flat.reshape(-1, 3) @ rotation.T

        vertex_index = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", vertex_index)

        normals: dict[tuple[int, int], Vector] = {}
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                normals[(polygon.index, int(vertex_index[loop_index]))] = Vector(rotated[loop_index])
        return normals

    @staticmethod
    def get_mesh_vertex_normals(
        bmesh_object: bmesh.types.BMesh,
        split_normals: dict[tuple[int, int], Vector] | None = None,
        duplicate_lookup: dict | None = None,
    ) -> tuple[list[int], list[list[float]]]:
        """One normal per vertex of the UV-split mesh, which is what a layout slot addresses.

        Splitting along UV islands has already separated most corners that disagree, so each
        surviving vertex takes the normal of any face it still belongs to.
        """
        split_normals = split_normals or {}
        duplicate_lookup = duplicate_lookup or {}
        indices = []
        normals = []

        for vert in bmesh_object.verts:
            original_index = duplicate_lookup.get(vert.index, vert.index)
            normal = next(
                (
                    split_normals[(face.index, original_index)]
                    for face in vert.link_faces
                    if (face.index, original_index) in split_normals
                ),
                vert.normal,
            )
            normals.append([normal.x, normal.y, normal.z])
            indices.append(vert.index)
        return indices, normals

    @staticmethod
    def get_mesh_vertex_groups(mesh_object: bpy.types.Object) -> dict[str, list[tuple[int, float]]]:
        if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
            return {}
        # Create a lookup table for the vertex group names by their index
        vertex_group_lookup = {vertex_group.index: vertex_group.name for vertex_group in mesh_object.vertex_groups}
        # Initialize the vertex groups dictionary
        vertex_groups = {vertex_group.name: [] for vertex_group in mesh_object.vertex_groups}

        # Loop through the vertices and get the vertex group names and the vertex and weights
        for vertex in mesh_object.data.vertices:
            vertex_group_names = [vertex_group_lookup.get(group.group, "") for group in vertex.groups]
            for vertex_group_name in vertex_group_names:
                vertex_group = mesh_object.vertex_groups.get(vertex_group_name)
                if vertex_group:
                    weight = vertex_group.weight(vertex.index)
                    if weight > 0:
                        vertex_groups[vertex_group_name].append((vertex.index, weight))

        return vertex_groups

    @staticmethod
    def get_mesh_vertex_uvs(bmesh_object: bmesh.types.BMesh) -> tuple[list[int], list[list[float]]]:
        uv_layer = bmesh_object.loops.layers.uv.active
        if not uv_layer:
            return [], []

        uv_indices = list(range(len(bmesh_object.verts)))
        uv_positions = []

        for face in bmesh_object.faces:
            for loop in face.loops:
                uv_indices[loop.vert.index] = loop.index
                uv_positions.append(list(loop[uv_layer].uv[:]))

        return uv_indices, uv_positions

    def set_dna_vertex_colors(self, mesh_index: int, bmesh_object: bmesh.types.BMesh):
        vertex_color_indices = list(range(len(bmesh_object.verts)))
        vertex_color_values = []
        color_layer = (
            bmesh_object.loops.layers.color.get(VERTEX_COLOR_ATTRIBUTE_NAME) or bmesh_object.loops.layers.color.active
        )
        if color_layer:
            for face in bmesh_object.faces:
                for loop in face.loops:
                    vertex_color_indices[loop.vert.index] = loop.index
                    vertex_color_values.append(list(loop[color_layer][:]))

            self._vertex_color_data[mesh_index]["indices"] = vertex_color_indices
            self._vertex_color_data[mesh_index]["values"] = vertex_color_values

    def set_dna_vertex_positions(
        self,
        mesh_index: int,
        positions: list[list[float]],
    ):
        self._dna_writer.setVertexPositions(meshIndex=mesh_index, positions=positions)

    def set_dna_faces(self, mesh_index: int, face_layouts: list[tuple[int, list[int]]]):
        for face_index, face_vertex_indices in face_layouts:
            self._dna_writer.setFaceVertexLayoutIndices(
                meshIndex=mesh_index, faceIndex=face_index, layoutIndices=face_vertex_indices
            )

    def set_dna_normals(self, mesh_index: int, normals: list[list[float]]):
        self._dna_writer.setVertexNormals(meshIndex=mesh_index, normals=normals)

    def set_dna_uvs(self, mesh_index: int, uvs: list[list[float]]):
        self._dna_writer.setVertexTextureCoordinates(meshIndex=mesh_index, textureCoordinates=uvs)

    def set_dna_vertex_groups(self, mesh_index: int, mesh_object: bpy.types.Object):
        if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
            logger.warning(
                f"Object '{mesh_object.name}' has no mesh data in the blender scene. Skipping vertex group export..."
            )
            return
        # Create a lookup table for the vertex group names by their index
        vertex_group_lookup = {vertex_group.index: vertex_group.name for vertex_group in mesh_object.vertex_groups}

        # Loop through the vertices and get the vertex group names and the vertex and weights
        for vertex in mesh_object.data.vertices:
            vertex_group_names = [vertex_group_lookup.get(group.group, "") for group in vertex.groups]
            bone_indices = []
            weights = []

            for vertex_group_name in vertex_group_names:
                bone_index = self._bone_index_lookup.get(vertex_group_name)
                vertex_group = mesh_object.vertex_groups.get(vertex_group_name)
                # ``bone_index`` can be 0 (the root joint), so test against ``None``
                # explicitly -- ``if bone_index`` would silently drop every weight
                # painted onto joint index 0.
                if bone_index is not None and vertex_group:
                    weight = vertex_group.weight(vertex.index)
                    if weight > 0:
                        bone_indices.append(bone_index)
                        weights.append(weight)

            self._dna_writer.setSkinWeightsJointIndices(
                meshIndex=mesh_index, vertexIndex=vertex.index, jointIndices=bone_indices
            )
            self._dna_writer.setSkinWeightsValues(meshIndex=mesh_index, vertexIndex=vertex.index, weights=weights)

    def set_dna_bones(
        self,
        indices: list[int],
        bone_names: list[str],
        hierarchy: list[int],
        translations: list[list[float]],
        rotations: list[list[float]],
    ):
        dna_x_rotations = self._dna_reader.getNeutralJointRotationXs()
        dna_y_rotations = self._dna_reader.getNeutralJointRotationYs()
        dna_z_rotations = self._dna_reader.getNeutralJointRotationZs()

        for index, bone_name in zip(indices, bone_names, strict=False):
            self._dna_writer.setJointName(index=index, name=bone_name)
            self._bone_index_lookup[bone_name] = index

            dna_x_rotations[index] = rotations[index][0]
            dna_y_rotations[index] = rotations[index][1]
            dna_z_rotations[index] = rotations[index][2]

        self._dna_writer.setJointHierarchy(hierarchy)
        self._dna_writer.setNeutralJointTranslations(translations)
        self._dna_writer.setNeutralJointRotations(
            [[x, y, z] for x, y, z in zip(dna_x_rotations, dna_y_rotations, dna_z_rotations, strict=False)]
        )

    def save_images(self):
        if not self._include_textures:
            return

        for image, file_name in self._images:
            new_image_path = self._target_dna_file.parent / "Maps" / file_name
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image.packed_file and not image.filepath:
                logger.warning(f"Image {image.name} is not packed or saved. Skipping export.")
                continue

            try:
                image.save(filepath=str(new_image_path))
            except Exception:
                try:
                    image.save_render(filepath=str(new_image_path))
                except Exception as error:
                    logger.error(f"Failed to export image {image.name}: {error}")
            logger.info(f"Image {image.name} exported successfully to: {new_image_path}")

    def save_vertex_colors(self):
        if self._include_vertex_colors:
            vertex_colors_file = self._target_dna_file.parent / f"{self._prefix}_vertex_colors.json"
            with vertex_colors_file.open("w") as f:
                json.dump(self._vertex_color_data, f)
                logger.info(f'Vertex colors exported successfully to: "{vertex_colors_file}"')

    def _reset_dna_geometry(self):
        """Clear the mesh and joint tables so they can be rewritten from the scene.

        The behavior layer (raw controls, PSDs, RBFs, GUI controls, joint groups,
        blend-shape channel wiring) is left untouched -- it was copied verbatim from
        the source DNA by ``setFrom`` in ``__init__`` and the overwrite path only
        rewrites geometry, joint transforms, skin weights, and blend-shape targets."""
        # Clear the mesh data
        self._dna_writer.clearMeshNames()
        self._dna_writer.clearMeshIndices()
        self._dna_writer.clearLODMeshMappings()
        self._dna_writer.clearMeshes()

        # Clear the bone data
        self._dna_writer.clearJointNames()
        self._dna_writer.clearJointIndices()
        self._dna_writer.clearLODJointMappings()

    def export_bones(self) -> list[int]:
        """Read the armature's neutral bone transforms and write them into the DNA.

        Returns the list of joint indices so the per-LOD joint mapping can reuse it."""
        bone_indices, bone_names, hierarchy, _is_leaf, translations, rotations = self.get_bone_transforms(
            armature_object=self._rig_object, extra_bones=self._extra_bones
        )

        # Set the bone data
        self.set_dna_bones(
            indices=bone_indices,
            bone_names=bone_names,
            hierarchy=hierarchy,
            translations=translations,
            rotations=rotations,
        )
        return bone_indices

    def export_meshes(self, bone_indices: list[int]):
        """Rewrite every included LOD mesh (geometry, layouts, skin weights, vertex
        colors) and the per-LOD joint/mesh index mappings from the scene."""
        for lod_index, mesh_objects in self._export_lods.items():
            # Set the joint indices
            self._dna_writer.setJointIndices(index=lod_index, jointIndices=bone_indices)
            self._dna_writer.setLODJointMapping(lod=lod_index, index=lod_index)

            # TODO: Currently we just copy this data from the default DNA file.
            # In the future maybe give the user control over this and the PSDs
            # self._dna_writer.setJointColumnCount
            # self._dna_writer.setJointRowCount
            # self._dna_writer.setJointGroupLODs
            # self._dna_writer.setJointGroupJointIndices
            # self._dna_writer.setJointGroupInputIndices
            # self._dna_writer.setJointGroupOutputIndices
            # self._dna_writer.setJointGroupValues

            # Set the mesh indices
            self._dna_writer.setMeshIndices(index=lod_index, meshIndices=[mesh_index for _, mesh_index in mesh_objects])
            self._dna_writer.setLODMeshMapping(lod=lod_index, index=lod_index)

            for mesh_object, mesh_index in mesh_objects:
                self._export_mesh(mesh_object=mesh_object, mesh_index=mesh_index)

    def _export_mesh(self, mesh_object: bpy.types.Object, mesh_index: int):
        """Write a single mesh's geometry (positions, faces, normals, uvs, vertex
        layouts), skin weights, and vertex colors into the DNA at ``mesh_index``."""
        real_name = utilities.remove_instance_prefix(mesh_object.name, self._prefix)

        logger.info(f'Exporting mesh: "{mesh_object.name}" to DNA as "{real_name}"...')
        # Read before the UV split, which renumbers the loops these are addressed by. With
        # normals excluded the layout still needs an array to point at, so
        # ``get_mesh_vertex_normals`` falls back to the geometry's own normals.
        split_normals = self.get_mesh_split_normals(mesh_object) if self._include_normals else {}
        self._dna_writer.clearFaceVertexLayoutIndices(meshIndex=mesh_index)
        self._dna_writer.clearSkinWeights(meshIndex=mesh_index)
        # Blend shape targets are (re)written by ``export_shape_keys`` -- either
        # recomputed from the scene's shape keys or copied from the source DNA -- so
        # they are intentionally not cleared here.

        # Set the mesh name
        self._dna_writer.setMeshName(index=mesh_index, name=real_name)
        bmesh_object = self.get_bmesh(mesh_object)
        # Split the mesh along UV islands so that we have all the UV loop indices needed for each vertex index
        split_to_original_vert_lookup = utilities.split_mesh_along_uv_islands(bmesh_object=bmesh_object)

        vertex_indices, vertex_positions = self.get_mesh_vertex_positions(
            bmesh_object=bmesh_object, duplicate_lookup=split_to_original_vert_lookup
        )
        normal_indices, normals = self.get_mesh_vertex_normals(
            bmesh_object=bmesh_object,
            split_normals=split_normals,
            duplicate_lookup=split_to_original_vert_lookup,
        )
        uv_indices, uvs = self.get_mesh_vertex_uvs(bmesh_object=bmesh_object)
        faces = self.get_mesh_faces(bmesh_object=bmesh_object)

        # Set the vertex color data so it can be saved to JSON later
        if self._include_vertex_colors:
            self.set_dna_vertex_colors(mesh_index=mesh_index, bmesh_object=bmesh_object)

        # Set the vertex layout so DNA knows how to read the vertex,
        # normal, and uv data from their respective arrays
        self._dna_writer.setVertexLayouts(
            meshIndex=mesh_index,
            layouts=[list(item) for item in zip(vertex_indices, uv_indices, normal_indices, strict=False)],
        )

        self.set_dna_vertex_positions(mesh_index, vertex_positions)
        self.set_dna_faces(mesh_index, faces)
        self.set_dna_normals(mesh_index, normals)
        self.set_dna_uvs(mesh_index, uvs)
        self.set_dna_vertex_groups(mesh_index, mesh_object)

        # Now free the BMesh from memory without applying the changes back to the mesh
        bmesh_object.free()

    def _source_mesh_index_by_name(self) -> dict[str, int]:
        """Map every source DNA mesh name to its source mesh index. Used to look up
        the blend-shape channels/targets that belong to a mesh being rewritten under
        a possibly different export mesh index."""
        return {self._dna_reader.getMeshName(index): index for index in range(self._dna_reader.getMeshCount())}

    def export_shape_keys(self):
        """Rewrite the blend-shape (shape key) targets and the mesh -> blend-shape
        channel mapping for every exported mesh.

        ``clearMeshes`` in :meth:`_reset_dna_geometry` wipes all per-mesh geometry
        (including the ``setFrom``-copied blend-shape targets), so every mesh's
        targets are re-established here: recomputed from the scene's shape keys when
        the mesh carries them, or copied verbatim from the source DNA otherwise (this
        preserves the original blend shapes when the artist never imported shape keys
        and correctly remaps them when meshes are deleted/reordered).

        Only channels that already exist in the source DNA are written (matched by
        name); creating brand-new channels would require wiring blend-shape behavior
        inputs/outputs, which is out of scope for the overwrite path. The blend-shape
        behavior (which control drives which channel output) is preserved verbatim
        from the source DNA."""
        # (export_mesh_index, channel_index) pairs collected across every mesh so the
        # definition-layer mesh -> blend-shape channel mapping can be rebuilt for the
        # (possibly renumbered) export mesh indices.
        self._mesh_blend_shape_channel_pairs = []

        if self._component_type != "head":
            # Only the head component carries blend shapes (mirrors the calibrator).
            return

        source_mesh_index_by_name = self._source_mesh_index_by_name()

        for mesh_objects in self._export_lods.values():
            for mesh_object, export_mesh_index in mesh_objects:
                real_name = utilities.remove_instance_prefix(mesh_object.name, self._prefix)
                source_mesh_index = source_mesh_index_by_name.get(real_name)
                if source_mesh_index is None:
                    # A brand new mesh (e.g. custom teeth under a name not in the
                    # source DNA) has no channels to reuse, so it carries no blend
                    # shapes.
                    logger.info(
                        f'Mesh "{real_name}" has no matching mesh in the source DNA; skipping shape key export.'
                    )
                    continue

                self._dna_writer.clearBlendShapeTargets(meshIndex=export_mesh_index)
                if self._mesh_has_shape_keys(mesh_object):
                    self._write_mesh_shape_keys_from_scene(
                        export_mesh_index=export_mesh_index,
                        source_mesh_index=source_mesh_index,
                        mesh_object=mesh_object,
                        real_name=real_name,
                    )
                else:
                    self._copy_mesh_shape_keys_from_source(
                        export_mesh_index=export_mesh_index,
                        source_mesh_index=source_mesh_index,
                        mesh_object=mesh_object,
                        real_name=real_name,
                    )

        self._rebuild_mesh_blend_shape_channel_mapping()

    @staticmethod
    def _mesh_has_shape_keys(mesh_object: bpy.types.Object) -> bool:
        """Whether the scene mesh carries usable shape key blocks (a ``Basis`` plus
        at least the shape-key datablock)."""
        return bool(
            mesh_object.data
            and isinstance(mesh_object.data, bpy.types.Mesh)
            and mesh_object.data.shape_keys
            and mesh_object.data.shape_keys.key_blocks.get(SHAPE_KEY_BASIS_NAME)
        )

    def _write_mesh_shape_keys_from_scene(
        self,
        export_mesh_index: int,
        source_mesh_index: int,
        mesh_object: bpy.types.Object,
        real_name: str,
    ):
        """Recompute one mesh's blend-shape targets from its scene shape keys. Each
        source target is preserved at its original target index (so the
        channel/target relationship stays 1:1 with the source), computing deltas from
        the matching scene shape key block when it exists and writing an empty target
        otherwise."""
        shape_key_basis = mesh_object.data.shape_keys.key_blocks.get(SHAPE_KEY_BASIS_NAME)  # type: ignore[union-attr]

        # The DNA blend-shape target vertex indices are position indices, which for an
        # exported mesh are the original Blender vertex indices (0..M-1) -- the UV
        # island split only appends duplicate vertices at higher indices, so no split
        # remapping is needed here.
        bmesh_object = self.get_bmesh(mesh_object)
        vertex_indices, _ = self.get_mesh_vertex_positions(bmesh_object=bmesh_object)
        bmesh_object.free()

        # DNA is Y-up, Blender is Z-up, so we need to rotate the deltas.
        rotation_matrix = Matrix.Rotation(math.radians(-90), 4, "X")  # type: ignore[arg-type]

        for target_index in range(self._dna_reader.getBlendShapeTargetCount(source_mesh_index)):
            channel_index = self._dna_reader.getBlendShapeChannelIndex(source_mesh_index, target_index)
            channel_name = self._dna_reader.getBlendShapeChannelName(channel_index)
            block_name = f"{real_name}__{channel_name}"

            dna_delta_vertex_indices: list[int] = []
            dna_delta_values: list[list[float]] = []

            # Blender caps shape key names at 63 characters, so channels whose block
            # name would exceed that limit are never imported into the scene; write
            # an empty target for them (they keep their channel wiring but no deltas).
            shape_key_block = None
            if len(block_name) <= SHAPE_KEY_NAME_MAX_LENGTH:
                shape_key_block = mesh_object.data.shape_keys.key_blocks.get(block_name)  # type: ignore[union-attr]

            if shape_key_block:
                for vertex_index in vertex_indices:
                    new_delta = rotation_matrix @ (
                        shape_key_block.data[vertex_index].co.copy() - shape_key_basis.data[vertex_index].co  # type: ignore[union-attr]
                    )
                    # Only store vertices that actually moved to avoid floating point drift.
                    if new_delta.length > SHAPE_KEY_DELTA_THRESHOLD:
                        converted_delta = new_delta / self._linear_modifier
                        dna_delta_vertex_indices.append(vertex_index)
                        # The writer's typemap requires a list of [x, y, z] lists of plain floats.
                        dna_delta_values.append(
                            [float(converted_delta.x), float(converted_delta.y), float(converted_delta.z)]
                        )
            else:
                logger.debug(
                    f"Shape key block '{block_name}' not found for mesh '{real_name}'. "
                    "Writing an empty blend shape target..."
                )

            self._dna_writer.setBlendShapeChannelIndex(
                meshIndex=export_mesh_index,
                blendShapeTargetIndex=target_index,
                blendShapeChannelIndex=channel_index,
            )
            self._dna_writer.setBlendShapeTargetVertexIndices(
                meshIndex=export_mesh_index,
                blendShapeTargetIndex=target_index,
                vertexIndices=dna_delta_vertex_indices,
            )
            self._dna_writer.setBlendShapeTargetDeltas(
                meshIndex=export_mesh_index,
                blendShapeTargetIndex=target_index,
                deltas=dna_delta_values,
            )
            self._mesh_blend_shape_channel_pairs.append((export_mesh_index, channel_index))

    def _copy_mesh_shape_keys_from_source(
        self,
        export_mesh_index: int,
        source_mesh_index: int,
        mesh_object: bpy.types.Object,
        real_name: str,
    ):
        """Copy a mesh's blend-shape targets verbatim from the source DNA to the
        (possibly renumbered) export mesh index. Used when the scene mesh has no
        shape keys, so the original blend shapes are preserved rather than dropped.

        Blend-shape target vertex indices are position indices into the mesh, so this
        is only safe when the vertex count is unchanged; if the artist swapped in a
        custom-topology mesh without shape keys, the source targets can't be reused
        and are dropped with a warning."""
        source_vertex_count = self._dna_reader.getVertexPositionCount(source_mesh_index)
        scene_vertex_count = len(mesh_object.data.vertices) if isinstance(mesh_object.data, bpy.types.Mesh) else 0
        if source_vertex_count != scene_vertex_count:
            logger.warning(
                f'Mesh "{real_name}" has no shape keys in the scene and its vertex count '
                f"({scene_vertex_count}) differs from the source DNA ({source_vertex_count}); "
                "the source blend shape targets cannot be safely reused and will be dropped."
            )
            return

        for target_index in range(self._dna_reader.getBlendShapeTargetCount(source_mesh_index)):
            channel_index = self._dna_reader.getBlendShapeChannelIndex(source_mesh_index, target_index)
            vertex_indices = [
                int(i) for i in self._dna_reader.getBlendShapeTargetVertexIndices(source_mesh_index, target_index)
            ]
            delta_xs = self._dna_reader.getBlendShapeTargetDeltaXs(source_mesh_index, target_index)
            delta_ys = self._dna_reader.getBlendShapeTargetDeltaYs(source_mesh_index, target_index)
            delta_zs = self._dna_reader.getBlendShapeTargetDeltaZs(source_mesh_index, target_index)
            deltas = [[float(x), float(y), float(z)] for x, y, z in zip(delta_xs, delta_ys, delta_zs, strict=False)]

            self._dna_writer.setBlendShapeChannelIndex(
                meshIndex=export_mesh_index,
                blendShapeTargetIndex=target_index,
                blendShapeChannelIndex=channel_index,
            )
            self._dna_writer.setBlendShapeTargetVertexIndices(
                meshIndex=export_mesh_index,
                blendShapeTargetIndex=target_index,
                vertexIndices=vertex_indices,
            )
            self._dna_writer.setBlendShapeTargetDeltas(
                meshIndex=export_mesh_index,
                blendShapeTargetIndex=target_index,
                deltas=deltas,
            )
            self._mesh_blend_shape_channel_pairs.append((export_mesh_index, channel_index))

    def _rebuild_mesh_blend_shape_channel_mapping(self):
        """Rebuild the definition-layer mesh -> blend-shape channel mapping from the
        pairs collected during shape key export. This must be rewritten from scratch
        because the export mesh indices can differ from the source DNA's, which would
        leave the ``setFrom``-copied mapping pointing at the wrong meshes. The
        per-LOD mapping ranges are derived by the writer from this flat list plus the
        already-written ``LODMeshMapping``, so no separate LOD setter is needed."""
        self._dna_writer.clearMeshBlendShapeChannelMappings()
        for mapping_index, (mesh_index, channel_index) in enumerate(self._mesh_blend_shape_channel_pairs):
            self._dna_writer.setMeshBlendShapeChannelMapping(
                index=mapping_index,
                meshIndex=mesh_index,
                blendShapeChannelIndex=channel_index,
            )

    def run(self) -> tuple[bool, str, str, Callable | None]:
        self.initialize_scene_data()
        if self._instance.output.run_validations:
            valid, title, message, fix = self.validate()
            if not valid:
                return False, title, message, fix

        self._reset_dna_geometry()

        # init the lod indices
        # TODO: Currently can't change this without messing up the joint behavior.
        # Default dna has 8 lods
        # self._dna_writer.setLODCount(len(self._export_lods.keys()))  # noqa: ERA001

        bone_indices = self.export_bones()
        self.export_meshes(bone_indices=bone_indices)

        if self._include_shape_keys:
            self._report("Exporting shape keys...")
            self.export_shape_keys()

        self._dna_writer.write()
        if not dna.Status.isOk():
            status = dna.Status.get()
            raise RuntimeError(f"Error saving DNA: {status.message}")
        logger.info(f'DNA exported successfully to: "{self._target_dna_file}"')

        self.save_images()
        self.save_vertex_colors()

        return True, "Success", "Export successful.", None
