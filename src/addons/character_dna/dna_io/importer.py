# standard library imports
import json
import logging
import math

from pathlib import Path

# third party imports
import bmesh
import bpy

from mathutils import Euler, Matrix, Vector

# local imports
from .. import utilities
from ..bindings import enums  # type: ignore[reportAttributeAccessIssue]
from ..constants import (
    CUSTOM_BONE_SHAPE_SCALE,
    EXTRA_BONES,
    FIRST_BONE_Y_LOCATION,
    MESH_VERTEX_COLORS_FILE_NAME,
    MESH_VERTEX_COLORS_FILE_PATH,
    NUMBER_OF_HEAD_LODS,
    SHAPE_KEY_BASIS_NAME,
    UV_MAP_NAME,
    VERTEX_COLOR_ATTRIBUTE_NAME,
    ComponentType,
)
from ..typing import *  # noqa: F403
from .misc import get_dna_reader


logger = logging.getLogger(__name__)


class DNAImporter:
    def __init__(
        self,
        instance: "RigInstance",
        import_properties: "CharacterImportProperties",
        linear_modifier: float,
        component_type: ComponentType = "head",
        create_extra_bones: bool = True,
        reader: "dna.BinaryStreamReader | None" = None,
        dna_file_path: Path | None = None,
    ):
        self.rig_object = None

        self._instance = instance
        self._import_properties = import_properties
        self._linear_modifier = linear_modifier

        if component_type == "head":
            self.source_dna_file = Path(bpy.path.abspath(str(dna_file_path) or instance.head_dna_file_path))
        elif component_type == "body":
            self.source_dna_file = Path(bpy.path.abspath(str(dna_file_path) or instance.body_dna_file_path))

        # Determine the file format of the DNA file
        file_format = "binary" if (dna_file_path or self.source_dna_file).suffix.lower() == ".dna" else "json"

        # Open a read to a DNA file if an existing reader is not provided
        if not reader:
            self._dna_reader = get_dna_reader(file_path=self.source_dna_file, file_format=file_format)
        else:
            self._dna_reader = reader

        self._create_extra_bones = create_extra_bones
        if component_type == "body":
            self._create_extra_bones = False

        self._prefix = self._instance.name
        self._import_lods = {}
        self._index_to_vert = {}
        self._index_to_face = {}
        self._vert_index_to_dna_index = {}
        self._face_index_to_dna_index = {}
        self._vertex_color_data = []
        self._default_vertex_color_layout = False
        self._component_type = component_type

    def _get_lod_settings(self) -> list[tuple[int, bool]]:
        return [(i, getattr(self._import_properties, f"import_lod{i}")) for i in range(NUMBER_OF_HEAD_LODS)]

    def initialize_dna_data(self):
        for lod_index, should_import in self._get_lod_settings():
            if should_import:
                self._import_lods[lod_index] = {}
                for mesh_index in self._dna_reader.getMeshIndicesForLOD(lod_index):
                    vertex_indices = self._dna_reader.getVertexLayoutPositionIndices(mesh_index)
                    mesh_name = self._dna_reader.getMeshName(mesh_index)
                    self._import_lods[lod_index][mesh_name] = {
                        "mesh_index": mesh_index,
                        "vertex_indices": vertex_indices,
                        "vertex_count": len(vertex_indices),
                        "shape_key_count": self._dna_reader.getBlendShapeTargetCount(mesh_index),
                    }

    def get_dna_faces(self, mesh_index: int) -> list[list[int]]:
        return [
            self._dna_reader.getFaceVertexLayoutIndices(mesh_index, i)
            for i in range(self._dna_reader.getFaceCount(mesh_index))
        ]

    def get_dna_vertex_positions(self, mesh_index: int) -> tuple[list[int], list[Vector]]:
        x_values = self._dna_reader.getVertexPositionXs(mesh_index)
        y_values = self._dna_reader.getVertexPositionYs(mesh_index)
        z_values = self._dna_reader.getVertexPositionZs(mesh_index)
        indices = []
        positions = []

        for index in self._dna_reader.getVertexLayoutPositionIndices(mesh_index):
            indices.append(index)
            positions.append(
                Vector(
                    (
                        x_values[index] * self._linear_modifier,
                        y_values[index] * self._linear_modifier,
                        z_values[index] * self._linear_modifier,
                    )
                )
            )

        return indices, positions

    def get_dna_vertex_colors(self, mesh_index: int) -> tuple[list[int], list[list[float]]]:
        if self._component_type == "body":
            return [], []

        # Avoid loading the vertex colors multiple times
        if not self._vertex_color_data:
            vertex_colors_file = self.source_dna_file.parent / f"{self._prefix}_{MESH_VERTEX_COLORS_FILE_NAME}"
            if not vertex_colors_file.exists():
                vertex_colors_file = MESH_VERTEX_COLORS_FILE_PATH
                self._default_vertex_color_layout = True

            with vertex_colors_file.open() as file:
                self._vertex_color_data = json.load(file)

        data = self._vertex_color_data[mesh_index]
        return (data["indices"], data["values"])

    def get_dna_vertex_normals(self, mesh_index: int) -> dict[int, Vector]:
        x_values = self._dna_reader.getVertexNormalXs(mesh_index)
        y_values = self._dna_reader.getVertexNormalYs(mesh_index)
        z_values = self._dna_reader.getVertexNormalZs(mesh_index)
        return {
            i: Vector(
                (
                    x_values[i] * self._linear_modifier,
                    y_values[i] * self._linear_modifier,
                    z_values[i] * self._linear_modifier,
                )
            )
            for i in self._dna_reader.getVertexLayoutNormalIndices(mesh_index)
        }

    def get_dna_vertex_uvs(self, mesh_index: int) -> tuple[list[int], list[tuple[float, float]]]:
        u_values = self._dna_reader.getVertexTextureCoordinateUs(mesh_index)
        v_values = self._dna_reader.getVertexTextureCoordinateVs(mesh_index)
        indices = self._dna_reader.getVertexLayoutTextureCoordinateIndices(mesh_index)
        uvs = [(u, v) for u, v in zip(u_values, v_values, strict=False)]
        return indices, uvs

    def get_dna_vertex_groups(self, mesh_index: int) -> dict[str, list[tuple[int, float]]]:
        vertex_groups = {}
        for layout_index, vertex_index in enumerate(self._dna_reader.getVertexLayoutPositionIndices(mesh_index)):
            vertex_bone_indices = self._dna_reader.getSkinWeightsJointIndices(mesh_index, vertex_index)
            vertex_weights = self._dna_reader.getSkinWeightsValues(mesh_index, vertex_index)
            for bone_index, weight in zip(vertex_bone_indices, vertex_weights, strict=False):
                if bone_index in vertex_groups:
                    vertex_group_indices, vertex_group_weights = vertex_groups[bone_index]
                else:
                    vertex_groups[bone_index] = vertex_group_indices, vertex_group_weights = [], []

                vertex_group_indices.append(layout_index)
                vertex_group_weights.append(weight)

        return vertex_groups

    def get_dna_shape_keys(self, mesh_index: int) -> dict:
        mapping = []
        shape_keys = {}
        for layout_index, vertex_index in enumerate(self._dna_reader.getVertexLayoutPositionIndices(mesh_index)):
            if vertex_index in mapping:
                mapping[vertex_index].append(layout_index)
            else:
                mapping[vertex_index] = [layout_index]

        for target_index in range(self._dna_reader.getBlendShapeTargetCount(mesh_index)):
            delta_x_values = self._dna_reader.getBlendShapeTargetDeltaXs(mesh_index, target_index)
            delta_y_values = self._dna_reader.getBlendShapeTargetDeltaYs(mesh_index, target_index)
            delta_z_values = self._dna_reader.getBlendShapeTargetDeltaYs(mesh_index, target_index)
            vertex_indices = self._dna_reader.getBlendShapeTargetVertexIndices(mesh_index, target_index)
            channel_index = self._dna_reader.getBlendShapeChannelIndex(mesh_index, target_index)
            shape_key_name = self._dna_reader.getBlendShapeChannelName(channel_index)
            deltas = []
            for vertex_index, delta_x, delta_y, delta_z in zip(
                vertex_indices, delta_x_values, delta_y_values, delta_z_values, strict=False
            ):
                for layout_index in mapping[vertex_index]:
                    deltas[layout_index] = (delta_x, delta_y, delta_z)

            shape_keys[shape_key_name] = deltas

        return shape_keys

    @staticmethod
    def set_shape_key(mesh_object: bpy.types.Object) -> bpy.types.Key:
        # clear all shape keys
        mesh_object.shape_key_clear()

        # create the basis shape key
        shape_key_block = mesh_object.shape_key_add(name=SHAPE_KEY_BASIS_NAME, from_mix=False)
        shape_key = shape_key_block.id_data

        # set the shape key name to the mesh object name
        if shape_key:
            shape_key.name = mesh_object.name

        return shape_key  # pyright: ignore[reportReturnType]

    def set_custom_bone_shape(self, pose_bone: bpy.types.PoseBone):
        if pose_bone.rotation_mode != "XYZ":
            pose_bone.rotation_mode = "XYZ"
        pose_bone.custom_shape = utilities.get_bone_shape()
        pose_bone.custom_shape_scale_xyz = CUSTOM_BONE_SHAPE_SCALE

    def set_vertex_groups(self, mesh_index: int, mesh_object: bpy.types.Object):
        if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
            logger.warning(
                f"Object '{mesh_object.name}' has no mesh data in the blender scene. Skipping vertex group export..."
            )
            return

        for vertex in mesh_object.data.vertices:
            vertex_bone_indices = self._dna_reader.getSkinWeightsJointIndices(mesh_index, vertex.index)
            vertex_weights = self._dna_reader.getSkinWeightsValues(mesh_index, vertex.index)

            for bone_index, weight in zip(vertex_bone_indices, vertex_weights, strict=False):
                vertex_group_name = self._dna_reader.getJointName(bone_index)
                vertex_group = mesh_object.vertex_groups.get(vertex_group_name)
                if not vertex_group:
                    vertex_group = mesh_object.vertex_groups.new(name=vertex_group_name)
                vertex_group.add(index=[vertex.index], weight=weight, type="REPLACE")

    def set_mesh_normals(self, mesh_index: int, mesh: bpy.types.Mesh):
        """Import the DNA's own normals as custom split normals.

        The DNA keeps normals in an array of their own, reached through the vertex layout the way
        UVs are, so a vertex on a hard edge carries a different normal per face corner. Reading
        them per vertex collapses those splits, and indexing the layout array by a position index
        reads some other vertex's normal outright -- the layout is longer than the position array.

        These are also what lets the head sit invisibly on the body: the head and body DNAs agree
        on the normal at every shared seam vertex, so normals Blender derives from the faces
        instead disagree by several degrees there and leave a shading crease along the join.
        """
        x_values = self._dna_reader.getVertexNormalXs(mesh_index)
        y_values = self._dna_reader.getVertexNormalYs(mesh_index)
        z_values = self._dna_reader.getVertexNormalZs(mesh_index)
        normal_indices = self._dna_reader.getVertexLayoutNormalIndices(mesh_index)

        loop_normals = [(0.0, 0.0, 1.0)] * len(mesh.loops)
        for polygon in mesh.polygons:
            layout_indices = self._dna_reader.getFaceVertexLayoutIndices(mesh_index, polygon.index)
            corner = dict(
                zip(
                    (mesh.loops[index].vertex_index for index in polygon.loop_indices),
                    layout_indices,
                    strict=False,
                )
            )
            for loop_index in polygon.loop_indices:
                layout_index = corner.get(mesh.loops[loop_index].vertex_index)
                if layout_index is None:
                    continue
                normal_index = normal_indices[layout_index]
                # A normal is a direction, so the DNA's linear unit does not apply to it.
                loop_normals[loop_index] = (
                    x_values[normal_index],
                    y_values[normal_index],
                    z_values[normal_index],
                )

        mesh.normals_split_custom_set(loop_normals)

    def set_mesh_vertex_positions(self, mesh_index: int, bmesh_object: bmesh.types.BMesh):
        x_values = self._dna_reader.getVertexPositionXs(mesh_index)
        y_values = self._dna_reader.getVertexPositionYs(mesh_index)
        z_values = self._dna_reader.getVertexPositionZs(mesh_index)

        self._index_to_vert.clear()
        for dna_index in self._dna_reader.getVertexLayoutPositionIndices(mesh_index):
            if not self._index_to_vert.get(dna_index):
                vert = bmesh_object.verts.new(
                    [
                        x_values[dna_index] * self._linear_modifier,
                        y_values[dna_index] * self._linear_modifier,
                        z_values[dna_index] * self._linear_modifier,
                    ]
                )
                self._index_to_vert[dna_index] = vert

        bmesh_object.verts.index_update()
        # flip the dictionary so that we can get the dna index from the vertex index
        self._vert_index_to_dna_index = {vert.index: index for index, vert in self._index_to_vert.items()}
        # sort the vertices so that they are in the same order as the DNA file
        bmesh_object.verts.sort(key=lambda v: self._vert_index_to_dna_index[v.index])
        bmesh_object.verts.ensure_lookup_table()

    def set_mesh_face_layout(self, mesh_index: int, bmesh_object: bmesh.types.BMesh):
        vertex_indices = self._dna_reader.getVertexLayoutPositionIndices(mesh_index)
        self._index_to_face.clear()
        for index in range(self._dna_reader.getFaceCount(mesh_index)):
            try:
                face = bmesh_object.faces.new(
                    [
                        bmesh_object.verts[vertex_indices[i]]
                        for i in self._dna_reader.getFaceVertexLayoutIndices(mesh_index, index)
                    ]
                )
                self._index_to_face[index] = face
            except (RuntimeError, Exception):
                logger.error(f"Face {index} failed to create on mesh index {mesh_index}")

        bmesh_object.faces.index_update()
        # flip the dictionary so that we can get the dna index from the face index
        self._face_index_to_dna_index = {face.index: index for index, face in self._index_to_face.items()}
        bmesh_object.faces.sort(key=lambda f: self._face_index_to_dna_index[f.index])
        bmesh_object.faces.ensure_lookup_table()

    def set_smooth(self, bmesh_object: bmesh.types.BMesh):
        # smooth faces all faces
        for face in bmesh_object.faces:
            face.smooth = True

    @staticmethod
    def init_uvs(mesh: bpy.types.Mesh):
        uv_layer = mesh.uv_layers.get(UV_MAP_NAME)
        if not uv_layer:
            uv_layer = mesh.uv_layers.new(name=UV_MAP_NAME)
        mesh.uv_layers.active = uv_layer

    def set_mesh_uvs(self, mesh_index: int, bmesh_object: bmesh.types.BMesh):
        u_values = self._dna_reader.getVertexTextureCoordinateUs(mesh_index)
        v_values = self._dna_reader.getVertexTextureCoordinateVs(mesh_index)
        uv_indices = self._dna_reader.getVertexLayoutTextureCoordinateIndices(mesh_index)
        uv_layer = bmesh_object.loops.layers.uv.active

        for face in bmesh_object.faces:
            face_vert_indices = [v.index for v in face.verts]
            dna_face_vert_indices = self._dna_reader.getFaceVertexLayoutIndices(mesh_index, face.index)
            lookup = dict(zip(face_vert_indices, dna_face_vert_indices, strict=False))
            for loop in face.loops:
                uv_index = uv_indices[lookup[loop.vert.index]]
                loop[uv_layer].uv.x = u_values[uv_index]
                loop[uv_layer].uv.y = v_values[uv_index]

    def set_vertex_colors(self, mesh_index: int, bmesh_object: bmesh.types.BMesh):
        vertex_color_indices, vertex_color_values = self.get_dna_vertex_colors(mesh_index)
        if not vertex_color_indices or not vertex_color_values:
            logger.debug(f"No vertex colors found for mesh index {mesh_index}. Skipping vertex color import.")
            return

        color_layer = bmesh_object.loops.layers.color.get(
            VERTEX_COLOR_ATTRIBUTE_NAME
        ) or bmesh_object.loops.layers.color.new(VERTEX_COLOR_ATTRIBUTE_NAME)

        if self._default_vertex_color_layout:
            # The bundled default layout is keyed by the standard MetaHuman mesh
            # ordering/vertex counts; when it is applied to a custom (overwrite)
            # DNA whose mesh set or topology differs, the lookups can fall out of
            # range. Guard every access so a mismatched layout skips gracefully
            # instead of raising and dropping the whole mesh from the import.
            vertex_count = len(bmesh_object.verts)
            index_count = len(vertex_color_indices)
            value_count = len(vertex_color_values)
            for vertex_index in self._dna_reader.getVertexLayoutPositionIndices(mesh_index):
                if vertex_index >= vertex_count or vertex_index >= index_count:
                    continue
                color_index = vertex_color_indices[vertex_index]
                if color_index >= value_count:
                    continue
                vert = bmesh_object.verts[vertex_index]
                value = vertex_color_values[color_index]
                for loop in vert.link_loops:
                    loop[color_layer] = Vector(value)

        # Todo: Implement the custom vertex color layout from exported JSON file
        # else:  # noqa: ERA001
        #     for face in bmesh_object.faces:
        #         face_vert_indices = [v.index for v in face.verts]  # noqa: ERA001
        #         dna_face_vert_indices = self._dna_reader.getFaceVertexLayoutIndices(mesh_index, face.index)  # noqa: E501, ERA001
        #         lookup = dict(zip(face_vert_indices, dna_face_vert_indices))  # noqa: ERA001
        #         for loop in face.loops:
        #             vertex_color_index = vertex_color_indices[lookup[loop.vert.index]]  # noqa: ERA001
        #             loop[color_layer] = Vector(vertex_color_values[vertex_color_index])  # noqa: ERA001

    def normalize_color_attributes(self, mesh: bpy.types.Mesh):
        """Ensure the mesh has a single color attribute named"""
        color_attributes = mesh.color_attributes
        if not color_attributes.get(VERTEX_COLOR_ATTRIBUTE_NAME):
            return

        # Remove every color attribute except the named one (collect first to avoid mutating
        # the collection while iterating).
        for attribute in [a for a in color_attributes if a.name != VERTEX_COLOR_ATTRIBUTE_NAME]:
            color_attributes.remove(attribute)

        for index, attribute in enumerate(color_attributes):
            if attribute.name == VERTEX_COLOR_ATTRIBUTE_NAME:
                color_attributes.active_color_index = index
                color_attributes.render_color_index = index
                break

    def create_mesh_object(self, lod_index: int, mesh_name: str) -> bpy.types.Object:
        name = f"{self._prefix}_{mesh_name}"
        mesh_index = self._import_lods[lod_index][mesh_name]["mesh_index"]

        # remove the mesh object if it already exists
        mesh_object = bpy.data.objects.get(name)
        mesh = bpy.data.meshes.get(name)
        if mesh_object:
            bpy.data.objects.remove(mesh_object)
        if mesh:
            bpy.data.meshes.remove(mesh)

        # Create the mesh object
        mesh = bpy.data.meshes.new(name=name)
        mesh_object = bpy.data.objects.new(name=name, object_data=mesh)

        # Link the mesh object to the scene
        utilities.deselect_all()
        if bpy.context.collection:
            bpy.context.collection.objects.link(mesh_object)
        if bpy.context.view_layer:
            bpy.context.view_layer.objects.active = mesh_object
        mesh_object.select_set(True)

        # Initialize the UV map
        self.init_uvs(mesh)

        # create an empty BMesh
        bmesh_object = bmesh.new()

        # fill it in from a Mesh
        bmesh_object.from_mesh(mesh=mesh)

        self.set_mesh_vertex_positions(mesh_index, bmesh_object)
        self.set_mesh_face_layout(mesh_index, bmesh_object)
        self.set_smooth(bmesh_object)

        # Add vertex colors
        # Todo: See if we can import vertex colors on all LODs.
        if self._import_properties.import_vertex_colors and lod_index == 0 and self._component_type == "head":
            self.set_vertex_colors(mesh_index, bmesh_object)

        # Add UVs
        self.set_mesh_uvs(mesh_index, bmesh_object)

        # send the data back to the mesh and free the BMesh from memory
        bmesh_object.to_mesh(mesh)
        bmesh_object.free()

        # Ensure a single, named color attribute that is both active and active-render
        self.normalize_color_attributes(mesh)

        # The DNA's own normals, which are what makes the head sit seamlessly on the body.
        if self._import_properties.import_normals:
            self.set_mesh_normals(mesh_index, mesh)

        if self._import_properties.import_vertex_groups:
            # Create the vertex groups
            self.set_vertex_groups(mesh_index, mesh_object=mesh_object)
            # Attach the mesh to the armature
            self.set_armature_modifier(mesh_object)

        # Rotate the mesh and apply to Z-up
        mesh_object.rotation_euler.x = math.radians(90)
        utilities.apply_transforms(mesh_object, rotation=True)
        return mesh_object

    def create_rig_object(self) -> bpy.types.Object | None:
        name = f"{self._prefix}_{self._component_type}_rig"
        # Remove the rig object if it already exists
        rig_object = bpy.data.objects.get(name)
        armature = bpy.data.armatures.get(name)
        if rig_object:
            for child in rig_object.children:
                child.parent = None

            bpy.data.objects.remove(rig_object)
        if armature:
            bpy.data.armatures.remove(armature)

        armature = bpy.data.armatures.new(name=name)
        rig_object = bpy.data.objects.new(name=name, object_data=armature)

        if bpy.context.scene and not bpy.context.scene.collection.objects.get(name):
            bpy.context.scene.collection.objects.link(rig_object)

        self.rig_object = rig_object
        return rig_object

    def get_bone_matrix(self, bone_name: str) -> Matrix | None:
        x_locations = self._dna_reader.getNeutralJointTranslationXs()
        y_locations = self._dna_reader.getNeutralJointTranslationYs()
        z_locations = self._dna_reader.getNeutralJointTranslationZs()
        x_rotations = self._dna_reader.getNeutralJointRotationXs()
        y_rotations = self._dna_reader.getNeutralJointRotationYs()
        z_rotations = self._dna_reader.getNeutralJointRotationZs()

        for index in range(self._dna_reader.getJointCount()):
            if bone_name == self._dna_reader.getJointName(index):
                location = Vector(
                    (
                        x_locations[index] * self._linear_modifier,
                        y_locations[index] * self._linear_modifier,
                        z_locations[index] * self._linear_modifier,
                    )
                )
                euler_rotation = Euler(
                    (
                        math.radians(x_rotations[index]),
                        math.radians(y_rotations[index]),
                        math.radians(z_rotations[index]),
                    ),
                    "XYZ",
                )

                # The first bone is in object space
                if index == 0:
                    rotation_matrix = Matrix.Rotation(math.radians(90), 4, "X").to_4x4()  # type: ignore[arg-type]
                    global_matrix = (
                        rotation_matrix @ Matrix.Translation(location[:])
                    ) @ euler_rotation.to_matrix().to_4x4()

                # Otherwise they are in parent space
                elif self.rig_object and self.rig_object.data and isinstance(self.rig_object.data, bpy.types.Armature):
                    parent_index = self._dna_reader.getJointParentIndex(index)
                    parent_bone_name = self._dna_reader.getJointName(parent_index)
                    parent_bone = self.rig_object.data.edit_bones[parent_bone_name]
                    # Calculate the global transformation matrix of the bone
                    local_matrix = Matrix.Translation(location[:]) @ euler_rotation.to_matrix().to_4x4()
                    global_matrix = parent_bone.matrix @ local_matrix

                return global_matrix
        return None

    def get_height_scale_factor(self) -> float:
        y_locations = self._dna_reader.getNeutralJointTranslationYs()
        # Determine the height scale factor based of how much the first bone's is moved
        height_scale_factor = 1.0
        if len(y_locations) > 0:
            height_scale_factor = y_locations[0] / FIRST_BONE_Y_LOCATION
        return height_scale_factor

    def create_extra_bones(self) -> bpy.types.EditBone | None:
        last_edit_bone = None
        if self._create_extra_bones:
            height_scale_factor = self.get_height_scale_factor()
            # Create the extra bones for the spine, pelvis, and root
            for bone_name, bone_data in EXTRA_BONES:
                location = bone_data["location"]
                rotation = bone_data["rotation"]
                if (
                    not self.rig_object
                    or not self.rig_object.data
                    or not isinstance(self.rig_object.data, bpy.types.Armature)
                ):
                    continue

                # Scale the location of the bones based on the height scale factor
                location.y = location.y * round(height_scale_factor, 4)

                extra_edit_bone = self.rig_object.data.edit_bones.new(bone_name)
                extra_edit_bone.length = self._linear_modifier
                global_matrix = Matrix.Translation(location) @ rotation.to_matrix().to_4x4()
                extra_edit_bone.matrix = global_matrix
                extra_edit_bone.parent = self.rig_object.data.edit_bones.get(bone_data["parent"] or "")
                last_edit_bone = extra_edit_bone

        return last_edit_bone

    def import_bones(self):
        if not self.rig_object or not self.rig_object.data or not isinstance(self.rig_object.data, bpy.types.Armature):
            return

        x_locations = self._dna_reader.getNeutralJointTranslationXs()
        y_locations = self._dna_reader.getNeutralJointTranslationYs()
        z_locations = self._dna_reader.getNeutralJointTranslationZs()
        x_rotations = self._dna_reader.getNeutralJointRotationXs()
        y_rotations = self._dna_reader.getNeutralJointRotationYs()
        z_rotations = self._dna_reader.getNeutralJointRotationZs()

        # Switch to edit mode
        utilities.switch_to_bone_edit_mode(self.rig_object)

        # remove all existing edit bones
        for edit_bone in self.rig_object.data.edit_bones:
            self.rig_object.data.edit_bones.remove(edit_bone)

        # Create the extra bones below the last bone in the DNA file
        extra_edit_bone = self.create_extra_bones()

        for index in range(self._dna_reader.getJointCount()):
            bone_name = self._dna_reader.getJointName(index)
            location = Vector(
                (
                    x_locations[index] * self._linear_modifier,
                    y_locations[index] * self._linear_modifier,
                    z_locations[index] * self._linear_modifier,
                )
            )
            euler_rotation = Euler(
                (
                    math.radians(x_rotations[index]),
                    math.radians(y_rotations[index]),
                    math.radians(z_rotations[index]),
                ),
                "XYZ",
            )

            # Create the new edit bone
            edit_bone = self.rig_object.data.edit_bones.new(name=bone_name)

            # The first bone is in object space
            if index == 0:
                edit_bone.length = self._linear_modifier
                edit_bone.matrix = Matrix.Translation(location[:]) @ euler_rotation.to_matrix().to_4x4()
                # The last extra bone should be the parent of first bone in the DNA file
                edit_bone.parent = extra_edit_bone

            # Otherwise they are in parent space
            else:
                parent_index = self._dna_reader.getJointParentIndex(index)
                parent_bone_name = self._dna_reader.getJointName(parent_index)
                parent_bone = self.rig_object.data.edit_bones[parent_bone_name]
                edit_bone.parent = parent_bone
                edit_bone.length = self._linear_modifier
                # Calculate the global transformation matrix of the bone
                local_matrix = Matrix.Translation(location[:]) @ euler_rotation.to_matrix().to_4x4()
                global_matrix = parent_bone.matrix @ local_matrix
                edit_bone.matrix = global_matrix

        # Set the custom bone shapes
        utilities.switch_to_object_mode()
        for pose_bone in self.rig_object.pose.bones:
            self.set_custom_bone_shape(pose_bone)
        self.rig_object.data.relation_line_position = "HEAD"

        # Rotate the armature and apply to Z-up
        self.rig_object.rotation_euler.x = math.radians(90)
        utilities.apply_transforms(self.rig_object, rotation=True)

    def setup_swing_bones(self):
        if not self.rig_object or not self.rig_object.pose:
            return

        for swing_index in range(self._dna_reader.getSwingCount()):
            input_indices = self._dna_reader.getSwingInputControlIndices(swing_index)
            output_indices = self._dna_reader.getSwingOutputJointIndices(swing_index)
            weights = self._dna_reader.getSwingBlendWeights(swing_index)
            axis = enums.TwistAxis(self._dna_reader.getSwingSetupTwistAxis(swing_index))  # pyright: ignore[reportCallIssue]

            # there should always be 4 input controls for a bone's quaternion rotation
            if len(input_indices) != 4:
                logger.error(f"Swing bone at index {swing_index} does not have 4 quaternion input controls. Skipping.")
                continue

            for input_index in input_indices:
                bone_name, _ = self._dna_reader.getRawControlName(input_index).split(".")
                pose_bone = self.rig_object.pose.bones.get(bone_name)
                if pose_bone:
                    _swing_bone_names = [self._dna_reader.getJointName(output_index) for output_index in output_indices]
                    pose_bone["swing_bone_names"] = _swing_bone_names
                    pose_bone["swing_blend_weights"] = weights
                    pose_bone["swing_axis"] = axis.name.lower()

    def setup_twist_bones(self):
        if not self.rig_object or not self.rig_object.pose:
            return

        for twist_index in range(self._dna_reader.getTwistCount()):
            input_indices = self._dna_reader.getTwistInputControlIndices(twist_index)
            output_indices = self._dna_reader.getTwistOutputJointIndices(twist_index)
            weights = self._dna_reader.getTwistBlendWeights(twist_index)
            axis = enums.TwistAxis(self._dna_reader.getTwistSetupTwistAxis(twist_index))  # pyright: ignore[reportCallIssue]

            # there should always be 4 input controls for a bone's quaternion rotation
            if len(input_indices) != 4:
                logger.error(f"Twist bone at index {twist_index} does not have 4 quaternion input controls. Skipping.")
                continue

            for input_index in input_indices:
                bone_name, _ = self._dna_reader.getRawControlName(input_index).split(".")
                pose_bone = self.rig_object.pose.bones.get(bone_name)
                if pose_bone:
                    _twist_bone_names = [self._dna_reader.getJointName(output_index) for output_index in output_indices]
                    pose_bone["twist_bone_names"] = _twist_bone_names
                    pose_bone["twist_blend_weights"] = weights
                    pose_bone["twist_axis"] = axis.name.lower()

    def set_armature_modifier(self, mesh_object: bpy.types.Object):
        armature_modifier = mesh_object.modifiers.get("Armature")
        if not armature_modifier or armature_modifier.type != "ARMATURE":
            armature_modifier = mesh_object.modifiers.new(name="Armature", type="ARMATURE")

        armature_modifier.object = self.rig_object  # type: ignore[attr-defined]

    def run(self) -> tuple[bool, str]:
        errors = []
        self.initialize_dna_data()

        if self._import_properties.import_bones:
            self.create_rig_object()
            self.import_bones()
            self.setup_swing_bones()
            self.setup_twist_bones()

        for lod_index, meshes in self._import_lods.items():
            lod_meshes = []
            for mesh_name in meshes:
                # Create the mesh object
                try:
                    if self._import_properties.import_mesh:
                        mesh_object = self.create_mesh_object(lod_index=lod_index, mesh_name=mesh_name)
                        mesh_object.parent = self.rig_object
                        lod_meshes.append(mesh_object)
                except (RuntimeError, Exception) as error:
                    message = f'Mesh "{mesh_name}" Error: {error}'
                    errors.append(message)
                    logger.error(message)

            # Make a collection per LOD
            utilities.move_to_collection(
                scene_objects=lod_meshes, collection_name=f"{self._prefix}_lod{lod_index}", exclusively=True
            )

        if errors:
            return False, "\n".join(errors)

        return True, f'Imported "{self.source_dna_file.stem}.dna" successfully!'
