# standard library imports
import json
import logging
import math

from pathlib import Path

# third party imports
import bpy

from mathutils import Euler, Vector

# local imports
from .. import utilities
from ..constants import (
    EXCLUDED_FACE_BOARD_CONTROLS,
    FACE_BOARD_SWITCHES,
    REGION_VERTEX_GROUP_PREFIX,
)
from ..fbx.reader import FbxAnimationClip
from ..rig_definition import HeadRigDefinition
from ..utilities import exclude_rig_instance_evaluation
from .base import CharacterComponentBase


logger = logging.getLogger(__name__)


class CharacterComponentHead(CharacterComponentBase):
    @exclude_rig_instance_evaluation
    def import_action(
        self,
        file_path: Path,
        is_face_board: bool = True,
        round_sub_frames: bool = True,
        match_frame_rate: bool = True,
        prefix_instance_name: bool = True,
        prefix_component_name: bool = True,
        clip: "FbxAnimationClip | None" = None,
    ):
        file_path = Path(file_path)

        if is_face_board and self.face_board_object:
            if file_path.suffix.lower() == ".json":
                utilities.import_face_board_action_from_json(file_path, self.face_board_object)
            elif file_path.suffix.lower() == ".fbx":
                utilities.import_face_board_action_from_fbx(
                    instance=self.rig_instance,
                    file_path=file_path,
                    armature=self.face_board_object,
                    round_sub_frames=round_sub_frames,
                    match_frame_rate=match_frame_rate,
                    prefix_instance_name=prefix_instance_name,
                    prefix_component_name=prefix_component_name,
                    clip=clip,
                )
        elif self.head_rig_object:
            utilities.import_action_from_fbx(
                instance=self.rig_instance,
                file_path=file_path,
                component="head",
                armature=self.head_rig_object,
                round_sub_frames=round_sub_frames,
                match_frame_rate=match_frame_rate,
                prefix_instance_name=prefix_instance_name,
                prefix_component_name=prefix_component_name,
                clip=clip,
            )

    def ingest(self, align: bool = True, constrain: bool = True) -> tuple[bool, str]:
        valid, message = self.dna_importer.run()
        self.rig_instance.head_rig = self.dna_importer.rig_object

        self._organize_viewport()
        self.import_materials()

        face_board_object = None
        # import the face board if one does not already exist in the scene
        if not any(i.face_board for i in self.scene_properties.rig_instance_list):
            if self.dna_import_properties.import_face_board:
                face_board_object = utilities.import_face_board(name=self.name)

        elif not self.rig_instance.face_board and not self.dna_import_properties.reuse_face_board:
            if self.dna_import_properties.import_face_board:
                face_board_object = utilities.duplicate_face_board(name=self.name)
        else:
            face_board_object = next(i.face_board for i in self.scene_properties.rig_instance_list if i.face_board)

        # set the references on the rig instance
        self.rig_instance.head_mesh = self.head_mesh_object
        self.rig_instance.head_rig = self.head_rig_object
        self.rig_instance.face_board = face_board_object

        # author rig-definition metadata: joint-group bone collections (with
        # per-bone colors) and region vertex groups on the head meshes
        self._build_rig_definition_metadata()

        if self.head_rig_object and self.head_mesh_object and self.head_rig_object.pose:
            if self.body_rig_object and self.body_rig_object.pose:
                # Add the additional driver bones from the head to the body rig,
                # since they share these same bones for driving the neck rbfs
                utilities.assign_body_bone_collections(
                    rig_object=self.body_rig_object,
                    driver_bone_names=tuple(
                        self.dna_reader.getRawControlName(i).split(".")[0]
                        for i in range(self.dna_reader.getRawControlCount())
                    ),
                )

                if align:
                    # Align the head rig with the body rig if it exists
                    body_object_head_bone = self.body_rig_object.pose.bones.get("head")
                    head_object_head_bone = self.head_rig_object.pose.bones.get("head")
                    if body_object_head_bone and head_object_head_bone:
                        # get the location offset between the body head bone and the head head bone
                        body_head_location = body_object_head_bone.matrix @ Vector((0, 0, 0))
                        head_head_location = head_object_head_bone.matrix @ Vector((0, 0, 0))
                        delta = body_head_location - head_head_location
                        # move the head rig object to align with the body rig head bone
                        self.head_rig_object.location += delta
            # if this isn't the first rig, move it to the right of the last head mesh
            elif len(self.scene_properties.rig_instance_list) > 1:
                last_instance = self.scene_properties.rig_instance_list[-2]
                if last_instance.head_mesh:
                    self.head_rig_object.location.x = utilities.get_bounding_box_left_x(last_instance.head_mesh) - (
                        utilities.get_bounding_box_width(last_instance.head_mesh) / 2
                    )

        # constrain the head rig to the body rig if it exists
        if constrain:
            self.constrain_head_to_body()

        # focus the view on head object
        if self.rig_instance.head_mesh and self._focus_on_import:
            utilities.select_only(self.rig_instance.head_mesh)
            utilities.focus_on_selected()

        # collapse the outliner
        utilities.toggle_expand_in_outliner()

        # switch to pose mode on the face gui object
        if face_board_object and utilities.set_active(face_board_object):
            utilities.position_face_board(
                head_mesh_object=self.head_mesh_object,
                head_rig_object=self.head_rig_object,
                face_board_object=face_board_object,
            )
            # a duplicated face board keeps the origin of the instance it was copied from
            utilities.align_face_board_origin(
                face_board_object=face_board_object,
                rig_object=self.rig_instance.body_rig or self.rig_instance.head_rig,
            )
            utilities.move_to_collection(scene_objects=[face_board_object], collection_name=self.name, exclusively=True)
            utilities.switch_to_pose_mode(face_board_object)
            # constrain the face board to the head rig if it was just created
            if not self.dna_import_properties.reuse_face_board:
                utilities.constrain_face_board_to_head(
                    face_board_object=face_board_object,
                    head_rig_object=self.rig_instance.head_rig,
                    body_rig_object=self.rig_instance.body_rig,
                    bone_name="CTRL_faceGUI",
                )
                utilities.constrain_face_board_to_head(
                    face_board_object=face_board_object,
                    head_rig_object=self.rig_instance.head_rig,
                    body_rig_object=self.rig_instance.body_rig,
                    bone_name="CTRL_C_eyesAim",
                )

        return valid, message

    def _build_rig_definition_metadata(self):
        """Author the joint-group bone collections and region vertex groups.

        Both passes are gated on their import toggles and degrade gracefully
        when the rig definition is unavailable.
        """
        if self.dna_import_properties.import_bone_collections:
            self._build_joint_group_collections()
        if self.dna_import_properties.import_region_vertex_groups:
            self._build_region_vertex_groups()

    def _build_region_vertex_groups(self):
        """Partition the LOD0 head mesh vertices into ``REGION_*`` vertex groups.

        The DNA does not store per-vertex region membership, so each vertex is
        assigned to the rig-definition region whose joints carry the largest
        summed skin weight on that vertex (a dominant-weight partition). Regions
        and their joints are read from the rig definition; this is skipped when
        the rig definition is unavailable.
        """
        rig_definition = self._get_rig_definition()
        if not isinstance(rig_definition, HeadRigDefinition) or not rig_definition.regions:
            return

        # map each joint name to the regions it belongs to
        regions_by_joint: dict[str, list[str]] = {}
        for region in rig_definition.regions:
            for joint_name in region.joints:
                regions_by_joint.setdefault(joint_name, []).append(region.name)
        if not regions_by_joint:
            return

        reader = self.dna_reader
        joint_name_cache: dict[int, str] = {}

        for mesh_index in reader.getMeshIndicesForLOD(0):
            mesh_name = reader.getMeshName(mesh_index)
            mesh_object = bpy.data.objects.get(f"{self.name}_{mesh_name}")
            if not mesh_object or not isinstance(mesh_object.data, bpy.types.Mesh):
                continue

            vertex_groups: dict[str, bpy.types.VertexGroup] = {}
            for vertex in mesh_object.data.vertices:
                joint_indices = reader.getSkinWeightsJointIndices(mesh_index, vertex.index)
                weights = reader.getSkinWeightsValues(mesh_index, vertex.index)

                region_weights: dict[str, float] = {}
                for joint_index, weight in zip(joint_indices, weights, strict=False):
                    joint_name = joint_name_cache.get(joint_index)
                    if joint_name is None:
                        joint_name = reader.getJointName(joint_index)
                        joint_name_cache[joint_index] = joint_name
                    for region_name in regions_by_joint.get(joint_name, ()):
                        region_weights[region_name] = region_weights.get(region_name, 0.0) + float(weight)

                if not region_weights:
                    continue

                dominant_region = max(region_weights, key=region_weights.__getitem__)
                vertex_group_name = f"{REGION_VERTEX_GROUP_PREFIX}{dominant_region}"
                vertex_group = vertex_groups.get(vertex_group_name)
                if not vertex_group:
                    vertex_group = mesh_object.vertex_groups.get(vertex_group_name)
                    if not vertex_group:
                        vertex_group = mesh_object.vertex_groups.new(name=vertex_group_name)
                    vertex_groups[vertex_group_name] = vertex_group
                vertex_group.add(index=[vertex.index], weight=1.0, type="REPLACE")

    def frame_rig_to_target(self, mesh_object: bpy.types.Object) -> Vector:
        """Translate the head rig so it is centered on ``mesh_object`` and reset
        all head-component origins to the world origin.

        Returns the translation ``delta`` applied (target center minus the
        current head-mesh center) so the caller can later offset the face board
        by the same amount. This is the shared framing step used by both the
        legacy ``Convert Selected to DNA`` path and the new Converter editor."""
        if not self.head_rig_object or not self.head_mesh_object:
            return Vector((0, 0, 0))

        target_center = utilities.get_bounding_box_center(mesh_object)
        head_center = utilities.get_bounding_box_center(self.head_mesh_object)
        delta = target_center - head_center

        # translate the head rig
        self.head_rig_object.location += delta

        # must be unhidden to switch to edit bone mode
        self.head_rig_object.hide_set(False)
        utilities.switch_to_bone_edit_mode(self.head_rig_object)
        # adjust the root bone so the root bone is still at zero
        root_bone = self.head_rig_object.data.edit_bones.get("root")  # type: ignore[union-attr]
        if root_bone:
            root_bone.head.z -= delta.z
            root_bone.tail.z -= delta.z

        # adjust the head rig origin to zero
        utilities.switch_to_object_mode()
        # select all the objects and set their origins to the 3d cursor
        utilities.deselect_all()
        for item in self.rig_instance.output.head_item_list:
            if item.scene_object:
                item.scene_object.hide_set(False)
                item.scene_object.select_set(True)
                if bpy.context.view_layer:
                    bpy.context.view_layer.objects.active = item.scene_object

        if self.face_board_object:
            self.face_board_object.select_set(True)
        self.head_rig_object.select_set(True)

        if bpy.context.scene:
            bpy.context.scene.cursor.location = Vector((0, 0, 0))
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        return delta

    def reposition_face_board(self, delta: Vector) -> None:
        """Reposition the face board onto the (possibly reshaped) head mesh and
        offset it by the framing ``delta`` applied in
        :meth:`frame_rig_to_target`."""
        if not (self.head_mesh_object and self.head_rig_object and self.face_board_object):
            return
        utilities.position_face_board(
            head_mesh_object=self.head_mesh_object,
            head_rig_object=self.head_rig_object,
            face_board_object=self.face_board_object,
        )
        self.face_board_object.location += delta

    def delete(self):
        for item in self.rig_instance.output.head_item_list:
            if item.scene_object:
                bpy.data.objects.remove(item.scene_object, do_unlink=True)
            if item.image_object:
                bpy.data.images.remove(item.image_object, do_unlink=True)

        self._delete_rig_instance()

    def reset_poses(self):
        if self.rig_instance.head_rig:
            utilities.reset_pose(self.rig_instance.head_rig)

        for pose_bone in self.rig_instance.face_board.pose.bones:
            if (
                not pose_bone.bone.children
                and pose_bone.name.startswith("CTRL_")
                and pose_bone.name not in FACE_BOARD_SWITCHES + EXCLUDED_FACE_BOARD_CONTROLS
            ):
                pose_bone.location = Vector((0.0, 0.0, 0.0))

    def set_pose(self):
        if not self.rig_instance.face_board:
            return
        thumbnail_file = Path(self.scene_properties.face_board.face_pose_previews)
        json_file_path = thumbnail_file.parent / "pose.json"
        if not json_file_path.exists():
            return

        logger.info(f"Applying face pose from {json_file_path}")
        with json_file_path.open() as file:
            data = json.load(file)

        self.reset_poses()

        # --- Face board controls (location-driven) ---
        face_board_data: dict = data.get("face_board", {})
        for bone_name, transform_data in face_board_data.items():
            pose_bone = self.rig_instance.face_board.pose.bones.get(bone_name)
            if pose_bone and bone_name not in FACE_BOARD_SWITCHES + EXCLUDED_FACE_BOARD_CONTROLS:
                pose_bone.location = Vector(transform_data["location"])

        # Apply body pose to the head bones directly when the head is not driven
        # by the body rig (no body rig, or the head-to-body constraint influence is
        # 0.0). When the head is constrained to the body, the body component poses
        # the body bones and the head follows via the copy-transforms constraints.
        if not self.rig_instance.head_constrained_to_body:
            body_data: dict = data.get("body", {})
            if body_data and self.rig_instance.head_rig and self.rig_instance.head_rig.pose:
                for bone_name, transform_data in body_data.items():
                    pose_bone = self.rig_instance.head_rig.pose.bones.get(bone_name)
                    if pose_bone is not None:
                        rotation_euler = Euler([math.radians(i) for i in transform_data["rotation"]], "XYZ")
                        pose_bone.rotation_mode = "QUATERNION"
                        pose_bone.rotation_quaternion = rotation_euler.to_quaternion()
