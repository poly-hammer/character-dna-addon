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
from ..fbx.reader import FbxAnimationClip
from ..utilities import exclude_rig_instance_evaluation
from .base import CharacterComponentBase


logger = logging.getLogger(__name__)


class CharacterComponentBody(CharacterComponentBase):
    @exclude_rig_instance_evaluation
    def import_action(
        self,
        file_path: Path,
        round_sub_frames: bool = True,
        match_frame_rate: bool = True,
        prefix_instance_name: bool = True,
        prefix_component_name: bool = True,
        clip: "FbxAnimationClip | None" = None,
    ):
        file_path = Path(file_path)

        if self.body_rig_object and self.body_rig_object.pose:
            # ensure the rig instance is initialized
            self.rig_instance.initialize()
            utilities.import_action_from_fbx(
                instance=self.rig_instance,
                file_path=file_path,
                component="body",
                armature=self.body_rig_object,
                round_sub_frames=round_sub_frames,
                match_frame_rate=match_frame_rate,
                prefix_instance_name=prefix_instance_name,
                prefix_component_name=prefix_component_name,
                clip=clip,
                # include animation only for body that are not driven by rig logic
                include_only_bones=[
                    b.name
                    for b in self.body_rig_object.pose.bones
                    if b.name
                    not in [
                        *self.rig_instance.body_driven_bone_names,
                        *self.rig_instance.body_swing_bone_names,
                        *self.rig_instance.body_twist_bone_names,
                    ]
                ],
            )

    def ingest(self, align: bool = True, constrain: bool = True) -> tuple[bool, str]:
        valid, message = self.dna_importer.run()
        self.rig_instance.body_rig = self.dna_importer.rig_object

        self._organize_viewport()
        self.import_materials()

        # set the references on the rig instance
        self.rig_instance.body_mesh = self.body_mesh_object
        self.rig_instance.body_rig = self.body_rig_object
        self.rig_instance.body_dna_file_path = str(self.dna_importer.source_dna_file)
        self.rig_instance.body_initialize()

        # create and assign the body bone collections from the bone names read from the DNA
        if self.body_rig_object:
            utilities.assign_body_bone_collections(
                rig_object=self.body_rig_object,
                swing_bone_names=tuple(self.rig_instance.body_swing_bone_names),
                twist_bone_names=tuple(self.rig_instance.body_twist_bone_names),
                driver_bone_names=tuple(self.rig_instance.body_driver_bone_names),
                driven_bone_names=tuple(self.rig_instance.body_driven_bone_names),
            )

        if self.body_rig_object and self.body_mesh_object and len(self.scene_properties.rig_instance_list) > 1:
            # if this isn't the first rig, move it to the right of the last body mesh
            last_instance = self.scene_properties.rig_instance_list[-2]
            if last_instance.body_mesh:
                self.body_rig_object.location.x = utilities.get_bounding_box_left_x(last_instance.body_mesh) - (
                    utilities.get_bounding_box_width(last_instance.body_mesh) / 2
                )

        # focus the view on body object
        if self.rig_instance.body_mesh and self._focus_on_import:
            utilities.select_only(self.rig_instance.body_mesh)
            utilities.focus_on_selected()

        # collapse the outliner
        utilities.toggle_expand_in_outliner()

        return valid, message

    def frame_rig_to_target(self, mesh_object: bpy.types.Object) -> float:
        """Uniformly scale the body rig so its height matches ``mesh_object``
        and reset all body-component origins to the world origin.

        Returns the scale ratio applied (target height / body height). This is
        the shared framing step used by both the legacy ``Convert Selected to
        DNA`` path and the new Converter editor."""
        if not self.body_rig_object or not self.body_mesh_object:
            return 1.0

        target_height = utilities.get_bounding_box_height(mesh_object)
        body_height = utilities.get_bounding_box_height(self.body_mesh_object)
        delta = target_height / body_height

        # scale the body rig and the body mesh to match the target height
        self.body_rig_object.scale.x *= delta
        self.body_rig_object.scale.y *= delta
        self.body_rig_object.scale.z *= delta

        self.body_rig_object.hide_set(False)
        utilities.apply_transforms(self.body_rig_object, scale=True, recursive=True)

        # adjust the body rig origin to zero
        utilities.switch_to_object_mode()
        # select all the objects and set their origins to the 3d cursor
        utilities.deselect_all()
        for item in self.rig_instance.output.body_item_list:
            if item.scene_object:
                item.scene_object.hide_set(False)
                item.scene_object.select_set(True)
                if bpy.context.view_layer:
                    bpy.context.view_layer.objects.active = item.scene_object
        self.body_rig_object.select_set(True)
        if bpy.context.scene:
            bpy.context.scene.cursor.location = Vector((0, 0, 0))
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        return delta

    def delete(self):
        for item in self.rig_instance.output.body_item_list:
            if item.scene_object:
                bpy.data.objects.remove(item.scene_object, do_unlink=True)
            if item.image_object:
                bpy.data.images.remove(item.image_object, do_unlink=True)

        self._delete_rig_instance()

    def reset_poses(self):
        if self.rig_instance.body_rig:
            utilities.reset_pose(self.rig_instance.body_rig)

    def set_pose(self):
        """Apply body bone rotations from the active pose.json to the body rig.
        No-op when no body rig is present; the head component handles any
        overlapping bones that exist on the head rig."""
        if not self.rig_instance.body_rig:
            return
        thumbnail_file = Path(self.scene_properties.face_board.face_pose_previews)
        json_file_path = thumbnail_file.parent / "pose.json"
        if not json_file_path.exists():
            return

        with json_file_path.open() as file:
            data = json.load(file)

        self.reset_poses()

        body_data: dict = data.get("body", {})
        if not body_data:
            return

        if self.rig_instance.body_rig.pose:
            for bone_name, transform_data in body_data.items():
                pose_bone = self.rig_instance.body_rig.pose.bones.get(bone_name)
                if pose_bone is not None:
                    rotation_euler = Euler([math.radians(i) for i in transform_data["rotation"]], "XYZ")
                    pose_bone.rotation_mode = "QUATERNION"
                    q = rotation_euler.to_quaternion()
                    pose_bone.rotation_quaternion = (q.w, q.x, -q.z, q.y)
