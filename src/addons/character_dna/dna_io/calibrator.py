# standard library imports
import logging
import math

from collections.abc import Callable

# third party imports
import bpy

from mathutils import Matrix, Vector

# local imports
from .. import utilities
from ..bindings import dna  # pyright: ignore[reportAttributeAccessIssue]
from ..constants import (
    BONE_DELTA_THRESHOLD,
    HEAD_TO_BODY_LOD_MAPPING,
    NORMAL_DELTA_THRESHOLD,
    SHAPE_KEY_BASIS_NAME,
    SHAPE_KEY_DELTA_THRESHOLD,
    SHAPE_KEY_NAME_MAX_LENGTH,
)
from ..typing import *  # noqa: F403  # noqa: F403
from .exporter import DNAExporter
from .importer import DNAImporter
from .misc import get_dna_reader


logger = logging.getLogger(__name__)


class DNACalibrator(DNAExporter, DNAImporter):
    @property
    def _auto_lod_enabled(self) -> bool:
        """Whether auto LOD calibration (LOD0 -> lower-LOD propagation) is active.

        This is only available in the Pro edition and when the output option is enabled."""
        return bool(self._instance.output.auto_update_lods) and self._instance.is_pro

    def validate(self) -> tuple[bool, str, str, Callable | None]:
        valid, title, message, fix = super().validate()
        if not valid:
            return (valid, title, message, fix)

        # A calibration writes back into the DNA's own vertex arrays by index, so a mesh that
        # gained vertices has nowhere to write them. Overwriting is unaffected, hence the override.
        mesh_index_lookup = {
            self._dna_reader.getMeshName(index): index for index in range(self._dna_reader.getMeshCount())
        }
        mismatched = []
        for mesh_objects in self._export_lods.values():
            for mesh_object, _ in mesh_objects:
                real_name = utilities.remove_instance_prefix(mesh_object.name, self._instance.name)
                mesh_index = mesh_index_lookup.get(real_name)
                if mesh_index is None:
                    continue
                dna_vertex_count = len(self._dna_reader.getVertexPositionXs(mesh_index))
                scene_vertex_count = len(mesh_object.data.vertices)
                if scene_vertex_count != dna_vertex_count:
                    mismatched.append(f'"{mesh_object.name}" has {scene_vertex_count}, DNA has {dna_vertex_count}')

        if mismatched:
            return (
                False,
                "Vertex count mismatch. Calibrating requires the DNA's topology.",
                "\n".join(mismatched)
                + "\n\nRemove the added or deleted vertices, or set the export method to Overwrite.",
                None,
            )

        return (True, "Success", "All validations passed.", None)

    def _get_body_mesh_lookup(
        self, lod_index: int, mesh_name: str, head_to_body_edge_loop_mapping: dict[str, dict[int, int]]
    ) -> dict[int, Vector]:
        # If this is the head, and the align head and body option is on, then we want to use the
        # exact same vertex positions for the body and head vertices where they overlap. This needs to
        # be precised to the exact floating point value.
        if mesh_name != f"{self._instance.name}_head_lod{lod_index}_mesh":
            return {}

        body_lod_index = HEAD_TO_BODY_LOD_MAPPING.get(lod_index)
        body_mesh_name = f"{self._instance.name}_body_lod{body_lod_index}_mesh"
        body_mesh_lod = bpy.data.objects.get(body_mesh_name)
        if (
            self._component_type == "head"
            and self._seam_follower == "head"
            and self._instance.output.method == "calibrate"
            and self._instance.output.align_head_and_body
            and body_mesh_lod
        ):
            bmesh_object = self.get_bmesh(body_mesh_lod)
            vertex_indices, vertex_positions = self.get_mesh_vertex_positions(bmesh_object)
            bmesh_object.free()
            vert_lookup = dict(zip(vertex_indices, vertex_positions, strict=False))

            try:
                return {
                    int(head_vertex_index): Vector(vert_lookup[body_vertex_index])
                    for head_vertex_index, body_vertex_index in head_to_body_edge_loop_mapping.get(
                        str(lod_index), {}
                    ).items()
                }
            except KeyError as error:
                logger.warning(
                    f"Head to body vertex mapping not found for LOD {lod_index}: {error}. A vertex on "
                    f"mesh {mesh_name} or {body_mesh_name} may have been deleted."
                )
        return {}

    def calibrate_vertex_positions(self):
        additional_meshes_by_lod = {}
        mesh_index_lookup = {
            self._dna_reader.getMeshName(index): index for index in range(self._dna_reader.getMeshCount())
        }
        head_to_body_edge_loop_mapping = utilities.get_head_to_body_edge_loop_mapping()

        # Collected for optional LOD0 -> lower-LOD propagation: the new LOD0 mesh
        # positions keyed by DNA mesh index, and the set of lower-LOD DNA mesh
        # indices that were calibrated directly from in-scene geometry (so the
        # propagation never overwrites a mesh the artist actually provided).
        lod0_mesh_writes: dict[int, list[list[float]]] = {}
        scene_calibrated_lower_lod_indices: set[int] = set()

        for lod_index, mesh_objects in self._export_lods.items():
            logger.info(f"Calibrating LOD {lod_index} vertex positions...")
            for mesh_object, _ in mesh_objects:
                body_mesh_lookup = self._get_body_mesh_lookup(
                    lod_index=lod_index,
                    mesh_name=mesh_object.name,
                    head_to_body_edge_loop_mapping=head_to_body_edge_loop_mapping,
                )

                real_name = utilities.remove_instance_prefix(mesh_object.name, self._instance.name)
                logger.info(f'Calibrating "{real_name}" vertex positions...')
                mesh_index = mesh_index_lookup.get(real_name)

                # If the mesh index is not found, we assume that the mesh is not part of the DNA
                # And we can add it to the additional meshes for this LOD
                if mesh_index is None:
                    additional_meshes_by_lod[lod_index] = additional_meshes_by_lod.get(lod_index, [])
                    additional_meshes_by_lod[lod_index].append(mesh_object)
                    logger.warning(f'Mesh "{real_name}" not found in DNA. This mesh will not be calibrated...')
                    continue

                bmesh_object = self.get_bmesh(mesh_object)
                vertex_indices, vertex_positions = self.get_mesh_vertex_positions(bmesh_object)
                bmesh_object.free()

                # Read these from the DNA file and modify these arrays so that they match the vertex indices match
                x_values = self._dna_reader.getVertexPositionXs(mesh_index)
                y_values = self._dna_reader.getVertexPositionYs(mesh_index)
                z_values = self._dna_reader.getVertexPositionZs(mesh_index)

                # Reachable when the user turned validations off, since the loop below indexes the
                # DNA's arrays with the scene mesh's own vertex indices.
                if vertex_indices and max(vertex_indices) >= len(x_values):
                    logger.warning(
                        f'Skipping "{real_name}", it has more vertices than the DNA mesh '
                        f"({max(vertex_indices) + 1} vs {len(x_values)}). Use the Overwrite method instead."
                    )
                    continue

                for vertex_index in vertex_indices:
                    # See if we can get the vertex position from the body mesh lookup first,
                    # so that we have an exact match
                    vertex_position = body_mesh_lookup.get(int(vertex_index), Vector(vertex_positions[vertex_index]))
                    dna_vertex_position = Vector(
                        (x_values[vertex_index], y_values[vertex_index], z_values[vertex_index])
                    )
                    delta = vertex_position - dna_vertex_position
                    # This ensures that we only modify the vertex positions that are different to
                    # avoid floating value drift
                    if delta.length > 1e-6:
                        x_values[vertex_index] = vertex_position.x
                        y_values[vertex_index] = vertex_position.y
                        z_values[vertex_index] = vertex_position.z

                positions = [[x, y, z] for x, y, z in zip(x_values, y_values, z_values, strict=False)]
                self._dna_writer.setVertexPositions(meshIndex=mesh_index, positions=positions)

                if lod_index == 0:
                    lod0_mesh_writes[mesh_index] = positions
                else:
                    scene_calibrated_lower_lod_indices.add(mesh_index)

        propagated_writes = self._propagate_lods_from_lod0(lod0_mesh_writes, scene_calibrated_lower_lod_indices)
        self._align_seams(lod0_mesh_writes, propagated_writes)

    def calibrate_normals(self):
        """Write the scene's custom split normals back over the DNA's own.

        Unlike the export this never rewrites the vertex layouts, so the DNA keeps its own
        splits and each face corner is reached the way :meth:`set_mesh_normals` reads it:
        through the layout index the face carries, then the normal index that layout holds.
        Several corners routinely share one normal index, so the last one wins -- which is
        correct, because they only share it while they agree.
        """
        mesh_index_lookup = {
            self._dna_reader.getMeshName(index): index for index in range(self._dna_reader.getMeshCount())
        }

        for lod_index, mesh_objects in self._export_lods.items():
            logger.info(f"Calibrating LOD {lod_index} normals...")
            for mesh_object, _ in mesh_objects:
                real_name = utilities.remove_instance_prefix(mesh_object.name, self._instance.name)
                mesh_index = mesh_index_lookup.get(real_name)
                if mesh_index is None:
                    continue

                mesh = mesh_object.data
                if not isinstance(mesh, bpy.types.Mesh) or not mesh.loops:
                    continue

                logger.info(f'Calibrating "{real_name}" normals...')
                split_normals = self.get_mesh_split_normals(mesh_object)

                x_values = self._dna_reader.getVertexNormalXs(mesh_index)
                y_values = self._dna_reader.getVertexNormalYs(mesh_index)
                z_values = self._dna_reader.getVertexNormalZs(mesh_index)
                normal_indices = self._dna_reader.getVertexLayoutNormalIndices(mesh_index)

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
                        normal = split_normals.get((polygon.index, mesh.loops[loop_index].vertex_index))
                        if normal is None:
                            continue

                        normal_index = normal_indices[layout_index]
                        dna_normal = Vector((x_values[normal_index], y_values[normal_index], z_values[normal_index]))
                        # Only touch what moved, so untouched normals keep their exact DNA value.
                        if (normal - dna_normal).length > NORMAL_DELTA_THRESHOLD:
                            x_values[normal_index] = normal.x
                            y_values[normal_index] = normal.y
                            z_values[normal_index] = normal.z

                normals = [[x, y, z] for x, y, z in zip(x_values, y_values, z_values, strict=False)]
                self._dna_writer.setVertexNormals(meshIndex=mesh_index, normals=normals)

    def _propagate_lods_from_lod0(
        self, lod0_mesh_writes: dict[int, list[list[float]]], skip_mesh_indices: set[int]
    ) -> dict[int, list[list[float]]]:
        """When ``output.auto_update_lods`` is enabled (Pro only), propagate the
        just-calibrated LOD0 mesh shapes to every lower-LOD mesh that was not
        calibrated from in-scene geometry, using the shared UV-space solver.

        Returns the positions written to each propagated lower-LOD mesh keyed by
        its DNA mesh index (empty when nothing was propagated), so the seam
        alignment pass can re-snap them without re-reading the writer.

        This is a no-op in the free edition (the shared ``editors`` submodule
        that owns the solver is absent) or when there are no LOD0 writes."""
        if not lod0_mesh_writes or not self._instance.output.auto_update_lods:
            return {}
        if not self._instance.is_pro:
            return {}
        try:
            from ..editors.shared.lod_propagation import propagate_lod0_to_lower_lods
        except ImportError:
            logger.error("LOD propagation is unavailable in this build; skipping auto LOD update.")
            return {}

        return propagate_lod0_to_lower_lods(
            self._dna_reader,
            self._dna_writer,
            lod0_mesh_writes,
            skip_mesh_indices=skip_mesh_indices,
            progress_callback=self._report,
        )

    @staticmethod
    def _head_lod_for_body_lod(body_lod_index: int) -> int | None:
        """The lowest (highest-fidelity) head LOD whose neck edge loop maps onto
        the given body LOD. Several head LODs share a body LOD (and share the same
        body seam vertices), so the lowest head LOD is the best reference."""
        candidates = sorted(head for head, body in HEAD_TO_BODY_LOD_MAPPING.items() if body == body_lod_index)
        return candidates[0] if candidates else None

    def _seam_reference_reader(self) -> "BinaryStreamReader | None":
        """The reader for the seam reference component (the one the follower snaps
        onto).

        When ``seam_reference_dna_path`` is set it is read fresh from disk -- this
        is the partner component that was just exported, so the follower conforms to
        the geometry that was *actually written* (auto LOD propagation regenerates
        each component's lower-LOD meshes independently, so the cached template no
        longer matches the exported partner). When no path is given a head follower
        falls back to the cached body DNA reader (the imported body); a body
        follower has no reference and seam alignment is skipped."""
        if self._seam_reference_dna_path is not None:
            if not self._seam_reference_dna_path.exists():
                logger.warning(
                    "Seam alignment skipped: reference DNA '%s' is unavailable.", self._seam_reference_dna_path
                )
                return None
            return get_dna_reader(file_path=self._seam_reference_dna_path)
        if self._seam_follower == "head":
            return self._instance.body_dna_reader
        return None

    @staticmethod
    def _read_mesh_positions(reader: "BinaryStreamReader", mesh_name: str) -> list[Vector] | None:
        """Read a reference mesh's vertex positions (DNA space, cm Y-up) by name,
        returning ``None`` when the mesh is absent from the reader."""
        for mesh_index in range(reader.getMeshCount()):
            if str(reader.getMeshName(mesh_index)) == mesh_name:
                xs = reader.getVertexPositionXs(mesh_index)
                ys = reader.getVertexPositionYs(mesh_index)
                zs = reader.getVertexPositionZs(mesh_index)
                return [Vector((xs[i], ys[i], zs[i])) for i in range(len(xs))]
        return None

    def _resolve_seam_pairs(
        self, mesh_name: str, edge_loop_mapping: dict[str, dict[int, int]]
    ) -> tuple[str, list[tuple[int, int]]] | None:
        """Resolve the seam reference mesh name and ``(follower_vertex,
        reference_vertex)`` pairs for a follower mesh, or ``None`` when the mesh is
        not the follower component's LOD mesh or has no head/body LOD partner.

        Only the head/body skin mesh carries the neck edge loop, so other meshes in
        the same LOD (teeth, eyes, ...) are ignored -- their vertex indices are
        unrelated to the head-to-body mapping."""
        lod_index = utilities.get_lod_index(mesh_name)
        if lod_index == -1:
            return None

        if self._seam_follower == "head":
            if mesh_name != f"head_lod{lod_index}_mesh":
                return None
            body_lod_index = HEAD_TO_BODY_LOD_MAPPING.get(lod_index)
            if body_lod_index is None:
                return None
            reference_mesh_name = f"body_lod{body_lod_index}_mesh"
            # (follower vertex, reference vertex) == (head vertex, body vertex)
            pairs = [
                (int(head_vertex), int(body_vertex))
                for head_vertex, body_vertex in edge_loop_mapping.get(str(lod_index), {}).items()
            ]
            return reference_mesh_name, pairs

        if mesh_name != f"body_lod{lod_index}_mesh":
            return None
        head_lod_index = self._head_lod_for_body_lod(lod_index)
        if head_lod_index is None:
            return None
        reference_mesh_name = f"head_lod{head_lod_index}_mesh"
        # (follower vertex, reference vertex) == (body vertex, head vertex)
        pairs = [
            (int(body_vertex), int(head_vertex))
            for head_vertex, body_vertex in edge_loop_mapping.get(str(head_lod_index), {}).items()
        ]
        return reference_mesh_name, pairs

    def _align_seams(
        self,
        lod0_mesh_writes: dict[int, list[list[float]]],
        propagated_writes: dict[int, list[list[float]]],
    ) -> None:
        """Snap the head/body neck edge loop so the two components share the exact
        same vertex positions where they overlap, across every LOD.

        LOD0 is already snapped in :meth:`calibrate_vertex_positions` for a head
        follower (via :meth:`_get_body_mesh_lookup`), so only the propagated
        lower-LOD meshes are aligned in that direction. For a body follower the
        main loop never snaps the body, so every body mesh (LOD0 plus the
        propagated lower LODs) is aligned onto the head."""
        follower = self._seam_follower
        if follower is None or follower != self._component_type:
            return
        if self._instance.output.method != "calibrate" or not self._instance.output.align_head_and_body:
            return

        reference_reader = self._seam_reference_reader()
        if reference_reader is None:
            return

        edge_loop_mapping = utilities.get_head_to_body_edge_loop_mapping()

        if follower == "head":
            meshes = propagated_writes
        else:
            meshes = {**lod0_mesh_writes, **propagated_writes}

        reference_cache: dict[str, list[Vector] | None] = {}

        for mesh_index, positions in meshes.items():
            mesh_name = str(self._dna_reader.getMeshName(mesh_index))
            resolved = self._resolve_seam_pairs(mesh_name, edge_loop_mapping)
            if resolved is None:
                continue
            reference_mesh_name, vertex_pairs = resolved

            if reference_mesh_name not in reference_cache:
                reference_cache[reference_mesh_name] = self._read_mesh_positions(reference_reader, reference_mesh_name)
            reference_positions = reference_cache[reference_mesh_name]
            if not reference_positions:
                continue

            modified = False
            try:
                for follower_vertex_index, reference_vertex_index in vertex_pairs:
                    reference_position = reference_positions[reference_vertex_index]
                    positions[follower_vertex_index] = [
                        reference_position.x,
                        reference_position.y,
                        reference_position.z,
                    ]
                    modified = True
            except IndexError as error:
                logger.warning(
                    f"Seam alignment skipped for mesh '{mesh_name}': vertex index out of range ({error}). "
                    f"A vertex on '{mesh_name}' or '{reference_mesh_name}' may have been deleted."
                )
                continue

            if modified:
                self._dna_writer.setVertexPositions(meshIndex=mesh_index, positions=positions)

    def calibrate_shape_keys(self):
        if self._component_type != "head":
            # TODO: in the future, we may want to support shape key calibration for other components
            # currently, we only calibrate shape keys for the head component
            return

        for lod_index in range(self._dna_reader.getLODCount()):
            # Skip LODs without blend shape channels
            if len(self._dna_reader.getBlendShapeChannelIndicesForLOD(lod_index)) == 0:
                continue

            logger.info(f"Calibrating shape keys for {self._component_type} component LOD {lod_index}...")

            for mesh_index in self._dna_reader.getMeshIndicesForLOD(lod_index):
                mesh_name = self._dna_reader.getMeshName(mesh_index)
                real_mesh_name = f"{self._prefix}_{mesh_name}"
                mesh_object = bpy.data.objects.get(real_mesh_name)
                if not mesh_object:
                    logger.error(f"Mesh object '{real_mesh_name}' not found for shape key calibration. Skipping...")
                    continue

                if (
                    not mesh_object.data
                    or not isinstance(mesh_object.data, bpy.types.Mesh)
                    or not mesh_object.data.shape_keys
                ):
                    logger.warning(
                        f"Mesh object '{mesh_object.name}' has no shape key data in the blender scene. "
                        "Skipping shape key calibration..."
                    )
                    continue

                shape_key_basis = mesh_object.data.shape_keys.key_blocks.get(SHAPE_KEY_BASIS_NAME)
                if not shape_key_basis:
                    raise RuntimeError(
                        f"Shape key '{SHAPE_KEY_BASIS_NAME}' not found for mesh '{real_mesh_name}'. "
                        "This is needed for calibration!"
                    )

                # helps to track the largest delta count for the shape keys
                largest_delta_count = 0

                # Get the vertex positions for the mesh object
                bmesh_object = self.get_bmesh(mesh_object)
                vertex_indices, _ = self.get_mesh_vertex_positions(bmesh_object)
                bmesh_object.free()

                # DNA is Y-up, Blender is Z-up, so we need to rotate the deltas
                rotation_matrix = Matrix.Rotation(math.radians(-90), 4, "X")  # type: ignore[arg-type]

                for index in range(self._dna_reader.getBlendShapeTargetCount(mesh_index)):
                    channel_index = self._dna_reader.getBlendShapeChannelIndex(mesh_index, index)
                    shape_key_name = self._dna_reader.getBlendShapeChannelName(channel_index)

                    # Currently, Blender has a limit of 63 characters for shape key names
                    if len(f"{mesh_name}__{shape_key_name}") > SHAPE_KEY_NAME_MAX_LENGTH:
                        continue

                    shape_key_block = mesh_object.data.shape_keys.key_blocks.get(f"{mesh_name}__{shape_key_name}")
                    if not shape_key_block:
                        logger.error(
                            f"Shape key '{shape_key_name}' not found for mesh '{real_mesh_name}'. "
                            "Skipping calibration..."
                        )
                        continue

                    dna_delta_vertex_indices = []
                    dna_delta_values = []

                    # the new shape key is the dna shape key with the deltas from the blender shape key applied
                    for vertex_index in vertex_indices:
                        # get the positions of the points
                        # Get the delta between the current shape key and the basis (rest) shape key
                        new_delta = rotation_matrix @ (
                            shape_key_block.data[vertex_index].co.copy() - shape_key_basis.data[vertex_index].co  # type: ignore[attr-defined]
                        )

                        # Only modify the vertex positions that are different to avoid floating value drift
                        if new_delta.length > SHAPE_KEY_DELTA_THRESHOLD:
                            # Apply the coordinate system conversion and linear modifier for the
                            # scene units to the delta
                            converted_delta = new_delta / self._linear_modifier
                            dna_delta_vertex_indices.append(vertex_index)
                            # The DNA writer's typemap requires a list of [x, y, z] lists of plain
                            # Python floats (matching ``setVertexPositions``); tuples or mathutils
                            # scalars are rejected with a SystemError/TypeError.
                            dna_delta_values.append(
                                [float(converted_delta.x), float(converted_delta.y), float(converted_delta.z)]
                            )

                    largest_delta_count = max(largest_delta_count, len(dna_delta_vertex_indices))

                    # Set the vertex indices for the delta values array for the shape key
                    self._dna_writer.setBlendShapeTargetVertexIndices(
                        meshIndex=mesh_index, blendShapeTargetIndex=index, vertexIndices=dna_delta_vertex_indices
                    )
                    # Set the actual delta value array for the shape key
                    self._dna_writer.setBlendShapeTargetDeltas(
                        meshIndex=mesh_index, blendShapeTargetIndex=index, deltas=dna_delta_values
                    )

                logger.debug(f"Largest Shape Key delta count for mesh {real_mesh_name} is {largest_delta_count}")

    def zero_blend_shape_deltas(self):
        """Clear every blend shape target's deltas in the DNA while preserving the
        blend shape channels and their rig wiring.

        Used by the converter's ``zero_shape_deltas`` option: reconforming the
        bone positions to new meshes invalidates the base DNA's blend shape
        deltas, so they are reset to empty. Each target keeps its channel index
        mapping but drives no vertex movement."""
        if self._component_type != "head":
            # Only the head component carries blend shapes.
            return

        logger.info("Zeroing blend shape deltas...")
        for mesh_index in range(self._dna_reader.getMeshCount()):
            for target_index in range(self._dna_reader.getBlendShapeTargetCount(mesh_index)):
                self._dna_writer.setBlendShapeTargetVertexIndices(
                    meshIndex=mesh_index, blendShapeTargetIndex=target_index, vertexIndices=[]
                )
                self._dna_writer.setBlendShapeTargetDeltas(
                    meshIndex=mesh_index, blendShapeTargetIndex=target_index, deltas=[]
                )

    def calibrate_vertex_groups(self):
        for lod_index in range(self._dna_reader.getLODCount()):
            logger.info(f"Calibrating vertex groups for {self._component_type} component LOD {lod_index}...")

            for mesh_index in self._dna_reader.getMeshIndicesForLOD(lod_index):
                mesh_name = self._dna_reader.getMeshName(mesh_index)
                real_mesh_name = f"{self._prefix}_{mesh_name}"
                mesh_object = bpy.data.objects.get(real_mesh_name)
                if not mesh_object:
                    # Lower-LOD meshes are expected to be absent from the scene when auto LOD
                    # calibration is enabled (they get propagated from LOD0), so only warn for
                    # LOD0 meshes or when auto LOD calibration is off.
                    message = f"Mesh object '{real_mesh_name}' not found for vertex group calibration. Skipping..."
                    if lod_index == 0 or not self._auto_lod_enabled:
                        logger.warning(message)
                    else:
                        logger.debug(message)
                    continue
                if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
                    logger.warning(
                        f"Object '{mesh_object.name}' has no mesh data in the blender scene. "
                        "Skipping vertex group calibration..."
                    )
                    continue

                self._dna_writer.clearSkinWeights(meshIndex=mesh_index)
                # Create a lookup table for the vertex group names by their index
                vertex_group_lookup = {
                    vertex_group.index: vertex_group.name for vertex_group in mesh_object.vertex_groups
                }
                # Create a lookup for bone indices by their names
                bone_index_lookup = {
                    self._dna_reader.getJointName(index): index for index in range(self._dna_reader.getJointCount())
                }

                # Loop through the vertices and get the vertex group names and the vertex and weights
                for vertex in mesh_object.data.vertices:
                    vertex_group_names = [vertex_group_lookup.get(group.group, "") for group in vertex.groups]
                    bone_indices = []
                    weights = []

                    for vertex_group_name in vertex_group_names:
                        bone_index = bone_index_lookup.get(vertex_group_name)
                        vertex_group = mesh_object.vertex_groups.get(vertex_group_name)
                        if bone_index is not None and vertex_group:
                            weight = vertex_group.weight(vertex.index)
                            bone_indices.append(bone_index)
                            weights.append(weight)

                    self._dna_writer.setSkinWeightsJointIndices(
                        meshIndex=mesh_index, vertexIndex=vertex.index, jointIndices=bone_indices
                    )
                    self._dna_writer.setSkinWeightsValues(
                        meshIndex=mesh_index, vertexIndex=vertex.index, weights=weights
                    )

    def calibrate_bone_transforms(self):
        ignored_bone_names = [i for i, _ in self._extra_bones]

        logger.info("Calibrating bones...")
        dna_x_translations = self._dna_reader.getNeutralJointTranslationXs()
        dna_y_translations = self._dna_reader.getNeutralJointTranslationYs()
        dna_z_translations = self._dna_reader.getNeutralJointTranslationZs()
        dna_x_rotations = self._dna_reader.getNeutralJointRotationXs()
        dna_y_rotations = self._dna_reader.getNeutralJointRotationYs()
        dna_z_rotations = self._dna_reader.getNeutralJointRotationZs()

        self._bone_index_lookup = {
            self._dna_reader.getJointName(index): index for index in range(self._dna_reader.getJointCount())
        }

        # Read the rig's own (already head-to-body reconciled) bone transforms in
        # this component's export space: joint 0 in world, every other joint
        # local to its Blender parent. The head and body skeletons do NOT share a
        # hierarchy (e.g. the head root ``spine_04`` is exported in world space
        # but is a mid-chain, parent-local joint in the body), so the overlapping
        # bones must be aligned in WORLD space on the rig itself (done by the
        # converter's head-to-body reconcile / by the source DNA on an assembled
        # character) -- never by swapping the body's parent-local values into the
        # head DNA, which corrupts the head root and shifts the whole skeleton.
        _, bone_names, _, _, translations, rotations = self.get_bone_transforms(
            self._rig_object, extra_bones=self._extra_bones
        )
        for bone_name, bone_translation, bone_rotation in zip(bone_names, translations, rotations, strict=False):
            if bone_name in ignored_bone_names:
                continue

            _bone_translation = Vector(bone_translation)
            _bone_rotation = Vector(bone_rotation)

            dna_bone_index = self._bone_index_lookup.get(bone_name)
            if dna_bone_index is not None:
                dna_bone_translation = Vector(
                    (
                        dna_x_translations[dna_bone_index],
                        dna_y_translations[dna_bone_index],
                        dna_z_translations[dna_bone_index],
                    )
                )
                translation_delta = _bone_translation - dna_bone_translation

                # Only modify the bone translations that are different to avoid floating point value drift
                if translation_delta.length > BONE_DELTA_THRESHOLD:
                    dna_x_translations[dna_bone_index] = _bone_translation[0]
                    dna_y_translations[dna_bone_index] = _bone_translation[1]
                    dna_z_translations[dna_bone_index] = _bone_translation[2]

                dna_bone_rotation = Vector(
                    (
                        dna_x_rotations[dna_bone_index],
                        dna_y_rotations[dna_bone_index],
                        dna_z_rotations[dna_bone_index],
                    )
                )
                rotation_delta = _bone_rotation - dna_bone_rotation

                # Only modify the bone rotations that are different to avoid floating point value drift
                # Also, handle angle wrapping (e.g., 180 vs -180 degrees) issues
                if BONE_DELTA_THRESHOLD < abs(rotation_delta.x) < 360:
                    dna_x_rotations[dna_bone_index] = _bone_rotation[0]
                if BONE_DELTA_THRESHOLD < abs(rotation_delta.y) < 360:
                    dna_y_rotations[dna_bone_index] = _bone_rotation[1]
                if BONE_DELTA_THRESHOLD < abs(rotation_delta.z) < 360:
                    dna_z_rotations[dna_bone_index] = _bone_rotation[2]
            else:
                logger.warning(f'No DNA bone index found for bone "{bone_name}". Ignored from calibration...')

        self._dna_writer.setNeutralJointTranslations(
            [[x, y, z] for x, y, z in zip(dna_x_translations, dna_y_translations, dna_z_translations, strict=False)]
        )
        self._dna_writer.setNeutralJointRotations(
            [[x, y, z] for x, y, z in zip(dna_x_rotations, dna_y_rotations, dna_z_rotations, strict=False)]
        )

    def run(self) -> tuple[bool, str, str, Callable | None]:
        self.initialize_scene_data()
        if self._instance.output.run_validations:
            valid, title, message, fix = self.validate()
            if not valid:
                return False, title, message, fix

        if self._include_meshes:
            self.calibrate_vertex_positions()
            if self._include_normals:
                self._report("Calibrating normals...")
                self.calibrate_normals()
        if self._include_shape_keys:
            self._report("Calibrating shape keys...")
            self.calibrate_shape_keys()
        elif self._zero_shape_deltas:
            self._report("Zeroing shape deltas...")
            self.zero_blend_shape_deltas()
        if self._include_vertex_groups:
            self._report("Calibrating skin weights...")
            self.calibrate_vertex_groups()
        if self._include_bones:
            self._report("Calibrating bones...")
            self.calibrate_bone_transforms()

        logger.info(f'Saving DNA to: "{self._target_dna_file}"...')
        self._dna_writer.write()

        if not dna.Status.isOk():
            status = dna.Status.get()
            raise RuntimeError(f"Error saving DNA: {status.message}")
        logger.info(f'DNA calibrated successfully to: "{self._target_dna_file}"')

        self.save_images()

        return True, "Success", f"Calibration of {self._component_type} successful.", None
