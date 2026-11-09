"""Read canonical component geometry through the scene merge/split boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import bpy
import numpy as np

from mathutils import Matrix

from .dna_io.vertex_indices import for_mesh


if TYPE_CHECKING:
    from .rig_component import RigComponent


@dataclass(frozen=True)
class MeshSample:
    """Owned evaluated positions, in mesh space, addressed by DNA position IDs."""

    positions: np.ndarray
    position_ids: tuple[int, ...]
    matrix_world: Matrix


@contextmanager
def _sampling_source(component: "RigComponent", source: bpy.types.Object) -> Iterator[bpy.types.Object]:
    # The converter owns both the correspondence and the reconstruction of
    # pending edits. Apply its split updates only to a disposable source copy.
    from .editors.converter.scene import merge

    surface = merge.proxy_for(component.instance, source)
    if surface is None:
        yield source
        return

    updates = [entry for entry in merge.prepare_mesh_edits(component.instance).values() if entry[0] == source]
    if not updates:
        yield source
        return

    copied = source.copy()
    mesh = None
    try:
        mesh = copied.data = source.data.copy()
        # A disposable sample must not become a runtime or component binding.
        for key in list(copied.keys()):
            if key.startswith(("ca_dna_", "dna_merge_")) or key == "dna_working_source":
                del copied[key]
        for _, name, positions, settings, sources, metadata in updates:
            merge._apply_updates(  # noqa: SLF001
                component.instance,
                {**sources, component.kind: copied},
                metadata,
                [(copied, name, positions, settings)],
                publish=False,
            )
        yield copied
    finally:
        bpy.data.objects.remove(copied, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def sample_mesh(component: "RigComponent", mesh_index: int) -> MeshSample:
    """Sample split or merged scene geometry without committing pending edits.

    The component reader is also the canonical partition of a merged DNA
    document. Its indices never address the combined document or welded scene
    mesh directly. Merged edits use exactly the converter's Split reconstruction.
    """
    source = component.mesh_index_lookup.get(mesh_index)
    if source is None:
        raise ValueError(f"Import {component.dna_reader.getMeshName(mesh_index)} before sampling its landmarks.")
    if source.mode != "OBJECT":
        raise ValueError(f"Return {source.name} to Object Mode before sampling its landmarks.")
    reader = component.dna_reader
    position_ids = tuple(for_mesh(reader, mesh_index, source.data))
    with _sampling_source(component, source) as obj:
        # Merged DNA archives canonical Objects outside every scene. Copies of
        # scene sources are also unlinked. Give either a temporary evaluation base.
        linked = obj.name not in bpy.context.view_layer.objects
        collection = bpy.context.scene.collection
        try:
            if linked:
                collection.objects.link(obj)
                obj.hide_set(True)
            bpy.context.view_layer.update()
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            if tuple(for_mesh(reader, mesh_index, evaluated.data)) != position_ids:
                raise ValueError(f"{source.name} changed DNA position order during evaluation.")
            positions = np.empty((len(position_ids), 3), dtype=np.float32)
            evaluated.data.vertices.foreach_get("co", positions.ravel())
            if not np.isfinite(positions).all():
                raise ValueError(f"{source.name} has non-finite evaluated vertices.")
            positions.flags.writeable = False
            return MeshSample(positions, position_ids, evaluated.matrix_world.copy())
        finally:
            if linked:
                collection.objects.unlink(obj)
                bpy.context.view_layer.update()
