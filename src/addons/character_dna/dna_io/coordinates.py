"""DNA's in-memory Blender basis and canonical interchange basis.

XZY is the XYZ Maya sequence expressed after the Y/Z axis exchange. Keeping
XYZ would change additive facial animation, even with matching neutral poses.
Units remain centimetres and degrees; scene adapters still handle unit scale.
"""

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import bpy

    from ..typing import dna

DNA_ROTATION_MODE = "XZY"
COORDINATE_VERSION = 1


def validate_rig_basis(rig: "bpy.types.ID | None") -> None:
    """Old rigs have Y-up bone-local axes despite their Z-up appearance."""
    if rig is not None and rig.get("dna_coordinate_version", 0) != COORDINATE_VERSION:
        raise ValueError(
            f"{rig.name} uses the legacy Y-up joint basis. Run Migrate Legacy Data "
            "to re-import its DNA in place and upgrade its animation and drivers."
        )


def configuration(data_layer: str = "All", *, blender: bool = True) -> "dna.Configuration":
    from ..bindings import dna

    system = dna.CoordinateSystem()
    system.x = dna.Direction_left
    system.y = dna.Direction_back if blender else dna.Direction_up
    system.z = dna.Direction_up if blender else dna.Direction_front
    config = dna.Configuration()
    config.layer = getattr(dna, f"DataLayer_{data_layer}")
    config.unknownLayerPolicy = dna.UnknownLayerPolicy_Preserve
    config.coordinateSystemTransformPolicy = dna.CoordinateSystemTransformPolicy_Transform
    config.coordinateSystem = system
    config.rotationSequence = dna.RotationSequence_xzy if blender else dna.RotationSequence_xyz
    return config


def transform_reader(reader: Any, *, blender: bool = True, data_layer: str = "All") -> "dna.BinaryStreamReader":
    """Convert every known DNA layer through the SDK, including JSON readers."""
    from ..bindings import dna

    stream = dna.MemoryStream()
    writer = dna.BinaryStreamWriter(stream)
    writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.write()
    if not dna.Status.isOk():
        raise RuntimeError(dna.Status.get().message)
    stream.seek(0)
    converted = dna.BinaryStreamReader(stream, configuration(data_layer, blender=blender), None)
    converted.read()
    if not dna.Status.isOk():
        raise RuntimeError(dna.Status.get().message)
    return converted


class CanonicalDNAWriter:
    """Stage edits in the reader's basis, then export a true Y-up/XYZ DNA.

    The destination is opened only after transformation succeeds. This avoids
    truncating a source file while an editor still owns its reader.
    """

    def __init__(self, file_path: Path, file_format: str) -> None:
        from ..bindings import dna

        self.file_path = file_path
        self.file_format = file_format
        self.stream = dna.MemoryStream()
        self.writer = dna.BinaryStreamWriter(self.stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)

    def write(self) -> None:
        from ..bindings import dna
        from ..runtime.engine import invalidate_model_file
        from .document import partition_backup, publish_partition, validate_write
        from .misc import release_dna_handle

        validate_write(self.file_path)
        previous_partition = partition_backup(self.file_path)

        self.writer.write()
        if not dna.Status.isOk():
            raise RuntimeError(dna.Status.get().message)
        self.stream.seek(0)
        reader = dna.BinaryStreamReader(self.stream, configuration(blender=False), None)
        reader.read()
        if not dna.Status.isOk():
            raise RuntimeError(dna.Status.get().message)
        invalidate_model_file(self.file_path)
        stream = dna.FileStream(str(self.file_path), dna.AccessMode_Write, dna.OpenMode_Binary, None)
        writer_class = dna.JSONStreamWriter if self.file_format == "json" else dna.BinaryStreamWriter
        output = writer_class(stream)
        try:
            try:
                output.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
                output.write()
                if not dna.Status.isOk():
                    raise RuntimeError(dna.Status.get().message)
            finally:
                release_dna_handle(output)
                release_dna_handle(stream)
                release_dna_handle(reader)
            publish_partition(self.file_path)
        except Exception:
            if previous_partition is not None:
                self.file_path.write_bytes(previous_partition)
            raise

    def close(self) -> None:
        from .misc import release_dna_handle

        release_dna_handle(self.writer)
        release_dna_handle(self.stream)
        self.writer = self.stream = None


def write_shape_targets(file_path: Path, targets: dict[tuple[int, int], tuple[list[int], list[list[float]]]]) -> None:
    """Patch existing targets in place, preserving every other native DNA field.

    Deltas use Blender's basis and the DNA's translation units, as returned by
    the configured reader. Only these vectors need the inverse basis transform;
    re-expressing the whole rig can otherwise change neutral Euler branches.
    Structural edits and exports continue through CanonicalDNAWriter.
    """
    from ..bindings import dna
    from ..runtime.engine import invalidate_model_file
    from .document import partition_backup, publish_partition, validate_write
    from .misc import release_dna_handle

    validate_write(file_path)
    previous_partition = partition_backup(file_path)
    try:
        with ExitStack() as handles:
            source = dna.FileStream(str(file_path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
            handles.callback(release_dna_handle, source)
            reader = dna.BinaryStreamReader(source, configuration(blender=False), None)
            handles.callback(release_dna_handle, reader)
            reader.read()
            if not dna.Status.isOk():
                raise RuntimeError(dna.Status.get().message)
            stream = dna.FileStream(str(file_path), dna.AccessMode_Write, dna.OpenMode_Binary, None)
            handles.callback(release_dna_handle, stream)
            writer = dna.BinaryStreamWriter(stream)
            handles.callback(release_dna_handle, writer)
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            for (mesh, target), (indices, deltas) in targets.items():
                if not 0 <= mesh < reader.getMeshCount() or not 0 <= target < reader.getBlendShapeTargetCount(mesh):
                    raise ValueError("A target-only edit cannot add a new mesh or shape target.")
                vertex_count = reader.getVertexPositionCount(mesh)
                if len(indices) != len(deltas) or any(index < 0 or index >= vertex_count for index in indices):
                    raise ValueError("Invalid sparse shape target vertex indices.")
                writer.setBlendShapeTargetVertexIndices(mesh, target, indices)
                writer.setBlendShapeTargetDeltas(mesh, target, [[float(x), float(z), -float(y)] for x, y, z in deltas])
            release_dna_handle(reader)
            release_dna_handle(source)
            invalidate_model_file(file_path)
            writer.write()
            if not dna.Status.isOk():
                raise RuntimeError(dna.Status.get().message)
        publish_partition(file_path, targets=set(targets))
    except Exception:
        if previous_partition is not None:
            file_path.write_bytes(previous_partition)
        raise
