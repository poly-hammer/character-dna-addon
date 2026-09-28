"""DNA's in-memory Blender basis and canonical interchange basis.

XZY is the XYZ Maya sequence expressed after the Y/Z axis exchange. Keeping
XYZ would change additive facial animation, even with matching neutral poses.
Units remain centimetres and degrees; scene adapters still handle unit scale.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import bpy

    from ..typing import dna

DNA_ROTATION_MODE = "XZY"
COORDINATE_VERSION = 1


def validate_rig_basis(rig: "bpy.types.ID | None") -> None:
    """Old rigs have Maya bone-local axes despite their Z-up appearance."""
    if rig is not None and rig.get("dna_coordinate_version", 0) != COORDINATE_VERSION:
        raise ValueError(
            f"{rig.name} uses the legacy Maya joint basis. Reimport its DNA into a new rig "
            "to use Blender-native coordinates; retain the old file for existing animation."
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
        from .misc import release_dna_handle

        self.writer.write()
        if not dna.Status.isOk():
            raise RuntimeError(dna.Status.get().message)
        self.stream.seek(0)
        reader = dna.BinaryStreamReader(self.stream, configuration(blender=False), None)
        reader.read()
        if not dna.Status.isOk():
            raise RuntimeError(dna.Status.get().message)
        stream = dna.FileStream(str(self.file_path), dna.AccessMode_Write, dna.OpenMode_Binary, None)
        writer_class = dna.JSONStreamWriter if self.file_format == "json" else dna.BinaryStreamWriter
        output = writer_class(stream)
        try:
            output.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            output.write()
            if not dna.Status.isOk():
                raise RuntimeError(dna.Status.get().message)
        finally:
            release_dna_handle(output)
            release_dna_handle(stream)
            release_dna_handle(reader)

    def close(self) -> None:
        from .misc import release_dna_handle

        release_dna_handle(self.writer)
        release_dna_handle(self.stream)
        self.writer = self.stream = None
