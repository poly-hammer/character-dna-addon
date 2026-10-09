# standard library imports
import logging
import re

from pathlib import Path
from typing import Any, Literal

# third party imports
import bpy
import numpy as np

# local imports
from ..constants import SHAPE_KEY_BASIS_NAME, ComponentType
from ..typing import *  # noqa: F403
from ..utilities import (
    exclude_rig_instance_evaluation,
    get_addon_window_manager_properties,
    switch_to_object_mode,
)
from .coordinates import CanonicalDNAWriter


logger = logging.getLogger(__name__)

FileFormat = Literal["binary", "json"]
DataLayer = Literal[
    "Descriptor",
    "Definition",
    "Behavior",
    "Geometry",
    "GeometryWithoutBlendShapes",
    "MachineLearnedBehavior",
    "RBFBehavior",
    "JointBehaviorMetadata",
    "TwistSwingBehavior",
    "All",
]


def _json_input_stream(file_path: Path, memory_resource: Any) -> Any:
    """Flush subnormal JSON floats that the SDK's C++ reader rejects on macOS."""
    from ..bindings import dna

    minimum = 2.0**-126  # Smallest normal IEEE 754 single-precision value.

    def normalize(match: re.Match[bytes]) -> bytes:
        token = match.group()
        # Match quoted strings first so names and opaque layer data stay intact.
        if token.startswith(b'"'):
            return token
        return b"0.0" if 0 < abs(float(token)) < minimum else token

    data = re.sub(rb'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?[eE]-\d+', normalize, file_path.read_bytes())
    stream = dna.MemoryStream(memory_resource)
    stream.write(data.decode("utf-8"), len(data))
    stream.seek(0)
    return stream


def release_dna_handle(handle: Any) -> None:
    """Destroy an OpenRigLogic handle now instead of waiting for garbage collection.

    ``dna``/``riglogic`` factory objects are caller-owned C++ allocations. The Python
    bindings only call ``destroy`` from the wrapper's ``__del__``, so dropping the last
    reference eventually frees them -- but not deterministically, which matters on Windows
    where a live reader/writer keeps the ``.dna`` file handle open.

    Args:
        handle: A wrapper returned by :func:`get_dna_reader` / :func:`get_dna_writer`, or
            any other RAII-wrapped binding object. ``None`` and raw SWIG proxies are ignored.
    """
    if handle is None:
        return
    if isinstance(handle, CanonicalDNAWriter):
        handle.close()
        return

    instance = getattr(handle, "_instance", None)
    if instance is None:
        return

    try:
        type(handle).destroy(instance)
    except Exception as error:
        logger.warning(f"Failed to destroy {type(handle).__name__}: {error}")
    handle._instance = None  # noqa: SLF001
    # The wrapper keeps its constructor arguments alive, which is what owns the stream a
    # reader/writer was built on. Dropping them here releases those in child-before-parent order.
    if getattr(handle, "_args", None):
        handle._args = ()  # noqa: SLF001


def get_dna_reader(
    file_path: Path,
    file_format: FileFormat = "binary",
    data_layer: DataLayer = "All",
    memory_resource: "dna.MemoryResource | None" = None,
) -> "dna.BinaryStreamReader":
    from ..bindings import dna  # type: ignore[reportAttributeAccessIssue]

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    # Construct via the class rather than `.create()`: the constructor returns the binding's
    # owning wrapper, which destroys the C++ object on release and keeps the stream alive for
    # exactly as long as the reader needs it. `.create()` returns a raw pointer that leaks.
    from .coordinates import configuration, transform_reader

    config = configuration(data_layer)

    if file_format.lower() == "json":
        # JSON has no Configuration overload. Transform it through a configured
        # binary reader below, respecting the JSON file's declared source basis.
        stream = _json_input_stream(file_path, memory_resource)
        reader = dna.JSONStreamReader(stream, memory_resource)
    elif file_format.lower() == "binary":
        stream = dna.FileStream(str(file_path), dna.AccessMode_Read, dna.OpenMode_Binary, memory_resource)
        reader = dna.BinaryStreamReader(stream, config, memory_resource)
    else:
        raise ValueError(f"Invalid file format '{file_format}'. Must be 'binary' or 'json'.")

    try:
        reader.read()
    except IndexError as error:
        logger.debug(f"Error reading DNA file '{file_path}': {error}")
        release_dna_handle(reader)
        return None  # pyright: ignore[reportReturnType]

    if not dna.Status.isOk():
        status = dna.Status.get()
        release_dna_handle(reader)
        raise RuntimeError(f'Error loading DNA: {status.message} from "{file_path}"')
    if file_format.lower() == "json":
        transformed = transform_reader(reader, data_layer=data_layer)
        release_dna_handle(reader)
        return transformed
    return reader


def get_dna_writer(file_path: Path, file_format: FileFormat = "binary") -> CanonicalDNAWriter:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format.lower() not in ("binary", "json"):
        raise ValueError(f"Invalid file format: {file_format}")
    return CanonicalDNAWriter(file_path, file_format.lower())


def get_dna_component_type(file_path: Path) -> ComponentType | None:
    """
    Determine the DNA component type based on the mesh names in the DNA file.

    Mesh names are the strongest signal, but some DNA files (for example clothing
    or custom body assets) do not include "head" or "body" in their mesh names. In
    that case we fall back to the joint names: head DNA files contain facial joints
    (``FACIAL_*``) while body DNA files are skinned to the body skeleton only.
    """
    component_type = None
    dna_reader = get_dna_reader(file_path=file_path, file_format="binary", data_layer="Definition")
    if dna_reader:
        for index in range(dna_reader.getMeshCount()):
            mesh_name = dna_reader.getMeshName(index)
            if "head" in mesh_name.lower():
                component_type = "head"
            elif "body" in mesh_name.lower():
                component_type = "body"

        # Fall back to joint names when the mesh names are inconclusive.
        if component_type is None and dna_reader.getJointCount() > 0:
            has_facial_joint = any(
                "facial" in dna_reader.getJointName(index).lower() for index in range(dna_reader.getJointCount())
            )
            component_type = "head" if has_facial_joint else "body"

        release_dna_handle(dna_reader)
    return component_type


@exclude_rig_instance_evaluation
def create_shape_key(
    index: int,
    mesh_index: int,
    mesh_object: bpy.types.Object,
    reader: "dna.BinaryStreamReader",
    name: str,
    prefix: str = "",
    is_neutral: bool = False,
    linear_modifier: float = 1.0,
) -> bpy.types.ShapeKey | None:
    if not mesh_object:
        logger.error(f"Mesh object not found for shape key {name}. Skipping creation.")
        return None
    if not mesh_object.data or not isinstance(mesh_object.data, bpy.types.Mesh):
        logger.error(
            f"Object '{mesh_object.name}' has no mesh data in the blender scene. Skipping shape key creation..."
        )
        return None
    if not mesh_object.data.shape_keys:
        mesh_object.shape_key_add(name=SHAPE_KEY_BASIS_NAME, from_mix=False)

    window_manager_properties = get_addon_window_manager_properties()
    window_manager_properties.progress_mesh_name = mesh_object.name
    # create the new key block on the shape key
    logger.debug(f"Creating shape key {name}")
    shape_key_name = f"{prefix}{name}"

    switch_to_object_mode()

    # remove any pre-existing key block with this name so re-imports overwrite cleanly
    shape_key = mesh_object.data.shape_keys.key_blocks.get(shape_key_name)  # type: ignore[attr-defined]
    if shape_key:
        shape_key.lock_shape = False
        mesh_object.shape_key_remove(shape_key)

    shape_key_block = mesh_object.shape_key_add(name=shape_key_name, from_mix=False)
    # zero the influence so the imported key is stored but not applied on top of the
    # basis (the delta geometry is written directly to the key block's data, so this
    # only affects the value slider, not the stored shape)
    shape_key_block.value = 0.0

    # Import the deltas if the shape key is not supposed to be neutral
    if not is_neutral:
        apply_blend_shape_deltas(
            mesh_object=mesh_object,
            shape_key_block=shape_key_block,
            reader=reader,
            mesh_index=mesh_index,
            index=index,
            name=name,
            linear_modifier=linear_modifier,
        )

    shape_key_block.lock_shape = True

    return shape_key_block


def apply_blend_shape_deltas(
    mesh_object: bpy.types.Object,
    shape_key_block: bpy.types.ShapeKey,
    reader: "dna.BinaryStreamReader",
    mesh_index: int,
    index: int,
    name: str,
    linear_modifier: float = 1.0,
) -> None:
    """Apply a DNA blend shape target's deltas onto ``shape_key_block`` using
    vectorized numpy + ``foreach_get``/``foreach_set`` for speed.

    Reads the basis (reference key) coordinates once, scales the sparse deltas
    from DNA units into Blender units, scatters them onto the
    affected vertices, and writes the whole shape key in a single bulk call.
    """
    vertex_indices = reader.getBlendShapeTargetVertexIndices(mesh_index, index)
    if len(vertex_indices) == 0:
        return

    reference_key = mesh_object.data.shape_keys.reference_key  # type: ignore[attr-defined]
    vertex_count = len(reference_key.data)

    # read the basis coordinates once as a flat (x, y, z) array
    base_flat = np.empty(vertex_count * 3, dtype=np.float32)
    reference_key.data.foreach_get("co", base_flat)
    new_flat = base_flat.copy()
    base = base_flat.reshape(-1, 3)
    new = new_flat.reshape(-1, 3)

    vertex_indices = np.asarray(vertex_indices, dtype=np.int64)
    deltas = np.empty((len(vertex_indices), 3), dtype=np.float32)
    deltas[:, 0] = reader.getBlendShapeTargetDeltaXs(mesh_index, index)
    deltas[:, 1] = reader.getBlendShapeTargetDeltaYs(mesh_index, index)
    deltas[:, 2] = reader.getBlendShapeTargetDeltaZs(mesh_index, index)

    scaled = deltas * linear_modifier

    # guard against vertex indices that no longer exist on the base mesh
    valid = vertex_indices < vertex_count
    if not valid.all():
        logger.warning(
            f'Some vertex indices are missing for shape key "{name}". '
            f'Were they deleted on the base mesh "{mesh_object.name}"?'
        )
        vertex_indices = vertex_indices[valid]
        scaled = scaled[valid]

    # the new vertex layout is the original vertex layout with the deltas from the dna applied
    new[vertex_indices] = base[vertex_indices] + scaled
    shape_key_block.data.foreach_set("co", new_flat)
