import json

from collections.abc import Sequence
from pathlib import Path

from character_dna.bindings import dna
from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
from constants import CI_BONE_SAMPLE_SIZE, CI_REQUIRED_BONE_NAMES, RUNNING_CI


def _ci_first_n(names: list[str], sample_size: int, required: Sequence[str] = ()) -> list[str]:
    """Outside CI return ``names`` unchanged. On CI keep the first ``sample_size``
    names plus any ``required`` names that are present but fell outside that slice,
    so the entities the assertions explicitly verify always survive the subsample.
    """
    if not RUNNING_CI:
        return names
    selected = list(names[:sample_size])
    for name in required:
        if name in names and name not in selected:
            selected.append(name)
    return selected


def _ci_only(names: list[str], required: Sequence[str]) -> list[str]:
    """Outside CI return ``names`` unchanged. On CI restrict to ``required`` names
    (preserving ``names`` order), used where only specific meshes need checking.
    """
    if not RUNNING_CI:
        return names
    required_set = set(required)
    return [name for name in names if name in required_set]


def get_dna_json_data(dna_file_path: Path, json_file_path: Path, data_layer: str = "All") -> dict:
    reader = get_dna_reader(dna_file_path, file_format="binary", data_layer=data_layer)
    writer = None
    try:
        writer = get_dna_writer(json_file_path, file_format="json")
        writer.setFrom(reader, getattr(dna, f"DataLayer_{data_layer}"), dna.UnknownLayerPolicy_Preserve, None)
        writer.write()
        if not dna.Status.isOk():
            status = dna.Status.get()
            raise RuntimeError(f"Error saving DNA: {status.message}")
    finally:
        release_dna_handle(writer)
        release_dna_handle(reader)

    with json_file_path.open() as file:
        text = file.read()
        text = "".join(text.split())
        text = text.replace('{"data":{"value":["A","N","D"]}}', "")
        data = json.loads(text)
        return data


def get_bone_names(dna_file_path: Path) -> list[str]:
    reader = get_dna_reader(file_path=dna_file_path, file_format="binary", data_layer="Definition")
    try:
        return [reader.getJointName(index) for index in range(reader.getJointCount())]
    finally:
        release_dna_handle(reader)


def get_mesh_names(dna_file_path: Path) -> list[str]:
    reader = get_dna_reader(file_path=dna_file_path, file_format="binary", data_layer="Definition")
    try:
        return [reader.getMeshName(index) for index in range(reader.getMeshCount())]
    finally:
        release_dna_handle(reader)


def get_mesh_vertex_count(dna_file_path: Path) -> list[int]:
    reader = get_dna_reader(file_path=dna_file_path, file_format="binary", data_layer="Geometry")
    try:
        return [reader.getVertexPositionCount(index) for index in range(reader.getMeshCount())]
    finally:
        release_dna_handle(reader)


def get_test_bone_definitions_params(dna_file_path: Path):
    bone_names = _ci_first_n(get_bone_names(dna_file_path), CI_BONE_SAMPLE_SIZE, CI_REQUIRED_BONE_NAMES)
    for bone_name in bone_names:
        attributes = ["neutralJointRotations", "neutralJointTranslations"]
        axis_names = ["x", "y", "z"]
        for attribute in attributes:
            for axis_name in axis_names:
                yield bone_name, attribute, axis_name


def get_test_bone_behaviors_params(dna_file_path: Path):
    yield from _ci_first_n(get_bone_names(dna_file_path), CI_BONE_SAMPLE_SIZE, CI_REQUIRED_BONE_NAMES)


def get_test_mesh_geometry_params(
    dna_file_path: Path,
    lods: list[int] | None = None,
    vertex_positions: bool = True,
    normals: bool = True,
    uvs: bool = True,
    ci_required: list[str] | None = None,
):
    mesh_names = get_mesh_names(dna_file_path)
    if ci_required is not None:
        mesh_names = _ci_only(mesh_names, ci_required)
    for mesh_name in mesh_names:
        if lods and not any(mesh_name.endswith(f"_lod{lod}_mesh") for lod in lods):
            # skip checking meshes that are not in the specified lods
            continue

        attributes = []
        if vertex_positions:
            attributes.append("positions")
        if normals:
            attributes.append("normals")
        if uvs:
            attributes.append("textureCoordinates")

        for attribute in attributes:
            axis_names = ["x", "y", "z"]
            if attribute == "textureCoordinates":
                axis_names = ["u", "v"]

            for axis_name in axis_names:
                yield mesh_name, attribute, axis_name


def get_test_skin_weights_params(
    dna_file_path: Path,
    lods: list[int] | None = None,
    ci_required: list[str] | None = None,
):
    required_set = set(ci_required) if (ci_required is not None and RUNNING_CI) else None
    for mesh_name, mesh_vertex_count in zip(
        get_mesh_names(dna_file_path), get_mesh_vertex_count(dna_file_path), strict=False
    ):
        if required_set is not None and mesh_name not in required_set:
            # on CI only check the explicitly required meshes
            continue
        if lods and not any(mesh_name.endswith(f"_lod{lod}_mesh") for lod in lods):
            # skip checking meshes that are not in the specified lods
            continue

        attributes = ["skinWeights"]
        for attribute in attributes:
            yield mesh_name, attribute, mesh_vertex_count
