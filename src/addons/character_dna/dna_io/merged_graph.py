"""Reversible native DNA composition with explicit component index domains.

JSON describes the small behavior layers; geometry uses typed native arrays.
The published file is an SDK binary containing the complete remapped graph.
"""

import base64
import hashlib
import json
import tempfile

from pathlib import Path
from typing import Any

import numpy as np

from ..bindings import dna
from ..constants import HEAD_TO_BODY_EDGE_LOOP_FILE_PATH, HEAD_TO_BODY_LOD_MAPPING
from . import native_geometry
from .coordinates import configuration
from .misc import release_dna_handle


KINDS = ("head", "body")
LAYERS = (
    "desc1.1",
    "defn1.1",
    "bhvr1.1",
    "geom1.1",
    "mlbh1.0",
    "rbfb1.0",
    "rbfe1.0",
    "jbmd1.0",
    "twsw1.0",
    "dsce1.0",
    "mlbe1.0",
)
NAME_DOMAINS = {
    "guiControlNames": "gui",
    "rawControlNames": "raw",
    "jointNames": "joint",
    "blendShapeChannelNames": "blend",
    "animatedMapNames": "animated",
}
LOD_DOMAINS = {
    "lodJointMapping": "joint",
    "lodBlendShapeMapping": "blend",
    "lodAnimatedMapMapping": "animated",
    "lodMeshMapping": "mesh",
    "lodSolverMapping": "solver",
    "lodNeuralNetworkMapping": "network",
}


def clone(value: Any) -> Any:
    """Copy mutable graph containers without visiting every numeric array element."""
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        if not value or not isinstance(value[0], (dict, list)):
            return list(value)
        return [clone(item) for item in value]
    return value


def _status() -> None:
    if not dna.Status.isOk():
        raise ValueError(dna.Status.get().message)


def from_bytes(content: bytes, *, geometry: bool = True) -> dict:
    """Read every SDK layer in canonical Y-up coordinates, including extensions."""
    with tempfile.TemporaryDirectory(prefix="dna-graph-") as directory:
        root = Path(directory)
        (root / "input.dna").write_bytes(content)
        stream = dna.FileStream(str(root / "input.dna"), dna.AccessMode_Read, dna.OpenMode_Binary, None)
        reader = dna.BinaryStreamReader(stream, configuration(blender=False), None)
        output = dna.FileStream(str(root / "output.json"), dna.AccessMode_Write, dna.OpenMode_Binary, None)
        writer = dna.JSONStreamWriter(output, 0, None)
        try:
            reader.read()
            _status()
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            writer.setFileFormatGeneration(2)
            writer.setFileFormatVersion(8)
            # The SDK JSON writer does not escape quotes inside metadata strings.
            # Transport descriptor strings through the native API instead.
            descriptor_metadata = [
                [reader.getMetaDataKey(index), reader.getMetaDataValue(reader.getMetaDataKey(index))]
                for index in range(reader.getMetaDataCount())
            ]
            writer.clearMetaData()
            writer.clearMeshes()
            for index in range(reader.getJointGroupCount()):
                writer.setJointGroupValues(index, [])
            writer.write()
            _status()
            graph = json.loads((root / "output.json").read_text(encoding="utf-8"))
            graph["desc1.1"]["metadata"] = descriptor_metadata
            graph["geom1.1"] = native_geometry.read(reader) if geometry else {"meshes": []}
            _float_fields(graph, reader=reader)
        finally:
            for handle in (writer, output, reader, stream):
                release_dna_handle(handle)
        return graph


def to_bytes(graph: dict, payload: dict | None = None) -> bytes:
    from . import metadata

    with tempfile.TemporaryDirectory(prefix="dna-graph-") as directory:
        root = Path(directory)
        serialized = {**graph, "desc1.1": {**graph["desc1.1"], "metadata": []}, "geom1.1": {"meshes": []}}
        behavior = graph["bhvr1.1"]
        serialized["bhvr1.1"] = {
            **behavior,
            "joints": {
                **behavior["joints"],
                # Keep the temporary matrix dimensions valid for SDK readers that
                # validate rows * columns before the native values are restored.
                "jointGroups": [
                    {**group, "inputIndices": [], "values": []} for group in behavior["joints"]["jointGroups"]
                ],
            },
        }
        # The SDK JSON reader consumes layers in archive order, not arbitrary map order.
        order = [
            entry["id"].to_bytes(4, "big").decode("ascii") + f"{entry['version'] >> 16}.{entry['version'] & 65535}"
            for entry in graph["index"]["entries"]
        ]
        serialized = {key: serialized[key] for key in ("signature", "version", "index", *order) if key in serialized}
        raw = json.dumps(serialized, separators=(",", ":"), allow_nan=False)
        # The SDK JSON float parser rejects subnormal values on some platforms.
        from .misc import _json_input_stream

        (root / "input.json").write_text(raw, encoding="utf-8")
        stream = _json_input_stream(root / "input.json", None)
        reader = dna.JSONStreamReader(stream, None)
        output = dna.FileStream(str(root / "output.dna"), dna.AccessMode_Write, dna.OpenMode_Binary, None)
        writer = dna.BinaryStreamWriter(output)
        try:
            reader.read()
            _status()
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            for index, group in enumerate(behavior["joints"]["jointGroups"]):
                writer.setJointGroupInputIndices(index, group["inputIndices"])
            _float_fields(graph, writer=writer)
            native_geometry.write(writer, graph["geom1.1"])
            for key, value in graph["desc1.1"]["metadata"]:
                if not isinstance(key, str) or not isinstance(value, str) or "\0" in key or "\0" in value:
                    raise ValueError("DNA metadata requires strings without literal NUL characters.")
                writer.setMetaData(key, value)
            if payload is not None:
                # Keep foreign metadata written above when replacing our namespace.
                for key, value in metadata.encode(payload).items():
                    writer.setMetaData(key, value)
            writer.write()
            _status()
        finally:
            for handle in (writer, output, reader, stream):
                release_dna_handle(handle)
        result = (root / "output.dna").read_bytes()
        if len(result) >= metadata.MAX_DOCUMENT_BYTES:
            raise ValueError("Combined DNA exceeds the supported file-size budget.")
        return result


def _float_fields(graph: dict, *, reader: Any = None, writer: Any = None) -> None:
    """Bypass SDK JSON's six-digit floats and its subnormal parser limitation."""

    def values(record: dict, field: str, api: str, *indices: int, scalar: bool = False) -> None:
        if reader is not None:
            value = getattr(reader, "get" + api)(*indices)
            record[field] = float(value) if scalar else np.asarray(value, dtype=np.float32).tolist()
        else:
            getattr(writer, "set" + api)(*indices, record[field])

    def vectors(record: dict, api: str, *indices: int, plural: str | None = None) -> None:
        if reader is not None:
            for axis in record:
                record[axis] = np.asarray(
                    getattr(reader, "get" + api + axis.capitalize())(*indices), dtype=np.float32
                ).tolist()
        else:
            array = np.asarray(list(record.values()), dtype=np.float32).T
            getattr(writer, "set" + (plural or api + "s"))(*indices, array.tolist())

    for field, api in (
        ("neutralJointTranslations", "NeutralJointTranslation"),
        ("neutralJointRotations", "NeutralJointRotation"),
    ):
        vectors(graph["defn1.1"][field], api)
    behavior = graph["bhvr1.1"]
    for table, api in (
        (behavior["controls"]["conditionals"], "GUIToRaw"),
        (behavior["animatedMaps"]["conditionals"], "AnimatedMap"),
    ):
        for field in ("fromValues", "toValues", "slopeValues", "cutValues"):
            values(table, field, api + field[0].upper() + field[1:])
    values(behavior["controls"]["psds"], "values", "PSDValues")
    for index, group in enumerate(behavior["joints"]["jointGroups"]):
        values(group, "values", "JointGroupValues", index)
    for index, solver in enumerate(graph["rbfb1.0"]["solvers"]):
        values(solver, "rawControlValues", "RBFSolverRawControlValues", index)
        values(solver, "radius", "RBFSolverRadius", index, scalar=True)
        values(solver, "weightThreshold", "RBFSolverWeightThreshold", index, scalar=True)
    for index, pose in enumerate(graph["rbfb1.0"]["poses"]):
        values(pose, "scale", "RBFPoseScale", index, scalar=True)
    for index, pose in enumerate(graph["rbfe1.0"]["poses"]):
        values(pose, "outputControlWeights", "RBFPoseOutputControlWeights", index)
    for label in ("twist", "swing"):
        for index, setup in enumerate(graph["twsw1.0"][label + "s"]):
            values(setup, label + "BlendWeights", label.capitalize() + "BlendWeights", index)
    for network_index, network in enumerate(graph["mlbh1.0"]["neuralNetworks"]):
        for index, layer in enumerate(network["layers"]):
            for field in ("biases", "weights"):
                values(layer, field, "NeuralNetworkLayer" + field.capitalize(), network_index, index)
            values(
                layer["activationFunction"],
                "parameters",
                "NeuralNetworkLayerActivationFunctionParameters",
                network_index,
                index,
            )


def digest(graph: dict) -> str:
    """Ignore archive byte offsets and our descriptor payload when comparing data."""
    from .metadata import PREFIX

    scalar_types = {bool, float, int, str, type(None)}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in {"index", "size", "offset"}}
        if isinstance(value, list):
            # SDK geometry and behavior arrays contain millions of scalar
            # values. Only containers need recursive archive-field filtering.
            if set(map(type, value)) <= scalar_types:
                return value
            return [clean(item) if isinstance(item, (dict, list)) else item for item in value]
        return value

    cleaned = clean(graph)
    cleaned["desc1.1"]["metadata"] = [
        pair for pair in cleaned["desc1.1"]["metadata"] if pair[0] != PREFIX and not pair[0].startswith(PREFIX + ".")
    ]
    return hashlib.sha256(json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def binary_digest(path: Path) -> str:
    """Hash native graph bytes without ownership chunks, avoiding JSON for reloads."""
    from .metadata import PREFIX

    stream = dna.FileStream(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    reader = dna.BinaryStreamReader(stream, configuration(blender=False), None)
    with tempfile.TemporaryDirectory(prefix="dna-revision-") as directory:
        output_path = Path(directory) / "graph.dna"
        output = dna.FileStream(str(output_path), dna.AccessMode_Write, dna.OpenMode_Binary, None)
        writer = dna.BinaryStreamWriter(output)
        try:
            reader.read()
            _status()
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            writer.clearMetaData()
            for index in range(reader.getMetaDataCount()):
                key = reader.getMetaDataKey(index)
                if key != PREFIX and not key.startswith(PREFIX + "."):
                    writer.setMetaData(key, reader.getMetaDataValue(key))
            writer.write()
            _status()
        finally:
            for handle in (writer, output, reader, stream):
                release_dna_handle(handle)
        return hashlib.sha256(output_path.read_bytes()).hexdigest()


def _counts(graph: dict) -> dict[str, int]:
    result = {domain: len(graph["defn1.1"][name]) for name, domain in NAME_DOMAINS.items()}
    result.update(
        psd=graph["bhvr1.1"]["controls"]["psdCount"],
        ml=len(graph["mlbh1.0"]["mlControlNames"]),
        rbf=len(graph["rbfe1.0"]["poseControlNames"]),
        pose=len(graph["rbfb1.0"]["poses"]),
        solver=len(graph["rbfb1.0"]["solvers"]),
        network=len(graph["mlbh1.0"]["neuralNetworks"]),
    )
    return result


def index_maps(parts: dict) -> dict:
    counts = {kind: _counts(graph) for kind, graph in parts.items()}
    maps = {kind: {} for kind in KINDS}
    for domain in counts["head"]:
        total = sum(counts[kind][domain] for kind in KINDS)
        if total > 65535:
            raise ValueError(f"Merged {domain} count exceeds the DNA uint16 capacity.")
        start = 0
        for kind in KINDS:
            maps[kind][domain] = list(range(start, start + counts[kind][domain]))
            start += counts[kind][domain]
    start = 0
    for kind in KINDS:
        maps[kind]["control"] = []
    for domain in ("raw", "psd", "ml", "rbf"):
        for kind in KINDS:
            maps[kind]["control"].extend(range(start, start + counts[kind][domain]))
            start += counts[kind][domain]
    if start > 65535 or sum(counts[k]["joint"] for k in KINDS) * 9 > 65535:
        raise ValueError("Merged control or joint-attribute indices exceed DNA uint16 capacity.")
    for kind in KINDS:
        maps[kind]["attribute"] = [joint * 9 + channel for joint in maps[kind]["joint"] for channel in range(9)]
    return maps


def _mapped(values: list, mapping: dict | list) -> list:
    try:
        if any(not isinstance(index, int) or index < 0 for index in values):
            raise ValueError("DNA relationship indices must be nonnegative integers.")
        return [mapping[index] for index in values]
    except (IndexError, KeyError) as error:
        raise ValueError("A DNA relationship crosses an unrecorded component index domain.") from error


def transform(graph: dict, maps: dict, *, inverse: bool = False) -> dict:  # noqa: PLR0912 - typed SDK index domains
    """Remap each known index domain independently; never offset arbitrary numbers."""
    geometry = graph["geom1.1"]
    graph = clone({key: value if key != "geom1.1" else {} for key, value in graph.items()})
    graph["geom1.1"] = {
        "meshes": [
            {
                **mesh,
                "skinWeights": [dict(weights) for weights in mesh["skinWeights"]],
                "blendShapeTargets": [dict(target) for target in mesh["blendShapeTargets"]],
            }
            for mesh in geometry["meshes"]
        ]
    }
    maps = (
        {domain: {value: index for index, value in enumerate(values)} for domain, values in maps.items()}
        if inverse
        else maps
    )

    def remap(node: dict, field: str, domain: str) -> None:
        node[field] = _mapped(node[field], maps[domain])

    definition = graph["defn1.1"]
    remap(definition, "jointHierarchy", "joint")
    remap(definition["meshBlendShapeChannelMapping"], "to", "blend")
    for key, domain in LOD_DOMAINS.items():
        for layer in (definition, graph["rbfb1.0"], graph["mlbh1.0"]):
            if key in layer and domain != "mesh":
                layer[key]["indices"] = [_mapped(row, maps[domain]) for row in layer[key]["indices"]]
    behavior = graph["bhvr1.1"]
    remap(behavior["controls"]["conditionals"], "inputIndices", "gui")
    remap(behavior["controls"]["conditionals"], "outputIndices", "control")
    for field in ("rows", "columns"):
        remap(behavior["controls"]["psds"], field, "control")
    for group in behavior["joints"]["jointGroups"]:
        for field, domain in (("inputIndices", "control"), ("outputIndices", "attribute"), ("jointIndices", "joint")):
            remap(group, field, domain)
    for table, domain in (
        (behavior["blendShapeChannels"], "blend"),
        (behavior["animatedMaps"]["conditionals"], "animated"),
    ):
        remap(table, "inputIndices", "control")
        remap(table, "outputIndices", domain)
    for mesh in graph["geom1.1"]["meshes"]:
        for weights in mesh["skinWeights"]:
            # SDK skin indices are already unsigned; avoid revalidating millions
            # of small arrays while applying this known component domain.
            weights["jointIndices"] = [maps["joint"][index] for index in weights["jointIndices"]]
        for target in mesh["blendShapeTargets"]:
            target["blendShapeChannelIndex"] = maps["blend"][target["blendShapeChannelIndex"]]
    for solver in graph["rbfb1.0"]["solvers"]:
        remap(solver, "rawControlIndices", "control")
        remap(solver, "poseIndices", "pose")
    for pose in graph["rbfe1.0"]["poses"]:
        for field in ("inputControlIndices", "outputControlIndices"):
            remap(pose, field, "control")
    for network in graph["mlbh1.0"]["neuralNetworks"]:
        for field in ("inputIndices", "outputIndices"):
            remap(network, field, "control")
    membership = graph["mlbh1.0"]["neuralNetworkToMeshRegion"]
    membership["indices"] = [[_mapped(row, maps["network"]) for row in mesh] for mesh in membership["indices"]]
    for label in ("twist", "swing"):
        for setup in graph["twsw1.0"][label + "s"]:
            remap(setup, label + "InputControlIndices", "control")
            remap(setup, label + "OutputJointIndices", "joint")
    return graph


def _lod_rows(mapping: dict, lods: list[int]) -> list:
    return [mapping["indices"][mapping["lods"][lod]] if mapping["lods"] else [] for lod in lods]


def _lod_map(rows: list) -> dict:
    return {"lods": list(range(len(rows))), "indices": rows}


def _join(head: Any, body: Any) -> Any:
    if isinstance(head, dict):
        return {key: _join(value, body[key]) for key, value in head.items()}
    if isinstance(head, list):
        return clone(head + body)
    if isinstance(head, (int, float)):
        return head + body
    if head != body:
        raise ValueError("Incompatible DNA layer descriptors.")
    return head


def _merge_table(parts: dict, lods: dict, *, animated: bool) -> tuple[dict, dict]:
    tables = {k: g["bhvr1.1"]["animatedMaps" if animated else "blendShapeChannels"] for k, g in parts.items()}
    rows = []
    for kind, table in tables.items():
        values = table["conditionals"] if animated else table
        for index in range(len(values["inputIndices"])):
            active = [
                lod for lod, source_lod in enumerate(lods[kind]) if table["lods"] and index < table["lods"][source_lod]
            ]
            rows.append((max(active, default=-1), kind, index))
    rows.sort(key=lambda row: -row[0])
    fields = list(
        tables["head"]["conditionals"]
        if animated
        else {key: value for key, value in tables["head"].items() if key != "lods"}
    )
    result = {
        field: [(tables[kind]["conditionals"] if animated else tables[kind])[field][index] for _, kind, index in rows]
        for field in fields
    }
    counts = [sum(last >= lod for last, _, _ in rows) for lod in range(len(lods["head"]))]
    ownership = {kind: [position for position, (_, owner, _) in enumerate(rows) if owner == kind] for kind in KINDS}
    return ({"lods": counts, "conditionals": result} if animated else {"lods": counts, **result}), ownership


def _weld(head: dict, body: dict, seam: dict) -> tuple[dict, dict]:
    count = len(body["positions"]["xs"])
    head_count = len(head["positions"]["xs"])
    if max(seam, default=-1) >= head_count or max(seam.values(), default=-1) >= count:
        raise ValueError("DNA surface topology does not match its LOD neckline mapping.")
    head_ids = []
    for index in range(head_count):
        if index in seam:
            head_ids.append(seam[index])
        else:
            head_ids.append(count)
            count += 1
    maps = {"head": head_ids, "body": list(range(len(body["positions"]["xs"])))}
    result = {
        **body,
        "positions": {axis: list(values) for axis, values in body["positions"].items()},
        "skinWeights": list(body["skinWeights"]),
    }
    for axis in ("xs", "ys", "zs"):
        result["positions"][axis] += [value for index, value in enumerate(head["positions"][axis]) if index not in seam]
    result["skinWeights"] += [weights for index, weights in enumerate(head["skinWeights"]) if index not in seam]
    result["maximumInfluencePerVertex"] = max(head["maximumInfluencePerVertex"], body["maximumInfluencePerVertex"])
    result["normals"] = _join(head["normals"], body["normals"])
    result["textureCoordinates"] = _join(head["textureCoordinates"], body["textureCoordinates"])
    result["layouts"] = {
        "positions": _mapped(head["layouts"]["positions"], maps["head"])
        + _mapped(body["layouts"]["positions"], maps["body"]),
        "textureCoordinates": head["layouts"]["textureCoordinates"]
        + [index + len(head["textureCoordinates"]["us"]) for index in body["layouts"]["textureCoordinates"]],
        "normals": head["layouts"]["normals"]
        + [index + len(head["normals"]["xs"]) for index in body["layouts"]["normals"]],
    }
    result["faces"] = clone(head["faces"]) + [
        {"layoutIndices": [index + len(head["layouts"]["positions"]) for index in face["layoutIndices"]]}
        for face in body["faces"]
    ]
    result["blendShapeTargets"] = []
    for kind, mesh in (("head", head), ("body", body)):
        for original in mesh["blendShapeTargets"]:
            target = clone(original)
            target["vertexIndices"] = _mapped(target["vertexIndices"], maps[kind])
            result["blendShapeTargets"].append(target)
    return result, maps


def compose(originals: dict, name: str, bridges: list | None = None) -> tuple[dict, dict]:  # noqa: PLR0912, PLR0915
    """Construct a native merged graph at every LOD and record its inverse domains."""
    for graph in originals.values():
        unexpected = set(graph) - {*LAYERS, "signature", "version", "index"}
        if unexpected:
            raise ValueError(f"Cannot remap unknown DNA layers: {', '.join(sorted(unexpected))}")
        extension = graph.get("mlbe1.0", {})
        if extension.get("mlbTypeData") or any(extension.get("mlbJoints", {}).values()):
            raise ValueError("This DNA uses ML operation extensions requiring a version-specific remapper.")
    head_lods = originals["head"]["desc1.1"]["lodCount"]
    body_lods = originals["body"]["desc1.1"]["lodCount"]
    lods = {"head": list(range(head_lods)), "body": [HEAD_TO_BODY_LOD_MAPPING[index] for index in range(head_lods)]}
    if set(lods["body"]) != set(range(body_lods)):
        raise ValueError("The RigDefinition LOD correspondence does not cover both components.")
    for field in ("translationUnit", "rotationUnit", "coordinateSystem"):
        if originals["head"]["desc1.1"][field] != originals["body"]["desc1.1"][field]:
            raise ValueError(f"DNA components have incompatible {field} conventions.")
    if originals["head"]["dsce1.0"] != originals["body"]["dsce1.0"]:
        raise ValueError("DNA components have incompatible rotation or winding conventions.")
    maps = index_maps(originals)
    parts = {kind: transform(graph, maps[kind]) for kind, graph in originals.items()}
    for kind, graph in parts.items():
        for group in graph["bhvr1.1"]["joints"]["jointGroups"]:
            group["lods"] = [group["lods"][lod] for lod in lods[kind]]
    result = clone({key: value if key != "geom1.1" else {} for key, value in parts["head"].items()})
    for layer in ("defn1.1", "bhvr1.1", "mlbh1.0", "rbfb1.0", "rbfe1.0", "jbmd1.0", "twsw1.0"):
        result[layer] = _join(parts["head"][layer], parts["body"][layer])
    result["desc1.1"]["name"] = name
    result["desc1.1"]["lodCount"] = head_lods
    result["bhvr1.1"]["joints"]["colCount"] = sum(len(maps[kind]["control"]) for kind in KINDS)
    info = {"maps": maps, "lods": lods, "surfaces": [], "tables": {}, "names": {}}
    for layer, field in [("defn1.1", field) for field in NAME_DOMAINS] + [
        ("mlbh1.0", "mlControlNames"),
        ("rbfe1.0", "poseControlNames"),
    ]:
        first = parts["head"][layer][field]
        second = parts["body"][layer][field]
        result[layer][field] = first + [f"DNA_body::{value}" if value in first else value for value in second]
        info["names"][layer + "/" + field] = result[layer][field]
    for layer, field in (("rbfb1.0", "solvers"), ("rbfb1.0", "poses")):
        first_names = {item["name"] for item in parts["head"][layer][field]}
        for item in result[layer][field][len(parts["head"][layer][field]) :]:
            if item["name"] in first_names:
                item["name"] = f"DNA_body::{item['name']}"
    for layer in ("defn1.1", "mlbh1.0", "rbfb1.0"):
        for field in LOD_DOMAINS:
            if field in result[layer]:
                rows = {kind: _lod_rows(parts[kind][layer][field], lods[kind]) for kind in KINDS}
                result[layer][field] = _lod_map([rows["head"][lod] + rows["body"][lod] for lod in range(head_lods)])
    for animated in (False, True):
        field = "animatedMaps" if animated else "blendShapeChannels"
        result["bhvr1.1"][field], info["tables"][field] = _merge_table(parts, lods, animated=animated)
    # Replace head surfaces in place. Each body surface can participate in two
    # head LODs; the inverse records both references to the same source mesh.
    names = {kind: parts[kind]["defn1.1"]["meshNames"] for kind in KINDS}
    meshes = list(parts["head"]["geom1.1"]["meshes"])
    merged_names = list(names["head"])
    mesh_maps = {"head": {str(index): [index] for index in range(len(names["head"]))}, "body": {}}
    necklines = json.loads(HEAD_TO_BODY_EDGE_LOOP_FILE_PATH.read_text())
    for lod in range(head_lods):
        try:
            head_index = names["head"].index(f"head_lod{lod}_mesh")
            body_index = names["body"].index(f"body_lod{lods['body'][lod]}_mesh")
        except ValueError as error:
            raise ValueError(f"Missing required head/body DNA surface for LOD{lod}.") from error
        seam = {int(key): value for key, value in necklines[str(lod)].items()}
        meshes[head_index], vertex_maps = _weld(
            parts["head"]["geom1.1"]["meshes"][head_index], parts["body"]["geom1.1"]["meshes"][body_index], seam
        )
        merged_names[head_index] = f"{name}_lod{lod}_mesh"
        mesh_maps["body"].setdefault(str(body_index), []).append(head_index)
        info["surfaces"].append(
            {
                "lod": lod,
                "head": head_index,
                "body": body_index,
                "mesh": head_index,
                "vertices": vertex_maps,
                "seam": list(seam),
            }
        )
    for index, mesh_name in enumerate(names["body"]):
        if str(index) not in mesh_maps["body"]:
            mesh_maps["body"][str(index)] = [len(meshes)]
            meshes.append(clone(parts["body"]["geom1.1"]["meshes"][index]))
            merged_names.append(f"DNA_body::{mesh_name}" if mesh_name in merged_names else mesh_name)
    info["meshes"] = mesh_maps
    result["geom1.1"] = {"meshes": meshes}
    result["defn1.1"]["meshNames"] = merged_names
    mesh_rows = []
    for lod in range(head_lods):
        row = []
        for kind in KINDS:
            source_indices = _lod_rows(parts[kind]["defn1.1"]["lodMeshMapping"], [lods[kind][lod]])[0]
            for source_index in source_indices:
                candidates = mesh_maps[kind][str(source_index)]
                row.extend(
                    index for index in candidates if len(candidates) == 1 or index == info["surfaces"][lod]["mesh"]
                )
        mesh_rows.append(list(dict.fromkeys(row)))
    result["defn1.1"]["lodMeshMapping"] = _lod_map(mesh_rows)
    mapping = {"from": [], "to": []}
    for index, mesh in enumerate(meshes):
        for target in mesh["blendShapeTargets"]:
            mapping["from"].append(index)
            mapping["to"].append(target["blendShapeChannelIndex"])
    if len(mapping["from"]) > 65535:
        raise ValueError("Merged mesh/channel mapping exceeds DNA uint16 capacity.")
    result["defn1.1"]["meshBlendShapeChannelMapping"] = mapping
    # Mesh regions are indexed by native mesh, independently of neural networks.
    membership = {"regionNames": [[] for _ in meshes], "indices": [[] for _ in meshes]}
    for kind in KINDS:
        source = parts[kind]["mlbh1.0"]["neuralNetworkToMeshRegion"]
        for index, regions in enumerate(source["regionNames"]):
            for destination in mesh_maps[kind][str(index)]:
                membership["regionNames"][destination].extend(regions)
                membership["indices"][destination].extend(source["indices"][index])
    result["mlbh1.0"]["neuralNetworkToMeshRegion"] = membership
    info["bridges"] = clone(bridges or [])
    _embed_bridges(result, originals, info)
    return result, info


def bridge_channel(originals: dict, bridge: dict) -> int:
    """Resolve a cross-component corrective's channel in its owning component."""
    definition = originals[bridge["source"]["kind"]]["defn1.1"]
    name = bridge["source"]["name"]
    for mesh_name in definition["meshNames"]:
        if name.startswith(mesh_name + "__"):
            name = name[len(mesh_name) + 2 :]
            break
    try:
        return definition["blendShapeChannelNames"].index(name)
    except ValueError as error:
        raise ValueError(f"Corrective has no owning DNA channel: {name}") from error


def bridge_delta(originals: dict, bridge: dict) -> np.ndarray:
    """Convert scene corrective coordinates into native DNA-space deltas."""
    graph = originals[bridge["kind"]]
    basis = np.asarray(list(graph["geom1.1"]["meshes"][bridge["mesh"]]["positions"].values())).T
    values = np.frombuffer(
        base64.b64decode(bridge.get("deltas", bridge["coordinates"]), validate=True), dtype="<f4"
    ).reshape(-1, 3)
    if values.shape != basis.shape or not np.isfinite(values).all():
        raise ValueError("Invalid cross-component corrective coordinates.")
    scale = 100.0 if graph["desc1.1"]["translationUnit"] == 0 else 1.0
    converted = values[:, [0, 2, 1]] * np.array([scale, scale, -scale])
    return converted if "deltas" in bridge else converted - basis


def _embed_bridges(graph: dict, originals: dict, info: dict) -> None:
    """Publish cross-component deltas as real native targets on every affected LOD."""
    for bridge in info["bridges"]:
        kind = bridge["kind"]
        channel = info["maps"][bridge["source"]["kind"]]["blend"][bridge_channel(originals, bridge)]
        delta = bridge_delta(originals, bridge)
        for destination in info["meshes"][kind][str(bridge["mesh"])]:
            record = next((surface for surface in info["surfaces"] if surface["mesh"] == destination), None)
            ids = record["vertices"][kind] if record else list(range(len(delta)))
            other = set(record["vertices"]["head" if kind == "body" else "body"]) if record else set()
            mesh = graph["geom1.1"]["meshes"][destination]
            targets = list(mesh["blendShapeTargets"])
            graph["geom1.1"]["meshes"][destination] = {**mesh, "blendShapeTargets": targets}
            target = next((item for item in targets if item["blendShapeChannelIndex"] == channel), None)
            if target is not None:
                index = targets.index(target)
                target = clone(target)
                targets[index] = target
            if target is None:
                target = {
                    "deltas": {axis: [] for axis in ("xs", "ys", "zs")},
                    "vertexIndices": [],
                    "blendShapeChannelIndex": channel,
                }
                targets.append(target)
            for vertex, native_vertex in enumerate(ids):
                if native_vertex in other or not np.any(np.abs(delta[vertex]) > 1e-5):
                    continue
                if native_vertex in target["vertexIndices"]:
                    raise ValueError("Corrective extension overlaps an existing native target owner.")
                target["vertexIndices"].append(native_vertex)
                for axis, value in zip(target["deltas"], delta[vertex], strict=True):
                    target["deltas"][axis].append(float(value))
    mapping = graph["defn1.1"]["meshBlendShapeChannelMapping"]
    mapping["from"], mapping["to"] = [], []
    for index, mesh in enumerate(graph["geom1.1"]["meshes"]):
        for target in mesh["blendShapeTargets"]:
            mapping["from"].append(index)
            mapping["to"].append(target["blendShapeChannelIndex"])
