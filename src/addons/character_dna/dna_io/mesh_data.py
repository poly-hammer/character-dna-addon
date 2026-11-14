"""Versioned component-owned UVs, colors, boundaries, and converter masks in DNA metadata.

Colors address DNA positions or face corners, never UV layout IDs. A merged
document carries explicit bindings back to each component's original topology.
"""

import base64
import json
import logging

from typing import Any

import numpy as np

from ..constants import UV_MAP_NAME
from . import metadata


SCHEMA = 1
PREFIX = "character_dna.mesh_data"
TYPE_PREFIX = "character_dna.topology"
SEAM_GROUP = "SEAM_neckline"
EXCLUSION_PREFIX = "EXCLUDE_"
PRESERVATION_PREFIX = "PRESERVE_"
KINDS = ("head", "body")
logger = logging.getLogger(__name__)


def is_exclusion_group(name: str) -> bool:
    """Identify converter-only masks, never armature deformation groups."""
    return name.upper().startswith(EXCLUSION_PREFIX)


def is_preservation_group(name: str) -> bool:
    """Identify weighted converter shape-preservation masks."""
    return name.upper().startswith(PRESERVATION_PREFIX)


def type_key(kind: str) -> str:
    return f"{TYPE_PREFIX}.{kind}.type"


def type_name(reader: Any, kind: str) -> str:
    return reader.getMetaDataValue(type_key(kind)) if reader else ""


class ValuesReader:
    """Read the same metadata codec from an in-memory graph descriptor."""

    def __init__(self, values: dict[str, str]):
        self.values = values

    def getMetaDataValue(self, key: str) -> str:
        return self.values.get(key, "")


def read(reader: Any, kind: str) -> dict | None:
    prefix = f"{PREFIX}.{kind}"
    if not reader.getMetaDataValue(prefix):
        return None
    payload = metadata.decode(reader, prefix=prefix)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("component") != kind
        or not isinstance(payload.get("meshes"), list)
    ):
        raise ValueError(f"Unsupported or invalid {kind} mesh metadata.")
    return payload


def graph_reader(graph: dict) -> ValuesReader:
    return ValuesReader(dict(graph["desc1.1"]["metadata"]))


def set_graph_payload(graph: dict, kind: str, payload: dict, name: str) -> None:
    prefix = f"{PREFIX}.{kind}"
    values = {k: v for k, v in graph["desc1.1"]["metadata"] if k != prefix and not k.startswith(prefix + ".")}
    values.update(metadata.encode(payload, prefix=prefix))
    values.update({f"{TYPE_PREFIX}.schema": str(SCHEMA), type_key(kind): name})
    metadata.validate_fields(values)
    graph["desc1.1"]["metadata"] = list(values.items())


def pack(values: Any, dtype: str) -> dict:
    if dtype not in ("|u1", "<f4", "<u4"):
        raise ValueError("Unsupported mesh metadata array type.")
    source = np.asarray(values)
    target = np.dtype(dtype)
    if source.size * target.itemsize > metadata.MAX_PAYLOAD_BYTES:
        raise ValueError("Mesh metadata array exceeds the memory budget.")
    if not np.isfinite(source).all():
        raise ValueError("Non-finite mesh metadata value.")
    if target.kind == "u" and (
        np.any(source < 0) or np.any(source > np.iinfo(target).max) or np.any(source != np.floor(source))
    ):
        raise ValueError(f"Mesh metadata value is outside the {dtype} integer range.")
    with np.errstate(over="ignore"):
        array = source.astype(target, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Mesh metadata value exceeds float32 range.")
    return {"dtype": dtype, "shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}


def unpack(record: dict) -> np.ndarray:
    if record["dtype"] not in ("|u1", "<f4", "<u4"):
        raise ValueError("Unsupported mesh metadata array type.")
    shape = record["shape"]
    if not isinstance(shape, list) or len(shape) > 2 or any(not isinstance(n, int) or n < 0 for n in shape):
        raise ValueError("Invalid mesh metadata array dimensions.")
    count = 1
    for size in shape:
        count *= size
    dtype = np.dtype(record["dtype"])
    if count * dtype.itemsize > metadata.MAX_PAYLOAD_BYTES:
        raise ValueError("Mesh metadata array exceeds the memory budget.")
    expected = count * dtype.itemsize
    if len(record["data"]) != 4 * ((expected + 2) // 3):
        raise ValueError("Truncated or oversized mesh metadata array.")
    raw = base64.b64decode(record["data"], validate=True)
    if len(raw) != count * dtype.itemsize:
        raise ValueError("Truncated mesh metadata array.")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite mesh metadata value.")
    return array


def topology(vertex_count: int, faces: list) -> dict:
    from ..bindings import load_converter_core

    return json.loads(load_converter_core().scene_topology_fingerprint(vertex_count, faces))


def reader_faces(reader: Any, index: int) -> list[list[int]]:
    positions = reader.getVertexLayoutPositionIndices(index)
    return [
        [int(positions[v]) for v in reader.getFaceVertexLayoutIndices(index, f)]
        for f in range(reader.getFaceCount(index))
    ]


def graph_faces(mesh: dict) -> list[list[int]]:
    positions = mesh["layouts"]["positions"]
    return [[positions[v] for v in face["layoutIndices"]] for face in mesh["faces"]]


def validate_record(record: dict, fingerprint: dict) -> None:
    if record["topology"] != fingerprint:
        raise ValueError(f"Mesh metadata topology mismatch for {record['name']}; re-export this mesh.")
    _validate_uv_layers(record, fingerprint)
    for color in record["colors"]:
        values = unpack(color["values"])
        indices = unpack(color["indices"])
        domain = color["domain"]
        if domain not in ("POINT", "CORNER") or color["data_type"] not in ("BYTE_COLOR", "FLOAT_COLOR"):
            raise ValueError("Unsupported color attribute domain or storage type.")
        dtype, space = ("|u1", "srgb") if color["data_type"] == "BYTE_COLOR" else ("<f4", "linear")
        count = fingerprint["vertices" if domain == "POINT" else "corners"]
        if color["values"]["dtype"] != dtype or color["color_space"] != space:
            raise ValueError("Invalid color attribute encoding.")
        if (
            color["indices"]["dtype"] != "<u4"
            or values.shape != (len(indices), 4)
            or indices.ndim != 1
            or np.any(indices >= count)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("Invalid color attribute indices.")
    seam = record.get("neckline")
    if seam is not None and (
        len(seam) < 3 or len(set(seam)) != len(seam) or any(not 0 <= v < fingerprint["vertices"] for v in seam)
    ):
        raise ValueError("Invalid neckline vertex indices.")
    exclusions = record.get("exclusions", {})
    if not isinstance(exclusions, dict):
        raise ValueError("Invalid converter exclusion groups.")
    for name, vertices in exclusions.items():
        if (
            not isinstance(name, str)
            or not is_exclusion_group(name)
            or not isinstance(vertices, list)
            or any(type(v) is not int or not 0 <= v < fingerprint["vertices"] for v in vertices)
            or len(set(vertices)) != len(vertices)
        ):
            raise ValueError(f"Invalid converter exclusion group: {name}")
    preservations = record.get("preservations", {})
    if not isinstance(preservations, dict):
        raise ValueError("Invalid converter preservation groups.")
    for name, mask in preservations.items():
        if not isinstance(name, str) or not is_preservation_group(name) or not isinstance(mask, dict):
            raise ValueError(f"Invalid converter preservation group: {name}")
        vertices, weights = mask.get("vertices"), mask.get("weights")
        if (
            not isinstance(vertices, list)
            or not isinstance(weights, list)
            or len(vertices) != len(weights)
            or any(type(v) is not int or not 0 <= v < fingerprint["vertices"] for v in vertices)
            or len(set(vertices)) != len(vertices)
            or any(type(w) not in (int, float) or not 0 <= w <= 1 for w in weights)
        ):
            raise ValueError(f"Invalid converter preservation group: {name}")


def _validate_uv_layers(record: dict, fingerprint: dict) -> None:
    layers = record.get("uv_layers")
    if layers is None:
        return
    if not isinstance(layers, list) or not layers or any(not isinstance(layer, dict) for layer in layers):
        raise ValueError("Invalid UV layer metadata.")
    names = [layer.get("name") for layer in layers]
    if any(not isinstance(name, str) or not name or "\0" in name for name in names) or len(set(names)) != len(names):
        raise ValueError("UV layer names must be nonempty and unique.")
    for flag in ("primary", "active", "render"):
        if any(type(layer.get(flag)) is not bool for layer in layers) or sum(layer[flag] for layer in layers) != 1:
            raise ValueError(f"UV layers require exactly one {flag} layer.")
    count = record.get("binding", {}).get("topology", fingerprint)["corners"]
    for layer in layers:
        if layer["primary"]:
            if "indices" in layer or "values" in layer:
                raise ValueError("Primary UV coordinates belong in the native DNA stream.")
            continue
        indices, values = unpack(layer["indices"]), unpack(layer["values"])
        if (
            layer["indices"]["dtype"] != "<u4"
            or layer["values"]["dtype"] != "<f4"
            or indices.shape != (count,)
            or values.shape != (count, 2)
            or len(np.unique(indices)) != count
            or np.any(indices >= fingerprint["corners"])
        ):
            raise ValueError("Invalid UV layer corner indices or coordinates.")


def primary_uv_layer(mesh: Any) -> Any:
    """Keep the canonical diffuse map stable while artists select other UV layers."""
    return (
        mesh.uv_layers.get(UV_MAP_NAME)
        or next((layer for layer in mesh.uv_layers if layer.active_render), None)
        or mesh.uv_layers.active
    )


def corner_order(mesh: Any, faces: list, position_ids: list[int]) -> list[int]:
    """Map DNA face-corner order to Blender loops, including sparse DNA positions."""
    corners = []
    for poly, face in zip(mesh.polygons, faces, strict=True):
        by_vertex = {position_ids[mesh.loops[i].vertex_index]: i for i in poly.loop_indices}
        if len(face) != len(poly.loop_indices) or set(face) != set(by_vertex):
            raise ValueError("Mesh metadata requires matching face-corner topology.")
        corners.extend(by_vertex[v] for v in face)
    return corners


def _capture_uv_layers(mesh: Any, corners: list[int]) -> list[dict]:
    primary = primary_uv_layer(mesh)
    layers = []
    for layer in mesh.uv_layers:
        record = {
            "name": layer.name,
            "primary": layer == primary,
            "active": layer == mesh.uv_layers.active,
            "render": layer.active_render,
        }
        if layer != primary:
            values = np.empty(len(mesh.loops) * 2, dtype=np.float32)
            layer.data.foreach_get("uv", values)
            record.update(
                indices=pack(np.arange(len(corners)), "<u4"), values=pack(values.reshape(-1, 2)[corners], "<f4")
            )
        layers.append(record)
    return layers


def apply_uv_layers(mesh: Any, record: dict, corners: list[int]) -> None:
    """Restore named extra UVs and layer selections around the native primary map."""
    layers = record.get("uv_layers")
    if layers is None:
        return
    primary = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", primary)
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    for layer in layers:
        uv = mesh.uv_layers.new(name=layer["name"], do_init=False)
        if uv is None or uv.name != layer["name"]:
            raise ValueError(f"Blender could not restore UV layer: {layer['name']}")
        values = primary.copy()
        if not layer["primary"]:
            values = values.reshape(-1, 2)
            values[np.asarray(corners)[unpack(layer["indices"])]] = unpack(layer["values"])
        uv.data.foreach_set("uv", values.reshape(-1))
    for layer in layers:
        if layer["active"]:
            mesh.uv_layers.active = mesh.uv_layers[layer["name"]]
        if layer["render"]:
            mesh.uv_layers[layer["name"]].active_render = True


def ordered_boundary(faces: list, vertices: set[int]) -> list[int]:
    """Delegate captured data to the native converter adapter."""
    from .native_mesh_data import ordered_boundary as native

    return native(faces, vertices)


def object_boundary(obj: Any) -> list[int] | None:
    group = obj.vertex_groups.get(SEAM_GROUP)
    if group is None:
        return None
    vertices = {v.index for v in obj.data.vertices if any(g.group == group.index and g.weight > 0 for g in v.groups)}
    return ordered_boundary([list(p.vertices) for p in obj.data.polygons], vertices)


def capture(
    obj: Any, mesh_index: int, name: str, vertex_count: int, faces: list, lods: list[int], *, colors: bool = True
) -> dict:
    """Bind Blender data to final DNA positions and corners after UV splitting."""
    mesh = obj.data
    used = sorted({v for face in faces for v in face})
    position_ids = used if len(used) == len(mesh.vertices) else list(range(len(mesh.vertices)))
    neckline = object_boundary(obj)
    record = {
        "mesh_index": mesh_index,
        "name": name,
        "lods": lods,
        "topology": topology(vertex_count, faces),
        "colors": [],
        "neckline": [position_ids[v] for v in neckline] if neckline is not None else None,
    }
    if len(faces) != len(mesh.polygons):
        raise ValueError("Mesh metadata requires the exported face correspondence.")
    corners = corner_order(mesh, faces, position_ids)
    if mesh.uv_layers:
        record["uv_layers"] = _capture_uv_layers(mesh, corners)
    groups = {group.index: group.name for group in obj.vertex_groups if is_exclusion_group(group.name)}
    if groups:
        record["exclusions"] = {name: [] for name in groups.values()}
        for vertex in mesh.vertices:
            for assignment in vertex.groups:
                if assignment.group in groups and assignment.weight > 0:
                    record["exclusions"][groups[assignment.group]].append(position_ids[vertex.index])
    preserve_groups = {group.index: group.name for group in obj.vertex_groups if is_preservation_group(group.name)}
    if preserve_groups:
        record["preservations"] = {name: {"vertices": [], "weights": []} for name in preserve_groups.values()}
        for vertex in mesh.vertices:
            for assignment in vertex.groups:
                if assignment.group in preserve_groups and assignment.weight > 0:
                    mask = record["preservations"][preserve_groups[assignment.group]]
                    mask["vertices"].append(position_ids[vertex.index])
                    mask["weights"].append(float(assignment.weight))
    if colors:
        for attr in mesh.color_attributes:
            byte = attr.data_type == "BYTE_COLOR"
            values = np.empty(len(attr.data) * 4, dtype=np.float32)
            attr.data.foreach_get("color_srgb" if byte else "color", values)
            values = values.reshape(-1, 4)
            indices = (
                np.asarray(position_ids, dtype=np.uint32)
                if attr.domain == "POINT"
                else np.arange(len(corners), dtype=np.uint32)
            )
            if attr.domain == "CORNER":
                values = values[corners]
            record["colors"].append(
                {
                    "name": attr.name,
                    "domain": attr.domain,
                    "data_type": attr.data_type,
                    "color_space": "srgb" if byte else "linear",
                    "active": attr == mesh.color_attributes.active_color,
                    "render": mesh.color_attributes.find(attr.name) == mesh.color_attributes.render_color_index,
                    "indices": pack(indices, "<u4"),
                    "values": pack(
                        np.rint(values * 255).astype(np.uint8) if byte else values, "|u1" if byte else "<f4"
                    ),
                }
            )
    validate_record(record, record["topology"])
    return record


def apply_colors(mesh: Any, record: dict, position_ids: list[int] | None = None) -> None:
    """Restore names, domains, byte precision, and active/render layer choices."""
    position_ids = position_ids if position_ids is not None else list(range(len(mesh.vertices)))
    for color in record["colors"]:
        old = mesh.color_attributes.get(color["name"])
        if old is not None and (old.domain != color["domain"] or old.data_type != color["data_type"]):
            mesh.color_attributes.remove(old)
            old = None
        attr = old or mesh.color_attributes.new(name=color["name"], type=color["data_type"], domain=color["domain"])
        indices, source = unpack(color["indices"]), unpack(color["values"])
        byte = color["data_type"] == "BYTE_COLOR"
        values = np.ones((len(attr.data), 4), dtype=np.float32)
        if color["domain"] == "POINT":
            inverse = {dna: local for local, dna in enumerate(position_ids)}
            pairs = [(inverse[int(dna)], row) for row, dna in enumerate(indices) if int(dna) in inverse]
            if pairs:
                dst, src = zip(*pairs, strict=True)
                values[list(dst)] = source[list(src)] / (255 if byte else 1)
        else:
            values[indices] = source / (255 if byte else 1)
        attr.data.foreach_set("color_srgb" if byte else "color", values.reshape(-1))
        if color.get("active"):
            mesh.color_attributes.active_color = attr
        if color.get("render"):
            mesh.color_attributes.render_color_index = mesh.color_attributes.find(attr.name)


def match_boundaries(
    head: list[int], body: list[int], head_points: Any, body_points: Any, *, tolerance: float
) -> dict[int, int]:
    """Delegate captured data to the native converter adapter."""
    from .native_mesh_data import match_boundaries as native

    return native(head, body, head_points, body_points, tolerance=tolerance)


def record_for(reader: Any, kind: str, mesh_index: int) -> dict | None:
    payload = read(reader, kind)
    if payload is None:
        return None
    if payload.get("space", "component") != "component":
        raise ValueError("Project the merged DNA component before importing its mesh metadata.")
    return next((record for record in payload["meshes"] if record["mesh_index"] == mesh_index), None)


def has_custom_topology(reader: Any, kind: str) -> bool:
    return bool(type_name(reader, kind) and type_name(reader, kind) != "MetaHuman")


def reader_seam(readers: dict, lod: int) -> dict[int, int]:
    """Validate and match embedded necklines from two component readers."""
    rings, points = {}, {}
    for kind, reader in readers.items():
        name = f"{kind}_lod{lod}_mesh"
        index = next((i for i in range(reader.getMeshCount()) if reader.getMeshName(i) == name), None)
        if index is None:
            raise ValueError(f"Missing {kind} LOD{lod} for neckline alignment.")
        record = record_for(reader, kind, index)
        if record is None or record.get("neckline") is None:
            raise ValueError(f"Missing {SEAM_GROUP} metadata on {name}.")
        faces = reader_faces(reader, index)
        validate_record(record, topology(reader.getVertexPositionCount(index), faces))
        rings[kind] = ordered_boundary(faces, set(record["neckline"]))
        points[kind] = np.asarray([getattr(reader, f"getVertexPosition{axis}s")(index) for axis in "XYZ"]).T
    tolerance = 0.1 if readers["head"].getTranslationUnit() == 0 else 0.001
    return match_boundaries(rings["head"], rings["body"], points["head"], points["body"], tolerance=tolerance)


def pair_lods(head_lods: list[int], body_lods: list[int], *, custom: bool) -> dict[int, int]:
    from ..constants import HEAD_TO_BODY_LOD_MAPPING

    pairs = {lod: lod if custom else HEAD_TO_BODY_LOD_MAPPING.get(lod, lod) for lod in head_lods}
    missing = [lod for lod, body in pairs.items() if body not in body_lods]
    if missing:
        raise ValueError(f"Missing body surface LODs for head LODs {missing}.")
    if set(body_lods) - set(pairs.values()):
        raise ValueError("Body surface LODs have no corresponding head surfaces.")
    return pairs


def graph_seam(
    graphs: dict, head_index: int, body_index: int, lod: int, *, default_necklines: Any = None
) -> dict[int, int]:
    from ..constants import HEAD_TO_BODY_EDGE_LOOP_FILE_PATH

    readers = {kind: graph_reader(graph) for kind, graph in graphs.items()}
    records = {
        kind: record_for(readers[kind], kind, index) for kind, index in (("head", head_index), ("body", body_index))
    }
    custom = any(has_custom_topology(readers[kind], kind) for kind in KINDS)
    if custom or any(record and record.get("neckline") is not None for record in records.values()):
        points = {}
        rings = {}
        for kind, index in (("head", head_index), ("body", body_index)):
            record = records[kind]
            if record is None or record.get("neckline") is None:
                raise ValueError(f"Missing {SEAM_GROUP} metadata on {kind} LOD{lod}.")
            mesh = graphs[kind]["geom1.1"]["meshes"][index]
            faces = graph_faces(mesh)
            validate_record(record, topology(len(mesh["positions"]["xs"]), faces))
            rings[kind] = ordered_boundary(faces, set(record["neckline"]))
            points[kind] = np.asarray([mesh["positions"][axis] for axis in ("xs", "ys", "zs")]).T
        # Native graph positions use the declared DNA units (cm or m).
        tolerance = 0.1 if graphs["head"]["desc1.1"]["translationUnit"] == 0 else 0.001
        return match_boundaries(rings["head"], rings["body"], points["head"], points["body"], tolerance=tolerance)
    return {
        int(h): int(b)
        for h, b in json.loads((default_necklines or HEAD_TO_BODY_EDGE_LOOP_FILE_PATH).read_text())[str(lod)].items()
    }


def compose(originals: dict, merged: dict, indices: dict) -> None:
    """Remap each component's attributes into native merged position/corner IDs."""
    for kind in KINDS:
        reader = graph_reader(originals[kind])
        payload = read(reader, kind)
        if payload is None:
            continue
        records = []
        for source in payload["meshes"]:
            source_index = source["mesh_index"]
            for destination in indices["meshes"][kind][str(source_index)]:
                record = json.loads(json.dumps(source))
                surface = next((s for s in indices["surfaces"] if s["mesh"] == destination), None)
                offset = 0
                vertices = list(range(source["topology"]["vertices"]))
                if surface:
                    vertices = surface["vertices"][kind]
                    if kind == "body":
                        head = originals["head"]["geom1.1"]["meshes"][surface["head"]]
                        offset = sum(len(face["layoutIndices"]) for face in head["faces"])
                record["binding"] = {key: source[key] for key in ("mesh_index", "name", "lods", "topology")}
                record["binding"]["corner_offset"] = offset
                record["mesh_index"] = destination
                record["name"] = merged["defn1.1"]["meshNames"][destination]
                mesh = merged["geom1.1"]["meshes"][destination]
                record["topology"] = topology(len(mesh["positions"]["xs"]), graph_faces(mesh))
                for layer in record.get("uv_layers", []):
                    if not layer["primary"]:
                        layer["indices"] = pack(unpack(layer["indices"]) + offset, "<u4")
                for color in record["colors"]:
                    old = unpack(color["indices"])
                    color["indices"] = pack(
                        old + offset if color["domain"] == "CORNER" else np.asarray(vertices)[old], "<u4"
                    )
                if record["neckline"] is not None:
                    record["neckline"] = [vertices[v] for v in record["neckline"]]
                if "exclusions" in record:
                    record["exclusions"] = {
                        name: [vertices[v] for v in mask] for name, mask in record["exclusions"].items()
                    }
                for mask in record.get("preservations", {}).values():
                    mask["vertices"] = [vertices[v] for v in mask["vertices"]]
                records.append(record)
        set_graph_payload(
            merged,
            kind,
            {"schema": SCHEMA, "component": kind, "space": "merged", "meshes": records},
            type_name(reader, kind),
        )


def project(native: dict, parts: dict, indices: dict) -> None:
    """Resolve current metadata, not the descriptor snapshot taken before Merge."""
    for kind, graph in parts.items():
        reader = graph_reader(native)
        payload = read(reader, kind)
        if payload is None:
            continue
        records = {}
        for merged_record in payload["meshes"]:
            record = json.loads(json.dumps(merged_record))
            binding = record.pop("binding")
            surface = next((s for s in indices["surfaces"] if s["mesh"] == record["mesh_index"]), None)
            vertices = surface["vertices"][kind] if surface else list(range(binding["topology"]["vertices"]))
            inverse = {v: i for i, v in enumerate(vertices)}
            for layer in record.get("uv_layers", []):
                if not layer["primary"]:
                    layer["indices"] = pack(unpack(layer["indices"]).astype(np.int64) - binding["corner_offset"], "<u4")
            for color in record["colors"]:
                old = unpack(color["indices"])
                color["indices"] = pack(
                    old - binding["corner_offset"] if color["domain"] == "CORNER" else [inverse[int(v)] for v in old],
                    "<u4",
                )
            if record["neckline"] is not None:
                record["neckline"] = [inverse[v] for v in record["neckline"]]
            if "exclusions" in record:
                record["exclusions"] = {name: [inverse[v] for v in mask] for name, mask in record["exclusions"].items()}
            for mask in record.get("preservations", {}).values():
                mask["vertices"] = [inverse[v] for v in mask["vertices"]]
            record.update({key: binding[key] for key in ("mesh_index", "name", "lods", "topology")})
            validate_record(record, record["topology"])
            index = record["mesh_index"]
            if index in records and records[index] != record:
                raise ValueError(f"Conflicting {kind} colors or seam edits across merged LODs.")
            records[index] = record
        set_graph_payload(
            graph,
            kind,
            {"schema": SCHEMA, "component": kind, "space": "component", "meshes": list(records.values())},
            type_name(reader, kind),
        )
