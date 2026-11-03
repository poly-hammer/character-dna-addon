"""Versioned component-owned colors and neck boundaries in DNA metadata.

Colors address DNA positions or face corners, never UV layout IDs. A merged
document carries explicit bindings back to each component's original topology.
"""

import base64
import hashlib
import json
import logging

from collections import Counter
from typing import Any

import numpy as np

from . import metadata


SCHEMA = 1
PREFIX = "character_dna.mesh_data"
TYPE_PREFIX = "character_dna.topology"
SEAM_GROUP = "SEAM_neckline"
KINDS = ("head", "body")
logger = logging.getLogger(__name__)


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
    graph["desc1.1"]["metadata"] = list(values.items())


def pack(values: Any, dtype: str) -> dict:
    array = np.asarray(values, dtype=dtype)
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
    raw = base64.b64decode(record["data"], validate=True)
    if len(raw) != count * dtype.itemsize:
        raise ValueError("Truncated mesh metadata array.")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite mesh metadata value.")
    return array


def topology(vertex_count: int, faces: list) -> dict:
    sizes = [len(face) for face in faces]
    digest = hashlib.sha256(np.asarray([vertex_count, len(faces), *sizes], dtype="<u4").tobytes())
    digest.update(np.asarray([v for face in faces for v in face], dtype="<u4").tobytes())
    return {"vertices": vertex_count, "faces": len(faces), "corners": sum(sizes), "sha256": digest.hexdigest()}


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
            values.shape != (len(indices), 4)
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


def ordered_boundary(faces: list, vertices: set[int]) -> list[int]:
    """Order exactly one closed boundary, rejecting branches and disconnected rings."""
    edges = Counter(tuple(sorted((a, b))) for face in faces for a, b in zip(face, [*face[1:], face[0]], strict=True))
    neighbors = {v: [] for v in vertices}
    for (a, b), count in edges.items():
        if count == 1 and a in vertices and b in vertices:
            neighbors[a].append(b)
            neighbors[b].append(a)
    if len(vertices) < 3 or any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError(f"{SEAM_GROUP} must select one complete closed mesh boundary.")
    ordered = [min(vertices)]
    previous, current = ordered[0], min(neighbors[ordered[0]])
    while current != ordered[0]:
        if current in ordered:
            raise ValueError(f"{SEAM_GROUP} contains a disconnected boundary.")
        ordered.append(current)
        following = next(v for v in neighbors[current] if v != previous)
        previous, current = current, following
    if len(ordered) != len(vertices):
        raise ValueError(f"{SEAM_GROUP} contains more than one boundary.")
    return ordered


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
    record = {
        "mesh_index": mesh_index,
        "name": name,
        "lods": lods,
        "topology": topology(vertex_count, faces),
        "colors": [],
        "neckline": object_boundary(obj),
    }
    if len(faces) != len(mesh.polygons):
        raise ValueError("Mesh metadata requires the exported face correspondence.")
    if colors:
        corners = []
        for poly, face in zip(mesh.polygons, faces, strict=True):
            by_vertex = {mesh.loops[i].vertex_index: i for i in poly.loop_indices}
            corners.extend(by_vertex[v] for v in face)
        for attr in mesh.color_attributes:
            byte = attr.data_type == "BYTE_COLOR"
            values = np.empty(len(attr.data) * 4, dtype=np.float32)
            attr.data.foreach_get("color_srgb" if byte else "color", values)
            values = values.reshape(-1, 4)
            indices = (
                np.arange(len(mesh.vertices), dtype=np.uint32)
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
    """Match ring rotation and direction, never a many-to-one nearest projection."""
    if len(head) != len(body) or len(head) < 3:
        raise ValueError("Head and body necklines must have the same vertex count to weld.")
    hp, bp = np.asarray(head_points)[head], np.asarray(body_points)[body]
    candidates = []
    for reverse in (False, True):
        ring = list(reversed(body)) if reverse else body
        points = bp[::-1] if reverse else bp
        for offset in range(len(body)):
            distances = np.linalg.norm(hp - np.roll(points, -offset, axis=0), axis=1)
            candidates.append((float(np.mean(distances**2)), float(np.max(distances)), ring[offset:] + ring[:offset]))
    candidates.sort(key=lambda row: row[0])
    if candidates[0][1] > tolerance:
        raise ValueError("Head/body neckline separation exceeds the matching tolerance; align the rest meshes.")
    if abs(candidates[1][0] - candidates[0][0]) < max(tolerance**2 * 1e-8, 1e-16):
        raise ValueError("Ambiguous neckline correspondence; inspect the boundary vertices.")
    return dict(zip(head, candidates[0][2], strict=True))


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
                for color in record["colors"]:
                    old = unpack(color["indices"])
                    color["indices"] = pack(
                        old + offset if color["domain"] == "CORNER" else np.asarray(vertices)[old], "<u4"
                    )
                if record["neckline"] is not None:
                    record["neckline"] = [vertices[v] for v in record["neckline"]]
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
            for color in record["colors"]:
                old = unpack(color["indices"])
                color["indices"] = pack(
                    old - binding["corner_offset"] if color["domain"] == "CORNER" else [inverse[int(v)] for v in old],
                    "<u4",
                )
            if record["neckline"] is not None:
                record["neckline"] = [inverse[v] for v in record["neckline"]]
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
