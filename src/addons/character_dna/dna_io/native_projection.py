"""Project bulk geometry in the SDK; Python resolves compact behavior recipes."""

import tempfile

from contextlib import ExitStack
from pathlib import Path

from ..bindings import load_native_runtime
from . import merged_graph, partitions
from .document_edits import _reader
from .misc import release_dna_handle


def plans(graphs: dict, indices: dict, records: dict) -> list[dict]:
    """Describe array slices and ownership, never copy bulk geometry into Python."""
    surfaces = {item["mesh"]: item for item in indices["surfaces"]}
    result = []
    for kind, graph in graphs.items():
        meshes = []
        for index, item in enumerate(records[kind]["meshes"]):
            sizes = item["sizes"]
            variants = []
            fields = ("normals", "uv", "layouts", "faces")
            for destination in indices["meshes"][kind][str(index)]:
                surface = surfaces.get(destination)
                head = records["head"]["meshes"][surface["head"]]["sizes"] if surface else sizes
                variants.append(
                    {
                        "mesh": destination,
                        "vertices": surface["vertices"][kind] if surface else list(range(sizes["vertices"])),
                        "starts": [head[field] if surface and kind == "body" else 0 for field in fields],
                        "counts": [sizes[field] for field in fields],
                    }
                )
            meshes.append(
                {
                    "variants": variants,
                    "skin_count": sizes["skinWeights"],
                    "influences": item["influences"],
                    "channels": item["channels"],
                    "seams": [{"vertex": int(vertex), **residual} for vertex, residual in item.get("seam", {}).items()],
                }
            )
        result.append(
            {
                "definition": merged_graph.to_bytes(graph),
                "joints": indices["maps"][kind]["joint"],
                "channels": indices["maps"][kind]["blend"],
                "meshes": meshes,
            }
        )
    return result


def _bridges(data: bytes, parts: dict[str, bytes], graphs: dict, records: list, indices: dict) -> list:
    """Read only the arrays needed by cross-component shapes, not entire meshes."""
    if not records:
        return []
    with tempfile.TemporaryDirectory(prefix="dna-bridge-view-") as directory, ExitStack() as handles:
        root = Path(directory)
        readers = {}
        for kind, content in {"native": data, **parts}.items():
            path = root / f"{kind}.dna"
            path.write_bytes(content)
            readers[kind] = _reader(path)
            handles.callback(release_dna_handle, readers[kind])
        native = {"geom1.1": {"meshes": {}}}
        for graph in graphs.values():
            graph["geom1.1"] = {"meshes": {}}
        source = readers["native"]
        for record in records:
            kind, index = record["kind"], record["mesh"]
            destination = indices["meshes"][kind][str(index)][0]
            graphs[kind]["geom1.1"]["meshes"][index] = {
                "positions": {
                    axis.lower(): list(getattr(readers[kind], "getVertexPosition" + axis)(index))
                    for axis in ("Xs", "Ys", "Zs")
                }
            }
            target = next(
                i
                for i in range(source.getBlendShapeTargetCount(destination))
                if source.getBlendShapeChannelIndex(destination, i) == record["channel"]
            )
            mesh = native["geom1.1"]["meshes"].setdefault(destination, {"blendShapeTargets": []})
            mesh["blendShapeTargets"].append(
                {
                    "blendShapeChannelIndex": record["channel"],
                    "vertexIndices": list(source.getBlendShapeTargetVertexIndices(destination, target)),
                    "deltas": {
                        axis.lower(): list(getattr(source, "getBlendShapeTargetDelta" + axis)(destination, target))
                        for axis in ("Xs", "Ys", "Zs")
                    },
                }
            )
        return partitions.expand_bridges(native, graphs, records, indices)


def project(data: bytes, payload: dict) -> tuple[dict[str, bytes], list]:
    sdk = load_native_runtime()
    if sdk.capabilities().get("component_projection") != 1:
        raise RuntimeError("Update the Blender RigLogic SDK and restart Blender to open combined DNA documents.")
    native = merged_graph.from_bytes(sdk.runtime_definition(data), geometry=False)
    graphs = partitions.project(native, payload["indices"], payload["components"], geometry=False)
    outputs = sdk.project_components(data, plans(graphs, payload["indices"], payload["components"]))
    parts = dict(zip(graphs, outputs, strict=True))
    bridges = _bridges(data, parts, graphs, payload["bridges"], payload["indices"])
    return parts, bridges
