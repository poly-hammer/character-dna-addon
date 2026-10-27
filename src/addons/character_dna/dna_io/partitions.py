"""Component projection recipes. Native DNA owns data; metadata owns correspondence.

Only values discarded by composition (descriptors, seam weights and seam offsets)
are retained here. Geometry, targets and behavior arrays are never recovery blobs.
"""

import base64

from typing import Any

import numpy as np

from . import merged_graph as graph_io


def _reference(path: list[str], start: int, stop: int) -> dict:
    return {"ref": path, "slice": [start, stop]}


def _joined_recipe(head: Any, body: Any, kind: str, path: list[str]) -> Any:
    value = head if kind == "head" else body
    if isinstance(value, dict):
        return {key: _joined_recipe(head[key], body[key], kind, [*path, key]) for key in value}
    if isinstance(value, list):
        start = 0 if kind == "head" else len(head)
        return _reference(path, start, start + len(value))
    return value


def _resolve(recipe: Any, native: dict) -> Any:
    if isinstance(recipe, dict):
        if set(recipe) == {"ref", "slice"}:
            value = native
            for key in recipe["ref"]:
                value = value[key]
            start, stop = recipe["slice"]
            if stop > len(value):
                raise ValueError("The native graph no longer matches its component ownership.")
            return graph_io.clone(value[start:stop])
        return {key: _resolve(value, native) for key, value in recipe.items()}
    return graph_io.clone(recipe)


def describe(originals: dict, native: dict, indices: dict) -> dict:
    """Record the inverse of composition without retaining either source file."""
    parts = {kind: graph_io.transform(value, indices["maps"][kind]) for kind, value in originals.items()}
    records = {}
    joined = ("defn1.1", "bhvr1.1", "mlbh1.0", "rbfb1.0", "rbfe1.0", "jbmd1.0", "twsw1.0")
    for kind, source in parts.items():
        recipe = {key: graph_io.clone(value) for key, value in source.items() if key not in {*joined, "geom1.1"}}
        recipe["desc1.1"]["metadata"] = [
            pair for pair in recipe["desc1.1"]["metadata"] if not pair[0].startswith("character_dna.shape_extensions")
        ]
        for layer in joined:
            recipe[layer] = _joined_recipe(parts["head"][layer], parts["body"][layer], kind, [layer])
        # These compact tables describe original ordering, names and LOD packing.
        # All mutable behavior values are referenced from the native graph.
        for layer, fields in (
            ("defn1.1", [*graph_io.LOD_DOMAINS, "meshNames", "meshBlendShapeChannelMapping", *graph_io.NAME_DOMAINS]),
            ("mlbh1.0", ["lodNeuralNetworkMapping", "neuralNetworkToMeshRegion", "mlControlNames"]),
            ("rbfb1.0", ["lodSolverMapping"]),
            ("rbfe1.0", ["poseControlNames"]),
        ):
            for field in fields:
                if field in source[layer]:
                    recipe[layer][field] = graph_io.clone(source[layer][field])
        for field in ("blendShapeChannels", "animatedMaps"):
            recipe["bhvr1.1"][field] = {"lods": source["bhvr1.1"][field]["lods"]}
        meshes = []
        for index, mesh in enumerate(source["geom1.1"]["meshes"]):
            record = next((item for item in indices["surfaces"] if item[kind] == index), None)
            sizes = {field: len(mesh[field]) for field in ("faces", "skinWeights")}
            sizes.update(
                vertices=len(mesh["positions"]["xs"]),
                normals=len(mesh["normals"]["xs"]),
                uv=len(mesh["textureCoordinates"]["us"]),
                layouts=len(mesh["layouts"]["positions"]),
            )
            values = {
                "sizes": sizes,
                "influences": mesh["maximumInfluencePerVertex"],
                "channels": [target["blendShapeChannelIndex"] for target in mesh["blendShapeTargets"]],
            }
            if record and kind == "head":
                merged = native["geom1.1"]["meshes"][record["mesh"]]
                values["seam"] = {
                    str(vertex): {
                        "source": [mesh["positions"][axis][vertex] for axis in ("xs", "ys", "zs")],
                        "merged": [
                            merged["positions"][axis][record["vertices"][kind][vertex]] for axis in ("xs", "ys", "zs")
                        ],
                        "weights": mesh["skinWeights"][vertex] if vertex < len(mesh["skinWeights"]) else None,
                    }
                    for vertex in record["seam"]
                }
            meshes.append(values)
        records[kind] = {
            "recipe": recipe,
            "meshes": meshes,
            "pose_names": [item["name"] for item in source["rbfb1.0"]["poses"]],
            "solver_names": [item["name"] for item in source["rbfb1.0"]["solvers"]],
        }
    return records


def _mesh(native: dict, item: dict, head_item: dict, record: dict | None, kind: str) -> dict:
    if record is None:
        return graph_io.clone(native)
    sizes, head_sizes = item["sizes"], head_item["sizes"]
    vertices = record["vertices"][kind]
    inverse = {value: index for index, value in enumerate(vertices)}
    result = {
        "positions": {axis: [values[index] for index in vertices] for axis, values in native["positions"].items()},
        "maximumInfluencePerVertex": item["influences"],
    }
    for field, count in (("normals", "normals"), ("textureCoordinates", "uv"), ("layouts", "layouts")):
        start = 0 if kind == "head" else head_sizes[count]
        result[field] = {axis: values[start : start + sizes[count]] for axis, values in native[field].items()}
    result["layouts"]["positions"] = [inverse[index] for index in result["layouts"]["positions"]]
    if kind == "body":
        for field, size in (("normals", "normals"), ("textureCoordinates", "uv")):
            result["layouts"][field] = [index - head_sizes[size] for index in result["layouts"][field]]
    start = 0 if kind == "head" else head_sizes["faces"]
    offset = 0 if kind == "head" else head_sizes["layouts"]
    result["faces"] = [
        {"layoutIndices": [index - offset for index in face["layoutIndices"]]}
        for face in native["faces"][start : start + sizes["faces"]]
    ]
    result["skinWeights"] = [native["skinWeights"][index] for index in vertices[: sizes["skinWeights"]]]
    for vertex, residual in item.get("seam", {}).items():
        for axis, old, baseline in zip(("xs", "ys", "zs"), residual["source"], residual["merged"], strict=True):
            current = result["positions"][axis][int(vertex)]
            result["positions"][axis][int(vertex)] = (
                old if current == baseline else float(np.float32(old + current - baseline))
            )
        if residual["weights"] is not None:
            result["skinWeights"][int(vertex)] = graph_io.clone(residual["weights"])
    targets = {target["blendShapeChannelIndex"]: target for target in native["blendShapeTargets"]}
    result["blendShapeTargets"] = []
    for channel in item["channels"]:
        target = targets[channel]
        selected = [i for i, vertex in enumerate(target["vertexIndices"]) if vertex in inverse]
        result["blendShapeTargets"].append(
            {
                "blendShapeChannelIndex": channel,
                "vertexIndices": [inverse[target["vertexIndices"][i]] for i in selected],
                "deltas": target["deltas"]
                if len(selected) == len(target["vertexIndices"])
                else {axis: [values[i] for i in selected] for axis, values in target["deltas"].items()},
            }
        )
    return result


def project(native: dict, indices: dict, records: dict, *, geometry: bool = True) -> dict:
    """Resolve current native values and invert each recorded component domain."""
    parts = {}
    for kind, record in records.items():
        result = _resolve(record["recipe"], native)
        lods = indices["lods"][kind]
        first = [lods.index(lod) for lod in range(result["desc1.1"]["lodCount"])]
        for group in result["bhvr1.1"]["joints"]["jointGroups"]:
            group["lods"] = [group["lods"][lod] for lod in first]
        for field in ("blendShapeChannels", "animatedMaps"):
            rows = indices["tables"][field][kind]
            table = native["bhvr1.1"][field]
            values = table["conditionals"] if field == "animatedMaps" else table
            projected = {key: [array[index] for index in rows] for key, array in values.items() if key != "lods"}
            counts = (
                [sum(index < table["lods"][lod] for index in rows) for lod in first]
                if result["bhvr1.1"][field]["lods"]
                else []
            )
            result["bhvr1.1"][field] = {
                "lods": counts,
                **({"conditionals": projected} if field == "animatedMaps" else projected),
            }
        for field, label in (("solvers", "solver_names"), ("poses", "pose_names")):
            for value, original in zip(result["rbfb1.0"][field], record[label], strict=True):
                if value["name"] == "DNA_body::" + original:
                    value["name"] = original
        meshes = []
        for index, item in enumerate(record["meshes"] if geometry else []):
            variants = []
            for destination in indices["meshes"][kind][str(index)]:
                surface = next((value for value in indices["surfaces"] if value["mesh"] == destination), None)
                head = records["head"]["meshes"][surface["head"]] if surface else item
                variants.append(_mesh(native["geom1.1"]["meshes"][destination], item, head, surface, kind))
            if any(value != variants[0] for value in variants[1:]):
                raise ValueError(f"Conflicting native edits to shared {kind} mesh {index} across LODs.")
            meshes.append(variants[0])
        result["geom1.1"] = {"meshes": meshes}
        parts[kind] = graph_io.transform(result, indices["maps"][kind], inverse=True)
    return parts


def compact_bridges(originals: dict, bridges: list, indices: dict) -> list:
    """Replace dense scene coordinates with native target references and seam residuals."""
    result = []
    for bridge in bridges:
        kind = bridge["kind"]
        channel = indices["maps"][bridge["source"]["kind"]]["blend"][graph_io.bridge_channel(originals, bridge)]
        delta = graph_io.bridge_delta(originals, bridge)
        surface = next((value for value in indices["surfaces"] if value[kind] == bridge["mesh"]), None)
        seam = []
        if surface:
            other = set(surface["vertices"]["body" if kind == "head" else "head"])
            seam = [index for index, value in enumerate(surface["vertices"][kind]) if value in other]
        result.append(
            {
                **{key: value for key, value in bridge.items() if key not in {"coordinates", "deltas"}},
                "channel": channel,
                "seam": {str(index): delta[index].tolist() for index in seam},
            }
        )
    return result


def expand_bridges(native: dict, parts: dict, records: list, indices: dict) -> list:
    """Build transient scene data from the one authoritative native target array."""
    result = []
    for record in records:
        kind, index = record["kind"], record["mesh"]
        destination = indices["meshes"][kind][str(index)][0]
        surface = next((value for value in indices["surfaces"] if value["mesh"] == destination), None)
        basis = np.asarray(list(parts[kind]["geom1.1"]["meshes"][index]["positions"].values())).T
        vertices = surface["vertices"][kind] if surface else list(range(len(basis)))
        inverse = {value: i for i, value in enumerate(vertices)}
        delta = np.zeros_like(basis)
        targets = native["geom1.1"]["meshes"][destination]["blendShapeTargets"]
        target = next(item for item in targets if item["blendShapeChannelIndex"] == record["channel"])
        for vertex, value in zip(target["vertexIndices"], np.asarray(list(target["deltas"].values())).T, strict=True):
            if vertex in inverse:
                delta[inverse[vertex]] = value
        for vertex, value in record["seam"].items():
            delta[int(vertex)] = value
        scale = 0.01 if parts[kind]["desc1.1"]["translationUnit"] == 0 else 1.0

        def scene(values: np.ndarray, scale: float = scale) -> np.ndarray:
            return (values[:, [0, 2, 1]] * [scale, -scale, scale]).astype("<f4")

        result.append(
            {
                **{key: value for key, value in record.items() if key not in {"channel", "seam"}},
                "coordinates": base64.b64encode(scene(basis + delta).tobytes()).decode("ascii"),
                "deltas": base64.b64encode(scene(delta).tobytes()).decode("ascii"),
            }
        )
    return result
