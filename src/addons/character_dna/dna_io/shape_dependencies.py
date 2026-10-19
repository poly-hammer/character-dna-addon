"""Partition cross-neck shapes with the raw/PSD inputs needed to evaluate them."""

# This is the dependency-partitioning half of the native graph codec.
# ruff: noqa: SLF001

from typing import Any

import numpy as np

from . import merged_graph as graph


def split_extensions(parts: dict[str, bytes], bridges: list[dict]) -> dict[str, bytes]:
    """Materialize representable dependencies in ordinary recipient DNA files.

    RBF and ML outputs require their driver/network closure, so those are rejected
    before either split file is published. The combined document remains intact.
    """
    originals = {kind: graph.from_bytes(content) for kind, content in parts.items()}
    result = dict(parts)
    for kind in graph.KINDS:
        owned = [bridge for bridge in bridges if bridge["kind"] == kind]
        if not owned:
            if _remove_extension_metadata(originals[kind]):
                result[kind] = graph.to_bytes(originals[kind])
            continue
        source_kind = "body" if kind == "head" else "head"
        if any(bridge["source"]["kind"] != source_kind for bridge in owned):
            raise ValueError("Invalid cross-component shape dependency.")
        destination = _extend(originals, kind, source_kind, owned)
        _remove_extension_metadata(destination)
        result[kind] = graph.to_bytes(destination)
    return result


def _remove_extension_metadata(value: dict) -> bool:
    from .document import EXTENSION_PREFIX

    original = value["desc1.1"]["metadata"]
    filtered = [
        pair for pair in original if pair[0] != EXTENSION_PREFIX and not pair[0].startswith(EXTENSION_PREFIX + ".")
    ]
    value["desc1.1"]["metadata"] = filtered
    return filtered != original


def rebind_scene(instance: Any, bridges: list[dict]) -> None:
    """Replace extension drivers with the split recipient's native shape channel."""
    import json

    for bridge in bridges:
        component = instance.component(bridge["kind"])
        obj = component.mesh_index_lookup.get(bridge["mesh"])
        if obj is None or obj.data.shape_keys is None:
            continue
        key = obj.data.shape_keys.key_blocks.get(bridge["name"])
        if key is None:
            continue
        name = bridge["source"]["name"].split("__", 1)[-1]
        reader = component.dna_reader
        channels = {
            reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(bridge["mesh"], index))
            for index in range(reader.getBlendShapeTargetCount(bridge["mesh"]))
        }
        alias = f"DNA_{bridge['source']['kind']}::{name}"
        if alias in channels:
            name = alias
        if name not in channels:
            raise ValueError(f"Split corrective channel is missing: {name}")
        key.driver_remove("value")
        key.name = reader.getMeshName(bridge["mesh"]) + "__" + name
        owners = json.loads(obj.get("dna_shape_bridges", "{}"))
        owners.pop(bridge["name"], None)
        if owners:
            obj["dna_shape_bridges"] = json.dumps(owners)
        else:
            obj.pop("dna_shape_bridges", None)
    instance.destroy_references()
    instance.sync_shape_key_list()


def _extend(originals: dict, kind: str, source_kind: str, bridges: list[dict]) -> dict:  # noqa: PLR0915
    source = originals[source_kind]
    recipient = originals[kind]
    src_definition = source["defn1.1"]
    channels = sorted({graph._bridge_channel(originals, bridge) for bridge in bridges})
    table = source["bhvr1.1"]["blendShapeChannels"]
    rows = [index for index, channel in enumerate(table["outputIndices"]) if channel in channels]
    raw_count = len(src_definition["rawControlNames"])
    psd_count = source["bhvr1.1"]["controls"]["psdCount"]
    psds = source["bhvr1.1"]["controls"]["psds"]
    required = set()

    def include(index: int, pending: set[int]) -> None:
        if index in pending:
            raise ValueError("Cyclic PSD dependency in corrective.")
        if index in required:
            return
        if index >= raw_count + psd_count:
            raise ValueError(
                "This corrective requires an RBF or ML dependency that cannot yet be represented "
                "in independent split DNA. The merged document is unchanged."
            )
        required.add(index)
        for row, column in zip(psds["rows"], psds["columns"], strict=True):
            if row == index:
                include(column, pending | {index})

    for row in rows:
        include(table["inputIndices"][row], set())
    raw = sorted(index for index in required if index < raw_count)
    psd = sorted(required - set(raw))
    counts = graph._counts(recipient)
    maps = {domain: list(range(count)) for domain, count in counts.items()}
    maps["attribute"] = list(range(counts["joint"] * 9))
    maps["control"] = (
        list(range(counts["raw"]))
        + [index + len(raw) for index in range(counts["raw"], counts["raw"] + counts["psd"])]
        + [
            index + len(raw) + len(psd)
            for index in range(counts["raw"] + counts["psd"], sum(counts[key] for key in ("raw", "psd", "ml", "rbf")))
        ]
    )
    destination = graph.transform(recipient, maps)
    definition = destination["defn1.1"]
    controls = destination["bhvr1.1"]["controls"]
    mapping = {old: counts["raw"] + index for index, old in enumerate(raw)}
    mapping.update({old: counts["raw"] + len(raw) + counts["psd"] + index for index, old in enumerate(psd)})
    for index in raw:
        name = src_definition["rawControlNames"][index]
        definition["rawControlNames"].append(
            f"DNA_{source_kind}::{name}" if name in definition["rawControlNames"] else name
        )
    controls["psdCount"] += len(psd)
    for index, row in enumerate(psds["rows"]):
        if row in psd:
            controls["psds"]["rows"].append(mapping[row])
            controls["psds"]["columns"].append(mapping[psds["columns"][index]])
            controls["psds"]["values"].append(psds["values"][index])
    gui = source["bhvr1.1"]["controls"]["conditionals"]
    gui_rows = [index for index, output in enumerate(gui["outputIndices"]) if output in raw]
    gui_inputs = sorted({gui["inputIndices"][index] for index in gui_rows})
    gui_map = {index: len(definition["guiControlNames"]) + offset for offset, index in enumerate(gui_inputs)}
    definition["guiControlNames"] += [src_definition["guiControlNames"][index] for index in gui_inputs]
    for index in gui_rows:
        for field, values in gui.items():
            value = values[index]
            controls["conditionals"][field].append(
                gui_map[value] if field == "inputIndices" else mapping[value] if field == "outputIndices" else value
            )
    channel_map = {channel: len(definition["blendShapeChannelNames"]) + index for index, channel in enumerate(channels)}
    for channel in channels:
        name = src_definition["blendShapeChannelNames"][channel]
        definition["blendShapeChannelNames"].append(
            f"DNA_{source_kind}::{name}" if name in definition["blendShapeChannelNames"] else name
        )
    recipient_table = destination["bhvr1.1"]["blendShapeChannels"]
    # Include the dependency rows at all recipient LODs. Geometry membership
    # still controls which targets can be visible at each LOD.
    extra = {
        "inputIndices": [mapping[table["inputIndices"][index]] for index in rows],
        "outputIndices": [channel_map[table["outputIndices"][index]] for index in rows],
    }
    for field, values in extra.items():
        recipient_table[field] = values + recipient_table[field]
    lod_count = destination["desc1.1"]["lodCount"]
    recipient_table["lods"] = [count + len(rows) for count in (recipient_table["lods"] or [0] * lod_count)]
    lod_rows = graph._lod_rows(definition["lodBlendShapeMapping"], list(range(lod_count)))
    definition["lodBlendShapeMapping"] = graph._lod_map([row + list(channel_map.values()) for row in lod_rows])
    destination["bhvr1.1"]["joints"]["colCount"] += len(raw) + len(psd)
    for bridge in bridges:
        delta = graph._bridge_delta(originals, bridge)
        indices = np.flatnonzero(np.any(np.abs(delta) > 1e-5, axis=1))
        channel = channel_map[graph._bridge_channel(originals, bridge)]
        target = {
            "deltas": {axis: delta[indices, index].tolist() for index, axis in enumerate(("xs", "ys", "zs"))},
            "vertexIndices": indices.tolist(),
            "blendShapeChannelIndex": channel,
        }
        destination["geom1.1"]["meshes"][bridge["mesh"]]["blendShapeTargets"].append(target)
        definition["meshBlendShapeChannelMapping"]["from"].append(bridge["mesh"])
        definition["meshBlendShapeChannelMapping"]["to"].append(channel)
    return destination
