"""Typed value edits on an existing native document; no graph/JSON reconstruction."""

from __future__ import annotations

import base64
import copy
import tempfile

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from ..bindings import dna
from . import metadata
from .coordinates import configuration
from .misc import release_dna_handle


def _reader(path: Path) -> dna.BinaryStreamReader:
    stream = dna.FileStream(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    reader = dna.BinaryStreamReader(stream, configuration(blender=False), None)
    reader.read()
    if not dna.Status.isOk():
        release_dna_handle(reader)
        raise ValueError(dna.Status.get().message)
    return reader


def _target_buffer(
    source: dna.BinaryStreamReader, buffers: dict, mesh: int, channel: int
) -> tuple[np.ndarray, np.ndarray]:
    identity = (mesh, channel)
    if identity not in buffers:
        target = next(
            (
                index
                for index in range(source.getBlendShapeTargetCount(mesh))
                if source.getBlendShapeChannelIndex(mesh, index) == channel
            ),
            None,
        )
        buffers[identity] = (
            (
                np.asarray(source.getBlendShapeTargetVertexIndices(mesh, target), dtype=np.int64),
                np.column_stack(
                    [getattr(source, "getBlendShapeTargetDelta" + axis)(mesh, target) for axis in ("Xs", "Ys", "Zs")]
                ),
            )
            if target is not None
            else (np.empty(0, dtype=np.int64), np.empty((0, 3)))
        )
    return buffers[identity]


def _update_target_buffers(
    source: dna.BinaryStreamReader, readers: dict, changes: set, payload: dict, buffers: dict
) -> None:
    """Remap changed component targets, retaining deltas owned by the other component."""
    info = payload["indices"]
    for kind, mesh, target in sorted(changes):
        reader = readers[kind]
        channel = info["maps"][kind]["blend"][reader.getBlendShapeChannelIndex(mesh, target)]
        expected = payload["components"][kind]["meshes"][mesh]["channels"]
        if reader.getBlendShapeTargetCount(mesh) != len(expected) or channel != expected[target]:
            raise ValueError("A target-only edit changed component structure.")
        vertices = np.asarray(reader.getBlendShapeTargetVertexIndices(mesh, target), dtype=np.int64)
        deltas = np.column_stack(
            [getattr(reader, "getBlendShapeTargetDelta" + axis)(mesh, target) for axis in ("Xs", "Ys", "Zs")]
        )
        for destination in info["meshes"][kind][str(mesh)]:
            surface = next((item for item in info["surfaces"] if item["mesh"] == destination), None)
            ids = np.asarray(surface["vertices"][kind] if surface else range(reader.getVertexPositionCount(mesh)))
            old_ids, old_deltas = _target_buffer(source, buffers, destination, channel)
            foreign = ~np.isin(old_ids, ids)
            buffers[(destination, channel)] = (
                np.concatenate((ids[vertices], old_ids[foreign])),
                np.concatenate((deltas, old_deltas[foreign])),
            )


def _update_bridge_buffers(
    source: dna.BinaryStreamReader, buffers: dict, payload: dict, bridges: list, previous_bridges: list
) -> list:
    """Update cross-component targets and retain their compact seam records."""
    info = payload["indices"]
    compact = []
    previous = {(item["kind"], item["mesh"], item["name"]): item for item in previous_bridges}
    recorded = {(item["kind"], item["mesh"], item["name"]): item for item in payload["bridges"]}
    for bridge in bridges:
        identity = bridge["kind"], bridge["mesh"], bridge["name"]
        if previous.get(identity) == bridge:
            compact.append(recorded[identity])
            continue
        kind, mesh = bridge["kind"], bridge["mesh"]
        source_kind = bridge["source"]["kind"]
        definition = payload["components"][source_kind]["recipe"]["defn1.1"]
        name = bridge["source"]["name"]
        prefix = next(mesh_name + "__" for mesh_name in definition["meshNames"] if name.startswith(mesh_name + "__"))
        channel = info["maps"][source_kind]["blend"][definition["blendShapeChannelNames"].index(name[len(prefix) :])]
        item = payload["components"][kind]["meshes"][mesh]
        count = item["sizes"]["vertices"]
        values = np.frombuffer(base64.b64decode(bridge["deltas"], validate=True), dtype="<f4").reshape(-1, 3)
        if len(values) != count or not np.isfinite(values).all():
            raise ValueError("Invalid cross-component corrective deltas.")
        scale = 100.0 if payload["components"][kind]["recipe"]["desc1.1"]["translationUnit"] == 0 else 1.0
        delta = values[:, [0, 2, 1]] * [scale, scale, -scale]
        seam = {}
        for destination in info["meshes"][kind][str(mesh)]:
            surface = next((item for item in info["surfaces"] if item["mesh"] == destination), None)
            ids = np.asarray(surface["vertices"][kind] if surface else range(count))
            shared = (
                np.isin(ids, surface["vertices"]["body" if kind == "head" else "head"])
                if surface
                else np.zeros(count, bool)
            )
            seam = {str(index): delta[index].tolist() for index in np.flatnonzero(shared)}
            selected = ~shared & np.any(np.abs(delta) > 1e-5, axis=1)
            old_ids, old_delta = _target_buffer(source, buffers, destination, channel)
            keep = ~np.isin(old_ids, ids[~shared])
            buffers[(destination, channel)] = (
                np.concatenate((old_ids[keep], ids[selected])),
                np.concatenate((old_delta[keep], delta[selected])),
            )
        compact.append(
            {
                **{key: value for key, value in bridge.items() if key not in {"coordinates", "deltas"}},
                "channel": channel,
                "seam": seam,
            }
        )
    return compact


def _write_target_buffers(source: dna.BinaryStreamReader, writer: dna.BinaryStreamWriter, buffers: dict) -> None:
    """Publish changed sparse arrays, adding channel mappings for new recipients."""
    counts = {}
    mapping = source.getMeshBlendShapeChannelMappingCount()
    for (mesh, channel), (ids, deltas) in buffers.items():
        target = next(
            (
                index
                for index in range(source.getBlendShapeTargetCount(mesh))
                if source.getBlendShapeChannelIndex(mesh, index) == channel
            ),
            None,
        )
        if target is None:
            target = counts.get(mesh, source.getBlendShapeTargetCount(mesh))
            counts[mesh] = target + 1
            writer.setBlendShapeChannelIndex(mesh, target, channel)
            writer.setMeshBlendShapeChannelMapping(mapping, mesh, channel)
            mapping += 1
        writer.setBlendShapeTargetVertexIndices(mesh, target, ids.tolist())
        writer.setBlendShapeTargetDeltas(mesh, target, deltas.tolist())


def write_targets(path: Path, parts: dict, changes: set, payload: dict, bridges: list, previous_bridges: list) -> bytes:
    """Remap authored sparse targets to every native LOD sharing the source mesh."""
    source = _reader(path)
    readers = {}
    writer = stream = None
    try:
        with tempfile.TemporaryDirectory(prefix="dna-target-edit-") as directory, ExitStack() as handles:
            root = Path(directory)
            for kind in {kind for kind, _, _ in changes}:
                part = root / f"{kind}.dna"
                part.write_bytes(parts[kind])
                readers[kind] = _reader(part)
                handles.callback(release_dna_handle, readers[kind])
            output = root / "published.dna"
            stream = dna.FileStream(str(output), dna.AccessMode_Write, dna.OpenMode_Binary, None)
            handles.callback(release_dna_handle, stream)
            writer = dna.BinaryStreamWriter(stream)
            handles.callback(release_dna_handle, writer)
            writer.setFrom(source, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            buffers = {}
            _update_target_buffers(source, readers, changes, payload, buffers)
            compact = _update_bridge_buffers(source, buffers, payload, bridges, previous_bridges)
            _write_target_buffers(source, writer, buffers)
            if compact != payload["bridges"]:
                payload = copy.deepcopy(payload)
                payload["bridges"] = compact
                metadata.write(writer, source, payload)
            writer.write()
            if not dna.Status.isOk():
                raise ValueError(dna.Status.get().message)
            release_dna_handle(writer)
            release_dna_handle(stream)
            writer = stream = None
            # Close temporary component files before TemporaryDirectory cleanup on Windows.
            for reader in readers.values():
                release_dna_handle(reader)
            readers.clear()
            return output.read_bytes()
    finally:
        for handle in (writer, stream, *readers.values(), source):
            release_dna_handle(handle)
