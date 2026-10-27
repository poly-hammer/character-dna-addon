"""Native merged DNA and rebuildable component views with revision-safe writes."""

import base64
import hashlib
import json
import logging
import os
import tempfile
import uuid

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..shape_names import ensure_name, find_key, logical_name


SCHEMA = 3
PARTITIONS = ("head", "body")
EXTENSION_PREFIX = "character_dna.shape_extensions"
_leases: dict[Path, tuple[Path, str, str]] = {}
_read_cache: OrderedDict[str, tuple[dict, dict[str, bytes]]] = OrderedDict()
MAX_VIEW_CACHE_BYTES = 256 * 1024**2
_pending: dict[Path, dict] = {}
_pending_targets: dict[Path, set | None] = {}


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_document(path: Path) -> bool:
    from .metadata import PREFIX
    from .misc import get_dna_reader, release_dna_handle

    with Path(path).open("rb") as stream:
        header = stream.read(3)
    if not header.startswith(b"DNA"):
        return False
    reader = get_dna_reader(path, data_layer="Descriptor")
    try:
        return any(reader.getMetaDataKey(index) == PREFIX for index in range(reader.getMetaDataCount()))
    finally:
        release_dna_handle(reader)


def _payload(path: Path) -> dict:
    from . import metadata
    from .misc import get_dna_reader, release_dna_handle

    reader = get_dna_reader(path, data_layer="Descriptor")
    try:
        payload = metadata.decode(reader)
        if payload.get("schema") != SCHEMA:
            raise ValueError("Unsupported combined DNA schema. Merge the component files again.")
        uuid.UUID(payload["manifest"]["id"])
        if set(payload["manifest"]["partitions"]) != set(PARTITIONS):
            raise ValueError("Invalid combined DNA ownership records.")
        return payload
    finally:
        release_dna_handle(reader)


def read_document(path: Path) -> tuple[dict, dict[str, bytes]]:
    """Project disposable component views from the authoritative native document."""
    from . import native_projection

    data = Path(path).read_bytes()
    revision = fingerprint(data)
    if revision in _read_cache:
        manifest, parts = _read_cache[revision]
        _read_cache.move_to_end(revision)
        return json.loads(json.dumps(manifest)), dict(parts)
    payload = _payload(path)
    manifest = payload["manifest"]
    parts, manifest["scene_bridges"] = native_projection.project(data, payload)
    _cache_read(revision, manifest, parts)
    return json.loads(json.dumps(manifest)), dict(parts)


def _cache_read(revision: str, manifest: dict, parts: dict[str, bytes]) -> None:
    for key, (existing, _) in list(_read_cache.items()):
        if existing["id"] == manifest["id"]:
            del _read_cache[key]
    _read_cache[revision] = (json.loads(json.dumps(manifest)), dict(parts))
    while (
        len(_read_cache) > 1
        and sum(len(data) for _, values in _read_cache.values() for data in values.values()) > MAX_VIEW_CACHE_BYTES
    ):
        _read_cache.popitem(last=False)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_document(
    path: Path, manifest: dict, parts: dict[str, bytes], *, expected_revision: str | None = None
) -> None:
    """Recompose structural changes, serializing native geometry only once."""
    from . import merged_graph, partitions

    _check_revision(path, expected_revision)
    manifest = json.loads(json.dumps(manifest))
    originals = {kind: merged_graph.from_bytes(content) for kind, content in parts.items()}
    bridges = manifest.pop("scene_bridges", [])
    graph, indices = merged_graph.compose(originals, manifest["name"], bridges)
    indices.pop("bridges", None)
    payload = {
        "schema": SCHEMA,
        "manifest": manifest,
        "indices": indices,
        "components": partitions.describe(originals, graph, indices),
        "bridges": partitions.compact_bridges(originals, bridges, indices),
    }
    published = merged_graph.to_bytes(graph, payload)
    _check_revision(path, expected_revision)
    _atomic_write(path, published)
    manifest["scene_bridges"] = bridges
    _cache_read(fingerprint(published), manifest, parts)


def _check_revision(path: Path, expected_revision: str | None) -> None:
    if expected_revision is not None and fingerprint(path.read_bytes()) != expected_revision:
        raise ValueError("The combined DNA changed on disk. Reload it before writing further edits.")


def merge_files(paths: dict[str, Path], destination: Path, name: str) -> None:
    """Create a new document; source files are never rewritten."""
    from ..rig_definition import get_rig_definition
    from .misc import get_dna_component_type, get_dna_reader, release_dna_handle

    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if destination.stem.lower() in PARTITIONS or not name.strip():
        raise ValueError("Use a rig instance name other than head or body for the combined DNA.")
    manifest: dict[str, Any] = {
        "version": 1,
        "id": str(uuid.uuid4()),
        "name": name,
        "partitions": {},
        "scene_bridges": [],
    }
    parts = {}
    for kind in PARTITIONS:
        path = paths[kind]
        if is_document(path) or get_dna_component_type(path) != kind:
            raise ValueError(f"Expected an ordinary {kind} DNA component: {path}")
        reader = get_dna_reader(path, data_layer="Definition")
        try:
            db_name = reader.getDBName()
            get_rig_definition(kind, db_name)
            manifest["partitions"][kind] = {"id": kind, "db_name": db_name, "source": str(path.resolve())}
            manifest["scene_bridges"].extend(_reader_extensions(reader, kind))
        finally:
            release_dna_handle(reader)
        parts[kind] = path.read_bytes()
    write_document(destination, manifest, parts)


def materialize(path: Path) -> tuple[dict, dict[str, Path]]:
    """Make disposable SDK working copies; the combined document is authoritative."""
    path = path.resolve()
    manifest, parts = read_document(path)
    revision = fingerprint(path.read_bytes())
    root = Path(tempfile.mkdtemp(prefix="character-dna-")) / str(uuid.UUID(manifest["id"])) / "rig"
    result = {}
    for kind in PARTITIONS:
        target = root / f"{kind}.dna"
        _atomic_write(target, parts[kind])
        _leases[target.resolve()] = (path, kind, revision)
        result[kind] = target
    return manifest, result


def validate_write(path: Path) -> None:
    lease = _leases.get(path.resolve())
    if lease is None:
        import bpy

        from ..constants import ToolInfo

        # A failed load/reconnect must not turn an old disposable cache into
        # an ordinary writable DNA file and silently bypass the document.
        for scene in bpy.data.scenes:
            properties = getattr(scene, ToolInfo.NAME, None)
            for instance in properties.rig_instance_list if properties else ():
                if instance.get("combined_dna_path") and any(
                    Path(bpy.path.abspath(instance.component(kind).dna_file_path)).resolve() == path.resolve()
                    for kind in PARTITIONS
                ):
                    raise ValueError("The combined DNA is disconnected. Reload its document before committing edits.")
    if lease and fingerprint(lease[0].read_bytes()) != lease[2]:
        raise ValueError("The combined DNA changed on disk. Reload it before committing edits.")


def publish_partition(path: Path, *, targets: set | None = None) -> None:
    """Publish a successfully written SDK partition back into its document."""
    lease = _leases.get(path.resolve())
    if lease is None:
        return
    document, kind, revision = lease
    if document in _pending:
        if targets is None:
            _pending_targets[document] = None
        elif _pending_targets[document] is not None:
            _pending_targets[document].update((kind, mesh, target) for mesh, target in targets)
        return
    manifest, parts = read_document(document)
    content = path.read_bytes()
    if content == parts[kind]:
        return
    parts[kind] = content
    _publish_edit(
        document, manifest, parts, revision, {(kind, mesh, target) for mesh, target in targets} if targets else None
    )
    _advance_lease(path, document)


@contextmanager
def edit_transaction(instance: Any):
    """Batch writes to one combined document; recover only the author's working file.

    Editor core methods already release their native handles before writing.
    This context does not delete, detach, or replace any Blender scene object.
    """
    import bpy

    authored = Path(bpy.path.abspath(instance.edit_target.dna_file_path)).resolve()
    previous = authored.read_bytes()
    combined = (
        Path(bpy.path.abspath(instance["combined_dna_path"])).resolve() if instance.get("combined_dna_path") else None
    )
    if combined:
        validate_write(authored)
        manifest, parts = read_document(combined)
        revision = fingerprint(combined.read_bytes())
        _pending[combined] = manifest
        _pending_targets[combined] = set()
    try:
        yield
        if combined:
            parts[instance.editor_component] = authored.read_bytes()
            _publish_edit(combined, _pending[combined], parts, revision, _pending_targets[combined])
            _advance_lease(authored, combined)
    except Exception:
        if authored.read_bytes() != previous:
            _atomic_write(authored, previous)
        raise
    finally:
        if combined:
            _pending.pop(combined, None)
            _pending_targets.pop(combined, None)


def _publish_edit(path: Path, manifest: dict, parts: dict[str, bytes], revision: str, targets: set | None) -> None:
    from .document_edits import write_targets

    previous, old_parts = read_document(path)
    if manifest == previous and parts == old_parts:
        return
    before = {key: value for key, value in previous.items() if key != "scene_bridges"}
    after = {key: value for key, value in manifest.items() if key != "scene_bridges"}

    def identity(bridge: dict) -> tuple[str, int, str]:
        return bridge["kind"], bridge["mesh"], bridge["name"]

    old_bridges = {identity(item) for item in previous.get("scene_bridges", [])}
    new_bridges = {identity(item) for item in manifest.get("scene_bridges", [])}
    if targets is not None and (targets or parts == old_parts) and before == after and old_bridges <= new_bridges:
        _check_revision(path, revision)
        published = write_targets(
            path, parts, targets, _payload(path), manifest.get("scene_bridges", []), previous.get("scene_bridges", [])
        )
        _check_revision(path, revision)
        _atomic_write(path, published)
        _cache_read(fingerprint(published), manifest, parts)
    else:
        write_document(path, manifest, parts, expected_revision=revision)


def _advance_lease(path: Path, document: Path) -> None:
    revision = fingerprint(document.read_bytes())
    for target, (owner, partition, _revision) in list(_leases.items()):
        if target.parent == path.resolve().parent:
            _leases[target] = (owner, partition, revision)


def partition_backup(path: Path) -> bytes | None:
    """Keep a working partition recoverable if publishing its document fails."""
    return path.read_bytes() if path.resolve() in _leases else None


def split_files(path: Path, directory: Path) -> dict[str, Path]:
    """Reconstruct component data using the recorded RigDefinition ownership."""
    from ..rig_definition import get_rig_definition
    from .misc import get_dna_component_type, get_dna_reader, release_dna_handle

    manifest, parts = read_document(path)
    if manifest.get("scene_bridges"):
        from .shape_dependencies import split_extensions

        parts = split_extensions(parts, manifest["scene_bridges"])
    outputs = {kind: directory / f"{kind}.dna" for kind in PARTITIONS}
    for target in outputs.values():
        if target.exists():
            raise ValueError(f"Split destination already exists: {target}")
    with tempfile.TemporaryDirectory(prefix="character-dna-validate-") as temporary:
        for kind, content in parts.items():
            candidate = Path(temporary) / f"{kind}.dna"
            candidate.write_bytes(content)
            if get_dna_component_type(candidate) != kind:
                raise ValueError(f"The {kind} partition no longer matches its RigDefinition ownership.")
            reader = get_dna_reader(candidate, data_layer="Definition")
            try:
                if reader.getDBName() != manifest["partitions"][kind]["db_name"]:
                    raise ValueError(f"The {kind} partition's RigDefinition changed.")
                get_rig_definition(kind, reader.getDBName())
            finally:
                release_dna_handle(reader)
    created = []
    try:
        for kind, target in outputs.items():
            _atomic_write(target, parts[kind])
            created.append(target)
    except Exception:
        for target in created:
            target.unlink()
        raise
    return outputs


def bind_instance(instance: Any, path: Path, *, restore_bridges: bool = True) -> None:
    """Bind an existing scene rig to the two SDK partitions of one document."""
    manifest, paths = materialize(path)
    instance.destroy()
    instance["combined_dna_path"] = str(path.resolve())
    instance["combined_dna_id"] = manifest["id"]
    for kind in PARTITIONS:
        instance[f"{kind}_dna_file_path"] = str(paths[kind])
    instance.initialize()
    if restore_bridges:
        restore_scene_bridges(instance, manifest)


def finish_import(instance: Any, path: Path | None, manifest: dict | None, *, import_mesh: bool) -> None:
    """Restore document ownership and the persistent presentation after component import."""
    from ..utilities import editors_available

    if path:
        instance["combined_dna_path"] = str(path.resolve())
        instance["combined_dna_id"] = manifest["id"]
        if hasattr(instance, "converter_merge_mode"):
            instance.converter_merge_mode = "DNA"
    if not editors_available():
        return
    from ..editors.converter import scene_merge
    from ..runtime import controller

    with controller.authoring_operation(instance):
        instance.initialize()
        extensions = manifest if path else scene_extensions(instance)
        if extensions.get("scene_bridges"):
            restore_scene_bridges(instance, extensions)
        if scene_merge.surfaces(instance):
            scene_merge.adopt_imported_lods(instance)
        elif path and import_mesh:
            scene_merge.merge(instance)


def publish_scene_bridges(instance: Any) -> None:
    """Keep cross-component scene correctives in the standalone addon document."""
    import bpy

    path = instance.get("combined_dna_path")
    working = Path(bpy.path.abspath(instance.head_dna_file_path))
    validate_write(working)
    manifest, parts = read_document(Path(path)) if path else (scene_extensions(instance), {})
    bridges = collect_scene_bridges(instance, manifest)
    if bridges == manifest.get("scene_bridges", []):
        return
    manifest["scene_bridges"] = bridges
    if not path:
        _publish_scene_extensions(instance, manifest)
        return
    if Path(path).resolve() in _pending:
        _pending[Path(path).resolve()] = manifest
        return
    _publish_edit(Path(path), manifest, parts, _leases[working.resolve()][2], set())
    _advance_lease(working, Path(path))


def collect_scene_bridges(instance: Any, manifest: dict | None = None) -> list[dict]:
    """Gather current relationships without writing to either authoring file."""
    import numpy as np

    path = instance.get("combined_dna_path")
    if manifest is None:
        manifest = read_document(Path(path))[0] if path else scene_extensions(instance)
    available = {
        kind: {
            name
            for mesh, _, name in instance.component(kind).shape_targets(lod=None)
            if instance.component(kind).get_shape_key_block(mesh, name) is not None
        }
        for kind in PARTITIONS
    }
    # Missing meshes or deferred source keys cannot author relationship deletions.
    # Retain their embedded relationships until their shapes are loaded.
    bridges = [
        item
        for item in manifest.get("scene_bridges", [])
        if item["mesh"] not in instance.component(item["kind"]).mesh_index_lookup
        or item["source"]["name"] not in available[item["source"]["kind"]]
    ]
    for kind in PARTITIONS:
        component = instance.component(kind)
        for index, obj in component.mesh_index_lookup.items():
            owners = json.loads(obj.get("dna_shape_bridges", "{}"))
            keys = obj.data.shape_keys
            for name, owner in owners.items():
                block = keys.key_blocks.get(name) if keys else None
                if block is None:
                    continue
                full_name = logical_name(keys, name)
                if owner["name"] not in {
                    key for _, _, key in instance.component(owner["kind"]).shape_targets(lod=None)
                }:
                    if path:
                        raise ValueError("A combined DNA bridge needs an existing source DNA shape channel.")
                    continue  # Custom scene keys remain scene data until assigned a DNA channel.
                values = np.empty(len(block.data) * 3, dtype="<f4")
                block.data.foreach_get("co", values)
                baseline = np.empty_like(values)
                keys.reference_key.data.foreach_get("co", baseline)
                # A previously retained relationship may now have a live recipient.
                bridges = [
                    item for item in bridges if (item["kind"], item["mesh"], item["name"]) != (kind, index, full_name)
                ]
                bridges.append(
                    {
                        "kind": kind,
                        "mesh": index,
                        "name": full_name,
                        "source": owner,
                        "coordinates": base64.b64encode(values.tobytes()).decode("ascii"),
                        "deltas": base64.b64encode((values - baseline).tobytes()).decode("ascii"),
                        "min": block.slider_min,
                        "max": block.slider_max,
                    }
                )
    return bridges


def _reader_extensions(reader: Any, kind: str) -> list[dict]:
    from . import metadata

    if not reader or not any(
        reader.getMetaDataKey(index) == EXTENSION_PREFIX for index in range(reader.getMetaDataCount())
    ):
        return []
    payload = metadata.decode(reader, prefix=EXTENSION_PREFIX)
    if (
        payload.get("owner") != kind
        or payload.get("version") != 1
        or any(bridge["source"]["kind"] != kind for bridge in payload["scene_bridges"])
    ):
        raise ValueError("Invalid component shape extension ownership.")
    return payload["scene_bridges"]


def scene_extensions(instance: Any) -> dict:
    """Collect corrective relationships from their owning component DNA metadata."""
    bridges = []
    for kind in PARTITIONS:
        bridges.extend(_reader_extensions(instance.component(kind).dna_reader, kind))
    return {"scene_bridges": bridges}


def _publish_scene_extensions(instance: Any, manifest: dict) -> None:
    """Write cross-neck correctives only to the selected authoring component."""
    import bpy

    from ..bindings import dna
    from . import metadata
    from .misc import get_dna_reader, get_dna_writer, release_dna_handle

    kind = instance.editor_component
    path = Path(bpy.path.abspath(instance.component(kind).dna_file_path))
    original = path.read_bytes()
    bridges = [bridge for bridge in manifest["scene_bridges"] if bridge["source"]["kind"] == kind]
    with tempfile.TemporaryDirectory(prefix="dna-extensions-") as directory:
        staged = Path(directory) / f"{kind}.dna"
        reader = get_dna_reader(path)
        writer = get_dna_writer(staged)
        try:
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            metadata.write(
                writer, reader, {"version": 1, "owner": kind, "scene_bridges": bridges}, prefix=EXTENSION_PREFIX
            )
            writer.write()
        finally:
            release_dna_handle(writer)
            release_dna_handle(reader)
        # These methods release SDK reader/manager caches; scene IDs are retained.
        getattr(instance, f"destroy_{kind}")()
        if path.read_bytes() != original:
            raise ValueError("The component DNA changed while preparing corrective metadata.")
        _atomic_write(path, staged.read_bytes())
        instance.component(kind).initialize()


def restore_scene_bridges(instance: Any, manifest: dict) -> None:
    """Restore cross-partition correctives only for shapes already loaded in the scene."""
    import numpy as np

    from .misc import create_shape_key

    bridges = manifest.get("scene_bridges", [])
    # Backups may predate a cross-component corrective. Remove only keys whose
    # ownership was recorded by this adapter, leaving unrelated custom keys alone.
    for kind in PARTITIONS:
        for mesh_index, obj in instance.component(kind).mesh_index_lookup.items():
            owners = json.loads(obj.get("dna_shape_bridges", "{}"))
            wanted = {item["name"] for item in bridges if item["kind"] == kind and item["mesh"] == mesh_index}
            for name in [name for name in owners if logical_name(obj.data.shape_keys, name) not in wanted]:
                block = obj.data.shape_keys.key_blocks.get(name) if obj.data.shape_keys else None
                if block is not None:
                    block.driver_remove("value")
                    obj.shape_key_remove(block)
                del owners[name]
            if owners:
                obj["dna_shape_bridges"] = json.dumps(owners)
            else:
                obj.pop("dna_shape_bridges", None)
    for bridge in bridges:
        source = instance.component(bridge["source"]["kind"])
        name = bridge["source"]["name"]
        source_target = next(
            ((mesh, channel) for mesh, channel, key in source.shape_targets(lod=None) if key == name), None
        )
        if source_target is None:
            raise ValueError(f"Combined shape bridge has no source DNA channel: {name}")
        mesh, channel = source_target
        source_obj = source.mesh_index_lookup.get(mesh)
        obj = instance.component(bridge["kind"]).mesh_index_lookup.get(bridge["mesh"])
        if source_obj is None or obj is None:
            continue  # A bones-only import has no shape outputs to bind.
        source_key = source.get_shape_key_block(mesh, name)
        block = find_key(obj.data.shape_keys, bridge["name"])
        if source_key is None and block is None:
            continue  # DNA relationships do not opt the user into loading shape keys.
        values = np.frombuffer(base64.b64decode(bridge["coordinates"], validate=True), dtype="<f4")
        if values.size != len(obj.data.vertices) * 3 or not np.isfinite(values).all():
            raise ValueError("Invalid combined shape bridge geometry.")
        if source_key is None:
            # An existing recipient still needs its driver after the source Key
            # datablock was replaced (including an interrupted shape reimport).
            reader = source.dna_reader
            target = next(
                i
                for i in range(reader.getBlendShapeTargetCount(mesh))
                if reader.getBlendShapeChannelIndex(mesh, i) == channel
            )
            from ..bindings import enums
            from ..constants import SCALE_FACTOR

            source_key = create_shape_key(
                target,
                mesh,
                source_obj,
                reader,
                reader.getBlendShapeChannelName(channel),
                prefix=f"{reader.getMeshName(mesh)}__",
                linear_modifier=1 / SCALE_FACTOR
                if enums.TranslationUnit(reader.getTranslationUnit()).name.lower() == "cm"
                else 1.0,
            )
        if obj.data.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        block = block or obj.shape_key_add(name=ensure_name(obj.data.shape_keys, bridge["name"]), from_mix=False)
        block.data.foreach_set("co", values)
        block.slider_min, block.slider_max = bridge["min"], bridge["max"]
        if block.id_data.animation_data and block.id_data.animation_data.drivers.find(block.path_from_id("value")):
            block.driver_remove("value")
        driver = block.driver_add("value").driver
        driver.type = "SUM"
        variable = driver.variables.new()
        variable.name = "source"
        variable.targets[0].id_type = "KEY"
        variable.targets[0].id = source_obj.data.shape_keys
        variable.targets[0].data_path = source_key.path_from_id("value")
        owners = json.loads(obj.get("dna_shape_bridges", "{}"))
        owners[block.name] = bridge["source"]
        obj["dna_shape_bridges"] = json.dumps(owners)
    instance.destroy_references()
    instance.sync_shape_key_list()


def restore_scene_documents() -> None:
    """Recover disposable paths after opening a .blend on another machine/session."""
    import bpy

    from ..constants import ToolInfo

    for scene in bpy.data.scenes:
        properties = getattr(scene, ToolInfo.NAME, None)
        for instance in properties.rig_instance_list if properties else ():
            if path := instance.get("combined_dna_path"):
                from ..runtime import engine

                try:
                    bind_instance(instance, Path(bpy.path.abspath(path)))
                    if not engine.carriers(instance) and instance.auto_evaluate:
                        engine.install(instance)
                except (ValueError, RuntimeError, OSError):
                    logging.getLogger(__name__).exception("Could not restore combined DNA for %s", instance.name)
