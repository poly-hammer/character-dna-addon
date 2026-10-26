"""Recover legacy truncated key identities without replacing sculpted geometry."""

import logging
import re

from typing import Any

import numpy as np

from .. import shape_names


logger = logging.getLogger(__name__)


def _truncated(name: str, limit: int) -> str:
    return name.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def recover_mesh(obj: Any, reader: Any, mesh: int, scale: float) -> dict[str, str]:
    """Match lost names only by a unique DNA delta match, never collection order.

    A modified/ambiguous old key is left intact and reported. Re-import or an
    explicit identity assignment is needed before it can be exported safely.
    """
    keys = obj.data.shape_keys
    if keys is None or len(keys.reference_key.data) != reader.getVertexPositionCount(mesh):
        return {}
    prefix = str(reader.getMeshName(mesh)) + "__"
    missing = {}
    for target in range(reader.getBlendShapeTargetCount(mesh)):
        channel = reader.getBlendShapeChannelIndex(mesh, target)
        full = prefix + str(reader.getBlendShapeChannelName(channel))
        if len(full.encode("utf-8")) > shape_names.LIMIT and shape_names.find_key(keys, full) is None:
            missing[full] = target
    if not missing:
        return {}
    candidates = {}
    for key in list(keys.key_blocks)[1:]:
        if shape_names.logical_name(keys, key.name) != key.name:
            continue
        suffix = re.search(r"\.\d{3,}$", key.name)
        stem = key.name[: suffix.start()] if suffix else key.name
        limit = shape_names.LIMIT - (len(suffix.group()) if suffix else 0)
        matches = [full for full in missing if stem == _truncated(full, limit)]
        if matches:
            candidates[key.name] = matches
    if not candidates:
        return {}
    basis = np.empty(len(keys.reference_key.data) * 3, dtype=np.float32)
    keys.reference_key.data.foreach_get("co", basis)
    basis = basis.reshape(-1, 3)
    expected = {}
    for full in {full for names in candidates.values() for full in names}:
        target = missing[full]
        coords = basis.copy()
        indices = np.asarray(reader.getBlendShapeTargetVertexIndices(mesh, target), dtype=np.intp)
        delta = np.asarray(
            [
                reader.getBlendShapeTargetDeltaXs(mesh, target),
                reader.getBlendShapeTargetDeltaYs(mesh, target),
                reader.getBlendShapeTargetDeltaZs(mesh, target),
            ],
            dtype=np.float32,
        ).T
        coords[indices] += delta * scale
        expected[full] = coords
    matches = {}
    values = np.empty(basis.size, dtype=np.float32)
    for old, names in candidates.items():
        keys.key_blocks[old].data.foreach_get("co", values)
        matches[old] = [full for full in names if np.allclose(values.reshape(-1, 3), expected[full], rtol=0, atol=2e-7)]
    renamed = {}
    for old, names in matches.items():
        if len(names) == 1 and sum(names[0] in other for other in matches.values()) == 1:
            renamed[old] = shape_names.rename_key(keys, keys.key_blocks[old], names[0], mesh=prefix[:-2])
        else:
            logger.warning("Cannot uniquely recover DNA identity for %s / %s; preserving its geometry", obj.name, old)
    return renamed


def migrate_instance(instance: Any) -> None:
    """Run only before rebuilding bindings, with no DNA file writes."""
    from ..bindings import enums
    from ..constants import SCALE_FACTOR

    # Snapshot old names before renaming so legacy surface provenance and drivers
    # can be updated without refreshing (and overwriting) an uncommitted sculpt.
    renamed = {}
    for kind in ("head", "body"):
        component = instance.component(kind)
        reader = component.dna_reader
        if reader is None:
            continue
        scale = 1 / SCALE_FACTOR if enums.TranslationUnit(reader.getTranslationUnit()).name.lower() == "cm" else 1.0
        for mesh, obj in component.mesh_index_lookup.items():
            changes = recover_mesh(obj, reader, mesh, scale)
            if changes:
                renamed[obj] = changes
    from ..utilities import editors_available

    if editors_available():
        from ..editors.converter.scene_merge import migrate_names

        migrate_names(instance, renamed)
    if renamed:
        for kind in ("head", "body"):
            for descriptor in (
                "shape_key_blocks",
                "shape_key_blocks_by_name",
                "rbf_preview_shape_plan",
                "shape_key_apply_plan",
                "shape_key_has_deltas",
            ):
                instance.data.pop(instance.cache_key(kind, descriptor), None)
        editor = getattr(instance, "shape_key_editor", None)
        selected = None
        if editor and 0 <= editor.shape_key_list_active_index < len(editor.shape_key_list):
            selected = editor.shape_key_list[editor.shape_key_list_active_index].name
        instance.sync_shape_key_list()
        if selected and (index := editor.shape_key_list.find(selected)) >= 0:
            editor["shape_key_list_active_index"] = index
    instance["dna_shape_name_version"] = 1
