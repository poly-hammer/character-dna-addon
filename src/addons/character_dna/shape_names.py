"""Lossless DNA shape identities at Blender's 63-byte name boundary.

Names are resolved when importing or building bindings, never while evaluating
those bindings. The persisted aliases contain strings only and survive Undo,
mesh copies and blend reloads without retaining RNA references in a global cache.
"""

import hashlib
import json

from functools import lru_cache
from typing import Any


PROPERTY = "dna_shape_names"
LIMIT = 63


@lru_cache(maxsize=256)
def _maps(serialized: str) -> tuple[dict[str, str], dict[str, str]]:
    aliases = json.loads(serialized)
    return aliases, {full: short for short, full in aliases.items()}


def logical_name(keys: Any, name: str) -> str:
    """Recover the complete DNA mesh/channel name, including any '__' in a channel."""
    return _maps(keys.get(PROPERTY, "{}"))[0].get(name, name) if keys else name


def stored_name(keys: Any, name: str) -> str:
    """Read an existing alias without allocating or changing scene data."""
    return _maps(keys.get(PROPERTY, "{}"))[1].get(name, name) if keys else name


def find_key(keys: Any, name: str) -> Any:
    """Resolve a logical DNA name. Use key_blocks.get for an actual Blender name."""
    return keys.key_blocks.get(stored_name(keys, name)) if keys else None


def mesh_name(name: str, catalog: Any) -> str | None:
    """Resolve ownership using exact known DNA mesh prefixes, longest first."""
    return next((mesh for mesh in sorted(catalog, key=len, reverse=True) if name.startswith(mesh + "__")), None)


def ensure_name(keys: Any, name: str, *, mesh: str | None = None) -> str:
    """Allocate a collision-checked alias, preserving the entire mesh prefix."""
    aliases, reverse = _maps(keys.get(PROPERTY, "{}"))
    if name in reverse:
        return reverse[name]
    if len(name.encode("utf-8")) <= LIMIT and (name not in aliases or aliases[name] == name):
        return name
    prefix = (mesh or name.partition("__")[0]) + "__"
    if not name.startswith(prefix):
        raise ValueError(f"Shape identity needs its DNA mesh prefix: {name}")
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    available = LIMIT - len((prefix + "h_").encode("utf-8"))
    for length in range(12, min(available, len(digest)) + 1):
        candidate = prefix + "h_" + digest[:length]
        if candidate not in aliases and keys.key_blocks.get(candidate) is None:
            keys[PROPERTY] = json.dumps({**aliases, candidate: name}, ensure_ascii=False)
            return candidate
    raise ValueError(f"Cannot safely encode the shape name with mesh prefix {prefix!r}.")


def rename_key(keys: Any, key: Any, name: str, *, mesh: str | None = None) -> str:
    """Rename without truncation; Blender updates animation paths on key rename."""
    candidate = ensure_name(keys, name, mesh=mesh)
    existing = keys.key_blocks.get(candidate)
    if existing is not None and existing != key:
        raise ValueError(f"Shape identity already exists: {name}")
    key.name = candidate
    if key.name != candidate:
        raise ValueError(f"Blender could not preserve the shape identity: {name}")
    return candidate


def copy_names(source: Any, destination: Any) -> None:
    """Restore the name registry with the geometry in an editor transaction."""
    if destination:
        destination[PROPERTY] = source.get(PROPERTY, "{}") if source else "{}"
