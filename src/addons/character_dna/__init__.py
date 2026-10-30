"""Addon metadata and entry points; domain modules can be used without Blender."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .rig_instance import RigInstance

bl_info = {
    "name": "Character DNA",
    "author": "Poly Hammer",
    "version": (0, 15, 3),
    "blender": (4, 5, 0),
    "location": "File > Import > MetaHuman DNA",
    "description": (
        "Imports MetaHuman head and body components from a their DNA files, "
        "lets you customize them, then send them back to MetaHuman Creator."
    ),
    "warning": "",
    "tracker_url": "https://github.com/orgs/poly-hammer/discussions/categories/character-dna-addon",
    "wiki_url": "https://docs.polyhammer.com/character-dna-addon/",
    "category": "Poly Hammer",
}

# Shared with integrations that listen for scene/instance setup and removal.
post_setup_scene_callbacks: list[Callable[["RigInstance | None"], None]] = []


def register():
    from . import registration

    registration.register()


def unregister():
    from . import registration

    registration.unregister()


def __getattr__(name: str) -> Any:
    # Keep the public registration surfaces used by integrations and reloads.
    if name in {"native_runtime", "classes", "app_handlers"}:
        from . import registration

        return getattr(registration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
