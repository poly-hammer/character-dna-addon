"""Inspect saved characters without running source scripts or addon startup."""

import argparse
import importlib.util
import json
import sys
import traceback

from pathlib import Path

import bpy


ADDON_IDS = ("meta_human_dna", "meta_human_dna_pro", "character_dna", "character_dna_pro")


class SavedInstance(bpy.types.PropertyGroup):
    """Expose persisted fields only, with no callbacks or runtime initialization."""


class SavedScene(bpy.types.PropertyGroup):
    """Read both generations of the saved instance list."""

    rig_instance_list: bpy.props.CollectionProperty(type=SavedInstance)  # pyright: ignore[reportInvalidTypeForm]
    rig_logic_instance_list: bpy.props.CollectionProperty(type=SavedInstance)  # pyright: ignore[reportInvalidTypeForm]


class SavedAssemblyScene(bpy.types.PropertyGroup):
    """Read Assembly's saved per-character records without enabling that addon."""

    rig_instance_proxies: bpy.props.CollectionProperty(type=SavedInstance)  # pyright: ignore[reportInvalidTypeForm]


def main() -> None:
    """Write a JSON descriptor; never enable an addon or save the opened blend."""
    import addon_utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--addon-folder")
    parser.add_argument("--addon-name")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    data_file = Path(args.data_file)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        for addon in list(bpy.context.preferences.addons.keys()):
            addon_utils.disable(addon, default_set=False)
        bpy.utils.register_class(SavedInstance)
        bpy.utils.register_class(SavedScene)
        bpy.utils.register_class(SavedAssemblyScene)
        bpy.types.Scene.character_assembly = bpy.props.PointerProperty(type=SavedAssemblyScene)
        for edition in ADDON_IDS:
            setattr(bpy.types.Scene, edition, bpy.props.PointerProperty(type=SavedScene))
        bpy.ops.wm.open_mainfile(filepath=args.blend_file, use_scripts=False)
        module_path = Path(__file__).resolve().parents[2] / "utilities" / "reference.py"
        spec = importlib.util.spec_from_file_location("character_reference_inspector", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = {}
        for scene in bpy.data.scenes:
            for edition in ADDON_IDS:
                group = getattr(scene, edition)
                for instance in (*group.rig_instance_list, *group.rig_logic_instance_list):
                    name = instance.get("name") or instance.get("instance_name")
                    if not name:
                        continue
                    if name in data:
                        data[name]["issues"].append("Ambiguous character name across saved scenes or editions")
                    else:
                        data[name] = module.describe_instance(scene, edition, instance)
        data_file.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        data_file.with_name(f"{data_file.stem}_error.log").write_text(traceback.format_exc(), encoding="utf-8")


if __name__ == "__main__":
    main()
