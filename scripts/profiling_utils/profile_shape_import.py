"""Profile deferred shape import and its synchronous completion on disposable DNA copies.

Run with the repository's bpy-enabled Python. The live Blender scene is never
accessed. Optional cProfile output measures completion only, not scene setup.
"""

# The benchmark deliberately instruments private implementation details.
# ruff: noqa: SLF001

import argparse
import contextlib
import cProfile
import ctypes
import functools
import hashlib
import json
import os
import queue
import shutil
import sys
import tempfile
import traceback

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from profile_document_io import Recorder, digest, peak_memory, summarize


@contextlib.contextmanager
def instrument_steps(recorder: Recorder, targets: list) -> Iterator[None]:
    """Measure generator execution without charging caller work between yields."""
    originals = []
    for module, name in targets:
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapper(
            *args: Any, _fn: Callable = original, _label: str = module.__name__ + "." + name, **kwargs: Any
        ) -> Iterator:
            iterator = _fn(*args, **kwargs)
            try:
                while True:
                    with recorder.phase(_label):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            return
                    yield item
            finally:
                iterator.close()

        setattr(module, name, functools.wraps(original)(wrapper))
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def surface_digest(instance: Any) -> str:
    """Compare every merged key's name and coordinates between benchmark runs."""
    from character_dna.editors.converter import scene_merge

    result = hashlib.sha256()
    for lod, obj in sorted(scene_merge.surfaces(instance).items()):
        result.update(str(lod).encode())
        result.update(scene_merge._coordinates(obj.data.vertices).tobytes())
        for block in obj.data.shape_keys.key_blocks:
            result.update(block.name.encode())
            result.update(scene_merge._coordinates(block.data).tobytes())
    return result.hexdigest()


def run_import(instance: Any, recorder: Recorder, profile_path: Path | None) -> dict:
    """Run the real import queue and finish hook, separating UI-ready work."""
    import bpy

    from character_dna.dna_io import document
    from character_dna.editors.converter import scene_merge, surface_session
    from character_dna.editors.shape_key_editor.operators import ImportShapeKeys
    from character_dna.rig_instance import RigInstance
    from character_dna.runtime import controller, engine, retention

    operation = SimpleNamespace(_timer=None, _import_instance=None, _native_was_active=False)
    commands = queue.Queue()
    component = ImportShapeKeys.get_component(operation)
    with recorder.phase("import.prepare"):
        ImportShapeKeys.set_commands_queue(operation, bpy.context, component, commands)
    queued = commands.qsize()
    with recorder.phase("import.queue"):
        while not commands.empty():
            index, mesh, _description, kwargs, callback = commands.get()
            callback(**kwargs(index, mesh))
    print(f"Queue finished: {queued} commands. Profiling completion...", flush=True)
    targets = [
        (RigInstance, "head_initialize"),
        (RigInstance, "body_initialize"),
        (RigInstance, "sync_shape_key_list"),
        (surface_session, "refresh"),
        (surface_session, "_refresh_context"),
        (scene_merge, "scatter"),
        (document, "read_document"),
        (document, "restore_scene_bridges"),
        (document, "scene_extensions"),
        (controller, "resume"),
        (engine, "install"),
    ]
    steps = [
        (surface_session, "refresh_steps"),
        (surface_session, "_refresh_keys"),
        (retention, "bind_imported_shapes"),
    ]
    profiler = cProfile.Profile() if profile_path else None
    if profiler:
        profiler.enable()
    try:
        with recorder.instrument(targets), instrument_steps(recorder, steps), recorder.phase("import.finish"):
            ImportShapeKeys.finish(operation, bpy.context)
        with recorder.phase("import.depsgraph_after_finish"):
            bpy.context.view_layer.update()
        with recorder.phase("import.first_list_values"):
            for item in instance.shape_key_editor.shape_key_list:
                _ = item.value, item.has_deltas
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(profile_path))
    assert not engine.binding_issues(instance)
    return {
        "commands": queued,
        "ui_rows": len(instance.shape_key_editor.shape_key_list),
        "merged_surfaces": len(scene_merge.surfaces(instance)),
        "surface_geometry_sha256": surface_digest(instance),
        "component_keys": {
            str(index): len(obj.data.shape_keys.key_blocks) - 1 if obj.data.shape_keys else 0
            for index, obj in instance.edit_target.mesh_index_lookup.items()
        },
    }


def benchmark(args: argparse.Namespace, directory: Path, recorder: Recorder) -> list[dict]:
    """Create the requested scene mode from copies, import all keys, then release it."""
    import bpy

    from character_dna.utilities import get_active_rig_instance

    bpy.ops.preferences.addon_enable(module="character_dna")
    bpy.context.preferences.addons["character_dna"].preferences.backup_manager.enabled = False
    bpy.ops.wm.read_homefile(use_empty=True)
    try:
        for name in ("head.dna", "body.dna", "ExportManifest.json"):
            if (args.input_dir / name).is_file():
                shutil.copyfile(args.input_dir / name, directory / name)
        with recorder.phase("setup.import"):
            bpy.ops.character_dna.import_dna(
                filepath=str(directory / "head.dna"),
                include_body=True,
                import_face_board=args.face_board,
                import_mesh=True,
                import_bones=True,
                import_shape_keys=False,
                import_materials=False,
                **{f"import_lod{i}": i == 0 or args.all_lods for i in range(8)},
            )
        instance = get_active_rig_instance()
        instance.editor_component = "head"
        if args.scene_mode != "separate":
            instance.converter_merge_mode = "DNA" if args.scene_mode == "combined" else "SCENE"
            with recorder.phase("setup.merge"):
                assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
        return [
            run_import(
                instance,
                recorder,
                args.output.with_suffix(f".{index}.prof") if args.cprofile else None,
            )
            for index in range(args.repeats)
        ]
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def main() -> None:
    """Write timings and verify that source DNA bytes were never changed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-mode", choices=("separate", "scene", "combined"), default="scene")
    parser.add_argument("--all-lods", action="store_true")
    parser.add_argument("--face-board", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cprofile", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/addons"))
    import bpy

    originals = {name: digest(args.input_dir / name) for name in ("head.dna", "body.dna")}
    recorder = Recorder()
    result = {"blender": bpy.app.version_string, "options": {key: str(value) for key, value in vars(args).items()}}
    try:
        with tempfile.TemporaryDirectory(prefix="dna-shape-profile-") as root:
            result["details"] = benchmark(args, Path(root), recorder)
    finally:
        result["samples"] = recorder.rows
        result["summary"] = summarize(recorder.rows)
        result["memory"] = peak_memory()
        result["source_files_unchanged"] = all(
            digest(args.input_dir / name) == value for name, value in originals.items()
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        assert result["source_files_unchanged"]
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip bpy DLL shutdown, as in the existing document profiler.
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.TerminateProcess(kernel.GetCurrentProcess(), status)
    os._exit(status)
