"""Real Shape Key Editor lifecycle workload for profile_document_io.py."""

import shutil

from argparse import Namespace
from pathlib import Path
from typing import Any


def benchmark(args: Namespace, root: Path, recorder: Any) -> dict:
    import bpy

    from character_dna.dna_io.misc import create_shape_key, get_dna_reader, release_dna_handle
    from character_dna.editors.backup_manager import core as backup_core
    from character_dna.editors.shape_key_editor import core
    from character_dna.editors.shape_key_editor.editor import ShapeKeyEditor
    from character_dna.runtime import controller
    from character_dna.utilities import get_active_rig_instance

    bpy.ops.preferences.addon_enable(module="character_dna")
    bpy.context.preferences.addons["character_dna"].preferences.backup_manager.enabled = args.backups
    bpy.ops.wm.read_homefile(use_empty=True)
    for name in ("head.dna", "body.dna", "Ada.dna", "ExportManifest.json"):
        if (args.input_dir / name).is_file():
            shutil.copyfile(args.input_dir / name, root / name)
    filepath = root / ("Ada.dna" if args.scene_mode == "combined" else "head.dna")
    with recorder.phase("editor.setup.import"):
        bpy.ops.character_dna.import_dna(
            filepath=str(filepath),
            include_body=True,
            import_face_board=False,
            import_mesh=True,
            import_bones=True,
            import_shape_keys=False,
            import_materials=False,
            **{f"import_lod{i}": i == 0 or args.all_lods for i in range(8)},
        )
    instance = get_active_rig_instance()
    if hasattr(instance, "editor_component"):
        instance.editor_component = "head"
    mesh = instance.head_mesh
    reader = instance.head_dna_reader
    mesh_index = next(i for i, obj in instance.head_mesh_index_lookup.items() if obj == mesh)
    channel = reader.getBlendShapeChannelIndex(mesh_index, 0)
    short_name = reader.getBlendShapeChannelName(channel)
    prefix = reader.getMeshName(mesh_index) + "__"
    with controller.authoring_operation(instance):
        create_shape_key(0, mesh_index, mesh, reader, short_name, prefix=prefix, linear_modifier=0.01)
        instance.destroy_references()
        instance.sync_shape_key_list()
    if args.all_keys:
        import queue

        from types import SimpleNamespace

        from character_dna.editors.shape_key_editor.operators import ImportShapeKeys

        operation = SimpleNamespace(_timer=None, _import_instance=None, _native_was_active=False)
        commands = queue.Queue()
        component = ImportShapeKeys.get_component(operation)
        with recorder.phase("editor.setup.import_all_shapes"):
            ImportShapeKeys.set_commands_queue(operation, bpy.context, component, commands)
            while not commands.empty():
                index, mesh_id, _description, kwargs, callback = commands.get()
                callback(**kwargs(index, mesh_id))
            ImportShapeKeys.finish(operation, bpy.context)
    if args.scene_mode == "scene":
        assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    elif args.scene_mode == "combined":
        from character_dna.editors.converter import surface_session

        # Direct single-key creation above stands in for the UI importer, whose
        # finish hook also refreshes the merged surface's key signatures.
        with controller.authoring_operation(instance):
            surface_session.refresh(instance)
    targets = [
        (core, "write_shape_key_to_dna"),
        (backup_core, "create_backup"),
        (controller, "resume"),
        (controller, "rebuild"),
    ]
    if args.scene_mode != "release":
        from character_dna.dna_io import document, merged_graph, metadata
        from character_dna.editors.converter import scene_merge, surface_session

        targets += [
            (document, name)
            for name in ("write_document", "publish_scene_bridges", "collect_scene_bridges", "read_document")
        ]
        targets += [(merged_graph, name) for name in ("from_bytes", "compose", "to_bytes", "binary_digest")]
        targets += [(metadata, name) for name in ("encode", "decode")]
        targets += [(scene_merge, name) for name in ("snapshot_sources", "apply_edits")]
        targets += [(surface_session, name) for name in ("finish", "refresh")]
        from character_dna.dna_io import misc

        if args.capture_json:
            original_stream = misc._json_input_stream

            def capture_stream(path, resource):
                if not args.capture_json.exists():
                    shutil.copyfile(path, args.capture_json)
                return original_stream(path, resource)

            misc._json_input_stream = capture_stream
        targets.append((misc, "_json_input_stream"))
        try:
            from character_dna.runtime import retention

            targets.extend(
                (retention, name) for name in ("resume", "_curves", "_scene_signature", "_read", "_set_mutes")
            )
        except ImportError:
            pass
    details = {
        "scene_mode": args.scene_mode,
        "all_lods": args.all_lods,
        "all_keys": args.all_keys,
        "backups": args.backups,
        "shape": prefix + short_name,
    }
    for iteration in range(args.repeats):
        from character_dna.runtime import engine

        before = {item["component"]: item["token"] for item in engine.carriers(instance)}
        instance.shape_key_editor.shape_key_list_active_index = instance.shape_key_editor.shape_key_list.find(
            prefix + short_name
        )
        editor = ShapeKeyEditor.for_instance(instance)
        with recorder.phase("editor.enter"):
            editor.enter(bpy.context)
        bpy.ops.object.mode_set(mode="OBJECT")
        edited_object = bpy.context.object
        # A merged surface stores body vertices first. Edit the same canonical
        # head vertex in every workload, rather than accidentally testing a
        # new cross-component corrective on the body's first vertex.
        head_ids = edited_object.data.attributes.get("DNA_head_vertex")
        vertex = next(i for i, item in enumerate(head_ids.data) if item.value == 0) if head_ids else 0
        edited_object.active_shape_key.data[vertex].co.x += 0.001
        with recorder.instrument(targets), recorder.phase("editor.commit"):
            editor.commit(bpy.context)
        after = {item["component"]: item["token"] for item in engine.carriers(instance)}
        details.setdefault("binding_reused", []).append(before == after)
        assert not instance.shape_key_editor.is_editing
        check = get_dna_reader(Path(instance.head_dna_file_path))
        try:
            target_index = core.find_blend_shape_target_index(check, mesh_index, short_name)
            indices = list(check.getBlendShapeTargetVertexIndices(mesh_index, target_index))
            assert 0 in indices
            # Verify persisted shape against the canonical scene key, including merged gather.
            expected_ids, expected_delta = core.compute_dna_deltas(
                mesh, mesh.data.shape_keys.key_blocks.get(prefix + short_name), core.linear_modifier_for(check)
            )
            import numpy as np

            np.testing.assert_array_equal(indices, expected_ids)
            np.testing.assert_allclose(
                check.getBlendShapeTargetDeltaXs(mesh_index, target_index),
                np.asarray(expected_delta)[:, 0],
                atol=1e-5,
                rtol=0,
            )
        finally:
            release_dna_handle(check)
        print(f"Completed {args.scene_mode} editor commit {iteration + 1}", flush=True)
    details["persisted_shape_check"] = True
    details["head_keys_loaded"] = len(mesh.data.shape_keys.key_blocks) - 1
    bpy.ops.wm.read_homefile(use_empty=True)
    return details
