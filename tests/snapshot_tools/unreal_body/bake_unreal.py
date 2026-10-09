"""Run bake(manifest, output_directory) inside the Unreal Editor Python interpreter."""

import hashlib
import json
import math

from pathlib import Path

import unreal


def _pose_options(mesh):
    options = unreal.AnimPoseEvaluationOptions()
    options.optional_skeletal_mesh = mesh
    options.should_retarget = False
    options.evaluation_type = unreal.AnimDataEvalType.RAW
    return options


def _new_clip(name, package, mesh):
    clip = unreal.load_asset(f"{package}/{name}")
    if clip is None:
        factory = unreal.AnimSequenceFactory()
        factory.target_skeleton = mesh.skeleton
        factory.preview_skeletal_mesh = mesh
        clip = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, package, unreal.AnimSequence, factory)
    clip.set_retarget_source_asset(mesh)
    clip.update_retarget_source_asset_data()
    compression = unreal.load_asset(f"{package}/ABC_RBF_Snapshots")
    if compression is None:
        compression = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "ABC_RBF_Snapshots",
            package,
            unreal.AnimBoneCompressionSettings,
            unreal.AnimBoneCompressionSettingsFactory(),
        )
        codec_class = unreal.load_class(None, "/Script/Engine.AnimCompress_BitwiseCompressOnly")
        codec = unreal.new_object(codec_class, outer=compression)
        compression.set_editor_property("codecs", [codec])
        unreal.EditorAssetLibrary.save_loaded_asset(compression)
    clip.set_editor_property("bone_compression_settings", compression)
    return clip


def _write_inputs(clip, mesh, manifest):
    controller = clip.controller
    controller.open_bracket("Author DNA driver poses", False)
    try:
        controller.remove_all_bone_tracks(False)
        controller.set_frame_rate(unreal.FrameRate(manifest["fps"], 1), False)
        # A guard key prevents Sequencer wrapping the last real pose to time zero.
        controller.set_number_of_frames(unreal.FrameNumber(len(manifest["frames"])), False)
        for joint in manifest["joints"]:
            name = joint["name"]
            neutral = unreal.Quat(*joint["unreal_rotation_xyzw"])
            rotations = []
            for frame in [*manifest["frames"], manifest["frames"][-1]]:
                x, y, z, w = frame["drivers"].get(name, [0, 0, 0, 1])
                rotations.append(neutral * unreal.Quat(-x, y, -z, w))
            controller.add_bone_curve(name, False)
            assert controller.set_bone_track_keys(
                name,
                [unreal.Vector(*joint["unreal_translation_cm"])] * len(rotations),
                rotations,
                [unreal.Vector(1, 1, 1)] * len(rotations),
                False,
            ), name
        # STEP may sample the previous frame at floating point frame boundaries.
        clip.set_editor_property("interpolation", unreal.AnimInterpolationType.LINEAR)
    finally:
        controller.close_bracket(False)
    reference = clip.get_anim_pose_at_frame(0, _pose_options(mesh))
    for joint in manifest["joints"]:
        transform = unreal.AnimPoseExtensions.get_ref_bone_pose(reference, joint["name"], unreal.AnimPoseSpaces.LOCAL)
        actual = [transform.translation.x, transform.translation.y, transform.translation.z]
        assert math.dist(actual, joint["unreal_translation_cm"]) < 0.001, f"DNA/mesh bind mismatch: {joint['name']}"
        rotation = transform.rotation
        dot = abs(sum(getattr(rotation, c) * v for c, v in zip("xyzw", joint["unreal_rotation_xyzw"], strict=True)))
        assert math.degrees(2 * math.acos(min(1, dot))) < 0.05, f"DNA/mesh rotation mismatch: {joint['name']}"
    unreal.EditorAssetLibrary.save_loaded_asset(clip)


def _bake_sequence(sequence, baked, binding, manifest):
    options = unreal.AnimSeqExportOption()
    settings = {
        "export_transforms": True,
        "export_morph_targets": False,
        "export_attribute_curves": False,
        "export_material_curves": False,
        "record_in_world_space": False,
        "evaluate_all_skeletal_mesh_components": True,
        "transact_recording": False,
        "use_custom_frame_rate": True,
        "custom_frame_rate": unreal.FrameRate(manifest["fps"], 1),
        "use_custom_time_range": True,
        "custom_display_rate": unreal.FrameRate(manifest["fps"], 1),
        "custom_start_frame": unreal.FrameNumber(0),
        "custom_end_frame": unreal.FrameNumber(len(manifest["frames"]) - 1),
        "warm_up_frames": unreal.FrameNumber(2),
    }
    for name, value in settings.items():
        options.set_editor_property(name, value)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    assert unreal.SequencerTools.export_anim_sequence(world, sequence, baked, options, binding, False)
    unreal.EditorAssetLibrary.save_loaded_asset(baked)


def _audit(baked, clip, mesh, manifest, output):
    """Record an independent Unreal reference for validating the FBX extraction."""
    frames = []
    max_driver_angle = 0.0
    options = _pose_options(mesh)
    for frame in manifest["frames"]:
        pose = baked.get_anim_pose_at_frame(frame["frame"], options)
        source = clip.get_anim_pose_at_frame(frame["frame"], options)
        locations = {}
        for joint in manifest["joints"]:
            name = joint["name"]
            transform = unreal.AnimPoseExtensions.get_bone_pose(pose, name, unreal.AnimPoseSpaces.WORLD)
            value = transform.translation
            locations[name] = [value.x / 100, -value.y / 100, value.z / 100]
        for name in frame["drivers"]:
            actual = unreal.AnimPoseExtensions.get_bone_pose(pose, name, unreal.AnimPoseSpaces.LOCAL).rotation
            expected = unreal.AnimPoseExtensions.get_bone_pose(source, name, unreal.AnimPoseSpaces.LOCAL).rotation
            dot = abs(sum(getattr(actual, c) * getattr(expected, c) for c in "xyzw"))
            angle = math.degrees(2 * math.acos(min(1, dot)))
            max_driver_angle = max(max_driver_angle, angle)
            assert angle < 0.01, (frame["frame"], name, angle)
        frames.append(locations)
    (output / "unreal_world_audit.json").write_text(json.dumps(frames), encoding="utf-8")
    return max_driver_angle


def bake(
    manifest_path, output_directory, blueprint="/Game/MetaHumans/Ada/BP_Ada", package="/Game/CharacterDNATests/AdaRBF"
):
    """Create driver/baked clips and a Blueprint preview sequence at LOD0."""
    manifest_path, output = Path(manifest_path), Path(output_directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = next((a for a in actors.get_all_level_actors() if "CharacterDNARBFTestCapture" in a.tags), None)
    if actor is None:
        actor = actors.spawn_actor_from_class(unreal.load_asset(blueprint).generated_class(), unreal.Vector(0, 0, 0))
        actor.tags = ["CharacterDNARBFTestCapture"]
        actor.set_actor_label("Ada_RBFSnapshotCapture")
    try:
        body = next(c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name() == "Body")
        mesh = body.skeletal_mesh_asset
        body.set_forced_lod(1)  # SkeletalMeshComponent uses 1 for LOD0.
        body.set_update_animation_in_editor(True)
        body.set_editor_property("disable_post_process_blueprint", False)
        for component in actor.get_components_by_class(unreal.LODSyncComponent):
            component.set_editor_property("forced_lod", 0)
        clip = _new_clip("AS_Ada_RBF_DriverPoses", package, mesh)
        _write_inputs(clip, mesh, manifest)
        baked = _new_clip("AS_Ada_RBF_UnrealBaked", package, mesh)
        sequence = unreal.load_asset(f"{package}/LS_Ada_RBF_Snapshots")
        if sequence is None:
            sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                "LS_Ada_RBF_Snapshots", package, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
            )
        for binding in sequence.get_bindings():
            binding.remove()
        sequence.set_display_rate(unreal.FrameRate(manifest["fps"], 1))
        sequence.set_playback_start(0)
        sequence.set_playback_end(len(manifest["frames"]))
        binding = sequence.add_possessable(actor)
        # Editing LODSync can rerun Blueprint construction and replace components.
        body = next(c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name() == "Body")
        body.set_forced_lod(1)
        body.set_update_animation_in_editor(True)
        body.set_editor_property("disable_post_process_blueprint", False)
        child = sequence.add_possessable(body)
        child.set_parent(binding)
        section = child.add_track(unreal.MovieSceneSkeletalAnimationTrack).add_section()
        section.set_range(0, len(manifest["frames"]))
        params = section.get_editor_property("params")
        params.animation = clip
        section.set_editor_property("params", params)
        unreal.EditorAssetLibrary.save_loaded_asset(sequence)
        _bake_sequence(sequence, baked, child, manifest)
        max_driver_angle = _audit(baked, clip, mesh, manifest, output)
        task = unreal.AssetExportTask()
        task.object = baked
        task.filename = str(output / "ada_unreal_rbf.fbx")
        task.automated = True
        task.prompt = False
        task.replace_identical = True
        task.exporter = unreal.AnimSequenceExporterFBX()
        task.options = unreal.FbxExportOption()
        task.options.export_preview_mesh = False
        task.options.force_front_x_axis = False
        task.options.ascii = False
        assert unreal.Exporter.run_asset_export_task(task)
        provenance = {
            "engine_version": unreal.SystemLibrary.get_engine_version(),
            "blueprint": blueprint,
            "mesh": mesh.get_path_name(),
            "post_process": mesh.post_process_anim_blueprint.get_path_name(),
            "input_animation": clip.get_path_name(),
            "baked_animation": baked.get_path_name(),
            "level_sequence": sequence.get_path_name(),
            "lod": 0,
            "maximum_baked_driver_error_degrees": max_driver_angle,
            "inputs_sha256": hashlib.sha256(manifest_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
            "fbx_sha256": hashlib.sha256(Path(task.filename).read_bytes()).hexdigest(),
        }
        (output / "capture.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        return provenance
    except BaseException:
        actors.destroy_actor(actor)
        raise
