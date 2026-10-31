import bpy
import pytest

from character_dna.ui.callbacks import get_active_rig_instance


FRAMES = range(1, 5)


@pytest.fixture
def freshly_imported_character(load_full_dna_for_animation):
    """A scene holding only this character.

    Rendering evaluates every rig instance present, so sharing the session scoped import would
    mean rendering whatever the previous test left behind, linked library data included.
    """
    yield


def _key_face_board_control(face_board, control_name: str, axis: str) -> tuple[bpy.types.Action, dict[int, float]]:
    """Key one face board control with a distinct value per frame.

    Goes through ``keyframe_insert`` rather than building f-curves by hand, since the slotted
    action API that would need is not the same on 4.5 as it is on 5.x. Every frame in the range
    is keyed, so the asserted values do not depend on the interpolation mode either.
    """
    pose_bone = face_board.pose.bones[control_name]
    index = "xyz".index(axis)

    if face_board.animation_data:
        face_board.animation_data.action = None

    values = {}
    for frame in FRAMES:
        values[frame] = round(0.1 * frame, 4)
        pose_bone.location[index] = values[frame]
        pose_bone.keyframe_insert(data_path="location", index=index, frame=frame)

    action = face_board.animation_data.action
    action.name = "render_sync_probe"
    return action, values


def test_rig_logic_reads_the_rendered_frame(freshly_imported_character, tmp_path):
    """Rig logic must consume the face board pose of the frame being rendered.

    Blender renders through its own dependency graph and never flushes the animated pose back
    to the original datablock, so reading ``face_board.pose`` directly leaves rig logic a frame
    behind (or frozen) for the whole render.
    """
    instance = get_active_rig_instance()
    scene = bpy.context.scene

    control_index, control_name, axis = next(
        (index, name, control_axis)
        for index, name, control_axis in instance.head_gui_control_plan
        if instance.face_board.pose.bones.get(name) and not name.endswith("_eye")
    )

    previous_action = instance.face_board.animation_data.action if instance.face_board.animation_data else None
    cycles = getattr(scene, "cycles", None)
    previous = {
        "engine": scene.render.engine,
        "resolution": (scene.render.resolution_x, scene.render.resolution_y),
        "filepath": scene.render.filepath,
        "range": (scene.frame_start, scene.frame_end),
        "camera": scene.camera,
        "frame": scene.frame_current,
        "samples": getattr(cycles, "samples", None),
        "device": getattr(cycles, "device", None),
    }
    probe_action, expected = _key_face_board_control(instance.face_board, control_name, axis)
    camera = bpy.data.objects.new("render_sync_camera", bpy.data.cameras.new("render_sync_camera"))
    observed: list[tuple[int, float]] = []

    def record(current_scene, _depsgraph):
        # Appended last, so it runs after the addon's frame change handler has evaluated.
        observed.append((current_scene.frame_current, round(instance.head_instance.getGUIControl(control_index), 4)))

    try:
        scene.collection.objects.link(camera)
        scene.camera = camera
        # Cycles on the CPU renders without a GPU draw context, which the headless CI runners do
        # not have. Workbench aborts there.
        scene.render.engine = "CYCLES"
        if cycles:
            cycles.device = "CPU"
            cycles.samples = 1
        scene.render.resolution_x = 8
        scene.render.resolution_y = 8
        scene.render.filepath = str(tmp_path / "frame_")
        scene.frame_start = FRAMES.start
        scene.frame_end = FRAMES.stop - 1

        bpy.app.handlers.frame_change_post.append(record)
        # Park on the first frame so a fully stale result is distinguishable from a one frame lag.
        scene.frame_set(FRAMES.start)
        observed.clear()

        bpy.ops.render.render(animation=True)
    finally:
        if record in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(record)
        bpy.data.objects.remove(camera, do_unlink=True)
        scene.render.engine = previous["engine"]
        if cycles:
            cycles.samples = previous["samples"]
            cycles.device = previous["device"]
        scene.render.resolution_x, scene.render.resolution_y = previous["resolution"]
        scene.render.filepath = previous["filepath"]
        scene.frame_start, scene.frame_end = previous["range"]
        scene.camera = previous["camera"]
        if instance.face_board.animation_data:
            instance.face_board.animation_data.action = previous_action
        bpy.data.actions.remove(probe_action)
        scene.frame_set(previous["frame"])

    # Blender restores the frame after the render, which fires the handler again; keep the first
    # observation per frame so that trailing call cannot mask the rendered value.
    first_seen: dict[int, float] = {}
    for frame, value in observed:
        first_seen.setdefault(frame, value)

    assert set(first_seen) == set(FRAMES), f"handler did not run for every rendered frame: {observed}"
    for frame in FRAMES:
        assert first_seen[frame] == pytest.approx(expected[frame], abs=1e-3), (
            f"frame {frame}: rig logic saw {first_seen[frame]} but the frame's pose is {expected[frame]}"
        )
