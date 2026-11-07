"""Background compute and cosmetic playback have independent geometry ownership."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.editors.converter import viewport_playback
from character_dna.editors.converter.lifecycle import ConversionLifecycle
from character_dna.editors.converter.pipeline import Transition, positions, write_positions


@pytest.fixture
def preview_scene(monkeypatch, request):
    # The native probe injects the actual extension. Standalone Blender tests
    # exercise the display driver with an interpolation test double; the extension
    # separately proves interpolation against NumPy, including exact endpoints.
    if request.config.getoption("--converter-native", default=None) is None:

        def interpolate(before, after, factor):
            if factor == 0:
                return before.tobytes()
            if factor == 1:
                return after.tobytes()
            return (before.astype(np.float64) * (1 - factor) + after * factor).astype(np.float32).tobytes()

        monkeypatch.setattr(viewport_playback, "_core", lambda: SimpleNamespace(interpolate_positions=interpolate))
    data = bpy.data.meshes.new("preview contract")
    data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new("preview contract", data)
    obj["dna_mesh_name"] = "must not appear on preview"
    bpy.context.scene.collection.objects.link(obj)
    pipeline = SimpleNamespace(objects={"body": obj}, operator=SimpleNamespace(_components={}), report={"stages": []})
    playback = viewport_playback.ViewportPlayback(bpy.context)
    try:
        yield obj, pipeline, playback
    finally:
        playback.close()
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)


def test_compute_advances_before_display_and_reads_only_committed_results(preview_scene):
    obj, pipeline, playback = preview_scene
    before = positions(obj)
    target = before + 2
    transition = Transition("Copy observations")
    transition.add_mesh(obj, target)
    captured, polls, published = [], [], []
    job = SimpleNamespace(label="Background work", previews=[], poll=lambda: polls.append(True) or False)

    def stages():
        yield transition
        captured.append(positions(obj))
        yield job

    operator = SimpleNamespace(
        _compute_done=False,
        _transition=None,
        _stages=stages(),
        _pipeline=pipeline,
        _playback=playback,
        _reporter=SimpleNamespace(update=lambda *_args, **_kwargs: None),
        _finalize=lambda _context: published.append(True),
    )
    for tick in range(3):
        assert ConversionLifecycle._background_tick(operator, bpy.context, tick * 0.01) == {"RUNNING_MODAL"}
    assert captured and polls
    np.testing.assert_array_equal(captured[0], target)
    np.testing.assert_array_equal(positions(obj), target)
    preview = playback.objects[obj]
    assert preview.data is not obj.data
    assert "dna_mesh_name" not in preview
    assert not np.array_equal(positions(preview), target)
    assert obj.hide_get()
    assert not published
    playback.tick(1.0)
    np.testing.assert_array_equal(positions(preview), target)
    np.testing.assert_array_equal(positions(obj), target)


def test_preview_pressure_retains_exact_final_geometry_for_every_mesh(preview_scene):
    obj, pipeline, playback = preview_scene
    playback.max_frames = 2
    playback.max_bytes = positions(obj).nbytes * 2
    playback.capture(pipeline)
    for index in range(12):
        playback.enqueue(str(index), [(obj, positions(obj) + index)])
    assert len(playback.frames) <= 2
    assert playback.pending_bytes <= playback.max_bytes
    assert playback.dropped_frames == 10
    exact = positions(obj) + np.float32(1.234567)
    write_positions(obj, exact)
    # Even an endpoint too large for the cosmetic queue must be applied exactly.
    playback.max_bytes = 1
    playback.finish(pipeline)
    for tick in range(10):
        if playback.tick(float(tick)):
            break
    else:
        pytest.fail("Playback did not drain")
    np.testing.assert_array_equal(positions(playback.objects[obj]), exact)
    np.testing.assert_array_equal(positions(obj), exact)


def test_early_worker_preview_cannot_commit_a_conversion_stage(preview_scene, monkeypatch):
    obj, pipeline, playback = preview_scene
    original = positions(obj)
    phases = []
    commit = Transition.commit

    def observe(transition):
        phases.append(transition.label)
        commit(transition)

    monkeypatch.setattr(Transition, "commit", observe)
    playback.capture(pipeline)
    job = SimpleNamespace(previews=[(1, "Template surface", original + 1, "body")])
    playback.job(pipeline, job)
    assert not playback.tick(0.0)
    assert playback.tick(1.0)
    assert phases == []
    np.testing.assert_array_equal(positions(obj), original)
    np.testing.assert_array_equal(positions(playback.objects[obj]), original + 1)
    accepted = Transition("Fit supplied surfaces")
    accepted.add_mesh(obj, original + 2)
    accepted.commit()
    assert phases == ["Fit supplied surfaces"]
    np.testing.assert_array_equal(positions(obj), original + 2)


def test_cancel_removes_preview_datablocks_and_restores_output_visibility(preview_scene):
    obj, pipeline, playback = preview_scene
    original = positions(obj)
    objects, meshes, collections = set(bpy.data.objects), set(bpy.data.meshes), set(bpy.data.collections)
    playback.capture(pipeline)
    playback.enqueue("Interrupted", [(obj, original + 10)])
    playback.tick(0.0)
    playback.tick(0.1)
    assert not np.array_equal(positions(playback.objects[obj]), original)
    playback.close()
    np.testing.assert_array_equal(positions(obj), original)
    assert not obj.hide_get()
    assert set(bpy.data.objects) == objects
    assert set(bpy.data.meshes) == meshes
    assert set(bpy.data.collections) == collections


def test_fast_compute_finishes_while_snapshots_are_still_playing(preview_scene):
    obj, pipeline, playback = preview_scene
    transition = Transition("Fast conversion")
    transition.add_mesh(obj, positions(obj) + 3)
    published = []
    operator = SimpleNamespace(
        _compute_done=False,
        _transition=None,
        _stages=iter([transition]),
        _pipeline=pipeline,
        _playback=playback,
        _reporter=SimpleNamespace(update=lambda *_args, **_kwargs: None),
        _finalize=lambda _context: published.append(True),
    )
    for tick in range(3):
        ConversionLifecycle._background_tick(operator, bpy.context, tick * 0.01)
    assert operator._compute_done
    assert not published
    for tick in range(1, 8):
        result = ConversionLifecycle._background_tick(operator, bpy.context, float(tick))
        if result == {"FINISHED"}:
            break
    assert result == {"FINISHED"}
    assert published == [True]
    assert operator._playback is None
    assert not obj.hide_get()


def test_enter_skips_only_display_and_keeps_the_exact_final_result(preview_scene):
    obj, pipeline, playback = preview_scene
    playback.capture(pipeline)
    before = positions(obj)
    playback.enqueue("Pending preview", [(obj, before + 5)])
    playback.tick(0.0)
    operator = SimpleNamespace(_playback=playback)
    assert ConversionLifecycle.modal(operator, bpy.context, SimpleNamespace(type="RET")) == {"RUNNING_MODAL"}
    assert playback.skipped and not playback.frames and playback.active is None
    np.testing.assert_array_equal(positions(obj), before)
    final = before + np.float32(0.234567)
    write_positions(obj, final)
    playback.finish(pipeline)
    assert playback.tick(1.0)
    np.testing.assert_array_equal(positions(playback.objects[obj]), final)
    np.testing.assert_array_equal(positions(obj), final)
