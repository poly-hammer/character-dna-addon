"""Shape import stays responsive and retains unchanged native rig outputs."""

import queue

from types import MethodType, SimpleNamespace

import bpy
import pytest

from test_merge_split import import_shape_keys
from test_runtime_retention import identities

from character_dna import shape_names
from character_dna.editors.converter import scene_merge
from character_dna.editors.shape_key_editor import utilities
from character_dna.editors.shape_key_editor.operators import ImportShapeKeys
from character_dna.runtime import controller, engine, retention
from fixtures.compact_dna import STATE_BONES
from fixtures.rigs import RigProfile


@pytest.fixture
def character(request, rig_factory, tmp_path):
    mode = getattr(request.node, "callspec", None)
    mode = mode.params.get("mode", "separate") if mode else "separate"
    lods = tuple(range(8)) if getattr(request, "param", None) == "all_lods" else (0,)
    marker = request.node.get_closest_marker("rig_profile")
    options = (
        {}
        if marker and marker.kwargs.get("full_geometry")
        else {"asset": str(request.getfixturevalue("compact_dna")), "pose_bones": STATE_BONES}
    )
    with rig_factory.lease(RigProfile(document_mode=mode, lods=lods, **options), tmp_path) as instance:
        yield instance


@pytest.mark.parametrize(
    "mode", [pytest.param("separate", marks=pytest.mark.rig_profile(full_geometry=True)), "scene", "combined"]
)
def test_import_reuses_bones_and_evaluates_shapes(character, mode, monkeypatch):
    instance = character
    assert instance.components_merged == (mode != "separate")
    before = identities(instance)
    curves = [curve.as_pointer() for _owner, curve, _mute in retention._curves(engine.carriers(instance))]
    for _repeat in range(2):
        # Reimport replaces Key datablocks, but must not replace bone drivers/models.
        import_shape_keys(instance)
        assert identities(instance) == before
        assert set(curves) <= {
            curve.as_pointer() for _owner, curve, _mute in retention._curves(engine.carriers(instance))
        }
        assert not controller.is_suspended(instance)
        assert not retention.retained(instance)
        assert not engine.binding_issues(instance)
        reader = instance.head_dna_reader
        channel = reader.getBlendShapeChannelIndex(0, 0)
        name = f"head_lod0_mesh__{reader.getBlendShapeChannelName(channel)}"
        assert engine.preview_shape(instance, channel, 0.73)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        source = instance.head_mesh.data.shape_keys.evaluated_get(graph)
        assert shape_names.find_key(source, name).value == pytest.approx(0.73, abs=1e-5)
        if proxy := scene_merge.proxy_for(instance):
            visible = proxy.data.shape_keys.evaluated_get(graph)
            assert shape_names.find_key(visible, name).value == pytest.approx(0.73, abs=1e-5)
            metadata = scene_merge.read_metadata(proxy)
            for kind in scene_merge.KINDS:
                assert metadata["sources"][kind]["shape_signature"] == scene_merge._shape_signature(
                    proxy[f"dna_merge_{kind}"]
                )
    expected = utilities.compute_head_shape_key_deltas(instance)
    actual = instance.data[instance.cache_key("head", "shape_key_has_deltas")]
    assert all(actual[name] == value for name, value in expected.items())
    monkeypatch.setattr(
        utilities, "compute_head_shape_key_deltas", lambda *_: pytest.fail("Redraw rescanned shape data")
    )
    assert all(item.has_deltas == actual[item.name] for item in instance.shape_key_editor.shape_key_list)


def test_modal_cancel_finishes_partial_import_before_unlocking(character):
    instance = character
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    operation = SimpleNamespace(_timer=None, _commands_queue=queue.Queue(), _time_budget=0.0001, reports=[])
    operation.finish = MethodType(ImportShapeKeys.finish, operation)
    operation.report = lambda level, text: operation.reports.append((level, text))
    ImportShapeKeys.set_commands_queue(
        operation, bpy.context, ImportShapeKeys.get_component(operation), operation._commands_queue
    )
    # An interrupted import may already have replaced Basis and several Keys.
    for _ in range(4):
        index, mesh, _, kwargs, callback = operation._commands_queue.get()
        callback(**kwargs(index, mesh))
    progress = bpy.context.window_manager.character_dna
    progress.progress = 0.25
    assert ImportShapeKeys.modal(operation, bpy.context, SimpleNamespace(type="ESC")) == {"RUNNING_MODAL"}
    samples = []
    for _ in range(10000):
        state = ImportShapeKeys.modal(operation, bpy.context, SimpleNamespace(type="TIMER"))
        if state == {"FINISHED"}:
            break
        samples.append(progress.progress)
        assert progress.progress < 1
        assert controller.is_suspended(instance)
    else:
        pytest.fail("Import finalization never completed")
    assert len(samples) > 3
    assert samples == sorted(samples)
    assert progress.progress == 1
    assert not operation.reports
    assert not controller.is_suspended(instance)
    assert not engine.binding_issues(instance)


@pytest.mark.rig_profile(full_geometry=True)
def test_surface_uv_fallback_rejects_ambiguous_islands(character):
    instance = character
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    proxy, source = scene_merge.proxy_for(instance), instance.head_mesh
    coordinates = scene_merge._basis(source)
    proxy.data.attributes.remove(proxy.data.attributes["DNA_head_vertex"])
    with pytest.raises(ValueError, match="Ambiguous overlapping UV islands"):
        scene_merge.prepare_scatter(proxy, source, "head")(coordinates)


def test_import_binding_failure_releases_runtime_ownership(character, monkeypatch):
    from character_dna.runtime import bindings

    instance = character
    install = bindings.install_targets_iter

    def fail(carrier, targets, **kwargs):
        iterator = install(carrier, targets, **kwargs)
        try:
            for index, item in enumerate(iterator):
                yield item
                if kwargs.get("append") and index == 2:
                    raise RuntimeError("Simulated shape binding failure")
        finally:
            iterator.close()

    monkeypatch.setattr(bindings, "install_targets_iter", fail)
    with pytest.raises(RuntimeError, match="Simulated shape binding failure"):
        import_shape_keys(instance)
    assert not controller.is_suspended(instance)
    assert not retention.retained(instance)
    assert engine.active(instance)
    assert not engine.binding_issues(instance)


@pytest.mark.parametrize("state", ["disabled", "rebuild"])
def test_import_locks_component_without_retained_bindings(character, monkeypatch, state):
    from character_dna.editors.shape_key_editor.import_session import ShapeImportSession

    instance = character
    if state == "disabled":
        instance.auto_evaluate = False
        engine.discard(instance)
    else:
        monkeypatch.setattr(retention, "pause", lambda _instance: False)
    commands = queue.Queue()
    component = ImportShapeKeys.get_component(None)
    session = ShapeImportSession(component)
    try:
        session.begin(component, commands, neutral=False)
        assert controller.is_suspended(instance)
        assert not retention.retained(instance)
        instance.editor_component = "body"
        assert instance.editor_component == "head"
        while not commands.empty():
            index, mesh, _, kwargs, callback = commands.get()
            callback(**kwargs(index, mesh))
        for _ in session.steps():
            pass
    finally:
        session.close()
    assert not controller.is_suspended(instance)
    assert engine.active(instance) == (state == "rebuild")
    assert instance.head_mesh.data.shape_keys
    assert not engine.binding_issues(instance)


@pytest.mark.parametrize("character", ["all_lods"], indirect=True)
def test_import_refreshes_every_merged_lod_without_changing_visibility(character):
    instance = character
    assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
    instance.view_options.active_lod = "lod3"
    registry = scene_merge.surfaces(instance)
    before = identities(instance)
    import_shape_keys(instance)
    assert identities(instance) == before
    assert len(registry) == 8
    assert instance.view_options.active_lod == "lod3"
    for lod, proxy in registry.items():
        assert proxy.hide_get() == (lod != 3)
        metadata = scene_merge.read_metadata(proxy)
        for kind in scene_merge.KINDS:
            source = proxy[f"dna_merge_{kind}"]
            assert metadata["sources"][kind]["shape_signature"] == scene_merge._shape_signature(source)
            for name in scene_merge._keys(source):
                if name != "Basis":
                    assert name in scene_merge._keys(proxy)
    assert not engine.binding_issues(instance)
