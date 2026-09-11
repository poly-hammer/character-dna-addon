"""Open editor panels opt into redraws without touching evaluation or scene data."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from character_dna.runtime import engine, ui_refresh


def handle(pointer, **kwargs):
    """Construct an RNA-like handle with a stable identity for scheduler tests."""
    return SimpleNamespace(as_pointer=lambda: pointer, **kwargs)


@pytest.fixture
def sidebar(monkeypatch):
    """Use fake UI handles so tests cannot enter Blender's real timer loop."""
    region = handle(3, type="UI", width=300, height=600, tag_redraw=Mock())
    viewport = handle(4, type="WINDOW", width=900, height=600, tag_redraw=Mock())
    area = handle(
        2,
        type="VIEW_3D",
        regions=[region, viewport],
        spaces=SimpleNamespace(active=SimpleNamespace(show_region_ui=True)),
    )
    screen = SimpleNamespace(areas=[area], is_animation_playing=True)
    window = handle(1, screen=screen)
    context = SimpleNamespace(window=window, area=area, region=region, window_manager=SimpleNamespace(windows=[window]))
    timers = SimpleNamespace(is_registered=Mock(return_value=False), register=Mock(), unregister=Mock())
    monkeypatch.setattr(ui_refresh, "bpy", SimpleNamespace(context=context, app=SimpleNamespace(timers=timers)))
    monkeypatch.setattr(ui_refresh.engine, "active", Mock(return_value=True))
    clock = [0.0]
    monkeypatch.setattr(ui_refresh.time, "monotonic", lambda: clock[0])
    ui_refresh._subscribers.clear()
    yield SimpleNamespace(context=context, region=region, viewport=viewport, timers=timers, clock=clock, screen=screen)
    ui_refresh._subscribers.clear()


def test_visible_panels_share_one_region_timer(sidebar):
    ui_refresh.watch(sidebar.context, object())
    sidebar.timers.is_registered.return_value = True
    ui_refresh.watch(sidebar.context, object())
    assert sidebar.timers.register.call_count == 1
    assert ui_refresh._refresh() == ui_refresh.PLAYBACK_INTERVAL
    sidebar.region.tag_redraw.assert_called_once()
    sidebar.viewport.tag_redraw.assert_not_called()


def test_collapsed_panel_subscription_expires(sidebar):
    ui_refresh.watch(sidebar.context, object())
    sidebar.clock[0] = ui_refresh.SUBSCRIPTION_LIFETIME + 0.01
    assert ui_refresh._refresh() is None
    assert not ui_refresh._subscribers
    sidebar.region.tag_redraw.assert_not_called()


def test_hidden_sidebar_stops_immediately(sidebar):
    ui_refresh.watch(sidebar.context, object())
    sidebar.context.area.spaces.active.show_region_ui = False
    assert ui_refresh._refresh() is None
    sidebar.region.tag_redraw.assert_not_called()


def test_idle_refresh_is_throttled(sidebar):
    ui_refresh.watch(sidebar.context, object())
    sidebar.screen.is_animation_playing = False
    assert ui_refresh._refresh() == ui_refresh.IDLE_INTERVAL


def test_legacy_rig_does_not_start_timer(sidebar):
    ui_refresh.engine.active.return_value = False
    ui_refresh.watch(sidebar.context, object())
    sidebar.timers.register.assert_not_called()


def test_removed_window_and_lifecycle_cleanup(sidebar):
    ui_refresh.watch(sidebar.context, object())
    sidebar.context.window_manager.windows.clear()
    assert ui_refresh._refresh() is None
    sidebar.timers.is_registered.return_value = True
    ui_refresh.clear()
    sidebar.timers.unregister.assert_any_call(ui_refresh._refresh)
    sidebar.timers.unregister.assert_any_call(ui_refresh._refresh_migration)
    assert sidebar.timers.unregister.call_count == 2


def test_raw_ui_snapshot_is_lazy_and_graph_local(monkeypatch):
    """One snapshot per visible graph revision; no UI reads of another render graph."""
    graphs = [handle(20, mode="VIEWPORT"), handle(21, mode="RENDER")]
    sessions = [object(), object()]
    rig = object()
    carrier = SimpleNamespace(
        evaluated_get=lambda graph: handle(graph.as_pointer() + 100),
        get={"component": "head", "rig": rig}.get,
    )
    contexts = {
        (7, graph.as_pointer(), graph.as_pointer() + 100, graph.mode): {"session": session, "revision": 3}
        for graph, session in zip(graphs, sessions, strict=True)
    }
    record = {"identity": "ada", "component": "head", "carrier": carrier, "contexts": contexts}
    monkeypatch.setattr(engine, "_generation", 7)
    monkeypatch.setattr(engine, "_records", {"ada-head": record})
    native = SimpleNamespace(
        control_snapshot=Mock(side_effect=lambda session: {"raw": [0.25 if session == sessions[0] else 0.75]})
    )
    monkeypatch.setattr(engine, "_module", native)
    instance = {"native_runtime_id": "ada", "head_rig": rig}
    assert engine.ui_raw_control_value(instance, 0, graphs[0]) == 0.25
    assert engine.ui_raw_control_value(instance, 0, graphs[0]) == 0.25
    assert native.control_snapshot.call_count == 1
    assert engine.ui_raw_control_value(instance, 0, graphs[1]) == 0.75
    assert native.control_snapshot.call_count == 2
    contexts[(7, 20, 120, "VIEWPORT")]["revision"] = 4
    assert engine.ui_raw_control_value(instance, 0, graphs[0]) == 0.25
    assert native.control_snapshot.call_count == 3
    assert engine.ui_raw_control_value({"native_runtime_id": "bruce"}, 0, graphs[0]) is None
    assert native.control_snapshot.call_count == 3


def test_raw_ui_read_never_initializes_a_missing_graph(monkeypatch):
    rig = object()
    record = {
        "identity": "ada",
        "component": "head",
        "contexts": {},
        "carrier": SimpleNamespace(evaluated_get=lambda _graph: handle(30), get={"component": "head", "rig": rig}.get),
    }
    monkeypatch.setattr(engine, "_records", {"head": record})
    native = Mock()
    monkeypatch.setattr(engine, "_module", native)
    instance = {"native_runtime_id": "ada", "head_rig": rig}
    assert engine.ui_raw_control_value(instance, 0, handle(20, mode="VIEWPORT")) == 0.0
    native.control_snapshot.assert_not_called()


def test_raw_ui_uses_rig_ownership_when_identities_collide(monkeypatch):
    """Copied persisted IDs must not alias another rig's UI snapshot."""
    graph = handle(20, mode="VIEWPORT")
    rigs = [object(), object()]
    records = {}
    for index, rig in enumerate(rigs):
        pointer = 100 + index
        carrier = SimpleNamespace(
            get={"component": "head", "rig": rig}.get,
            evaluated_get=lambda _graph, pointer=pointer: handle(pointer),
        )
        records[pointer] = {
            "identity": "copied-id",
            "component": "head",
            "carrier": carrier,
            "contexts": {(7, 20, pointer, "VIEWPORT"): {"session": index, "revision": 1}},
        }
    monkeypatch.setattr(engine, "_generation", 7)
    monkeypatch.setattr(engine, "_records", records)
    monkeypatch.setattr(
        engine, "_module", SimpleNamespace(control_snapshot=lambda session: {"raw": [0.2 + session * 0.6]})
    )
    for rig, expected in zip(rigs, (0.2, 0.8), strict=True):
        assert engine.ui_raw_control_value(
            {"native_runtime_id": "copied-id", "head_rig": rig}, 0, graph
        ) == pytest.approx(expected)


def test_migration_panel_defers_validation_until_outside_draw(monkeypatch):
    """Panel polling must never walk the saved output drivers, even on a cache miss."""
    from character_dna import utilities
    from character_dna.ui import view_3d

    scene = handle(100)
    context = SimpleNamespace(scene=scene)
    validation = Mock(return_value=True)
    timers = SimpleNamespace(is_registered=Mock(return_value=False), register=Mock(), unregister=Mock())
    monkeypatch.setattr(
        ui_refresh, "bpy", SimpleNamespace(app=SimpleNamespace(timers=timers), data=SimpleNamespace(scenes=[scene]))
    )
    monkeypatch.setattr(utilities, "detect_legacy_data", Mock(return_value=None))
    monkeypatch.setattr(utilities, "detect_runtime_migration", validation)
    ui_refresh.clear()
    try:
        assert not view_3d.CHARACTER_DNA_PT_migrate_legacy_data.poll(context)
        validation.assert_not_called()
        callback = timers.register.call_args.args[0]
        callback()
        validation.assert_called_once_with(scene)
        for _ in range(10):
            assert view_3d.CHARACTER_DNA_PT_migrate_legacy_data.poll(context)
        validation.assert_called_once()
        ui_refresh.clear()
        assert not view_3d.CHARACTER_DNA_PT_migrate_legacy_data.poll(context)
        validation.assert_called_once()
    finally:
        ui_refresh.clear()


def test_ui_ownership_does_not_scan_scene_objects(monkeypatch):
    """Repeated activity and protection reads scale with bindings, not scene object count."""
    from character_dna.ui.callbacks import is_reference_readonly

    rig = SimpleNamespace(library=None, override_library=None)
    instance = SimpleNamespace(bl_rna=True, head_rig=rig, body_rig=None, get=lambda _key, default=None: default)
    carrier = SimpleNamespace(get={"component": "head", "rig": rig}.get, library=None, override_library=None)
    monkeypatch.setattr(engine, "_records", {123: {"carrier": carrier}})
    monkeypatch.setattr(engine, "carriers", lambda *_args: pytest.fail("UI reads must not scan scene objects"))
    for _ in range(10):
        assert engine.active(instance)
        assert not is_reference_readonly(instance)
