"""Different component types require confirmation before export side effects."""

from types import SimpleNamespace
from unittest.mock import Mock

import bpy
import pytest

from character_dna import operators


@pytest.fixture
def confirmation_setup(monkeypatch, tmp_path):
    instance = SimpleNamespace(
        output=SimpleNamespace(
            method="overwrite",
            head_topology_type="3D Scan Store",
            body_topology_type="MetaHuman",
            folder_path=str(tmp_path),
            export_normals=False,
            run_validations=False,
        ),
        head_mesh=object(),
        body_mesh=object(),
        head_rig=object(),
        body_rig=object(),
        auto_evaluate_head=True,
        auto_evaluate_body=True,
        editor_component="head",
        as_pointer=lambda: 42,
        component=lambda _kind: SimpleNamespace(dna_reader=SimpleNamespace(getMetaDataValue=lambda _key: "")),
    )
    dialog = Mock(return_value={"RUNNING_MODAL"})
    context = SimpleNamespace(window_manager=SimpleNamespace(invoke_props_dialog=dialog))
    # Exercise the interactive branch without opening native windows in pytest.
    monkeypatch.setattr(operators, "bpy", SimpleNamespace(app=SimpleNamespace(background=False), path=bpy.path))
    monkeypatch.setattr(operators.callbacks, "get_active_rig_instance", lambda: instance)
    monkeypatch.setattr(operators.utilities, "get_current_context", lambda: None)
    monkeypatch.setattr(operators.utilities, "set_context", lambda _: None)
    monkeypatch.setattr(
        operators.utilities, "get_addon_ops_module", lambda: SimpleNamespace(force_evaluate=lambda: None)
    )
    for kind in ("head", "body"):
        component = SimpleNamespace(
            component_type=kind, linear_modifier=100, dna_reader=object(), write_export_manifest=Mock()
        )
        monkeypatch.setattr(operators.utilities, f"get_active_{kind}", lambda c=component: c)
    exporter = Mock()
    exporter.return_value.validate_scene.return_value = (True, "Success", "Valid", None)
    exporter.return_value.run.return_value = (True, "Success", "Exported", None)
    monkeypatch.setattr(operators, "DNAExporter", exporter)
    return instance, context, dialog, exporter


@pytest.mark.parametrize("operator_class", [operators.ExportSelectedComponent, operators.SendToMetaHumanCreator])
def test_cancel_writes_nothing_and_confirmation_allows_one_export(confirmation_setup, operator_class):
    instance, context, dialog, exporter = confirmation_setup

    class Probe(operators.TopologyTypeExportConfirmation):
        execute = operator_class.execute
        report = Mock()

    operator = Probe()
    assert operator.invoke(context, None) == {"RUNNING_MODAL"}
    exporter.assert_not_called()
    assert instance.auto_evaluate_head and instance.auto_evaluate_body
    assert dialog.call_args.kwargs["confirm_text"] == "Export Anyway"
    assert dialog.call_args.kwargs["cancel_default"] is True
    assert operator._pending_topology_confirmation[1:] == ("3D Scan Store", "MetaHuman")
    operator.cancel(context)
    exporter.assert_not_called()
    assert operator.execute(context) == {"RUNNING_MODAL"}
    exporter.assert_not_called()
    # Confirm: the dialog invokes execute again on this operator instance.
    assert operator.execute(context) == {"FINISHED"}
    assert exporter.call_count == (2 if operator_class is operators.SendToMetaHumanCreator else 1)
    assert instance.auto_evaluate_head and instance.auto_evaluate_body
    exporter.reset_mock()
    assert operator.execute(context) == {"RUNNING_MODAL"}
    exporter.assert_not_called()


def test_changed_names_require_new_confirmation(confirmation_setup):
    instance, context, dialog, exporter = confirmation_setup
    operator = operators.TopologyTypeExportConfirmation()
    assert operator._confirm_topology_types(context, instance) == {"RUNNING_MODAL"}
    instance.output.body_topology_type = "Another topology"
    assert operator._confirm_topology_types(context, instance) == {"RUNNING_MODAL"}
    assert dialog.call_count == 2
    exporter.assert_not_called()


@pytest.mark.parametrize("scenario", ["matching", "calibrate", "missing", "head_only", "inherited"])
def test_warning_scope_and_metadata_fallback(confirmation_setup, scenario):
    instance, context, dialog, _ = confirmation_setup
    if scenario == "matching":
        instance.output.body_topology_type = " 3D Scan Store "
    elif scenario == "calibrate":
        instance.output.method = "calibrate"
    elif scenario == "missing":
        instance.output.body_topology_type = ""
    elif scenario == "head_only":
        instance.body_mesh = None
    else:
        instance.output.body_topology_type = ""
        instance.component = lambda _kind: SimpleNamespace(
            dna_reader=SimpleNamespace(getMetaDataValue=lambda _key: "MetaHuman")
        )
    result = operators.TopologyTypeExportConfirmation()._confirm_topology_types(context, instance)
    if scenario == "inherited":
        assert result == {"RUNNING_MODAL"}
        dialog.assert_called_once()
    else:
        assert result is None
        dialog.assert_not_called()


def test_background_export_cannot_bypass_confirmation(confirmation_setup, monkeypatch):
    instance, context, dialog, exporter = confirmation_setup
    monkeypatch.setattr(operators.bpy.app, "background", True)
    operator = operators.TopologyTypeExportConfirmation()
    operator.report = Mock()
    assert operator._confirm_topology_types(context, instance) == {"CANCELLED"}
    dialog.assert_not_called()
    exporter.assert_not_called()
    operator.report.assert_called_once()
