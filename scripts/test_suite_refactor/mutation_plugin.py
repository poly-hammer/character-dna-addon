"""Inject opt-in faults during test calls, restoring production code before teardown."""

import ast
import copy
import hashlib
import importlib
import inspect
import json
import textwrap

from pathlib import Path
from types import FunctionType
from typing import Any

import pytest


EVIDENCE = pytest.StashKey[list[dict[str, Any]]]()


# Exact, reviewed source replacements fail closed when the production code changes.
# Swapping the existing function's code also reaches already-imported aliases.
TARGETS = {
    "resume": (
        "character_dna.editors.shared.editor",
        "Editor.enter",
        "resume(self.instance, native_active)",
        "pass",
    ),
    "rollback": (
        "character_dna.editors.shared.editor",
        "Editor.enter",
        "surface_session.finish(self.instance, commit=False)",
        "pass",
    ),
    "abort": (
        "character_dna.editors.shared.editor",
        "Editor.enter",
        "self._abort_enter()",
        "pass",
    ),
    "destination": (
        "character_dna.rig_component",
        "RigComponent.dna_file_path",
        'f"{self.kind}_dna_file_path"',
        "f\"{'body' if self.kind == 'head' else 'head'}_dna_file_path\"",
    ),
    "axis": (
        "character_dna.editors.swing_twist_editor.evaluation",
        "apply_offset",
        "Quaternion(direction, properties.preview_offset)",
        "Quaternion(direction, -properties.preview_offset)",
    ),
    "units": (
        "character_dna.editors.swing_twist_editor.evaluation",
        "apply_offset",
        "Quaternion(direction, properties.preview_offset)",
        "Quaternion(direction, properties.preview_offset * 180.0 / 3.141592653589793)",
    ),
    "rbf": (
        "character_dna.editors.swing_twist_editor.evaluation",
        "apply_selected",
        "rbfs=instance.evaluate_rbfs",
        "rbfs=False",
    ),
    "shape": (
        "character_dna.editors.swing_twist_editor.evaluation",
        "apply_selected",
        "target.update_shape_keys(state=evaluator.state)",
        "pass",
    ),
    "visibility": (
        "character_dna.editors.swing_twist_editor.editor",
        "SwingTwistEditor._restore_scene_context",
        'collection.is_visible = values["visible"] if isinstance(values, dict) else values',
        "pass",
    ),
    "publication": (
        "character_dna.dna_io.document",
        "write_document",
        "_atomic_write(path, published)",
        "_atomic_write(path, published)\n    _atomic_write(path, published)",
    ),
    "reader": (
        "character_dna.dna_io.document",
        "read_document",
        "data = Path(path).read_bytes()",
        "data = Path(path).read_bytes()",
    ),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--contract-fault", "--lifecycle-fault", choices=tuple(TARGETS), default=None)
    parser.addoption("--mutation-evidence", default=None)


def pytest_configure(config: pytest.Config) -> None:
    config.stash[EVIDENCE] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Keep fixtures pristine; inject only while the selected test body executes."""
    fault = item.config.getoption("contract_fault")
    if fault is None:
        yield
        return
    module_name, attribute, before, after = TARGETS[fault]
    target: Any = importlib.import_module(module_name)
    for name in attribute.split("."):
        target = getattr(target, name)
    if isinstance(target, property):
        target = target.fget
    target = inspect.unwrap(target)
    source = textwrap.dedent(inspect.getsource(target))
    assert source.count(before) == 1, (fault, attribute, source.count(before))
    tree = ast.parse(source.replace(before, after))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    function.decorator_list = []
    entry = 1 if ast.get_docstring(function) is not None else 0
    if fault == "reader":
        function.body = function.body[:entry] + ast.parse("return _contract_stale_read(path)").body
    function.body.insert(entry, ast.Expr(ast.Call(ast.Name("_contract_mutation_hit", ast.Load()), [], [])))
    record: dict[str, Any] = {
        "nodeid": item.nodeid,
        "fault": fault,
        "target": f"{module_name}.{attribute}",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "calls": 0,
    }

    def hit():
        record["calls"] += 1

    injections: dict[str, Any] = {"_contract_mutation_hit": hit}
    if fault == "reader":
        original_read = FunctionType(target.__code__, target.__globals__, target.__name__)
        cached_reads = {}

        def stale_read(path: Path) -> tuple[dict, dict[str, bytes]]:
            key = str(Path(path).resolve())
            if key not in cached_reads:
                cached_reads[key] = copy.deepcopy(original_read(path))
            return copy.deepcopy(cached_reads[key])

        injections["_contract_stale_read"] = stale_read

    namespace = {}
    exec(  # noqa: S102  # nosec B102 - reviewed local source mutation, never external code
        compile(ast.fix_missing_locations(tree), f"<intentional {fault} mutation>", "exec"),
        target.__globals__,
        namespace,
    )
    original = target.__code__
    sentinel = object()
    previous = {name: target.__globals__.get(name, sentinel) for name in injections}
    target.__globals__.update(injections)
    target.__code__ = namespace[target.__name__].__code__
    try:
        yield
    finally:
        target.__code__ = original
        for name, value in previous.items():
            if value is sentinel:
                target.__globals__.pop(name, None)
            else:
                target.__globals__[name] = value
        item.config.stash[EVIDENCE].append(copy.deepcopy(record))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    yield
    if call.when == "call" and call.excinfo is not None:
        for record in item.config.stash[EVIDENCE]:
            if record["nodeid"] == item.nodeid:
                record["exception"] = call.excinfo.typename
                record["message"] = str(call.excinfo.value)


def pytest_sessionfinish(session: pytest.Session) -> None:
    destination = session.config.getoption("mutation_evidence")
    if destination:
        Path(destination).write_text(json.dumps(session.config.stash[EVIDENCE], indent=2) + "\n", encoding="utf-8")
