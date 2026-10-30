"""Demonstrate that the cheap lifecycle contracts detect a missing runtime resume."""

import ast
import inspect
import textwrap

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--lifecycle-fault", choices=["resume", "rollback", "abort"], default="resume")


def pytest_collection_modifyitems(config: pytest.Config) -> None:
    from character_dna.editors.shared import editor

    tree = ast.parse(textwrap.dedent(inspect.getsource(editor.Editor.enter)))

    class DropResume(ast.NodeTransformer):
        count = 0

        def visit_Expr(self, node: ast.Expr) -> ast.AST:
            fault = config.getoption("--lifecycle-fault")
            function = node.value.func if isinstance(node.value, ast.Call) else None
            matches = (
                (fault == "resume" and isinstance(function, ast.Name) and function.id == "resume")
                or (fault == "abort" and isinstance(function, ast.Attribute) and function.attr == "_abort_enter")
                or (
                    fault == "rollback"
                    and isinstance(function, ast.Attribute)
                    and function.attr == "finish"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "surface_session"
                )
            )
            if matches:
                self.count += 1
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)

    mutate = DropResume()
    tree = mutate.visit(tree)
    assert mutate.count == 1
    namespace = {}
    exec(  # noqa: S102  # nosec B102 - intentional mutation of local Editor.enter, no external code
        compile(ast.fix_missing_locations(tree), "<intentional lifecycle mutation>", "exec"), editor.__dict__, namespace
    )
    editor.Editor.enter = namespace["enter"]
