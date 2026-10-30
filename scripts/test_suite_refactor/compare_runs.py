"""Compare complete runs without hiding missing cases, skips, or coverage losses."""

import argparse
import ast
import json
import re

from difflib import SequenceMatcher
from pathlib import Path

from coverage import CoverageData


HERE = Path(__file__).resolve().parent
SOURCE = Path("src/addons/character_dna")


def canonical_node(node: str, migration: dict) -> str:
    path, separator, case = node.partition("::")
    destination = migration["tests"].get(path, path)
    if isinstance(destination, dict):
        destination = destination["live_dna" if "matches_live_dna" in case else "matrix"]
    return destination + separator + case


def compare_cases(baseline: dict, candidate: dict, migration: dict) -> dict:
    expected = {canonical_node(node, migration): phases for node, phases in baseline["outcomes"].items()}
    actual = candidate["outcomes"]
    collected = {canonical_node(node, migration) for node in baseline["items"]}
    missing = sorted(collected - set(candidate["items"]))
    unfinished = sorted(node for node in candidate["items"] if "teardown" not in actual.get(node, {}))
    baseline_unfinished = sorted(node for node in collected if "teardown" not in expected.get(node, {}))
    regressions, new_skips, improvements = [], [], []
    for node, before in expected.items():
        after = actual.get(node, {})
        if any(outcome == "failed" and before.get(phase) != "failed" for phase, outcome in after.items()):
            regressions.append(node)
        if "skipped" not in before.values() and "skipped" in after.values():
            new_skips.append(node)
        if "failed" in before.values() and after.get("call") == "passed" and "failed" not in after.values():
            improvements.append(node)
    regressions.extend(node for node in set(actual) - set(expected) if "failed" in actual[node].values())
    return {
        "baseline_cases": len(collected),
        "candidate_cases": len(candidate["items"]),
        "missing": missing,
        "unfinished": unfinished,
        "baseline_unfinished": baseline_unfinished,
        "regressions": sorted(regressions),
        "new_skips": sorted(new_skips),
        "fixed": sorted(improvements),
    }


def matching_lines(before: list[str], after: list[str], old_start: int = 1, new_start: int = 1) -> dict[int, int]:
    result = {}
    for block in SequenceMatcher(
        None, [line.strip() for line in before], [line.strip() for line in after], autojunk=False
    ).get_matching_blocks():
        result.update({old_start + block.a + offset: new_start + block.b + offset for offset in range(block.size)})
    return result


def reader_alias_lines(relative: str, source: str, migration: dict) -> list[str]:
    lines = source.splitlines()
    for function, attribute in migration.get("reader_aliases", {}).get(relative, {}).items():
        matches = [
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == function
        ]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous reader alias mapping: {relative}:{function}")
        node = matches[0]
        for index in range(node.lineno - 1, node.end_lineno):
            lines[index] = lines[index].replace(attribute, "reader")
    return lines


def source_mapping(baseline_root: Path, candidate_root: Path, migration: dict) -> dict:
    """Map unchanged lines and explicitly extracted definitions; unknowns stay visible."""
    mappings = {}
    for old in (baseline_root / SOURCE).rglob("*.py"):
        relative = old.relative_to(baseline_root / SOURCE).as_posix()
        new_relative = "registration.py" if relative == "__init__.py" else relative
        new = candidate_root / SOURCE / new_relative
        if new.exists():
            old_source = old.read_text(encoding="utf-8")
            old_lines = reader_alias_lines(relative, old_source, migration)
            for start, end in matching_lines(old_lines, new.read_text(encoding="utf-8").splitlines()).items():
                mappings[relative, start] = (new_relative, end)
            # coverage records a synthetic line zero for an empty module.
            if (
                not ast.parse(old.read_text(encoding="utf-8")).body
                and not ast.parse(new.read_text(encoding="utf-8")).body
            ):
                mappings[relative, 0] = (new_relative, 0)
                mappings[relative, 1] = (new_relative, 1)
    naming = candidate_root / SOURCE / "naming.py"
    naming_lines = naming.read_text(encoding="utf-8").splitlines()
    definitions = {}
    for node in ast.parse(naming.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    for key, definition in migration["functions"].items():
        relative, start = key.rsplit(":", 1)
        start, end = int(start), definition["old_end"]
        node = definitions[definition["name"]]
        old_lines = (baseline_root / SOURCE / relative).read_text(encoding="utf-8").splitlines()
        # The extraction renamed these identifiers, including calls between
        # extracted functions. The adapter/naming matrices verify the aliases.
        for old_name, new_name in migration.get("renamed_identifiers", {}).get(relative, {}).items():
            old_lines = [re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, line) for line in old_lines]
        for old_line, new_line in matching_lines(
            old_lines[start - 1 : end], naming_lines[node.lineno - 1 : node.end_lineno], start, node.lineno
        ).items():
            mappings[relative, old_line] = ("naming.py", new_line)
        # Renamed signatures retain the same entry/exit, verified by adapter tests.
        mappings[relative, start] = ("naming.py", node.lineno)
    for extraction in migration.get("declarations", []):
        old_file, new_file, name = (extraction[key] for key in ("before", "after", "name"))
        sources = [
            (root / SOURCE / file).read_text(encoding="utf-8")
            for root, file in ((baseline_root, old_file), (candidate_root, new_file))
        ]

        def declaration(source: str, name: str = name) -> ast.Assign | ast.AnnAssign:
            for node in ast.parse(source).body:
                targets = getattr(node, "targets", [getattr(node, "target", None)])
                if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                    return node
            raise ValueError(f"Missing declared extraction {name}")

        old_node, new_node = map(declaration, sources)
        for before, after in matching_lines(
            sources[0].splitlines()[old_node.lineno - 1 : old_node.end_lineno],
            sources[1].splitlines()[new_node.lineno - 1 : new_node.end_lineno],
            old_node.lineno,
            new_node.lineno,
        ).items():
            mappings[old_file, before] = (new_file, after)
        mappings[old_file, old_node.lineno] = (new_file, new_node.lineno)
    for transfer in (*migration.get("imports", []), *migration.get("statements", [])):

        def find_import(root: Path, file: str, text: str) -> int:
            matches = [
                index
                for index, line in enumerate((root / SOURCE / file).read_text(encoding="utf-8").splitlines(), 1)
                if line.strip() == text
            ]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous statement mapping: {file}: {text}")
            return matches[0]

        before, after = transfer["before"], transfer["after"]
        mappings[before[0], find_import(baseline_root, *before)] = (after[0], find_import(candidate_root, *after))
    return mappings


def module_declarations(path: Path) -> set[int]:
    """Import/declaration sequencing is structural, not a conditional branch."""
    lines = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines.add(node.lineno)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    if isinstance(default, ast.Constant):
                        lines.add(default.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Module-level calls still have ordinary sequencing. Conditional
            # expressions, lambdas and comprehensions retain exact arc checks.
            if any(
                isinstance(
                    child,
                    (ast.IfExp, ast.BoolOp, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
                )
                for child in ast.walk(node.value)
            ):
                continue
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def covered_path(arc: tuple[int, int], executed: set[tuple[int, int]], mapped_lines: set[int]) -> bool:
    """Allow inserted executed statements, without crossing another old branch."""
    pending, visited = [arc[0]], set()
    successors = {}
    for start, end in executed:
        successors.setdefault(start, set()).add(end)
    while pending:
        start = pending.pop()
        if start in visited:
            continue
        visited.add(start)
        for end in successors.get(start, ()):
            if end == arc[1]:
                return True
            if end > 0 and end not in mapped_lines:
                pending.append(end)
    return False


def normalized_data(path: Path, root: Path) -> dict:
    data = CoverageData(basename=str(path.resolve()))
    data.read()
    result = {}
    for file in data.measured_files():
        absolute = Path(file)
        try:
            relative = absolute.relative_to(root / SOURCE).as_posix()
        except ValueError:
            continue
        result[relative] = {"lines": set(data.lines(file) or ()), "arcs": set(data.arcs(file) or ())}
    return result


def compare_coverage(
    before_path: Path, after_path: Path, baseline_root: Path, candidate_root: Path, migration: dict
) -> dict:
    before = normalized_data(before_path, baseline_root)
    after = normalized_data(after_path, candidate_root)
    mapping = source_mapping(baseline_root, candidate_root, migration)
    mapped_lines = {}
    for file, line in mapping.values():
        mapped_lines.setdefault(file, set()).add(line)
    missing_lines, missing_arcs, unmapped_lines, unmapped_arcs = [], [], [], []
    structural_arcs, inserted_paths = [], []
    for file, values in before.items():
        declarations = module_declarations(baseline_root / SOURCE / file)
        for line in sorted(values["lines"]):
            target = mapping.get((file, line))
            if target is None:
                unmapped_lines.append([file, line])
            elif target[1] not in after.get(target[0], {}).get("lines", set()):
                missing_lines.append([file, line, *target])
        for arc in sorted(values["arcs"]):
            endpoints = [mapping.get((file, abs(point))) for point in arc]
            if None in endpoints:
                unmapped_arcs.append([file, *arc])
                continue
            if endpoints[0][0] != endpoints[1][0]:
                if all(point in declarations for point in arc) and all(
                    target[1] in after.get(target[0], {}).get("lines", set()) for target in endpoints
                ):
                    structural_arcs.append([file, *arc])
                else:
                    unmapped_arcs.append([file, *arc])
                continue
            new_arc = tuple(target[1] * (-1 if old < 0 else 1) for target, old in zip(endpoints, arc, strict=True))
            target_file = endpoints[0][0]
            executed = after.get(target_file, {}).get("arcs", set())
            if new_arc not in executed:
                if all(point in declarations for point in arc) and all(
                    target[1] in after.get(target[0], {}).get("lines", set()) for target in endpoints
                ):
                    # A deferred import no longer executes beside adjacent
                    # module declarations, even when it stays in the same file.
                    structural_arcs.append([file, *arc])
                elif covered_path(new_arc, executed, mapped_lines.get(target_file, set())):
                    inserted_paths.append([file, *arc, target_file, *new_arc])
                else:
                    missing_arcs.append([file, *arc, target_file, *new_arc])
    return {
        "baseline_files": len(before),
        "candidate_files": len(after),
        "missing_lines": missing_lines,
        "missing_arcs": missing_arcs,
        "unmapped_lines": unmapped_lines,
        "unmapped_arcs": unmapped_arcs,
        "structural_declaration_arcs": structural_arcs,
        "executed_inserted_paths": inserted_paths,
    }


def compare_reports(
    before: dict,
    after: dict,
    baseline_root: Path,
    candidate_root: Path,
    baseline_coverage: Path | None = None,
    candidate_coverage: Path | None = None,
) -> dict:
    migration = json.loads((HERE / "migration-map.json").read_text())
    comparison = compare_cases(before, after, migration)
    comparison["runtime"] = {
        "wall_ratio": after["wall_seconds"] / before["wall_seconds"],
        "phase_ratio": sum(after["phase_seconds"].values()) / sum(before["phase_seconds"].values()),
    }
    comparison["environment_matches"] = before["environment"] == after["environment"]
    if baseline_coverage and candidate_coverage:
        comparison["coverage"] = compare_coverage(
            baseline_coverage, candidate_coverage, baseline_root.resolve(), candidate_root.resolve(), migration
        )
    comparison["accepted"] = (
        comparison["environment_matches"]
        and all(
            not comparison[key] for key in ("missing", "unfinished", "baseline_unfinished", "regressions", "new_skips")
        )
        and all(value <= 0.25 for value in comparison["runtime"].values())
        and "coverage" in comparison
        and comparison["coverage"]["baseline_files"] > 0
        and all(
            not comparison["coverage"][key]
            for key in ("missing_lines", "missing_arcs", "unmapped_lines", "unmapped_arcs")
        )
    )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-coverage", type=Path)
    parser.add_argument("--candidate-coverage", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = (json.loads(path.read_text()) for path in (args.baseline, args.candidate))
    comparison = compare_reports(
        before, after, args.baseline_root, args.candidate_root, args.baseline_coverage, args.candidate_coverage
    )
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: len(value) if isinstance(value, list) else value
                for key, value in comparison.items()
                if key != "coverage"
            },
            indent=2,
        )
    )
    return 0 if comparison["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
