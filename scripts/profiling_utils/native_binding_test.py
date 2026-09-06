"""Reject malformed native bindings without crashing or leaving stale writers."""

from __future__ import annotations

import argparse
import json
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any

import bpy


def run(fixture: Path, output: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(fixture))
    head = next(obj for obj in bpy.data.objects if "_riglogic_head" in obj)
    original = head["_riglogic_head"].to_dict()
    checks = []

    def reject(name: str, change: Callable[[Any], None]) -> None:
        head["_riglogic_head"] = original
        change(head["_riglogic_head"])
        try:
            bpy.app.riglogic.rebuild_bindings()
        except ValueError as error:
            checks.append({"case": name, "error": str(error)})
        else:
            raise AssertionError(f"Invalid binding accepted: {name}")
        head["_riglogic_head"] = original
        bpy.app.riglogic.rebuild_bindings()
        bpy.context.view_layer.update()

    reject("bad LOD", lambda group: group.__setitem__("lod", 100))
    reject("missing driver", lambda group: group["drivers"][0].__setitem__("bone", "__missing__"))
    reject(
        "bad shape channel",
        lambda group: group["keys"][0].__setitem__("channels", [999999] * len(group["keys"][0]["channels"])),
    )
    reject("missing texture node", lambda group: group.__setitem__("texture_node", "__missing__"))
    reject("duplicate output", lambda group: group.__setitem__("joints", original["joints"] + [original["joints"][0]]))
    variable = next(index for index, joint in enumerate(original["joints"]) if not joint["constant"])
    reject("false constant", lambda group: group["joints"][variable].__setitem__("constant", 1))
    duplicate = head.copy()
    duplicate.data = head.data.copy()
    bpy.context.scene.collection.objects.link(duplicate)
    try:
        bpy.app.riglogic.rebuild_bindings()
    except ValueError as error:
        checks.append({"case": "shared target writers", "error": str(error)})
    else:
        raise AssertionError("Conflicting native writers accepted")
    bpy.data.objects.remove(duplicate, do_unlink=True)
    bpy.app.riglogic.rebuild_bindings()
    bpy.context.scene.frame_set(47)
    graph = bpy.context.evaluated_depsgraph_get()
    if head.evaluated_get(graph)["_riglogic_head"]["debug_eye"][0] < 1:
        raise AssertionError("Valid native evaluation did not recover")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"passed": True, "checks": checks}, stream, indent=2)
    print(f"NATIVE BINDINGS PASS: {len(checks)} rejected invalid cases, recovery passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.fixture, args.output)
