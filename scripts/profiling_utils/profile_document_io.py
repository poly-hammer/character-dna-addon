"""Isolated DNA I/O benchmark; writes only temporary copies and JSON/profile results.

Run with the repository's bpy-enabled Python. Use --source-root to compare an
archived release (including its editor submodule and native bindings) with HEAD.
OS file caches are deliberately not flushed. 'Cold' means addon cache cold.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import cProfile
import ctypes
import functools
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import traceback
import zlib

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Recorder:
    """Measure nested phase time without the per-call overhead of cProfile."""

    def __init__(self):
        self.rows = []
        self.stack = []

    @contextlib.contextmanager
    def phase(self, label: str):
        frame = [time.perf_counter(), time.process_time(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            wall = time.perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += wall
            self.rows.append(
                dict(phase=label, wall_s=wall, cpu_s=time.process_time() - frame[1], exclusive_s=wall - frame[2])
            )

    @contextlib.contextmanager
    def instrument(self, targets: list):
        originals = []
        for module, name in targets:
            original = getattr(module, name)
            originals.append((module, name, original))

            def wrapper(*args, _fn=original, _label=module.__name__ + "." + name, **kwargs):
                with self.phase(_label):
                    return _fn(*args, **kwargs)

            setattr(module, name, functools.wraps(original)(wrapper))
        try:
            yield
        finally:
            for module, name, original in originals:
                setattr(module, name, original)


def summarize(rows: list[dict]) -> dict:
    result = {}
    for label in dict.fromkeys(row["phase"] for row in rows):
        selected = [row for row in rows if row["phase"] == label]
        result[label] = {
            "count": len(selected),
            "median_s": statistics.median(row["wall_s"] for row in selected),
            "min_s": min(row["wall_s"] for row in selected),
            "max_s": max(row["wall_s"] for row in selected),
            "total_s": sum(row["wall_s"] for row in selected),
            "exclusive_s": sum(row["exclusive_s"] for row in selected),
        }
    return result


def peak_memory() -> dict:
    """Return Windows process high-water memory without an optional dependency."""
    if sys.platform != "win32":
        return {}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    if not function(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {}
    return {"peak_working_set_bytes": counters.PeakWorkingSetSize, "peak_pagefile_bytes": counters.PeakPagefileUsage}


def patch_target(path: Path, recorder: Recorder, prefix: str) -> None:
    """Same target delta API used by the Shape Key Editor, on a disposable file."""
    import numpy as np

    from character_dna.bindings import dna
    from character_dna.dna_io import coordinates
    from character_dna.dna_io.misc import get_dna_reader, get_dna_writer, release_dna_handle

    with recorder.phase(prefix + ".read"):
        reader = get_dna_reader(path)
    with recorder.phase(prefix + ".clone_patch"):
        fast_write = getattr(coordinates, "write_shape_targets", None)
        writer = None
        if fast_write is None:
            writer = get_dna_writer(path)
            writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        mesh = next((i for i in range(reader.getMeshCount()) if reader.getBlendShapeTargetCount(i)), None)
        if mesh is None:
            raise ValueError("Benchmark input must contain at least one shape target.")
        values = np.column_stack([getattr(reader, "getBlendShapeTargetDelta" + axis + "s")(mesh, 0) for axis in "XYZ"])
        indices = list(reader.getBlendShapeTargetVertexIndices(mesh, 0))
        if not indices:
            indices, values = [0], np.zeros((1, 3), dtype=np.float32)
        values[0, 0] += 0.125
        if writer is not None:
            writer.setBlendShapeTargetVertexIndices(mesh, 0, indices)
            writer.setBlendShapeTargetDeltas(mesh, 0, values.tolist())
        release_dna_handle(reader)
    with recorder.phase(prefix + ".write"):
        if fast_write is not None:
            fast_write(path, {(mesh, 0): (indices, values.tolist())})
        else:
            writer.write()
            release_dna_handle(writer)
    # Verification is deliberately outside the measured write.
    with recorder.phase(prefix + ".verify"):
        check = get_dna_reader(path)
        try:
            np.testing.assert_allclose(check.getBlendShapeTargetDeltaXs(mesh, 0), values[:, 0], rtol=0, atol=1e-6)
        finally:
            release_dna_handle(check)


def sdk_benchmark(args: argparse.Namespace, root: Path, recorder: Recorder) -> dict:
    from character_dna.bindings import dna
    from character_dna.dna_io.misc import get_dna_reader, release_dna_handle

    for name in ("head", "body", "Ada"):
        source = args.input_dir / (name + ".dna")
        for iteration in range(args.repeats):
            copy = root / (name + ".dna")
            shutil.copyfile(source, copy)
            with recorder.phase(name + ".sdk_read"):
                reader = get_dna_reader(copy)
            output = dna.FileStream(str(root / "clone.dna"), dna.AccessMode_Write, dna.OpenMode_Binary, None)
            writer = dna.BinaryStreamWriter(output)
            with recorder.phase(name + ".sdk_clone"):
                writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
            with recorder.phase(name + ".sdk_write"):
                writer.write()
            for handle in (writer, output, reader):
                release_dna_handle(handle)
            if name != "Ada":
                with recorder.phase(name + ".target_edit_total"):
                    patch_target(copy, recorder, name)
            print(f"Completed {name} SDK sample {iteration + 1}", flush=True)
            gc.collect()
    return {}


def file_anatomy(args: argparse.Namespace, root: Path, recorder: Recorder) -> dict:
    from character_dna.bindings import dna
    from character_dna.dna_io import document, metadata
    from character_dna.dna_io.coordinates import configuration
    from character_dna.dna_io.misc import get_dna_reader, release_dna_handle

    path = args.input_dir / "Ada.dna"
    reader = get_dna_reader(path, data_layer="Descriptor")
    with recorder.phase("metadata.decode"):
        payload = metadata.decode(reader)
    header = json.loads(reader.getMetaDataValue(metadata.PREFIX))
    total_metadata = sum(
        len(reader.getMetaDataKey(i).encode()) + len(reader.getMetaDataValue(reader.getMetaDataKey(i)).encode())
        for i in range(reader.getMetaDataCount())
    )
    release_dna_handle(reader)
    result = {
        "metadata_header": header,
        "all_metadata_strings_bytes": total_metadata,
        "payload_keys": list(payload),
        "sections": {},
        "indices_sections": {},
    }

    def size(value: Any) -> dict:
        raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
        packed = zlib.compress(raw)
        return {"json_bytes": len(raw), "zlib_bytes": len(packed), "base64_zlib_bytes": len(base64.b64encode(packed))}

    for key, value in payload.items():
        result["sections"][key] = size(value)
    for key, value in payload["indices"].items():
        result["indices_sections"][key] = size(value)
    result["recovery"] = {}
    extracted = (
        {kind: base64.b64decode(encoded) for kind, encoded in payload["recovery"].items()}
        if "recovery" in payload
        else document.read_document(path)[1]
    )
    for kind, content in extracted.items():
        result["recovery"][kind] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "equals_input": content == (args.input_dir / (kind + ".dna")).read_bytes(),
        }
        (root / (kind + ".dna")).write_bytes(content)
    if args.fixture_output:
        args.fixture_output.mkdir(parents=True, exist_ok=False)
        for kind in ("head", "body"):
            shutil.copyfile(root / (kind + ".dna"), args.fixture_output / (kind + ".dna"))
        shutil.copyfile(path, args.fixture_output / "Ada.dna")
        if (args.input_dir / "ExportManifest.json").is_file():
            shutil.copyfile(args.input_dir / "ExportManifest.json", args.fixture_output / "ExportManifest.json")
    stream = dna.FileStream(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    source = dna.BinaryStreamReader(stream, configuration(blender=False), None)
    source.read()
    out = dna.FileStream(
        str(root / "native_without_partition_metadata.dna"), dna.AccessMode_Write, dna.OpenMode_Binary, None
    )
    writer = dna.BinaryStreamWriter(out)
    writer.setFrom(source, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.clearMetaData()
    for i in range(source.getMetaDataCount()):
        key = source.getMetaDataKey(i)
        if key != metadata.PREFIX and not key.startswith(metadata.PREFIX + "."):
            writer.setMetaData(key, source.getMetaDataValue(key))
    writer.write()
    for handle in (writer, out, source, stream):
        release_dna_handle(handle)
    result["native_without_partition_metadata_bytes"] = (root / "native_without_partition_metadata.dna").stat().st_size
    result["scene_bridges"] = {
        "manifest": len(payload["manifest"].get("scene_bridges", [])),
        "indices": len(payload["indices"].get("bridges", [])),
    }
    compact = {key: value for key, value in payload.items() if key != "recovery"}
    result["without_recovery_estimate"] = size(compact)
    # This is a sizing experiment, NOT a valid proposed new document format.
    result["manifest_scene_bridges"] = size(payload["manifest"].get("scene_bridges", []))
    for label in ("cold", "warm"):
        for _ in range(args.repeats):
            if label == "cold":
                document._read_cache.clear()
            with recorder.phase("read_document." + label):
                manifest, parts = document.read_document(path)
            assert parts == extracted
    return result


def document_benchmark(args: argparse.Namespace, root: Path, recorder: Recorder) -> dict:
    from character_dna.dna_io import document, merged_graph, metadata

    target = root / "Ada.dna"
    shutil.copyfile(args.input_dir / "Ada.dna", target)
    document._read_cache.clear()
    with recorder.phase("setup.materialize"):
        _manifest, paths = document.materialize(target)
    instance = {"combined_dna_path": str(target)}

    class Instance(dict):
        pass

    instance = Instance(instance)
    modules = [
        (merged_graph, name)
        for name in ("from_bytes", "to_bytes", "_float_fields", "compose", "_weld", "binary_digest")
    ]
    modules += [(metadata, name) for name in ("encode", "decode")]
    modules += [
        (document, name)
        for name in ("read_document", "write_document", "_atomic_write", "validate_write", "_advance_lease")
    ]
    for kind in ("head", "body"):
        instance.editor_component = kind
        instance.edit_target = SimpleNamespace(dna_file_path=str(paths[kind]))
        for iteration in range(args.repeats):
            with recorder.instrument(modules), recorder.phase(kind + ".document_target_edit_total"):
                with document.edit_transaction(instance):
                    patch_target(paths[kind], recorder, kind)
            print(f"Completed {kind} document sample {iteration + 1}", flush=True)
            gc.collect()
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        with document.edit_transaction(instance):
            patch_target(paths["body"], Recorder(), "profile")
        profiler.disable()
        profiler.dump_stats(str(args.output.with_suffix(".prof")))
    # Invalidate memory cache and verify persisted partition data after all writes.
    document._read_cache.clear()
    with recorder.phase("verify.cold_reopen"):
        _manifest, parts = document.read_document(target)
    for kind in ("head", "body"):
        assert merged_graph.digest(merged_graph.from_bytes(parts[kind])) == merged_graph.digest(
            merged_graph.from_bytes(paths[kind].read_bytes())
        ), f"Persisted {kind} native fields differ from the authored component"
    result = {"published_bytes": target.stat().st_size, "persisted_partition_check": True}
    # materialize() creates a disposable directory outside this script's temporary root.
    for path in paths.values():
        document._leases.pop(path.resolve(), None)
    disposable = paths["head"].parents[2].resolve()
    assert disposable.parent == Path(tempfile.gettempdir()).resolve() and disposable.name.startswith("character-dna-")
    shutil.rmtree(disposable)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("sdk", "anatomy", "document", "editor"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--scene-mode", choices=("separate", "scene", "combined", "release"), default="separate")
    parser.add_argument("--all-keys", action="store_true")
    parser.add_argument("--all-lods", action="store_true")
    parser.add_argument("--backups", action="store_true")
    parser.add_argument("--capture-json", type=Path, help="Diagnostic editor run: retain the first generated SDK JSON")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root / "src/addons"))
    import bpy

    before = {
        name: {"bytes": (args.input_dir / name).stat().st_size, "sha256": digest(args.input_dir / name)}
        for name in ("Ada.dna", "head.dna", "body.dna")
    }
    recorder = Recorder()
    result = {
        "mode": args.mode,
        "source_root": str(args.source_root),
        "inputs": before,
        "python": sys.version,
        "blender": bpy.app.version_string,
        "platform": platform.platform(),
        "repeats": args.repeats,
        "cache_policy": "OS warm/uncontrolled; addon cold/warm explicitly labeled",
    }
    bindings_root = (
        args.source_root
        / "src/addons/character_dna/bindings/windows/x64"
        / f"py{sys.version_info.major}{sys.version_info.minor}"
    )
    result["native_bindings"] = {path.name: digest(path) for path in bindings_root.glob("*.pyd")}
    try:
        with tempfile.TemporaryDirectory(prefix="dna-io-profile-") as directory:
            from profile_document_editor import benchmark

            result["details"] = {
                "sdk": sdk_benchmark,
                "anatomy": file_anatomy,
                "document": document_benchmark,
                "editor": benchmark,
            }[args.mode](args, Path(directory), recorder)
    finally:
        result["loaded_sources"] = {
            name: module.__file__
            for name, module in sys.modules.items()
            if name in {"character_dna", "character_dna.dna_io.misc", "character_dna.editors.shape_key_editor.core"}
        }
        result["samples"] = recorder.rows
        result["summary"] = summarize(recorder.rows)
        result["memory"] = peak_memory()
        result["source_files_unchanged"] = all(
            digest(args.input_dir / name) == info["sha256"] for name, info in before.items()
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        assert result["source_files_unchanged"]
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # bpy DLL shutdown can crash after successful execution on Windows.
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.TerminateProcess(kernel.GetCurrentProcess(), status)
    os._exit(status)
