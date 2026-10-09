"""Platform selection and reload-safe, package-scoped native bindings."""

import importlib.machinery
import importlib.util
import os
import platform
import sys
import types

from pathlib import Path

from ..exceptions import UnsupportedPlatformError


BINDINGS_FOLDER = Path(__file__).parent
arch = "arm64" if "arm" in platform.processor().lower() else "x64"
os_name = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform)
if os_name is None or sys.version_info[:2] not in {(3, 11), (3, 13)}:
    raise UnsupportedPlatformError
python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
combo_folder = BINDINGS_FOLDER / os_name / arch / python_version
_ROOTDIR_ATTR = "__character_dna_rootdir__"

# importlib.reload retains the module dictionary. Keep native objects and wrapper
# classes together: re-executing a SWIG wrapper changes its registered proxy types.
_previous_loader = globals().get("IsolatedModuleLoader")
_modules: dict[str, types.ModuleType] = globals().get("_modules", dict(getattr(_previous_loader, "_loaded", {})))
_dll_directories: dict = globals().get("_dll_directories", {})
_loading_depth = 0


def _module_path(name: str, folder: Path) -> Path:
    for suffix in (".py", *importlib.machinery.EXTENSION_SUFFIXES):
        path = folder / (name + suffix)
        if path.is_file():
            return path
    raise ModuleNotFoundError(f"Could not load bindings module '{name}' from '{folder}'.")


def _reuse_spec(name: str, module: types.ModuleType) -> importlib.machinery.ModuleSpec:
    """Give Python a discoverable spec without reinitializing a loaded binary."""
    spec = importlib.machinery.ModuleSpec(name, IsolatedModuleLoader, origin=module.__file__)
    spec.has_location = True
    return spec


def _publish(name: str, module: types.ModuleType, path: Path) -> types.ModuleType:
    fullname = f"{__name__}.{name}"
    module.__name__ = fullname
    module.__package__ = __name__
    module.__file__ = str(path)
    module.__loader__ = IsolatedModuleLoader
    module.__spec__ = _reuse_spec(fullname, module)
    setattr(module, _ROOTDIR_ATTR, str(path.parent.resolve()))
    _modules[name] = module
    sys.modules[fullname] = module
    if sys.modules.get(name) is module:
        del sys.modules[name]
    return module


class IsolatedModuleLoader:
    """Bridge generated wrapper imports and reuse namespaced bindings on reload.

    The generated wrappers find this class by name on sys.meta_path. A temporary
    bare 'dna' import is supported only while those wrappers are executing.
    Public imports and development reloads use the full addon package name.
    """

    @classmethod
    def get_module_rootdir(cls, module_name: str) -> str | None:
        module = cls.cached(module_name)
        return getattr(module, _ROOTDIR_ATTR, None)

    @classmethod
    def cached(cls, module_name: str) -> types.ModuleType | None:
        return _modules.get(module_name)

    @classmethod
    def load_module(cls, module_name: str, rootdir: str | Path | None = None) -> types.ModuleType:
        global _loading_depth
        folder = Path(rootdir or combo_folder).resolve()
        fullname = f"{__name__}.{module_name}"
        existing = _modules.get(module_name) or sys.modules.get(fullname)
        if existing is not None:
            if getattr(existing, _ROOTDIR_ATTR, None) != str(folder):
                raise ImportError("Bindings already loaded from a different folder; restart Blender")
            return _publish(module_name, existing, _module_path(module_name, folder))

        path = _module_path(module_name, folder)
        _add_dll_directory_once(str(folder))
        spec = importlib.util.spec_from_file_location(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load bindings: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        # Preserve unrelated addons' aliases even if a generated wrapper fails
        # between removing an alias and restoring it.
        aliases = {name: sys.modules.get(name) for name in ("dna", "riglogic", "_py3dna")}
        _loading_depth += 1
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(fullname, None)
            raise
        finally:
            _loading_depth -= 1
            for name, previous in aliases.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        return _publish(module_name, module, path)

    @classmethod
    def find_spec(
        cls,
        fullname: str,
        path: object = None,  # noqa: ARG003
        target: types.ModuleType | None = None,  # noqa: ARG003
    ) -> importlib.machinery.ModuleSpec | None:
        prefix = __name__ + "."
        if fullname.startswith(prefix):
            name = fullname[len(prefix) :]
            module = _modules.get(name)
            # Adopt a native runtime loaded before the package itself was reloaded.
            if module is None:
                module = sys.modules.get(fullname)
                if getattr(module, _ROOTDIR_ATTR, None) != str(combo_folder.resolve()):
                    return None
            if module is not None:
                return _reuse_spec(fullname, module)
        elif _loading_depth and fullname in _modules:
            return _reuse_spec(fullname, _modules[fullname])
        return None

    @classmethod
    def create_module(cls, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return _modules.get(spec.name.rsplit(".", 1)[-1]) or sys.modules.get(spec.name)

    @classmethod
    def exec_module(cls, module: types.ModuleType) -> None:
        # Neither native initializers nor SWIG class registrations may run twice.
        # Restore the canonical spec after a temporary bare wrapper import.
        name = module.__name__.rsplit(".", 1)[-1]
        module.__spec__ = _reuse_spec(f"{__name__}.{name}", module)


def _register_loader() -> None:
    sys.meta_path[:] = [
        entry
        for entry in sys.meta_path
        if not (
            getattr(entry, "__name__", None) == "IsolatedModuleLoader"
            and getattr(entry, "__module__", None) == __name__
        )
    ]
    sys.meta_path.insert(0, IsolatedModuleLoader)


def _add_dll_directory_once(folder_path: str) -> None:
    """Keep each DLL directory handle alive and reuse it across addon reloads."""
    if not (sys.platform == "win32" and hasattr(os, "add_dll_directory")):
        return
    folder = str(Path(folder_path).resolve())
    if folder not in _dll_directories:
        _dll_directories[folder] = os.add_dll_directory(folder)


def load_native_runtime() -> types.ModuleType:
    """Load or reuse the runtime under the current addon/extension namespace."""
    return IsolatedModuleLoader.load_module("_riglogic_blender")


def _make_fake_module(module_name: str, attributes: tuple[str, ...]) -> types.ModuleType:
    """Build a placeholder module used when the real bindings are absent."""
    module = types.ModuleType(f"{__name__}.{module_name}")
    module.__is_fake__ = True  # type: ignore[attr-defined]
    for attr in attributes:
        setattr(module, attr, object)
    return module


_register_loader()
try:
    dna = IsolatedModuleLoader.load_module("dna")
    riglogic = IsolatedModuleLoader.load_module("riglogic")
except ModuleNotFoundError:
    if os.environ.get("RUNNING_CI"):
        raise
    dna = _make_fake_module(
        "dna",
        (
            "BinaryStreamReader",
            "BinaryStreamWriter",
            "JSONStreamReader",
            "JSONStreamWriter",
            "FileStream",
            "Status",
            "MemoryResource",
        ),
    )
    riglogic = _make_fake_module("riglogic", ("RigLogic", "RigInstance", "Configuration"))
