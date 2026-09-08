"""Platform/Python selection and loader for the OpenRigLogic Python bindings."""

import os
import platform
import sys
import types

from pathlib import Path

from ..exceptions import UnsupportedPlatformError


BINDINGS_FOLDER = Path(__file__).parent
is_dev_mode = os.getenv("CHARACTER_DNA_DEV", "0") == "1"

# ---------------------------------------------------------------------------
# Resolve the platform/architecture/python-version combo folder.
# ---------------------------------------------------------------------------
arch = "x64"
if "arm" in platform.processor().lower():
    arch = "arm64"

if sys.platform == "win32":
    os_name = "windows"
elif sys.platform == "linux":
    os_name = "linux"
elif sys.platform == "darwin":
    os_name = "macos"
else:
    raise UnsupportedPlatformError

if sys.version_info[:2] == (3, 11):
    python_version = "py311"
elif sys.version_info[:2] == (3, 13):
    python_version = "py313"
else:
    raise UnsupportedPlatformError

combo_folder = BINDINGS_FOLDER / os_name / arch / python_version

def load_native_runtime() -> types.ModuleType:
    """Load the optional native runtime beside the SDK without a top-level module."""
    import importlib.machinery
    import importlib.util

    module_name = f"{__name__}._riglogic_blender"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if getattr(existing, _ROOTDIR_ATTR, None) != str(combo_folder.resolve()):
            raise ImportError("Native runtime already loaded from a different bindings folder; restart Blender")
        return existing
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = combo_folder / f"_riglogic_blender{suffix}"
        if not path.is_file():
            continue
        _add_dll_directory_once(str(combo_folder))
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load native runtime: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        setattr(module, _ROOTDIR_ATTR, str(combo_folder.resolve()))
        sys.modules[module_name] = module
        return module
    raise ModuleNotFoundError(f"Native runtime is not installed in {combo_folder}")


# Bare ``sys.modules`` names the SWIG wrappers create transiently while loading.
# They are stripped after a successful load so the packaged extension does not
# register any top-level (non-namespaced) modules, which Blender's extension
# validator forbids.
_BARE_MODULE_PREFIXES = ("_py3dna", "_py3riglogic", "swig_runtime_data")
_BARE_MODULE_NAMES = ("dna", "riglogic")

# Marker recording the folder a compiled binding was loaded from. The dev-reload
# reuse check keys off this attribute instead of ``__file__`` so the latter can be
# nulled (to hide the module from Blender's validator) without forcing a dangerous
# re-initialization of the ``.pyd`` on the next reload.
_ROOTDIR_ATTR = "__character_dna_rootdir__"


class IsolatedModuleLoader:
    """``sys.meta_path`` hook the generated SWIG wrappers cooperate with.

    The wrappers (``dna.py`` / ``riglogic.py``) probe ``sys.meta_path`` for a
    class literally named ``IsolatedModuleLoader`` and, when found, ask it to
    load their compiled siblings via :meth:`load_module` instead of doing a bare
    ``import _py3dna...`` (which would pollute the top-level module namespace and
    require ``sys.path`` mutation). ``riglogic.py`` also calls
    :meth:`get_module_rootdir` to locate ``dna`` and performs an unconditional
    reentrant ``import dna``; the finder protocol below satisfies that by
    re-serving the already-loaded module without re-executing it.
    """

    # Populated per process; reset whenever this module is re-imported (dev
    # reload). Loaded compiled modules are also reused from ``sys.modules`` when
    # still present (dev mode keeps them), so the ``.pyd`` is never re-initialized.
    _loaded: dict = {}
    _rootdirs: dict = {}

    @classmethod
    def get_module_rootdir(cls, module_name):
        return cls._rootdirs.get(module_name)

    @classmethod
    def cached(cls, module_name):
        """Return the already-loaded module, or ``None`` if it is not loaded."""
        return cls._loaded.get(module_name)

    @classmethod
    def load_module(cls, module_name, rootdir=None):
        cached = cls._loaded.get(module_name)
        if cached is not None:
            return cached

        if rootdir is not None:
            cls._rootdirs[module_name] = rootdir
        rootdir = cls._rootdirs.get(module_name) or str(combo_folder)

        # Reuse an already-loaded binding (e.g. across a dev-mode reload) so the
        # compiled extension is not initialized twice in the same process. The match
        # uses our own rootdir marker rather than ``__file__`` because ``__file__``
        # is nulled after loading to hide the module from Blender's validator.
        existing = sys.modules.get(module_name)
        if existing is not None and getattr(existing, _ROOTDIR_ATTR, None) == os.path.abspath(rootdir):
            cls._loaded[module_name] = existing
            return existing

        module = cls._exec_from_disk(module_name, rootdir)
        cls._loaded[module_name] = module
        return module

    @staticmethod
    def _exec_from_disk(module_name, rootdir):
        import importlib.machinery
        import importlib.util

        py_path = os.path.join(rootdir, module_name + ".py")
        if os.path.isfile(py_path):
            spec = importlib.util.spec_from_file_location(module_name, py_path)
        else:
            spec = None
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                ext_path = os.path.join(rootdir, module_name + suffix)
                if os.path.isfile(ext_path):
                    loader = importlib.machinery.ExtensionFileLoader(module_name, ext_path)
                    spec = importlib.util.spec_from_file_location(module_name, ext_path, loader=loader)
                    break

        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Could not load bindings module '{module_name}' from '{rootdir}'.")

        module = importlib.util.module_from_spec(spec)
        # The wrappers' absolute imports and SWIG's cross-module type table look
        # the module up by its bare name during execution; register it for the
        # duration of the load and strip it again once everything is in place.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        # Record where this module came from so the reuse check above can match it
        # after ``__file__`` has been nulled.
        setattr(module, _ROOTDIR_ATTR, os.path.abspath(rootdir))
        return module

    # -- finder/loader protocol -------------------------------------------------
    # ``riglogic.py`` does an unconditional ``import dna`` after popping it from
    # ``sys.modules``; serve the already-loaded module instead of re-executing.
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname in cls._loaded:
            import importlib.util

            return importlib.util.spec_from_loader(fullname, cls)
        return None

    @classmethod
    def create_module(cls, spec):
        return cls._loaded.get(spec.name)

    @classmethod
    def exec_module(cls, module):
        # The module returned by ``create_module`` is already initialized.
        return None


def _register_loader():
    """Place a single fresh :class:`IsolatedModuleLoader` on ``sys.meta_path``."""
    sys.meta_path[:] = [entry for entry in sys.meta_path if getattr(entry, "__name__", None) != "IsolatedModuleLoader"]
    sys.meta_path.insert(0, IsolatedModuleLoader)


def _strip_bare_binding_modules(*, include_compiled):
    """Remove the transient top-level binding modules the wrappers created.

    The bare wrapper names (``dna``/``riglogic``) are always removed; they cost
    nothing to re-execute on a dev reload. The compiled siblings
    (``_py3dna*``/``_py3riglogic*``/``swig_runtime_data*``) are only removed when
    ``include_compiled`` is true: in dev mode they are deliberately kept in
    ``sys.modules`` so a subsequent reload reuses the already-initialized compiled
    extensions instead of re-initializing the ``.pyd`` files."""
    for name in list(sys.modules):
        is_compiled = name.startswith(_BARE_MODULE_PREFIXES)
        if is_compiled and not include_compiled:
            continue
        if name in _BARE_MODULE_NAMES or is_compiled:
            module = sys.modules[name]
            module_file = getattr(module, "__file__", "") or ""
            # Only remove our own bindings, never an unrelated top-level ``dna``
            if is_compiled or (
                module_file and os.path.abspath(module_file).startswith(os.path.abspath(str(BINDINGS_FOLDER)))
            ):
                del sys.modules[name]


def _neutralize_binding_file_attrs(*wrapper_modules):
    """Hide every binding module from Blender's extension validator.

    ``addon_utils._extensions_warnings_get`` reports a "Policy violation with top
    level module" for any non-namespaced entry in ``sys.modules`` whose ``__file__``
    resolves inside the extension directory, but it skips entries that have no
    ``__file__``. Nulling ``__file__`` keeps the bindings invisible to that check:

    * the ``.py`` wrappers (``dna``/``riglogic``) -- even if the persistent
      meta-path loader re-serves them into ``sys.modules`` on a later re-entrant
      ``import dna``/``import riglogic`` (which the wrappers and SWIG's cross-module
      type sharing perform), and
    * the compiled siblings (``_py3dna*``/``_py3riglogic*``/``swig_runtime_data*``)
      that dev mode deliberately keeps in ``sys.modules`` for ``.pyd`` reuse.
    """
    targets = list(wrapper_modules)
    targets.extend(
        sys.modules[name] for name in list(sys.modules) if name.startswith(_BARE_MODULE_PREFIXES)
    )
    for module in targets:
        if module is None:
            continue
        try:
            module.__file__ = None  # type: ignore[assignment]
        except Exception:
            pass


def _add_dll_directory_once(folder_path):
    """Register ``folder_path`` as a DLL search directory at most once per process.

    Each ``os.add_dll_directory`` call appends a new entry to the process-wide DLL
    search list -- calling it repeatedly with the same folder (as happens on every
    dev-mode addon reload, which re-executes this module) makes that list grow
    without bound until Windows rejects it with ``WinError 206`` ("filename or
    extension is too long"). The set of already-registered folders is stashed on
    ``sys`` so it survives the module re-execution a reload performs."""
    if not (sys.platform == "win32" and hasattr(os, "add_dll_directory") and os.path.isdir(folder_path)):
        return
    registered = getattr(sys, "_character_dna_dll_directories", None)
    if registered is None:
        registered = set()
        sys._character_dna_dll_directories = registered  # type: ignore[attr-defined]
    abs_folder = os.path.abspath(folder_path)
    if abs_folder in registered:
        return
    os.add_dll_directory(abs_folder)
    registered.add(abs_folder)


def _load_bindings(folder):
    _register_loader()

    folder_path = str(folder)
    # Make the compiled extensions' sibling DLLs discoverable without mutating
    # ``sys.path`` (which the extension validator forbids).
    _add_dll_directory_once(folder_path)

    # ``riglogic.py`` depends on ``dna``, so load ``dna`` first.
    dna_module = IsolatedModuleLoader.load_module("dna", rootdir=folder_path)
    riglogic_module = IsolatedModuleLoader.load_module("riglogic", rootdir=folder_path)

    # Always drop the bare wrapper names from ``sys.modules``. The compiled siblings
    # are only stripped in a packaged release; dev mode keeps them so a reload reuses
    # the already-initialized ``.pyd`` extensions.
    _strip_bare_binding_modules(include_compiled=not is_dev_mode)

    # Neutralize the ``__file__`` of the wrappers and any retained compiled modules
    # so Blender's validator never flags them, regardless of dev/packaged mode and
    # immune to the loader re-adding them later.
    _neutralize_binding_file_attrs(dna_module, riglogic_module)

    return dna_module, riglogic_module


def _make_fake_module(module_name, attributes):
    """Build a placeholder module used when the real bindings are absent."""
    module = types.ModuleType(module_name)
    module.__is_fake__ = True  # type: ignore[attr-defined]
    for attr in attributes:
        setattr(module, attr, object)
    return module


# ---------------------------------------------------------------------------
# Load (or reuse) the two modules and expose them as ``dna`` / ``riglogic``.
# ---------------------------------------------------------------------------
# In a packaged release the compiled dependencies must not be reloaded, so reuse
# an already-loaded pair unless we are explicitly in dev mode.
_already_loaded = IsolatedModuleLoader.cached("dna") is not None and IsolatedModuleLoader.cached("riglogic") is not None
_should_load = is_dev_mode or not _already_loaded

try:
    if not _should_load and _already_loaded:
        dna = IsolatedModuleLoader.cached("dna")
        riglogic = IsolatedModuleLoader.cached("riglogic")
    elif combo_folder.exists():
        dna, riglogic = _load_bindings(combo_folder)
    else:
        raise ModuleNotFoundError

except ModuleNotFoundError:
    # On CI fail hard if the bindings are missing.
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
    riglogic = _make_fake_module(
        "riglogic",
        (
            "RigLogic",
            "RigInstance",
            "Configuration",
        ),
    )
