"""Binding reloads must retain native state and use Blender's extension namespace."""

import importlib
import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest

from character_dna import bindings


pytestmark = pytest.mark.skipif(
    getattr(bindings.dna, "__is_fake__", False) or getattr(bindings.riglogic, "__is_fake__", False),
    reason="Real compiled bindings are required to exercise module ownership.",
)


def test_binding_modules_are_namespaced_and_rediscoverable():
    """Python can reload the bindings without executing their initializers again."""
    native = bindings.load_native_runtime()
    before = dict(bindings._modules)
    for name, module in before.items():
        fullname = f"{bindings.__name__}.{name}"
        assert module.__name__ == module.__spec__.name == fullname
        assert Path(module.__file__).is_file()
        assert sys.modules[fullname] is module
        assert sys.modules.get(name) is not module
        assert importlib.reload(module) is module
    assert bindings.load_native_runtime() is native
    assert bindings.riglogic.dna is bindings.dna


def test_native_runtime_reload_recovers_its_spec():
    module = bindings.load_native_runtime()
    module.__spec__ = None
    assert importlib.reload(module) is module
    assert module.__spec__.name == f"{bindings.__name__}._riglogic_blender"
    assert module.capabilities()["api_version"] == 1


def test_loading_from_a_different_folder_requires_restart(monkeypatch, tmp_path):
    bindings.load_native_runtime()
    monkeypatch.setattr(bindings, "combo_folder", tmp_path)
    with pytest.raises(ImportError, match="different folder; restart Blender"):
        bindings.load_native_runtime()


@pytest.mark.parametrize("dev_mode", ["0", "1"])
def test_extension_namespace_reloads_without_policy_warnings(dev_mode):
    """Run the real Blender warning scanner against extension-qualified imports."""
    addon_folder = str(bindings.BINDINGS_FOLDER.parent)
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.machinery
        import sys
        import types
        from pathlib import Path
        import bpy
        import addon_utils

        namespace = 'bl_ext.binding_test.character_dna'
        root = Path({addon_folder!r})
        for name in ('bl_ext', 'bl_ext.binding_test', namespace):
            package = types.ModuleType(name)
            package.__path__ = [str(root)] if name == namespace else []
            package.__package__ = name
            package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            if name == namespace:
                package.__file__ = str(root / '__init__.py')
            sys.modules[name] = package
        # An unrelated addon's absolute imports must remain intact.
        unrelated = {{name: types.ModuleType(name) for name in ('dna', 'riglogic', '_py3dna')}}
        sys.modules.update(unrelated)
        paths_before = list(sys.path)
        bindings = importlib.import_module(namespace + '.bindings')
        native = bindings.load_native_runtime()
        modules = dict(bindings._modules)
        capabilities = native.capabilities
        handles = tuple(bindings._dll_directories.values())
        original_extension_exec = importlib.machinery.ExtensionFileLoader.exec_module
        original_source_exec = importlib.machinery.SourceFileLoader.exec_module

        def guarded_extension_exec(loader, module):
            if Path(loader.path).is_relative_to(root / 'bindings'):
                raise AssertionError('Native binding initializer ran again')
            return original_extension_exec(loader, module)

        def guarded_source_exec(loader, module):
            if Path(loader.path).name in ('dna.py', 'riglogic.py'):
                raise AssertionError('SWIG proxy classes were recreated')
            return original_source_exec(loader, module)

        importlib.machinery.ExtensionFileLoader.exec_module = guarded_extension_exec
        importlib.machinery.SourceFileLoader.exec_module = guarded_source_exec
        for cycle in range(3):
            # Match the helper's deepest-first traversal, including the loaded runtime.
            for name in sorted(tuple(sys.modules), key=lambda value: value.count('.'), reverse=True):
                if name == namespace + '.bindings' or name.startswith(namespace + '.bindings.'):
                    importlib.reload(sys.modules[name])
            assert bindings.load_native_runtime() is native
            assert native.capabilities is capabilities
            assert capabilities()['api_version'] == 1
            assert bindings._modules == modules
            assert tuple(bindings._dll_directories.values()) == handles
            assert bindings.riglogic.dna is bindings.dna
            assert all(sys.modules[name] is module for name, module in unrelated.items())
            assert sys.path == paths_before
            assert sum(getattr(entry, '__module__', None) == bindings.__name__
                       and getattr(entry, '__name__', None) == 'IsolatedModuleLoader'
                       for entry in sys.meta_path) == 1
            for name, module in tuple(sys.modules.items()):
                file = getattr(module, '__file__', None)
                if file and Path(file).is_relative_to(root / 'bindings'):
                    assert name == bindings.__name__ or name.startswith(namespace + '.bindings.'), name
            addon_utils._extensions_warnings_get._is_first = True
            warnings = addon_utils._extensions_warnings_get()
            assert not warnings.get(namespace), warnings
        print('EXTENSION_BINDINGS_PASSED', flush=True)
        """
    )
    environment = dict(os.environ, CHARACTER_DNA_DEV=dev_mode)
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(Path(__file__).parent / "utilities" / "process.py"), script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "EXTENSION_BINDINGS_PASSED" in output, output
