"""Exit Blender test processes after flushing their actual result, before DLL teardown."""

import faulthandler
import os
import sys

from typing import NoReturn


def exit_blender(status: int) -> NoReturn:
    """Preserve the test exit status without running bpy's unstable shutdown cleanup."""
    sys.stdout.flush()
    sys.stderr.flush()
    faulthandler.disable()
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), status)
    os._exit(status)


if __name__ == "__main__":
    import traceback

    script = sys.argv.pop(1)
    exit_status = 0
    try:
        exec(compile(script, "<blender-test>", "exec"), {"__name__": "__main__"})  # noqa: S102
    except BaseException:
        traceback.print_exc()
        exit_status = 1
    exit_blender(exit_status)
