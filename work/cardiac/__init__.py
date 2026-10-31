import os as _os
import sys as _sys

# Only for a conda env launched without `conda activate`: put its DLL folders on PATH, otherwise cuDNN and
# MKL fail to load their sub-DLLs. Pip/venv installs (the current setup, see py.cmd) need nothing here.
_root = _os.path.dirname(_sys.executable)
_libbin = _os.path.join(_root, "Library", "bin")
if _os.path.isdir(_libbin) and _libbin.lower() not in _os.environ.get("PATH", "").lower():
    _dirs = [_root] + [_os.path.join(_root, *p) for p in
                       (("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"), ("Library", "bin"), ("Scripts",))]
    _os.environ["PATH"] = _os.pathsep.join(_dirs + [_os.environ.get("PATH", "")])
_os.environ.setdefault("ECHO_DATA", r"E:\echo_project")
_os.environ.setdefault("TORCH_HOME", _os.path.join(_os.environ["ECHO_DATA"], "torch-cache"))
