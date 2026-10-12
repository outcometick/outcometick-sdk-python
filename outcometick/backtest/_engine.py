"""Load the replay engine used by the sandbox image.

Register otengine, otfeed, otreplay and otharness under their bare import names.
Installed packages contain copies in _sandbox/; source checkouts load them from
runner/harness/python.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_NAMES = ("otengine", "otfeed", "otmaker", "otreplay", "otharness")


def _engine_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(here, "_sandbox")
    if os.path.isfile(os.path.join(packaged, "otengine.py")):
        return packaged
    repo = os.path.normpath(os.path.join(here, "..", "..", "runner", "harness", "python"))
    if os.path.isfile(os.path.join(repo, "otengine.py")):
        return repo
    raise ImportError("outcometick.backtest: the replay engine files are missing from this install")


def load():
    d = _engine_dir()
    for name in _NAMES:
        path = os.path.join(d, f"{name}.py")
        existing = sys.modules.get(name)
        if existing is not None:
            if os.path.abspath(getattr(existing, "__file__", "") or "") == path:
                continue
            raise ImportError(f"a different module named {name!r} is already imported "
                              f"({getattr(existing, '__file__', '?')}); outcometick.backtest needs its own")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return {name: sys.modules[name] for name in _NAMES}
