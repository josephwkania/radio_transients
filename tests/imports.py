#!/usr/bin/env python
"""Import each module named on the command line, in one interpreter.

Importing tensorflow, cupy and numba costs seconds apiece, so paying the
interpreter start-up once for the whole list rather than once per module
takes the import sweep from minutes to seconds.

Prints one pipe-delimited line per module for container_test.sh to turn into
TAP, and always exits 0 -- the caller decides what a failure means.

    OK|numpy|1.26.4
    FAIL|cupy|ModuleNotFoundError: No module named 'cupy'
"""

import importlib
import sys
import traceback
import warnings

# A broken install should show up as an ImportError, not as a wall of
# DeprecationWarnings from an unrelated dependency.
warnings.filterwarnings("ignore")


def version_of(module):
    """Best-effort version string, or '' if the module does not carry one."""
    for attr in ("__version__", "version", "VERSION"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return ""


def main(names):
    failed = False
    for name in names:
        try:
            module = importlib.import_module(name)
        except BaseException:  # noqa: BLE001 -- some libs raise SystemExit
            exc_type, exc, _ = sys.exc_info()
            # Keep it to one line: the exception type and message are enough
            # to tell a missing package from a broken shared library.
            detail = (
                "".join(traceback.format_exception_only(exc_type, exc))
                .strip()
                .replace("\n", " ")
            )
            print(f"FAIL|{name}|{detail}", flush=True)
            failed = True
        else:
            print(f"OK|{name}|{version_of(module)}", flush=True)

    return failed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: imports.py MODULE [MODULE...]")
    main(sys.argv[1:])
    sys.exit(0)
