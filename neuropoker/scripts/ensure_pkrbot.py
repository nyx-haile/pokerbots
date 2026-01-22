#!/usr/bin/env python3
"""
Ensure pkrbot is installed from a local wheel (offline-safe).

This script is intended to be called by commands.json on the server. It:
1) Checks whether pkrbot can already be imported.
2) Finds a local pkrbot wheel (pkrbot-*.whl) in the cwd.
3) Bootstraps pip via ensurepip if needed.
4) Installs the wheel using --no-index and --find-links to stay offline.
5) Verifies import of pkrbot.
"""

from __future__ import print_function

import glob
import os
import subprocess
import sys


def _log(message):
    print("[pkrbot] " + message, flush=True)


def _bootstrap_pip():
    try:
        import ensurepip  # noqa: F401
    except Exception as exc:
        _log("ensurepip unavailable: {0}".format(exc))
        return False
    try:
        import ensurepip as _ensurepip
        _ensurepip.bootstrap(upgrade=True)
        return True
    except Exception as exc:
        _log("ensurepip bootstrap failed: {0}".format(exc))
        return False


def _install_wheel(wheel_path):
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        os.path.dirname(os.path.abspath(wheel_path)) or ".",
        wheel_path,
    ]
    _log("installing {0}".format(os.path.basename(wheel_path)))
    subprocess.check_call(cmd)


def main():
    try:
        import pkrbot  # noqa: F401
        _log("pkrbot already available")
        return 0
    except Exception:
        pass

    wheels = sorted(glob.glob("pkrbot-*.whl"))
    if not wheels:
        _log("no pkrbot wheel found in cwd")
        return 1

    if not _bootstrap_pip():
        return 1

    try:
        _install_wheel(wheels[0])
    except Exception as exc:
        _log("wheel install failed: {0}".format(exc))
        return 1

    try:
        import pkrbot  # noqa: F401
    except Exception as exc:
        _log("import failed after install: {0}".format(exc))
        return 1

    _log("installed and import verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
