"""Selects the OS-specific module. Everything else imports `from core.osal import osx`."""
import sys

if sys.platform == "win32":
    from core import os_win as osx  # noqa: F401
else:
    from core import os_linux as osx  # noqa: F401
