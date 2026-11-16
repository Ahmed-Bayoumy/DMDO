"""Backward-compatible alias of :mod:`DMDO.app.main`.

Kept so existing imports (``DMDO.DMDO``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .app import main as _module

sys.modules[__name__] = _module
