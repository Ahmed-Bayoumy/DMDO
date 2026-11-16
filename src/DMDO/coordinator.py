"""Backward-compatible alias of :mod:`DMDO.coordination.admm`.

Kept so existing imports (``DMDO.coordinator``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .coordination import admm as _module

sys.modules[__name__] = _module
