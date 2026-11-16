"""Backward-compatible alias of :mod:`DMDO.core.variables`.

Kept so existing imports (``DMDO.variables``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .core import variables as _module

sys.modules[__name__] = _module
