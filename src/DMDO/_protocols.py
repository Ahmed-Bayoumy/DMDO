"""Backward-compatible alias of :mod:`DMDO.core.protocols`.

Kept so existing imports (``DMDO._protocols``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .core import protocols as _module

sys.modules[__name__] = _module
