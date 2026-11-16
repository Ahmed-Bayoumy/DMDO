"""Backward-compatible alias of :mod:`DMDO.core.enums`.

Kept so existing imports (``DMDO._globals``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .core import enums as _module

sys.modules[__name__] = _module
