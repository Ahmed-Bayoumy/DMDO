"""Backward-compatible alias of :mod:`DMDO.ai.decisor`.

Kept so existing imports (``DMDO._decisor``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .ai import decisor as _module

sys.modules[__name__] = _module
