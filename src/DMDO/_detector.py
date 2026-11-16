"""Backward-compatible alias of :mod:`DMDO.ai.detector`.

Kept so existing imports (``DMDO._detector``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .ai import detector as _module

sys.modules[__name__] = _module
