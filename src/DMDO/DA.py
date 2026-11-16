"""Backward-compatible alias of :mod:`DMDO.analysis.da`.

Kept so existing imports (``DMDO.DA``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .analysis import da as _module

sys.modules[__name__] = _module
