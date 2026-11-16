"""Backward-compatible alias of :mod:`DMDO.analysis.mda`.

Kept so existing imports (``DMDO.MDA``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .analysis import mda as _module

sys.modules[__name__] = _module
