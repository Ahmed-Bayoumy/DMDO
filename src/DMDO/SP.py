"""Backward-compatible alias of :mod:`DMDO.subproblem.subproblem`.

Kept so existing imports (``DMDO.SP``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .subproblem import subproblem as _module

sys.modules[__name__] = _module
