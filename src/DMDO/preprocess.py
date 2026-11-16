"""Backward-compatible alias of :mod:`DMDO.app.builder`.

Kept so existing imports (``DMDO.preprocess``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .app import builder as _module

sys.modules[__name__] = _module
