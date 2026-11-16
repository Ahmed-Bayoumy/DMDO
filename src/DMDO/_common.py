"""Backward-compatible alias of :mod:`DMDO.utils.log`.

Kept so existing imports (``DMDO._common``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .utils import log as _module

sys.modules[__name__] = _module
