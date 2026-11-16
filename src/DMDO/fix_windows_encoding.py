"""Backward-compatible alias of :mod:`DMDO.utils.fix_windows_encoding`.

Kept so existing imports (``DMDO.fix_windows_encoding``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .utils import fix_windows_encoding as _module

sys.modules[__name__] = _module
