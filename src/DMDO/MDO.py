"""Backward-compatible alias of :mod:`DMDO.workflow.mdo`.

Kept so existing imports (``DMDO.MDO``) and checkpoints pickled by earlier DMDO versions keep working.
"""
import sys

from .workflow import mdo as _module

sys.modules[__name__] = _module
