"""Regression guard: reference runs must reproduce their recorded outer-iteration histories.

The traces are platform-specific (floating-point and OMADS precision settings differ between OSes), so this suite
is opt-in: set ``DMDO_GOLDEN=1`` to run it, typically on the machine that recorded the traces.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _harness import CASES, compare, load_trace, run_case, trace_path  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("DMDO_GOLDEN", "0") != "1",
                                reason="golden traces are opt-in; set DMDO_GOLDEN=1")


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_trace(name):
  if not os.path.exists(trace_path(name)):
    pytest.skip(f"no recorded trace for {name}; run tests/golden/record_traces.py")
  diffs = compare(load_trace(name), run_case(name))
  assert not diffs, f"{name} diverged from its golden trace:\n  " + "\n  ".join(diffs)
