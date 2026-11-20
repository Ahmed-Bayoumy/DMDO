# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#
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
