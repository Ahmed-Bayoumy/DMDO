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
"""Record (or verify) golden traces.

Usage:
  python tests/golden/record_traces.py            # record all cases
  python tests/golden/record_traces.py sellar_scipy gp_mads
  python tests/golden/record_traces.py --check    # run twice per case, report non-determinism
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _harness import CASES, compare, run_case, save_trace  # noqa: E402


def main(argv):
  check = "--check" in argv
  names = [a for a in argv if not a.startswith("--")] or list(CASES)
  for name in names:
    first = run_case(name)
    if check:
      diffs = compare(first, run_case(name))
      print(f"[{name}] deterministic: {not diffs}" + ("".join(f"\n    {d}" for d in diffs)))
    save_trace(name, first)
    print(f"[{name}] recorded: {first['iterations']} iterations, f={first['objective_history'][-1]!r}")


if __name__ == "__main__":
  if sys.platform.startswith("win"):
    from multiprocessing import freeze_support
    freeze_support()
  main(sys.argv[1:])
