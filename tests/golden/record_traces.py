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
