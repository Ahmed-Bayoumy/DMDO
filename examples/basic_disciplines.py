"""Analyses and objectives of the basic two-subproblem example (examples/basic_problem.yaml).

The two analyses are written as :class:`DMDO.Discipline` objects (named inputs in, named outputs out), the form an
adapter around a model from another framework takes. They live in an importable module so that worker processes
can unpickle them.
"""
import numpy as np


class Analysis1:
  """a = log10(u) + log10(v) + log10(b)"""

  def compute(self, inputs):
    u, v, b = (max(float(inputs[n]), 1e-12) for n in ("u", "v", "b"))
    return {"a": np.log10(u) + np.log10(v) + np.log10(b)}


class Analysis2:
  """b = 1/u + 1/w + 1/a"""

  def compute(self, inputs):
    u, w, a = (max(float(inputs[n]), 1e-12) for n in ("u", "w", "a"))
    return {"b": 1.0 / u + 1.0 / w + 1.0 / a}


def opt1(x, y, c=None):
  """Subproblem 1 objective: u + v + b + a (no constraint)."""
  return [sum(x) + y[0], [0.]]


def opt2(x, y, c=None):
  """Subproblem 2: no objective, constraint w + b - 10 <= 0."""
  return [0., [x[1] + y[0] - 10.]]


A1 = Analysis1()
A2 = Analysis2()

# Names used in basic_problem.yaml. Over the HTTP service, give them as import paths ("basic_disciplines:A1", ...).
BLACKBOXES = {"A1": A1, "A2": A2, "opt1": opt1, "opt2": opt2}
