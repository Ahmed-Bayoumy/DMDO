"""System-wide functions of augmented Lagrangian coordination (Tosserams et al. 2010, Sec. 2.2, Eqs. 5-8).

A system-wide constraint depends on variables of more than one subproblem. Instead of a target-response pair it is
relaxed directly with an augmented Lagrangian penalty:

- inequality ``g0(x) <= 0``:  phi(g0 + s) = v (g0 + s) + (w (g0 + s))^2, with a slack variable s >= 0;
- equality   ``h0(x) = 0``:   phi(h0)     = v h0 + (w h0)^2.

``g0 + s`` and ``h0`` (times ``scaling``) extend the inconsistency vector q = [c; g0 + s; h0]; their multipliers are
updated with the same method-of-multipliers rule as the target-response pairs (Eqs. 7-8). Every subproblem that owns
an input of the function (or its slack) adds the penalty to its objective (Eq. 6).

The function is evaluated from the coordinator's master vector, so inside a subproblem it sees that subproblem's
trial values and the latest values of the other subproblems.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

KINDS = ("ineq", "eq")


@dataclass
class SystemFunction:
  """One system-wide constraint.

  :param name: label used in logs.
  :param kind: ``"ineq"`` (g0 <= 0, needs ``slack``) or ``"eq"`` (h0 = 0).
  :param blackbox: callable taking the list of input values (vector variables flattened, in ``inputs`` order) and
    returning the function value.
  :param inputs: 1-based indices of the master variables passed to ``blackbox``.
  :param slack: 1-based index of the slack variable (a local variable, lower bound 0, of one subproblem); inequality
    only.
  :param scaling: multiplies the function value before it enters q (same role as the pair scaling).
  """
  name: str
  kind: str
  blackbox: Callable[[List[float]], Any]
  inputs: List[int]
  slack: Optional[int] = None
  scaling: float = 1.0
  subproblems: List[int] = field(default_factory=list)

  def __post_init__(self):
    self.kind = str(self.kind).lower()
    if self.kind not in KINDS:
      raise ValueError(f"System-wide function '{self.name}': type must be one of {KINDS}, not '{self.kind}'.")
    if self.kind == "ineq" and self.slack is None:
      raise ValueError(f"System-wide inequality '{self.name}' needs a slack variable (lower bound 0).")
    if self.kind == "eq" and self.slack is not None:
      raise ValueError(f"System-wide equality '{self.name}' cannot have a slack variable.")

  def bind(self, master_vars: Sequence[Any]) -> None:
    """Record the subproblems whose objectives receive this penalty: owners of an input or of the slack."""
    owners = {int(master_vars[i - 1].sp_index) for i in self.inputs}
    if self.slack is not None:
      owners.add(int(master_vars[self.slack - 1].sp_index))
    self.subproblems = sorted(owners)

  def value(self, master_vars: Sequence[Any]) -> float:
    """Scaled ``g0 + s`` or ``h0`` at the master vector."""
    x: List[float] = []
    for i in self.inputs:
      val = master_vars[i - 1].value
      x.extend(list(val) if isinstance(val, (list, tuple, np.ndarray)) else [val])
    out = self.blackbox(copy.deepcopy(x))
    out = float(np.ravel(np.asarray(out, dtype=float))[0])
    if self.slack is not None:
      out += float(master_vars[self.slack - 1].value)
    return out * float(self.scaling)


def penalty(v: np.ndarray, w: np.ndarray, q: np.ndarray) -> np.ndarray:
  """Augmented Lagrangian penalty per function: v q + (w q)^2 (Eq. 5)."""
  return np.multiply(v, q) + np.multiply(np.multiply(w, w), np.multiply(q, q))
