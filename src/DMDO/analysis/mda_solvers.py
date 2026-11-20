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
"""Feedback detection and fixed-point solvers for a multidisciplinary analysis (used by ``MON: MDF``).

Feedback detection
  Inside one MDA block, analysis A feeds analysis B when a variable key in A's ``outputs`` is also in B's ``inputs``
  (matched by the variable's unique ``index``, never by name). The strongly connected components of that directed
  graph (Tarjan) are the feedback loops: a component with two or more analyses, or an analysis that reads its own
  output. Only those are handed to the MDA solver; the rest is feed-forward and runs once, in dependency order.

Solvers (``MDA.<id>.solver``: a name, or a mapping ``{name: ..., tol: ..., maxiter: ...}``)
  ``GS``      Gauss-Seidel fixed point: each analysis uses the latest outputs of the others (default).
  ``Jacobi``  Jacobi fixed point: every analysis of a sweep uses the outputs of the previous sweep.
  ``Newton``  Newton's method on the residual R(y) = y - F(y), finite-difference Jacobian, backtracking step.

Options: ``tol`` (default 1e-10) on the scaled residual ``max_i |y_i - F_i(y)| / max(1, |y_i|)``, ``maxiter`` (default 100),
``relaxation`` (GS/Jacobi, default 1.0: y <- y + relaxation (F(y) - y)), ``fd_step`` (Newton, default 1e-7, relative).
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..registry import mda_solvers

__all__ = ["Component", "MDAResult", "MDASolver", "GaussSeidelSolver", "JacobiSolver", "NewtonSolver",
           "coupling_components", "create_mda_solver", "DEFAULT_MDA_SOLVER"]

DEFAULT_MDA_SOLVER = "GS"

State = Dict[int, Any]
Evaluate = Callable[[Any, State], State]


# --------------------------------------------------------------------------------------------- graph
@dataclass
class Component:
  """A strongly connected group of analyses, in execution order."""
  analyses: List[Any]
  #: True for a feedback loop (two or more analyses, or one that reads its own output).
  feedback: bool
  #: Indices of the variables that are both produced and consumed inside the component (the unknowns of the loop).
  couplings: List[int] = field(default_factory=list)


def _outputs(da) -> List[int]:
  return [int(v.index) for v in da.outputs if v is not None and v.index is not None]


def _inputs(da) -> List[int]:
  return [int(v.index) for v in da.inputs if v is not None and v.index is not None]


def _strongly_connected(succ: List[List[int]]) -> List[List[int]]:
  """Tarjan's algorithm (iterative): strongly connected components in topological order."""
  n = len(succ)
  index_of = [-1] * n
  low = [0] * n
  on_stack = [False] * n
  stack: List[int] = []
  sccs: List[List[int]] = []
  counter = 0
  for root in range(n):
    if index_of[root] != -1:
      continue
    work = [(root, 0)]
    index_of[root] = low[root] = counter
    counter += 1
    stack.append(root)
    on_stack[root] = True
    while work:
      v, k = work[-1]
      if k < len(succ[v]):
        work[-1] = (v, k + 1)
        w = succ[v][k]
        if index_of[w] == -1:
          index_of[w] = low[w] = counter
          counter += 1
          stack.append(w)
          on_stack[w] = True
          work.append((w, 0))
        elif on_stack[w]:
          low[v] = min(low[v], index_of[w])
        continue
      work.pop()
      if work:
        low[work[-1][0]] = min(low[work[-1][0]], low[v])
      if low[v] == index_of[v]:
        comp = []
        while True:
          w = stack.pop()
          on_stack[w] = False
          comp.append(w)
          if w == v:
            break
        sccs.append(sorted(comp))
  sccs.reverse()
  return sccs


def coupling_components(analyses: Sequence[Any]) -> List[Component]:
  """Split the analyses of an MDA into strongly connected components in dependency (topological) order."""
  n = len(analyses)
  produced_by: Dict[int, List[int]] = {}
  for i, da in enumerate(analyses):
    for idx in _outputs(da):
      produced_by.setdefault(idx, []).append(i)
  succ: List[List[int]] = [[] for _ in range(n)]
  for j, da in enumerate(analyses):
    for idx in _inputs(da):
      for i in produced_by.get(idx, ()):
        if j not in succ[i]:
          succ[i].append(j)

  sccs = _strongly_connected(succ)

  out = []
  for comp in sccs:
    feedback = len(comp) > 1 or comp[0] in succ[comp[0]]
    produced = {idx for i in comp for idx in _outputs(analyses[i])}
    consumed = {idx for i in comp for idx in _inputs(analyses[i])}
    couplings = sorted(produced & consumed) if feedback else []
    out.append(Component(analyses=[analyses[i] for i in comp], feedback=feedback, couplings=couplings))
  return out


# --------------------------------------------------------------------------------------------- helpers
def _as_array(value: Any) -> np.ndarray:
  return np.atleast_1d(np.asarray(value, dtype=float)).ravel()


def _flatten(state: State, indices: Sequence[int]) -> np.ndarray:
  if not indices:
    return np.zeros(0)
  return np.concatenate([_as_array(state[i]) for i in indices])


def _unflatten(state: State, indices: Sequence[int], vec: np.ndarray) -> State:
  out = dict(state)
  pos = 0
  for i in indices:
    size = _as_array(state[i]).size
    chunk = vec[pos:pos + size]
    pos += size
    out[i] = list(map(float, chunk)) if isinstance(state[i], (list, tuple, np.ndarray)) else float(chunk[0])
  return out


def _scaled_residual(y: np.ndarray, fy: np.ndarray) -> float:
  if y.size == 0:
    return 0.0
  return float(np.max(np.abs(y - fy) / np.maximum(1.0, np.abs(y))))


def _relax(old: Any, new: Any, w: float) -> Any:
  if w == 1.0:
    return copy.deepcopy(new)
  if isinstance(old, (list, tuple, np.ndarray)):
    return list(map(float, _as_array(old) + w * (_as_array(new) - _as_array(old))))
  return float(old) + w * (float(new) - float(old))


@dataclass
class MDAResult:
  """Outcome of an MDA solve."""
  converged: bool = True
  iterations: int = 0
  residual: float = 0.0
  evaluations: int = 0
  #: Per-component results of the feedback loops, in execution order.
  loops: List["MDAResult"] = field(default_factory=list)


# --------------------------------------------------------------------------------------------- solvers
class MDASolver:
  """Base class of an MDA solver. ``solve`` converges one feedback component in place on ``state``."""
  name = "base"
  _options = ("tol", "maxiter")

  def __init__(self, tol: float = 1e-10, maxiter: int = 100, **options):
    unknown = set(options) - set(self._options)
    if unknown:
      raise ValueError(f"Unknown option(s) {sorted(unknown)} for the {self.name} MDA solver; "
                       f"allowed: {sorted(set(self._options) | {'tol', 'maxiter'})}.")
    self.tol = float(tol)
    self.maxiter = int(maxiter)
    for key, value in options.items():
      setattr(self, key, value)

  def solve(self, component: Component, state: State, evaluate: Evaluate) -> MDAResult:  # pragma: no cover
    raise NotImplementedError

  def __repr__(self) -> str:
    return f"{type(self).__name__}(tol={self.tol}, maxiter={self.maxiter})"


@mda_solvers.register("GS", aliases=("gauss_seidel", "gauss-seidel", "gaussseidel", "nlbgs"))
class GaussSeidelSolver(MDASolver):
  """Nonlinear block Gauss-Seidel: each analysis immediately sees the outputs computed before it in the sweep."""
  name = "GS"
  _options = ("tol", "maxiter", "relaxation")
  relaxation = 1.0
  jacobi = False

  def solve(self, component: Component, state: State, evaluate: Evaluate) -> MDAResult:
    w = float(self.relaxation)
    res = MDAResult(converged=False)
    for it in range(1, self.maxiter + 1):
      y_old = _flatten(state, component.couplings)
      base = dict(state)
      for da in component.analyses:
        outs = evaluate(da, base if self.jacobi else state)
        res.evaluations += 1
        for idx, val in outs.items():
          state[idx] = _relax(state[idx], val, w) if idx in component.couplings and idx in state else val
      res.iterations = it
      res.residual = _scaled_residual(y_old, _flatten(state, component.couplings))
      if res.residual <= self.tol:
        res.converged = True
        break
    return res


@mda_solvers.register("Jacobi", aliases=("nlbj",))
class JacobiSolver(GaussSeidelSolver):
  """Nonlinear block Jacobi: every analysis of a sweep uses the outputs of the previous sweep."""
  name = "Jacobi"
  jacobi = True


@mda_solvers.register("Newton", aliases=("newton_raphson", "nr"))
class NewtonSolver(MDASolver):
  """Newton's method on R(y) = y - F(y) over the loop couplings, forward-difference Jacobian, backtracking step."""
  name = "Newton"
  _options = ("tol", "maxiter", "fd_step", "max_backtracks")
  fd_step = 1e-7
  max_backtracks = 8

  def solve(self, component: Component, state: State, evaluate: Evaluate) -> MDAResult:
    idx = component.couplings
    res = MDAResult(converged=False)
    last_outs: State = {}

    def F(y: np.ndarray) -> np.ndarray:
      trial = _unflatten(state, idx, y)
      outs: State = {}
      for da in component.analyses:   # Jacobi-style: all analyses read the same trial couplings
        outs.update(evaluate(da, trial))
        res.evaluations += 1
      last_outs.clear()
      last_outs.update(outs)
      return _flatten({**trial, **outs}, idx)

    y = _flatten(state, idx)
    fy = F(y)
    r = y - fy
    for it in range(1, self.maxiter + 1):
      res.iterations = it
      res.residual = _scaled_residual(y, fy)
      if res.residual <= self.tol:
        res.converged = True
        break
      n = y.size
      J = np.eye(n)
      for k in range(n):
        h = float(self.fd_step) * max(1.0, abs(y[k]))
        yp = y.copy()
        yp[k] += h
        J[:, k] -= (F(yp) - fy) / h
      try:
        step = np.linalg.solve(J, -r)
      except np.linalg.LinAlgError:
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
      alpha, norm0 = 1.0, float(np.linalg.norm(r))
      for _ in range(int(self.max_backtracks) + 1):
        y_new = y + alpha * step
        f_new = F(y_new)
        if float(np.linalg.norm(y_new - f_new)) < norm0 or alpha < 2.0 ** -int(self.max_backtracks):
          break
        alpha *= 0.5
      y, fy = y_new, f_new
      r = y - fy
    else:
      res.residual = _scaled_residual(y, fy)
      res.converged = res.residual <= self.tol
    # Finish with the fixed-point update y <- F(y) at the final couplings (as GS and Jacobi do), so the outputs respond
    # to input changes below the tolerance; finite-difference gradients of the optimizer would otherwise vanish.
    fy = F(y)
    state.update(_unflatten(state, idx, fy))
    for k, v in last_outs.items():
      if k not in idx:
        state[k] = v
    return res


# --------------------------------------------------------------------------------------------- factory
def create_mda_solver(spec: Any = None) -> MDASolver:
  """Build an MDA solver from ``MDA.<id>.solver``: ``None`` (GS), a name, a mapping ``{name: ..., <options>}`` or an
  :class:`MDASolver` instance."""
  if isinstance(spec, MDASolver):
    return spec
  if spec is None:
    return mda_solvers.get(DEFAULT_MDA_SOLVER)()
  if isinstance(spec, str):
    return mda_solvers.get(spec.strip())()
  if isinstance(spec, Mapping):
    options = dict(spec)
    name = options.pop("name", options.pop("type", DEFAULT_MDA_SOLVER))
    return mda_solvers.get(str(name).strip())(**options)
  raise TypeError(f"MDA solver must be a name or a mapping with a 'name' key, not {spec!r}.")


def resolve_name(spec: Any) -> Optional[str]:
  """The registered (canonical) name of a solver spec, for logging."""
  if spec is None:
    return DEFAULT_MDA_SOLVER
  if isinstance(spec, MDASolver):
    return spec.name
  name = spec if isinstance(spec, str) else dict(spec).get("name", dict(spec).get("type", DEFAULT_MDA_SOLVER))
  return mda_solvers.canonical_name(str(name).strip())
