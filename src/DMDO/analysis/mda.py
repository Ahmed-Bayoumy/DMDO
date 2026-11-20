"""
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
# """



import copy
from dataclasses import dataclass
from genericpath import isfile
import json
from typing import Any, Callable, Dict, List

from .da import DA
from ..core.protocols import Process_data
from .mda_solvers import Component, MDAResult, MDASolver, coupling_components, create_mda_solver


@dataclass
class MDA_data(Process_data):
  nAnalyses: int
  analyses: List[DA]
  index: int = None
  #: ``MDA.<id>.solver`` of the problem definition (a name or a mapping); used only by ``MON: MDF``
  #: (:meth:`run_coupled`). ``None`` means Gauss-Seidel.
  solver: Any = None


def evaluate_analysis(da: DA, state: Dict[int, Any]) -> Dict[int, Any]:
  """Run one analysis with the inputs found in ``state`` (keyed by variable index); return its outputs by index.

  Inputs are replaced by independent copies (as :meth:`MDA.run` does), so the shared variable objects are not
  modified through them."""
  inputs = da.inputs
  for k in range(len(inputs)):
    idx = inputs[k].index
    if idx in state:
      v = inputs[k].clone() if hasattr(inputs[k], "clone") else copy.deepcopy(inputs[k])
      v.value = copy.deepcopy(state[idx])
      inputs[k] = v
  da.run()
  return {o.index: copy.deepcopy(o.value) for o in da.outputs}


@dataclass
class MDA(MDA_data):

  def setup(self, input):
    data: Dict = {}
    if isfile(input):
      data = json.load(input)

    if isinstance(input, dict):
      self(**data)

  def run(self):
    # Each analysis input gets an independent copy of the variable with the same index (the last one wins, as in
    # the former all-pairs loop).
    by_index = {}
    for v in self.variables:
      by_index[v.index] = v
    for i in range(self.nAnalyses):
      inputs = self.analyses[i].inputs
      for k in range(len(inputs)):
        src = by_index.get(inputs[k].index)
        if src is not None:
          inputs[k] = src.clone() if hasattr(src, "clone") else copy.deepcopy(src)
      self.analyses[i].run()

  def components(self) -> List[Component]:
    """The analyses grouped into feed-forward and feedback components, in dependency order."""
    key = tuple(id(da) for da in self.analyses)
    cached = self.__dict__.get("_components")
    if cached is None or cached[0] != key:
      cached = (key, coupling_components(self.analyses))
      self.__dict__["_components"] = cached
    return cached[1]

  def has_feedback(self) -> bool:
    return any(c.feedback for c in self.components())

  def mda_solver(self) -> MDASolver:
    cached = self.__dict__.get("_mda_solver")
    if cached is None or cached[0] is not self.solver:
      cached = (self.solver, create_mda_solver(self.solver))
      self.__dict__["_mda_solver"] = cached
    return cached[1]

  def run_coupled(self, state: Dict[int, Any], evaluate: Callable = evaluate_analysis) -> MDAResult:
    """Converge the analyses on ``state`` (variable index -> value), updated in place.

    Feed-forward analyses run once in dependency order; each feedback loop is converged with :attr:`solver`."""
    total = MDAResult()
    for comp in self.components():
      if not comp.feedback:
        for da in comp.analyses:
          state.update(evaluate(da, state))
          total.evaluations += 1
        continue
      res = self.mda_solver().solve(comp, state, evaluate)
      total.loops.append(res)
      total.evaluations += res.evaluations
      total.iterations = max(total.iterations, res.iterations)
      total.residual = max(total.residual, res.residual)
      total.converged = total.converged and res.converged
    return total

  def validation(self, vType: int):
    self.term_status = []
    for i in range(len(self.term_critteria)):
      if self.term_type[i] == vType:
          self.term_status.append(self.term_critteria[i])

  def setInputs(self, values: List[Any]):
    # run() copies every variable it hands to an analysis, so a shallow list is enough here.
    self.variables = list(values)

  def getOutputs(self):
    out = []
    if self.responses is None:
      return [None]
    for i in range(len(self.responses)):
      out.append(self.responses[i].value)
    return out


