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
"""Unit tests for the plug-in architecture: registries, solver strategies, Discipline adapters, blackbox resolution."""
import numpy as np
import pytest

from DMDO import (
  COUPLING_TYPE, DA, BaseSolver, Discipline, Registry, SolverOutcome, SubproblemSolver, blackboxes,
  register_blackbox, solvers, variableData,
)
from DMDO.app.builder import problemSetup
from DMDO.registry import RegistryError
from DMDO.solvers import MADSSolver, PollSolver, ScipySolver, SearchSolver, create_solver


# --------------------------------------------------------------------------------------------- registry
def test_registry_is_case_insensitive_with_aliases():
  reg = Registry("thing")
  reg.register("Alpha", 1, aliases=("a", "ALF"))
  assert reg.get("alpha") == 1 and reg.get("A") == 1 and reg.get("alf") == 1
  assert reg.canonical_name("ALF") == "alpha"
  assert "alpha" in reg and "beta" not in reg


def test_registry_decorator_and_override():
  reg = Registry("thing")

  @reg.register("f")
  def f():
    return 1

  assert reg.get("f") is f
  with pytest.raises(ValueError):
    reg.register("f", lambda: 2)
  reg.register("f", 3, override=True)
  assert reg.get("f") == 3


def test_registry_import_path_and_unknown_names():
  reg = Registry("thing")
  assert reg.get("numpy:sqrt") is np.sqrt
  assert "numpy:sqrt" in reg
  assert "numpy:not_a_function" not in reg
  with pytest.raises(RegistryError):
    reg.get("nothing_here")


def test_registry_loads_entry_points_lazily(monkeypatch):
  class FakeEP:
    name = "plugged"

    def load(self):
      return "loaded"

  from DMDO import registry as registry_module
  monkeypatch.setattr(registry_module.metadata, "entry_points", lambda group: [FakeEP()] if group == "grp" else [])
  reg = Registry("thing", "grp")
  assert reg.names() == ["plugged"]
  assert reg.get("PLUGGED") == "loaded"


# --------------------------------------------------------------------------------------------- solvers
@pytest.mark.parametrize("name, cls", [("mads", MADSSolver), ("OMADS", MADSSolver), ("omads", MADSSolver),
                                       ("poll", PollSolver), ("search", SearchSolver), ("scipy", ScipySolver)])
def test_builtin_solvers_are_registered(name, cls):
  impl = create_solver(name)
  assert isinstance(impl, cls)
  assert isinstance(impl, SubproblemSolver)


def test_solver_contract_flags():
  assert create_solver("scipy").objective_form == "scalar"
  assert create_solver("mads").objective_form == "vector"
  assert create_solver("mads").rejects_out_of_bounds and not create_solver("scipy").rejects_out_of_bounds
  assert create_solver("poll").refresh_dependent_baseline and not create_solver("mads").refresh_dependent_baseline
  assert create_solver("scipy").report_hmin(0.5) == [0.5]
  assert create_solver("mads").report_hmin([0.1, 0.2]) == [0.1, 0.2]


def test_scipy_settings_defaults_when_incomplete():
  warnings = []
  settings = create_solver("scipy").prepare_settings(None, 3, warnings.append)
  assert settings["method"] == "SLSQP" and settings["is_con"] is False
  assert warnings and "subproblem # 3" in warnings[0]
  given = {"method": "COBYLA", "options": {}, "tol": 1e-6, "is_con": True}
  assert create_solver("scipy").prepare_settings(given, 1, warnings.append) is given


def _toy_subproblem(solver):
  from DMDO import ADMM, MDA, SubProblem, w_scheme
  x = variableData(index=1, sp_index=1, name="x", dim=1, coupling_type=COUPLING_TYPE.UNCOUPLED, link=None,
                   baseline=1., scaling=1., lb=0., value=1., ub=2., type="REAL")
  y = variableData(index=2, sp_index=1, name="y", dim=1, coupling_type=COUPLING_TYPE.UNCOUPLED, link=None,
                   baseline=1., scaling=1., lb=0., value=1., ub=2., type="REAL")
  da = DA(inputs=[x], outputs=[y], blackbox=lambda v: [v[0]], links=None, coupling_type=COUPLING_TYPE.UNCOUPLED)
  mda = MDA(nAnalyses=1, analyses=[da], variables=[x], responses=[y])
  coord = ADMM(beta=1.3, nsp=1, budget=1, index_of_master_SP=1, display=False, scaling=[1., 1.], mode="serial",
               M_update_scheme=w_scheme.NORMAL, store_q_io=False)
  return SubProblem(nv=1, index=1, vars=[x], resps=[y], is_main=1, analysis=mda, coordination=coord,
                    opt=lambda a, b, c=None: [a[0], [0.]], fmin_nop=np.inf, budget=5, display=False, solver=solver)


def test_custom_solver_plugs_into_subproblem():
  class MySolver(BaseSolver):
    def solve(self, request):
      return SolverOutcome(x=list(request.x0), fmin=0., hmin=0., raw={"ok": True})

  solvers.register("unit_test_solver", MySolver, override=True)
  try:
    sp = _toy_subproblem("Unit_Test_Solver")
    assert sp.solver == "unit_test_solver"
    assert isinstance(sp._solver_impl(), MySolver)
  finally:
    solvers.unregister("unit_test_solver")


def test_unknown_solver_falls_back_to_mads_with_warning(caplog):
  with caplog.at_level("WARNING"):
    sp = _toy_subproblem("does-not-exist")
  assert sp.solver == "mads"
  assert "Inappropriate solver method" in caplog.text


def test_omads_default_name_is_canonicalised_without_scipy_settings():
  sp = _toy_subproblem("OMADS")
  assert sp.solver == "mads" and sp.scipy is None


def test_solver_cache_is_not_pickled():
  sp = _toy_subproblem("poll")
  sp._solver_impl()
  state = sp.__getstate__()
  assert "_solver_cache" not in state
  restored = type(sp).__new__(type(sp))
  restored.__setstate__(state)
  assert isinstance(restored._solver_impl(), PollSolver)


# --------------------------------------------------------------------------------------------- disciplines
def _var(name, dim=1, value=1.):
  return variableData(index=1, sp_index=1, name=name, dim=dim, coupling_type=COUPLING_TYPE.UNCOUPLED, link=None,
                      baseline=value, scaling=1., lb=0., value=value, ub=10., type="REAL")


class _Doubler:
  def __init__(self):
    self.calls = []

  def compute(self, inputs):
    self.calls.append(dict(inputs))
    return {"y": 2 * inputs["a"], "z": inputs["a"] + inputs["b"], "unused": 0.}


def test_discipline_protocol_detection():
  assert isinstance(_Doubler(), Discipline)
  assert not isinstance(lambda x: x, Discipline)


def test_da_runs_a_discipline_by_name():
  disc = _Doubler()
  da = DA(inputs=[_var("a", value=3.), _var("b", value=4.)], outputs=[_var("y"), _var("z")], blackbox=disc,
          links=None, coupling_type=COUPLING_TYPE.UNCOUPLED)
  da.run()
  assert disc.calls == [{"a": 3., "b": 4.}]
  assert [o.value for o in da.outputs] == [6., 7.]


def test_da_discipline_vector_output_and_missing_output():
  class Vec:
    def compute(self, inputs):
      return {"v": np.array([1., 2., 3.])}

  da = DA(inputs=[_var("a")], outputs=[_var("v", dim=3, value=[0., 0., 0.])], blackbox=Vec(), links=None,
          coupling_type=COUPLING_TYPE.UNCOUPLED)
  da.run()
  assert list(da.outputs[0].value) == [1., 2., 3.]

  bad = DA(inputs=[_var("a")], outputs=[_var("w")], blackbox=Vec(), links=None, coupling_type=COUPLING_TYPE.UNCOUPLED)
  with pytest.raises(KeyError, match="w"):
    bad.run()


# --------------------------------------------------------------------------------------------- blackbox resolution
def test_problem_setup_resolves_callable_names():
  def local_fn(x):
    return x

  setup = problemSetup(data={}, blackboxes={"local": local_fn})
  assert setup.setBlackboxes("local", "callable") is local_fn
  assert setup.setBlackboxes(local_fn, "callable") is local_fn
  assert setup.setBlackboxes("numpy:cos", "callable") is np.cos
  # Unresolved names are kept so they can still be bound after the build (legacy workflow).
  assert setup.setBlackboxes("not_registered_anywhere", "callable") == "not_registered_anywhere"

  register_blackbox("unit_test_bb", local_fn)
  try:
    assert problemSetup(data={}).setBlackboxes("unit_test_bb", "callable") is local_fn
  finally:
    blackboxes.unregister("unit_test_bb")
