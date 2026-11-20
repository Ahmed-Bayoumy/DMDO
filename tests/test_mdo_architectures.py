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
"""MDO architecture selection (``MDO.architecture``), MDA feedback detection and solvers, MDF and ATC drivers."""

import pytest
import yaml

from DMDO import MDO_ARCHITECTURE, ArchitectureError, ArchitectureSpec, parse_architecture
from DMDO.architectures import DEFAULT_ARCHITECTURE


# --------------------------------------------------------------------------------------------- parser
@pytest.mark.parametrize("text, path", [
  ("MON: MDF", ("MON", "MDF")),
  ("mon: mdf", ("MON", "MDF")),
  ("MON:\n  MDF:", ("MON", "MDF")),
  ("DIST:\n  IDF: ATC", ("DIST", "IDF", "ATC")),
  ("DIST:\n  IDF: NHATC", ("DIST", "IDF", "NHATC")),
  ("DIST:\n  IDF:\n    ATC:", ("DIST", "IDF", "ATC")),
])
def test_known_architecture_paths(text, path):
  spec, warning = parse_architecture(yaml.safe_load(text))
  assert warning is None
  assert spec.path == path


@pytest.mark.parametrize("value", ["IDF", "MDF", "MON:MDF", {"MON": "XYZ"}, {"DIST": "IDF"}, ["MON", "MDF"],
                                   {"MON": "MDF", "DIST": "IDF"}])
def test_unrecognised_values_fall_back_to_nhatc_with_a_warning(value):
  spec, warning = parse_architecture(value)
  assert spec == DEFAULT_ARCHITECTURE and spec.is_nhatc
  assert warning is not None and "NHATC" in warning


def test_missing_key_is_the_silent_default():
  assert parse_architecture(None) == (DEFAULT_ARCHITECTURE, None)


@pytest.mark.parametrize("text", ["MON: IDF", "MON:\n  IDF: BLISS", "DIST:\n  MDF: BLISS"])
def test_out_of_scope_architectures_raise(text):
  with pytest.raises(ArchitectureError, match="not supported"):
    parse_architecture(yaml.safe_load(text))


def test_spec_flags_and_label():
  mdf, _ = parse_architecture({"MON": "MDF"})
  atc, _ = parse_architecture({"DIST": {"IDF": "ATC"}})
  assert mdf.is_monolithic and mdf.is_mdf and not mdf.is_atc and mdf.label == "MON: MDF"
  assert atc.is_atc and not atc.is_monolithic and str(atc) == "DIST: IDF: ATC"
  assert isinstance(DEFAULT_ARCHITECTURE, ArchitectureSpec)


def test_mdo_treats_legacy_enum_as_nhatc():
  from DMDO import MDO
  mdo = MDO.__new__(MDO)
  mdo.Architecture = MDO_ARCHITECTURE.IDF
  assert mdo.architecture.is_nhatc


class _RecordingLog:
  def __init__(self):
    self.messages = []

  def log_msg(self, msg, msg_type):
    self.messages.append((msg_type, msg))


@pytest.mark.parametrize("value, n_warnings", [("IDF", 1), ({"MON": "MDF"}, 0), (None, 0)])
def test_builder_logs_fallback_warning_once(value, n_warnings):
  from DMDO.app.builder import problemSetup
  from DMDO.utils.log import MSG_TYPE

  log = _RecordingLog()
  setup = problemSetup(data={"MDO": {"architecture": value}}, log=log)
  spec = setup.architectureSetup()
  assert spec == parse_architecture(value)[0]
  warnings = [m for t, m in log.messages if t == MSG_TYPE.WARNING.value]
  assert len(warnings) == n_warnings
  # MDOSetup reuses the parsed spec instead of parsing (and warning) again.
  assert setup.architecture is spec


# --------------------------------------------------------------------------------------------- MDA feedback + solvers
import numpy as np  # noqa: E402

from DMDO import DA, MDA, variableData  # noqa: E402
from DMDO.analysis.mda_solvers import (  # noqa: E402
  GaussSeidelSolver, JacobiSolver, NewtonSolver, coupling_components, create_mda_solver,
)
from DMDO.registry import RegistryError  # noqa: E402


def _var(index, name, value=1.0, dim=1):
  return variableData(name=name, sp_index=1, coupling_type=None, link=None, dim=dim, value=value, baseline=value,
                      scaling=1.0, lb=-1e9, ub=1e9, index=index)


def _da(index, inputs, outputs, fn):
  return DA(inputs=inputs, outputs=outputs, blackbox=fn, links=[], coupling_type=None, index=index)


def _sellar_mda(order=(1, 2), solver=None):
  """Sellar's coupled analyses at x = 1, z = (5, 2), y = (y1, y2) unknown."""
  x, z1, z2, y1, y2 = _var(1, "x", 1.0), _var(2, "z1", 5.0), _var(3, "z2", 2.0), _var(4, "y1", 1.0), _var(5, "y2", 1.0)
  da1 = _da(1, [x, z1, z2, y2], [y1], lambda v: [v[1] ** 2 + v[2] + v[0] - 0.2 * v[3]])
  da2 = _da(2, [z1, z2, y1], [y2], lambda v: [v[0] + v[1] + np.sqrt(v[2])])
  das = {1: da1, 2: da2}
  mda = MDA(nAnalyses=2, analyses=[das[i] for i in order], variables=[x, z1, z2], responses=[y1, y2], index=1,
            solver=solver)
  state = {1: 1.0, 2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0}
  return mda, state


def _sellar_reference():
  y1, y2 = 1.0, 1.0
  for _ in range(200):
    y1 = 25 + 2 + 1 - 0.2 * y2
    y2 = 5 + 2 + np.sqrt(y1)
  return y1, y2


def test_feedback_loop_is_detected_by_variable_index():
  mda, _ = _sellar_mda()
  comps = mda.components()
  assert len(comps) == 1 and comps[0].feedback
  assert comps[0].couplings == [4, 5]
  assert mda.has_feedback()


def test_feed_forward_analyses_run_in_dependency_order():
  a, b, c = _var(1, "a", 2.0), _var(2, "b"), _var(3, "c")
  first = _da(1, [a], [b], lambda v: [3 * v[0]])
  second = _da(2, [b], [c], lambda v: [v[0] + 1])
  comps = coupling_components([second, first])        # listed in the wrong order
  assert [c_.analyses[0].index for c_ in comps] == [1, 2]
  assert not any(c_.feedback for c_ in comps)
  mda = MDA(nAnalyses=2, analyses=[second, first], variables=[a], responses=[b, c], index=1)
  state = {1: 2.0, 2: 0.0, 3: 0.0}
  res = mda.run_coupled(state)
  assert state[3] == pytest.approx(7.0) and res.converged and res.evaluations == 2


def test_self_reference_is_a_feedback_loop():
  y = _var(1, "y", 1.0)
  da = _da(1, [y], [y], lambda v: [np.cos(v[0])])
  comps = coupling_components([da])
  assert comps[0].feedback and comps[0].couplings == [1]
  for name in ("GS", "Jacobi", "Newton"):
    mda = MDA(nAnalyses=1, analyses=[da], variables=[], responses=[y], index=1, solver={"name": name, "tol": 1e-12})
    state = {1: 1.0}
    res = mda.run_coupled(state)
    assert res.converged, name
    assert state[1] == pytest.approx(0.7390851332, abs=1e-9), name


@pytest.mark.parametrize("solver", ["GS", "Jacobi", "Newton", {"name": "GS", "relaxation": 0.8},
                                    {"name": "newton", "tol": 1e-12}])
def test_solvers_converge_sellar_couplings(solver):
  ref = _sellar_reference()
  mda, state = _sellar_mda(solver=solver)
  res = mda.run_coupled(state)
  assert res.converged
  assert state[4] == pytest.approx(ref[0], rel=1e-7) and state[5] == pytest.approx(ref[1], rel=1e-7)


def test_newton_needs_fewer_iterations_than_fixed_point():
  iters = {}
  for name in ("GS", "Jacobi", "Newton"):
    mda, state = _sellar_mda(solver={"name": name, "tol": 1e-10})
    iters[name] = mda.run_coupled(state).iterations
  assert iters["Newton"] < iters["GS"] <= iters["Jacobi"]


def test_order_of_the_loop_does_not_change_the_solution():
  a, sa = _sellar_mda(order=(1, 2))
  b, sb = _sellar_mda(order=(2, 1))
  a.run_coupled(sa)
  b.run_coupled(sb)
  assert sa[4] == pytest.approx(sb[4], rel=1e-7) and sa[5] == pytest.approx(sb[5], rel=1e-7)


def test_vector_coupling_with_newton():
  # y = A y + b with y of dimension 2 (one analysis reads and writes the vector): y* = (I - A)^-1 b.
  A, bvec = np.array([[0.2, 0.1], [0.3, 0.4]]), np.array([1.0, 2.0])
  y = _var(1, "y", [0.0, 0.0], dim=2)
  da = _da(1, [y], [y], lambda v: list(A @ np.asarray(v) + bvec))
  mda = MDA(nAnalyses=1, analyses=[da], variables=[], responses=[y], index=1, solver="Newton")
  state = {1: [0.0, 0.0]}
  assert mda.run_coupled(state).converged
  assert np.allclose(state[1], np.linalg.solve(np.eye(2) - A, bvec))


def test_not_converged_is_reported():
  mda, state = _sellar_mda(solver={"name": "GS", "maxiter": 2, "tol": 1e-14})
  res = mda.run_coupled(state)
  assert not res.converged and res.iterations == 2 and res.residual > 1e-14


def test_mda_solver_factory():
  assert isinstance(create_mda_solver(None), GaussSeidelSolver)
  assert isinstance(create_mda_solver("gauss_seidel"), GaussSeidelSolver)
  assert isinstance(create_mda_solver("JACOBI"), JacobiSolver)
  assert isinstance(create_mda_solver({"name": "Newton", "fd_step": 1e-6}), NewtonSolver)
  with pytest.raises(ValueError, match="Unknown option"):
    create_mda_solver({"name": "Newton", "relaxation": 0.5})
  with pytest.raises(RegistryError):
    create_mda_solver("Broyden")


# --------------------------------------------------------------------------------------------- MON: MDF
import copy as _copy  # noqa: E402
import os  # noqa: E402

from DMDO import Session, build_mdo, load_setup, solve_task  # noqa: E402

_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
SELLAR_F_STAR = 3.18339


def _sellar_blackboxes():
  import importlib.util
  spec = importlib.util.spec_from_file_location("sellar_mdf_bb", os.path.join(_EXAMPLES_DIR, "sellar_mdf.py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod.BLACKBOXES


@pytest.fixture(scope="module")
def mdf_data():
  return load_setup(os.path.join(_EXAMPLES_DIR, "sellar_mdf.yaml"))


def _build_mdf(data, tmp_path, name="mdf"):
  mdo = build_mdo(data, working_dir=str(tmp_path), mdo_name=name, blackboxes=_sellar_blackboxes())
  mdo.display = False
  return mdo


@pytest.mark.parametrize("solver", ["GS", "Jacobi", "Newton"])
def test_mdf_sellar_reaches_the_optimum(mdf_data, tmp_path, solver):
  data = _copy.deepcopy(mdf_data)
  data["MDA"]["MDA1"]["solver"] = solver
  mdo = _build_mdf(data, tmp_path, f"mdf_{solver}")
  assert mdo.architecture.is_mdf and mdo.Coordinator is None
  out = mdo.run()
  res = out["subproblems"][0]
  assert out["architecture"] == "MON: MDF"
  assert res["feasible"] and res["mda_converged"]
  assert res["fmin"] == pytest.approx(SELLAR_F_STAR, abs=1e-4)
  assert res["x"]["z1"] == pytest.approx(1.97764, abs=1e-3)
  assert res["y"]["y1"] == pytest.approx(3.16, abs=1e-5)
  # The history file is written and the run's variables hold the solution.
  assert os.path.isfile(os.path.join(mdo.post_dir, "MDF_history.csv"))
  assert {v.name: v.value for v in mdo.variables}["y1"] == pytest.approx(3.16, abs=1e-5)


def test_mdf_newton_uses_fewer_mda_iterations(mdf_data, tmp_path):
  iters = {}
  for solver in ("GS", "Newton"):
    data = _copy.deepcopy(mdf_data)
    data["MDA"]["MDA1"]["solver"] = solver
    iters[solver] = _build_mdf(data, tmp_path, f"it_{solver}").run()["subproblems"][0]["mda_iterations"]
  assert iters["Newton"] < iters["GS"]


def test_mdf_rejects_linked_subproblems(mdf_data, tmp_path):
  data = _copy.deepcopy(mdf_data)
  data["subproblem"]["sp2"] = dict(_copy.deepcopy(data["subproblem"]["sp1"]), index=2, is_main=False)
  data["MDO"]["subproblems"] = [1, 2]
  data["variables"]["v1"][2] = 2                      # z1 of subproblem 1 linked to subproblem 2
  with pytest.raises(ValueError, match="standalone"):
    _build_mdf(data, tmp_path).run()


def test_mdf_has_no_ask_tell_loop(mdf_data, tmp_path):
  mdo = _build_mdf(_copy.deepcopy(mdf_data), tmp_path)
  with pytest.raises(NotImplementedError, match="MDO.run"):
    next(mdo.coordinate())


def test_mda_solver_key_is_ignored_outside_mdf(mdf_data, tmp_path):
  data = _copy.deepcopy(mdf_data)
  data["MDA"]["MDA1"]["solver"] = "Newton"
  data["MDO"]["architecture"] = "IDF"                 # legacy value -> NHATC; the MDA solver is not attached
  data["coord"] = {"c1": {"index": 1, "type": "ADMM", "nsp": 1, "budget": 1, "index_of_master_SP": 1,
                          "display": False, "scaling": None, "mode": "serial", "M_update_scheme": "median",
                          "store_q_io": False}}
  data["subproblem"]["sp1"]["coordinator"] = 1
  data["MDO"]["coordinator"] = 1
  mdo = build_mdo(data, working_dir=str(tmp_path), mdo_name="legacy", blackboxes=_sellar_blackboxes())
  assert mdo.architecture.is_nhatc
  assert mdo.subProblems[0].MDA_process.solver is None
  assert isinstance(Session(mdo), Session)


# --------------------------------------------------------------------------------------------- DIST: IDF: ATC
from types import SimpleNamespace  # noqa: E402

from DMDO.architectures.atc import HierarchyError, build_hierarchy  # noqa: E402


def _sp(index, links, is_main=False):
  """A stand-in subproblem: one variable per link target."""
  vars_ = [SimpleNamespace(name=f"v{index}_{k}", link=target) for k, target in enumerate(links)]
  return SimpleNamespace(index=index, is_main=is_main, vars=vars_, MDA_process=SimpleNamespace(responses=[]))


def test_hierarchy_of_a_star_and_a_chain():
  star = build_hierarchy([_sp(1, [2, 3, 4], True), _sp(2, [1]), _sp(3, [1]), _sp(4, [1])])
  assert star.root == 1 and star.levels == [[1], [2, 3, 4]] and star.parent == {2: 1, 3: 1, 4: 1}
  # SBJ (d): root is aero (2), linked to 1 and 3; list links count once per target.
  chain = build_hierarchy([_sp(1, [2]), _sp(2, [1, 3, [1, 3]], True), _sp(3, [2])])
  assert chain.levels == [[2], [1, 3]] and chain.children[2] == [1, 3]
  deep = build_hierarchy([_sp(1, [2], True), _sp(2, [1, 3]), _sp(3, [2, 4]), _sp(4, [3])])
  assert deep.levels == [[1], [2], [3], [4]] and deep.describe() == "[SP1] -> [SP2] -> [SP3] -> [SP4]"


@pytest.mark.parametrize("sps, match", [
  ([_sp(1, [2, 3], True), _sp(2, [1, 3]), _sp(3, [1, 2])], "cycle"),                 # GP: 1-2, 1-3, 2-3
  ([_sp(1, [2], True), _sp(2, [1, 3]), _sp(3, [2, 4]), _sp(4, [3, 2])], "cycle"),
  ([_sp(1, [2], True), _sp(2, [1]), _sp(3, [])], "not linked"),
  ([_sp(1, [2], True), _sp(2, [1], True)], "exactly one"),
  ([_sp(1, [5], True)], "not in MDO.subproblems"),
])
def test_hierarchy_rejects_non_trees(sps, match):
  with pytest.raises(HierarchyError, match=match):
    build_hierarchy(sps)


def _star_opt_root(x, y, *args):      # root: (a - b)^2
  return [(x[0] - x[1]) ** 2, [0.0]]


def _star_opt_2(x, y, *args):         # child 2: (a - 3)^2
  return [(x[0] - 3.0) ** 2, [0.0]]


def _star_opt_3(x, y, *args):         # child 3: (b + 1)^2
  return [(x[0] + 1.0) ** 2, [0.0]]


STAR_BLACKBOXES = {"opt_root": _star_opt_root, "opt_2": _star_opt_2, "opt_3": _star_opt_3}
# f = (a - b)^2 + (a - 3)^2 + (b + 1)^2: a* = 5/3, b* = 1/3, f* = 16/3.
STAR_A, STAR_B, STAR_F = 5.0 / 3.0, 1.0 / 3.0, 16.0 / 3.0


def _star_problem(architecture, budget=100):
  scipy = {"method": "SLSQP", "is_con": False, "tol": 1e-12, "options": {"maxiter": 200, "ftol": 1e-14}}

  def sp(index, vars_, opt, is_main=False):
    return {"nv": len(vars_), "index": index, "vars": vars_, "resps": [], "is_main": is_main, "MDA": index,
            "coordinator": 1, "opt": opt, "type": "callable", "fmin_nop": "inf", "budget": 200, "display": False,
            "solver": "scipy", "scipy": scipy}

  return {
    "variables": {
      "v1": ["a", 1, 2, "s", -10.0, 0.0, 10.0, 1, "REAL"],
      "v2": ["b", 1, 3, "s", -10.0, 0.0, 10.0, 1, "REAL"],
      "v3": ["a", 2, 1, "s", -10.0, 0.0, 10.0, 1, "REAL"],
      "v4": ["b", 3, 1, "s", -10.0, 0.0, 10.0, 1, "REAL"],
    },
    "DA": {},
    "MDA": {f"MDA{i}": {"index": i, "nAnalyses": 0, "analyses": [], "variables": v, "responses": []}
            for i, v in ((1, ["v1", "v2"]), (2, ["v3"]), (3, ["v4"]))},
    # One multiplier update per outer iteration (the legacy default updates twice, which drives w up until the
    # inner solves cannot reduce q any further and the run stalls short of the optimum).
    "coord": {"c1": {"index": 1, "type": "ADMM", "beta": 1.2, "gamma": 0.8, "nsp": 3, "budget": budget,
                     "index_of_master_SP": 1, "display": False, "scaling": 1.0, "mode": "serial",
                     "M_update_scheme": "median", "store_q_io": False, "single_multiplier_update": True}},
    "subproblem": {"sp1": sp(1, ["v1", "v2"], "opt_root", True), "sp2": sp(2, ["v3"], "opt_2"),
                   "sp3": sp(3, ["v4"], "opt_3")},
    "USER": None,
    "MDO": {"architecture": architecture, "coordinator": 1, "subproblems": [1, 2, 3], "responses": [],
            "fmin": "inf", "hmin": "inf", "display": False, "inc_stop": 1e-9, "stop": "Iteration budget exhausted",
            "tab_inc": [], "noprogress_stop": 100},
  }


def _star_solution(mdo):
  values = {(v.name, v.sp_index): v.value for v in mdo.Coordinator.master_vars}
  a, b = values[("a", 1)], values[("b", 1)]
  f = (a - b) ** 2 + (values[("a", 2)] - 3.0) ** 2 + (values[("b", 3)] + 1.0) ** 2
  return a, b, f, mdo.Coordinator.max_inconsistency()


def test_atc_star_reaches_the_optimum(tmp_path):
  mdo = build_mdo(_star_problem({"DIST": {"IDF": "ATC"}}), working_dir=str(tmp_path), mdo_name="atc",
                  blackboxes=STAR_BLACKBOXES)
  assert mdo.architecture.is_atc and mdo.atc_hierarchy.levels == [[1], [2, 3]]
  mdo.run()
  a, b, f, qmax = _star_solution(mdo)
  assert qmax < 1e-8
  assert a == pytest.approx(STAR_A, abs=1e-5) and b == pytest.approx(STAR_B, abs=1e-5)
  assert f == pytest.approx(STAR_F, abs=1e-6)


def test_atc_and_nhatc_agree_on_the_star(tmp_path):
  sols = {}
  for name, arch in (("atc", {"DIST": {"IDF": "ATC"}}), ("nhatc", {"DIST": {"IDF": "NHATC"}})):
    mdo = build_mdo(_star_problem(arch), working_dir=str(tmp_path), mdo_name=name, blackboxes=STAR_BLACKBOXES)
    mdo.run()
    sols[name] = _star_solution(mdo)
  # On a star the level order equals the Gauss-Seidel order and siblings share no pairs: identical runs.
  assert sols["atc"] == pytest.approx(sols["nhatc"], abs=1e-12)


def test_atc_level_batches_through_session(tmp_path):
  session = Session.from_dict(_star_problem({"DIST": {"IDF": "ATC"}}, budget=3), working_dir=str(tmp_path),
                              mdo_name="atc_session", blackboxes=STAR_BLACKBOXES)
  batches = []
  while not session.done:
    tasks = session.ask()
    if not tasks:
      break
    batches.append(sorted(t.subproblem.index for t in tasks))
    session.tell([solve_task(t) for t in tasks])
  assert batches[:2] == [[1], [2, 3]]          # root first, then both children in one batch
  session.close()


def test_atc_rejects_a_cyclic_link_graph(tmp_path):
  data = _star_problem({"DIST": {"IDF": "ATC"}})
  data["variables"]["v5"] = ["c", 2, 3, "s", -10.0, 0.0, 10.0, 1, "REAL"]   # sibling link 2-3
  data["variables"]["v6"] = ["c", 3, 2, "s", -10.0, 0.0, 10.0, 1, "REAL"]
  data["subproblem"]["sp2"]["vars"].append("v5")
  data["subproblem"]["sp3"]["vars"].append("v6")
  data["MDA"]["MDA2"]["variables"].append("v5")
  data["MDA"]["MDA3"]["variables"].append("v6")
  with pytest.raises(HierarchyError, match="cycle"):
    build_mdo(data, working_dir=str(tmp_path), mdo_name="cyclic", blackboxes=STAR_BLACKBOXES)


def test_atc_parallel_executor_matches_serial(tmp_path):
  sols = {}
  for mode in ("serial", "parallel"):
    mdo = build_mdo(_star_problem({"DIST": {"IDF": "ATC"}}, budget=4), working_dir=str(tmp_path),
                    mdo_name=f"atc_{mode}", blackboxes=STAR_BLACKBOXES)
    mdo.run(mode=mode)
    sols[mode] = _star_solution(mdo)
  assert sols["parallel"] == pytest.approx(sols["serial"], abs=1e-12)
