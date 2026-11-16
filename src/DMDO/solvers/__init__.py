"""Built-in subproblem solver strategies, registered in :data:`DMDO.registry.solvers`.

=========  ============================================  =========================
name       strategy                                      aliases
=========  ============================================  =========================
mads       :class:`MADSSolver` (OMADS search + poll)     ``OMADS``, ``omads``
poll       :class:`PollSolver` (OMADS poll step)
search     :class:`SearchSolver` (OMADS search step)
scipy      :class:`ScipySolver` (scipy.optimize)
=========  ============================================  =========================

Third-party solvers register through :func:`DMDO.registry.solvers.register` or the ``dmdo.solvers`` entry-point group.
"""
from ..registry import solvers
from .base import BaseSolver
from .omads_solver import MADSSolver, OMADSSolver, PollSolver, SearchSolver
from .scipy_solver import ScipySolver

DEFAULT_SOLVER = "mads"

for _name, _cls, _aliases in (
  ("mads", MADSSolver, ("omads",)),
  ("poll", PollSolver, ()),
  ("search", SearchSolver, ()),
  ("scipy", ScipySolver, ()),
):
  _cls.name = _name
  solvers.register(_name, _cls, aliases=_aliases, override=True)


def create_solver(name: str) -> BaseSolver:
  """Instantiate the solver strategy registered under ``name`` (aliases and import paths accepted)."""
  return solvers.get(name)()


__all__ = ["BaseSolver", "OMADSSolver", "MADSSolver", "PollSolver", "SearchSolver", "ScipySolver",
           "DEFAULT_SOLVER", "create_solver"]
