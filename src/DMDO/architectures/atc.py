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
"""Hierarchical analytical target cascading (``architecture: {DIST: {IDF: ATC}}``).

The subproblem with ``is_main: True`` is the system-level element (the root). The other subproblems hang below it
through the variable links: the subproblems linked to the root are level 1, the subproblems linked to those (other
than their parent) are level 2, and so on. The links must form a tree, so that every target/response pair connects a
parent to one of its children; a cycle (for example three subproblems linked to each other) is rejected.

The relaxation and the multiplier updates are the ADMM coordinator's (augmented Lagrangian ``v q + |w q|^2``, method
of multipliers): ATC with ADMM is the alternating-direction variant of Tosserams et al. (2006), with one pass over
the hierarchy per outer iteration. Each outer iteration solves the hierarchy level by level from the root down; the
subproblems of one level share no pairs, so they are solved as one batch (in parallel with a parallel executor) from
the same coordinator state.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


class HierarchyError(ValueError):
  """The subproblem links do not form a tree rooted at the ``is_main`` subproblem."""


@dataclass
class Hierarchy:
  """The ATC tree of a run."""
  #: Subproblem index of the root (the ``is_main`` subproblem).
  root: int
  #: Parent subproblem index of every non-root subproblem.
  parent: Dict[int, int]
  #: Subproblem indices per level, root first; within a level in the order of ``MDO.subproblems``.
  levels: List[List[int]]
  children: Dict[int, List[int]] = field(default_factory=dict)

  def describe(self) -> str:
    return " -> ".join("[" + ", ".join(f"SP{s}" for s in level) + "]" for level in self.levels)


def _links(var) -> List[int]:
  link = var.link
  out = []
  for item in (link if isinstance(link, (list, tuple)) else [link]):
    if item is None or str(item) == "None":
      continue
    out.append(int(item))
  return out


def subproblem_edges(subproblems: Sequence) -> List[Tuple[int, int]]:
  """Undirected subproblem links ``(a, b)``, ``a < b``, from the links of their variables and responses."""
  indices = {int(sp.index) for sp in subproblems}
  edges = set()
  for sp in subproblems:
    own = int(sp.index)
    for v in list(sp.vars) + list(getattr(sp.MDA_process, "responses", None) or []):
      for other in _links(v):
        if other == own:
          continue
        if other not in indices:
          raise HierarchyError(f"DIST: IDF: ATC: variable '{v.name}' of subproblem {own} is linked to subproblem "
                               f"{other}, which is not in MDO.subproblems.")
        edges.add((min(own, other), max(own, other)))
  return sorted(edges)


def build_hierarchy(subproblems: Sequence, root: Optional[int] = None) -> Hierarchy:
  """The ATC tree from the subproblem links, rooted at the ``is_main`` subproblem (or ``root``)."""
  order = [int(sp.index) for sp in subproblems]
  if root is None:
    mains = [int(sp.index) for sp in subproblems if sp.is_main]
    if len(mains) != 1:
      raise HierarchyError(f"DIST: IDF: ATC needs exactly one system-level subproblem (is_main: True); found {mains}.")
    root = mains[0]
  adjacency: Dict[int, List[int]] = {s: [] for s in order}
  for a, b in subproblem_edges(subproblems):
    adjacency[a].append(b)
    adjacency[b].append(a)
  for s in adjacency:
    adjacency[s].sort(key=order.index)

  parent: Dict[int, int] = {}
  levels: List[List[int]] = [[root]]
  seen = {root}
  while True:
    nxt: List[int] = []
    for s in levels[-1]:
      for t in adjacency[s]:
        if t == parent.get(s):
          continue
        if t in seen:
          raise HierarchyError(
            f"DIST: IDF: ATC: the subproblem links contain a cycle through SP{s} and SP{t}, so they do not form a "
            f"tree. ATC only allows parent-child links; remove the link between SP{s} and SP{t} or use "
            f"DIST: IDF: NHATC. Links: {subproblem_edges(subproblems)}.")
        seen.add(t)
        parent[t] = s
        nxt.append(t)
    if not nxt:
      break
    levels.append(sorted(nxt, key=order.index))
  missing = [s for s in order if s not in seen]
  if missing:
    raise HierarchyError(f"DIST: IDF: ATC: subproblem(s) {missing} are not linked to the hierarchy below SP{root}.")
  children = {s: [t for t in adjacency[s] if parent.get(t) == s] for s in order}
  return Hierarchy(root=root, parent=parent, levels=levels, children=children)
