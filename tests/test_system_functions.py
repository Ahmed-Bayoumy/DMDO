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
"""System-wide functions of augmented Lagrangian coordination (Tosserams et al. 2010, Eqs. 5-8)."""
from types import SimpleNamespace

import numpy as np
import pytest

from DMDO.coordination.admm import ADMM
from DMDO.coordination.system_functions import SystemFunction


def _mv(values, sps):
  return [SimpleNamespace(value=v, sp_index=s) for v, s in zip(values, sps)]


def _coord(funcs, master, master_old=None):
  c = ADMM.__new__(ADMM)
  c.system_functions = funcs
  c.master_vars = master
  c.master_vars_old = master_old if master_old is not None else master
  c.q, c.qold = np.array([0.1, -0.2]), np.array([0.1, -0.2])
  c.beta, c.gamma, c.w_init = 2.0, 0.5, 3.0
  return c


def test_inequality_value_slack_scaling_and_owners():
  master = _mv([1.0, 2.0, 0.25], [1, 2, 3])
  g = SystemFunction("g", "ineq", lambda x: x[0] + x[1] - 4.0, inputs=[1, 2], slack=3, scaling=2.0)
  g.bind(master)
  assert g.subproblems == [1, 2, 3]
  assert g.value(master) == pytest.approx(2.0 * (3.0 - 4.0 + 0.25))


def test_kind_and_slack_validation():
  with pytest.raises(ValueError):
    SystemFunction("g", "ineq", sum, inputs=[1])
  with pytest.raises(ValueError):
    SystemFunction("h", "eq", sum, inputs=[1], slack=2)
  with pytest.raises(ValueError):
    SystemFunction("x", "le", sum, inputs=[1])


def test_penalty_only_in_dependent_subproblems_and_extended_q():
  master = _mv([1.0, 3.0], [1, 2])
  h = SystemFunction("h", "eq", lambda x: x[0] - 0.5, inputs=[1])
  h.bind(master)
  c = _coord([h], master)
  # q = 0.5, v = 0, w = w_init = 3: phi = (3 * 0.5)^2
  assert c.system_penalty(1) == pytest.approx(2.25)
  assert c.system_penalty(2) == 0.0
  assert c.max_inconsistency() == pytest.approx(0.5)
  assert list(c.q_extended()) == pytest.approx([0.1, -0.2, 0.5])


def test_multiplier_update():
  old = _mv([0.6], [1])
  new = _mv([1.0], [1])
  h = SystemFunction("h", "eq", lambda x: x[0] - 0.5, inputs=[1])
  h.bind(new)
  c = _coord([h], new, old)
  c.update_system_multipliers()
  # v += 2 w^2 q with w = 3, q = 0.5; |q| = 0.5 > gamma |q_old| = 0.05 -> w *= beta
  assert c.v_sw[0] == pytest.approx(9.0)
  assert c.w_sw[0] == pytest.approx(6.0)
  assert c.q_sw_old[0] == pytest.approx(0.1)


def test_no_system_functions_leaves_q_unchanged():
  c = _coord([], _mv([1.0], [1]))
  assert c.system_penalty(1) == 0.0
  assert list(c.q_extended()) == pytest.approx([0.1, -0.2])
  assert c.max_inconsistency() == pytest.approx(0.2)


def test_initial_weight_rule():
  master = _mv([1.0], [1])
  h = SystemFunction("h", "eq", lambda x: x[0] - 0.5, inputs=[1])
  h.bind(master)
  c = _coord([h], master)
  c.w, c.v = [0.001, 0.001], [1.0, 1.0]
  c.w_init_f_hat = None
  assert c.apply_initial_weight_rule() is False
  c.w_init_f_hat, c.w_init_alpha = 1.0e4, 0.1
  assert c.apply_initial_weight_rule() is True
  # ||q0||_inf = 0.5 (system-wide function); w = sqrt(0.1 * 1e4) / 0.5
  assert c.w == pytest.approx([np.sqrt(1000.0) / 0.5] * 2)
  assert c.v == [0.0, 0.0]
  assert c.w_sw[0] == pytest.approx(np.sqrt(1000.0) / 0.5) and c.v_sw[0] == 0.0
