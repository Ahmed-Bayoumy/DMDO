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
"""Sellar with monolithic MDF: ``python sellar_mdf.py [GS|Jacobi|Newton]``."""
import os
import sys

import numpy as np

from DMDO import build_mdo, load_setup


def sellar_a1(x):
  return [x[0] + x[1] ** 2 + x[2] - 0.2 * x[3]]


def sellar_a2(x):
  return [x[0] + x[1] + np.sqrt(x[2])]


def sellar_mdf_opt(x, y, *args):
  """x = [x, z1, z2], y = [y1, y2]."""
  return [x[0] ** 2 + x[2] + y[0] + np.exp(-y[1]), [3.16 - y[0], y[1] - 24.0]]


BLACKBOXES = {"sellar_a1": sellar_a1, "sellar_a2": sellar_a2, "sellar_mdf_opt": sellar_mdf_opt}


def run(solver: str = "GS"):
  csd = os.path.dirname(os.path.abspath(__file__))
  data = load_setup(os.path.join(csd, "sellar_mdf.yaml"))
  data["MDA"]["MDA1"]["solver"] = solver
  mdo = build_mdo(data, working_dir=csd, mdo_name=f"sellar_mdf_{solver}", blackboxes=BLACKBOXES)
  return mdo.run()


if __name__ == "__main__":
  print(run(sys.argv[1] if len(sys.argv) > 1 else "GS"))
