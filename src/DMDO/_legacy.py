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
"""Load the legacy module aliases (``DMDO.SP``, ``DMDO.coordinator``, ...) of the pre-subpackage layout.

Each legacy path is a thin module that aliases its new location (e.g. ``DMDO.SP`` -> ``DMDO.subproblem.subproblem``),
so old imports and checkpoints pickled by earlier versions keep working.

They are imported here, before ``DMDO/__init__.py`` binds its public names, to reproduce the old import order: the
package attributes ``DMDO.MDO``, ``DMDO.DA``, ``DMDO.MDA`` (classes) and ``DMDO.coordinator`` (protocol) then take
precedence over the same-named legacy modules. Importing a legacy alias for the first time later, for instance while
unpickling an old checkpoint, would otherwise rebind the package attribute to the module.
"""
import importlib

LEGACY_MODULES = ("_globals", "variables", "_protocols", "_common", "_decisor", "_detector", "coordinator", "DA",
                  "MDA", "SP", "MDO", "preprocess", "DMDO")

for _name in LEGACY_MODULES:
  importlib.import_module(f"{__package__}.{_name}")
