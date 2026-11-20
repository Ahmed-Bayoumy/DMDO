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
"""ask/tell with your own scheduler: solve each Jacobi batch on a concurrent.futures process pool.

Any scheduler works the same way (Dask, Ray, a job queue, another framework's parallel evaluator): ship each
SubproblemTask to a worker that calls DMDO.solve_task(task) and hand the SubproblemResults back with tell().

Run from the repository root:  python examples/ask_tell_process_pool.py
"""
import os
from concurrent.futures import ProcessPoolExecutor

from basic_disciplines import BLACKBOXES

from DMDO import Session, load_setup, solve_task

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
  data = load_setup(os.path.join(HERE, "basic_problem.yaml"))
  data["coord"]["c1"]["budget"] = 8
  data["MDO"]["display"] = False     # no per-iteration console lines
  with Session.from_dict(data, working_dir=os.path.join(HERE, "_runs"), mdo_name="pool", blackboxes=BLACKBOXES,
                         sweep="jacobi") as session, ProcessPoolExecutor(max_workers=2) as pool:
    while not session.done:
      tasks = session.ask()
      record = session.tell(list(pool.map(solve_task, tasks)))
      if record is not None:
        print(f"iteration {record.iteration}: max|q| = {record.max_inconsistency:.3e}")
    print(session.result().to_dict()["variables"])


if __name__ == "__main__":  # required for process pools on Windows/macOS
  main()
