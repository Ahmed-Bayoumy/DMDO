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
"""Build a DMDO problem from YAML and drive it through the Session API: run, step, observers and ask/tell.

Run from the repository root:  python examples/session_basics.py
"""
import os

from basic_disciplines import BLACKBOXES

from DMDO import RunObserver, Session, load_setup, solve_task

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.join(HERE, "basic_problem.yaml")
WORK = os.path.join(HERE, "_runs")


class Progress(RunObserver):
  """Receives one IterationRecord per outer iteration (could push to a dashboard or a message queue)."""

  def on_iteration_end(self, run, record):
    print(f"  [observer] iteration {record.iteration}: objective={float(record.objective):.6f} "
          f"max|q|={record.max_inconsistency:.3e}")


def main():
  data = load_setup(SETUP)
  data["coord"]["c1"]["budget"] = 8   # problem definitions are plain dictionaries: adjust before building
  data["MDO"]["display"] = False     # no per-iteration console lines

  # 1. Run to the end.
  result = Session.from_dict(data, working_dir=WORK, mdo_name="run", blackboxes=BLACKBOXES).run()
  print(f"run():  {result.status}, {result.iterations} iterations, objective {result.objective}, "
        f"max|q| {result.max_inconsistency:.3e} ({result.stop_reason})")

  # 2. One outer iteration at a time, with an observer.
  with Session.from_dict(data, working_dir=WORK, mdo_name="step", blackboxes=BLACKBOXES,
                         observers=[Progress()]) as session:
    while (record := session.step()) is not None:
      if record.max_inconsistency < 3e-2:
        session.request_stop()          # e.g. an outer driver decides it is good enough
    print(f"step(): stopped after {session.iteration} iterations ({session.result().stop_reason})")

  # 3. ask/tell: the caller solves the subproblems (here in-process; could be another framework or a cluster).
  with Session.from_dict(data, working_dir=WORK, mdo_name="asktell", blackboxes=BLACKBOXES,
                         sweep="jacobi") as session:
    while not session.done:
      tasks = session.ask()                       # independent tasks: solve them anywhere, in any order
      results = [solve_task(task) for task in tasks]
      session.tell(results)
    print(f"ask/tell (Jacobi): objective {session.result().objective}")


if __name__ == "__main__":
  main()
