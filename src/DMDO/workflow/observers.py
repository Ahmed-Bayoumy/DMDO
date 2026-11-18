"""Built-in run observers: coordination history file, console/log progress and resume checkpoints.

Each one is a :class:`DMDO.ports.RunObserver` attached by :meth:`DMDO.MDO.run`; user observers are called after them
(see :func:`default_observers`). They keep no state of their own (everything lives on the run), so a resumed run gets
fresh instances and the checkpoint never has to pickle them.
"""
import copy
import csv
import os
import pickle
import time
from typing import Any, Iterable, List, Optional

import numpy as np

from ..core.enums import FEASIBILITY_TOL
from ..ports import IterationRecord, RunObserver
from ..utils.log import MSG_TYPE

HISTORY_FILE = "Coordination_history.out"
CHECKPOINT_FILE = "coord_chk.pkl"


def notify(observers: Iterable[Any], hook: str, *args) -> None:
  """Call ``hook(*args)`` on every observer that defines it, in order."""
  for observer in observers:
    method = getattr(observer, hook, None)
    if method is not None:
      method(*args)


class CheckpointWriter(RunObserver):
  """Pickles the run to ``<post_dir>/coord_chk.pkl`` at the start of every outer iteration after the first."""

  def on_iteration_start(self, run, iteration: int) -> None:
    if iteration > 0:
      with open(os.path.join(run.post_dir, CHECKPOINT_FILE), "wb") as fh:
        pickle.dump(run, fh)


class ConsoleReporter(RunObserver):
  """Prints and logs one progress line per outer iteration (when the run's ``display`` option is on)."""

  @staticmethod
  def format_line(iter, qmax, obj, dx, wmax) -> str:
    """Format log message with dynamic spacing for vertical alignment"""
    # Field widths are set based on typical value ranges
    iter_width = 4  # 4 digits for iteration number
    qmax_width = 16  # Enough for scientific notation
    obj_width = 16  # Enough for large objective values
    dx_width = 16  # Enough for large dx values
    wmax_width = 16  # Enough for large wmax values

    iter_str = f'{iter}'.rjust(iter_width)
    qmax_str = f'qmax: {qmax:.8f}'.rjust(qmax_width)
    obj_str = f'Obj: {obj:.8f}'.rjust(obj_width)
    dx_str = f'dx: {dx:.8f}'.rjust(dx_width)
    wmax_str = f'max(w): {wmax:.8f}'.rjust(wmax_width)
    return f'{iter_str} || {qmax_str} || {obj_str} || {dx_str} || {wmax_str}'

  def on_iteration_end(self, run, record: IterationRecord) -> None:
    if not run.display:
      return
    coord = run.Coordinator
    objective = record.objective[0] if isinstance(record.objective, list) else record.objective
    formatted_msg = self.format_line(record.iteration, record.max_inconsistency, objective, record.variables_change,
                                     record.max_penalty_weight)
    run.log.log_msg(msg=formatted_msg, msg_type=MSG_TYPE.INFO.value)
    print(formatted_msg)
    label = worst_link_label(coord, record.worst_link)
    run.log.log_msg(msg=f'Highest inconsistency : {label}', msg_type=MSG_TYPE.INFO.value)
    print(f'Highest inconsistency : {label}')


def worst_link_label(coord, link: int) -> str:
  """``<name>_<subproblem> to <name>_<subproblem>`` for a link (a column of ``coord._linker``, the index used by
  :meth:`~DMDO.ADMM.batch_q` and ``IterationRecord.worst_link``): the two copies of the variable and the subproblems
  that own them."""
  a = coord.master_vars[coord._linker[0, link] - 1]
  b = coord.master_vars[coord._linker[1, link] - 1]
  return f'{a.name}_{a.sp_index} to {b.name}_{b.sp_index}'


class CoordinationHistoryRecorder(RunObserver):
  """Writes ``Coordination_history.out``: a header at the first outer iteration, then one row per iteration.

  With the AI regularizer the file also carries the detector outputs and the adaptation scales actually applied.
  """

  # ------------------------------------------------------------------------------------------------ columns
  @staticmethod
  def hmin_columns(run) -> List[str]:
    return [f'{f"hmin_{sp.index}".rjust(30)}' for sp in run.subProblems]

  def columns(self, run) -> List[str]:
    coord = run.Coordinator
    hmin_cols = self.hmin_columns(run)
    if coord.is_AI_regularizer:
      adaptive_cols = (
        [f'{"Eff_spike_w_scale".rjust(30)}',
         f'{"Eff_spike_gamma_scale".rjust(30)}',
         f'{"Eff_budget_scale".rjust(30)}',
         f'{"Adapt_event".rjust(30)}']
        if getattr(coord, "spike_adaptive_scales", True)
        else []
      )
      return [f'{"Time"}', f'{"Iteration #".rjust(30)}',
              f'{"Max. inconsistency".rjust(30)}',
              f'{"spike_location".rjust(30)}',
              f'{"alarm_detected".rjust(30)}',
              f'{"spike_detected".rjust(30)}',
              f'{"Objective".rjust(30)}',
              f'{"Spike_prob".rjust(30)}',
              f'{"Forecast_prob".rjust(30)}',
              f'{"Non_improvement_prob".rjust(30)}',
              f'{"Obj_non_improvement_prob".rjust(30)}',
              f'{"Constraint_non_improvement_prob".rjust(30)}',
              f'{"Detector_uncertainty".rjust(30)}',
              f'{"Constraint_pressure".rjust(30)}',
              f'{"Status".rjust(30)}',
              f'{"Variables change".rjust(30)}',
              f'{"max(w)".rjust(30)}',
              f'{"Coupling_with_qmax".rjust(30)}'] + \
        adaptive_cols + \
        hmin_cols + \
        [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in coord.master_vars]

    return [f'{"Time"}', f'{"Iteration #".rjust(30)}',
            f'{"Max. inconsistency".rjust(30)}',
            f'{"alarm_detected".rjust(30)}',
            f'{"spike_detected".rjust(30)}',
            f'{"Objective".rjust(30)}',
            f'{"Status".rjust(30)}',
            f'{"Variables change".rjust(30)}',
            f'{"max(w)".rjust(30)}',
            f'{"Coupling_with_qmax".rjust(30)}'] + \
      hmin_cols + \
      [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in coord.master_vars]

  # ------------------------------------------------------------------------------------------------ file output
  def write_header(self, run, file: Optional[str] = None) -> None:
    if not hasattr(run.Coordinator, "master_vars"):
      return
    file = file or run.Ol_file
    mode = 'x' if not os.path.exists(file) else 'w'
    with open(file, mode=mode) as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=self.columns(run))
      writer.writeheader()

  def append_row(self, run, row: dict) -> None:
    with open(run.Ol_file, mode='a') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=self.columns(run))
      writer.writerow(row)

  def build_row(self, run, record: IterationRecord) -> dict:
    coord = run.Coordinator
    iter, dx, index = record.iteration, record.variables_change, record.worst_link
    hmin_cols = self.hmin_columns(run)
    keys = self.columns(run)
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    hmin = max(run.hmin) if isinstance(run.hmin, list) and run.hmin is not None and len(run.hmin) > 0 else run.hmin
    hstatus = "Feasible" if hmin <= FEASIBILITY_TOL else "Infeasible"
    status = copy.deepcopy(hstatus) if run.fmin != np.inf else "Error"
    cmax = worst_link_label(coord, index)
    if coord.is_AI_regularizer:
      values = [curr_time, iter, coord.max_inconsistency(), coord.spike_location,
                int(bool(getattr(coord, "alarm_detected", False))), int(bool(getattr(coord, "spike_detected", False))),
                run.fmin, coord.spike_prob, coord.forecast_prob, coord.non_improvement_prob,
                coord.objective_non_improvement_prob, coord.constraint_non_improvement_prob,
                coord.detector_uncertainty, coord.constraint_pressure, status, dx, np.max(coord.w), cmax]
    else:
      values = [curr_time, iter, coord.max_inconsistency(), int(bool(getattr(coord, "alarm_detected", False))),
                int(bool(getattr(coord, "spike_detected", False))), run.fmin, status, dx, np.max(coord.w), cmax]
    row = {keys[0]: str(values[0])}
    for k, value in zip(keys[1:len(values)], values[1:]):
      row[k] = str(value).rjust(30)
    if coord.is_AI_regularizer and getattr(coord, "spike_adaptive_scales", True):
      # Scales actually applied by the last penalty update (recomputing them here would see the cool-down).
      _eff_w, _eff_g, _eff_b, _event = getattr(coord, "_applied_scales", (1.0, 1.0, 1.0, ""))
      row[f'{"Eff_spike_w_scale".rjust(30)}'] = f'{_eff_w:.6f}'.rjust(30)
      row[f'{"Eff_spike_gamma_scale".rjust(30)}'] = f'{_eff_g:.6f}'.rjust(30)
      row[f'{"Eff_budget_scale".rjust(30)}'] = f'{_eff_b:.6f}'.rjust(30)
      row[f'{"Adapt_event".rjust(30)}'] = str(_event or "none").rjust(30)

    for sp, hmin_col in zip(run.subProblems, hmin_cols):
      row[hmin_col] = str(run.hmin_dict.get(sp.index, [0.])).rjust(30)
    lr = len(row)
    for cv, v in enumerate(coord.master_vars):
      row[keys[lr+cv]] = str(v.value).rjust(30)
    return row

  # ------------------------------------------------------------------------------------------------ hooks
  def on_iteration_start(self, run, iteration: int) -> None:
    if iteration == 0:
      self.write_header(run)

  def on_iteration_end(self, run, record: IterationRecord) -> None:
    self.append_row(run, self.build_row(run, record))

  def on_resume(self, run, iteration: int) -> None:
    self.truncate_from(run, iteration)

  @staticmethod
  def truncate_from(run, iteration: int) -> None:
    """Drop the rows of ``iteration`` and later (written after the checkpoint a run is resumed from)."""
    file = getattr(run, "Ol_file", None)
    if not file or not os.path.exists(file):
      return
    with open(file, newline="") as fh:
      rows = [row for row in csv.reader(fh) if row]
    if not rows:
      return

    def row_iteration(row):
      try:
        return int(float(row[1]))
      except (IndexError, ValueError):
        return -1

    header, body = rows[0], rows[1:]
    kept = [row for row in body if row_iteration(row) < iteration]
    if len(kept) == len(body):
      return
    with open(file, mode='w') as csv_file:  # same mode as the writer above, so the line endings match
      writer = csv.writer(csv_file)
      writer.writerow(header)
      writer.writerows(kept)


def default_observers(run) -> List[Any]:
  """DMDO's own observers for ``run``, in the order they are called."""
  from .ai_hooks import AIRegularizerHooks, RollbackGuard

  observers: List[Any] = []
  if run.Coordinator.is_AI_regularizer:
    observers += [AIRegularizerHooks(), RollbackGuard()]
  observers += [CheckpointWriter(), ConsoleReporter(), CoordinationHistoryRecorder()]
  return observers
