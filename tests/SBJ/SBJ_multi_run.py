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
"""Run 100 SBJ optimization instances with randomized baseline values."""
import copy
import csv
import shutil
from pathlib import Path

import numpy as np
import yaml

from DMDO import MDA, MDO, main

# Reuse SBJ blackboxes/objectives from the existing runner.
import SBJ_main


def _cleanup_previous_runs(base_dir: Path) -> None:
    """Delete all existing run* directories before starting a new batch."""
    for path in base_dir.glob("run*"):
        if path.is_dir():
            shutil.rmtree(path)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _randomize_baselines(yaml_data: dict, seed: int, perturbation: float = 0.10) -> dict:
    """
    Randomize baseline values ±perturbation% while respecting [lb, ub] bounds.
    Returns a copy with modified baseline values.
    """
    updated = copy.deepcopy(yaml_data)
    rng = np.random.RandomState(seed)
    variables = updated.get("variables", {})

    for var_key, var_list in variables.items():
        # var_list format: [name, sp_index, link, coupling, lb, bl, ub, dim, var_type]
        if len(var_list) >= 7:
            lb, bl, ub = var_list[4], var_list[5], var_list[6]

            # Skip dummy variables or zero baselines
            if bl == 0.0 or ub == 0.0:
                continue

            # Random perturbation: sample uniformly from [-perturbation, +perturbation]
            alpha = rng.uniform(-perturbation, perturbation)
            new_bl = bl * (1.0 + alpha)

            # Clip to [lb, ub] and convert to Python float for YAML serialization
            new_bl = float(np.clip(new_bl, lb, ub))
            var_list[5] = new_bl

    return updated


def _bind_callable_blackboxes(mdao: MDO) -> None:
    for sp in mdao.subProblems:
        mda: MDA = sp.MDA_process
        for analysis in mda.analyses:
            analysis.blackbox = getattr(SBJ_main, analysis.blackbox)
        sp.opt = getattr(SBJ_main, sp.opt)


def _collect_run_summary(run_dir: Path, run_name: str, run_id: int) -> dict:
    history_file = run_dir / "Coordination_history.out"
    summary = {
        "run_name": run_name,
        "run_id": int(run_id),
        "final_objective": "",
        "final_max_inconsistency": "",
        "iterations_consumed": 0,
    }

    if not history_file.exists():
        return summary

    last_row = None
    row_count = 0
    with history_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            row_count += 1
            last_row = row

    if row_count == 0 or last_row is None:
        return summary

    # Keys in Coordination_history.out are right-justified with spaces.
    normalized = {str(k).strip(): v for k, v in last_row.items()}

    objective_raw = normalized.get("Objective", "")
    inconsistency_raw = normalized.get("Max. inconsistency", "")
    iter_raw = normalized.get("Iteration #", "")

    try:
        summary["final_objective"] = float(str(objective_raw).strip())
    except (TypeError, ValueError):
        summary["final_objective"] = ""

    try:
        summary["final_max_inconsistency"] = float(str(inconsistency_raw).strip())
    except (TypeError, ValueError):
        summary["final_max_inconsistency"] = ""

    try:
        # Iteration index is zero-based; consumed count is last index + 1.
        summary["iterations_consumed"] = int(float(str(iter_raw).strip())) + 1
    except (TypeError, ValueError):
        summary["iterations_consumed"] = int(row_count)

    return summary


def _write_batch_summary(summary_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_name",
        "run_id",
        "final_objective",
        "final_max_inconsistency",
        "iterations_consumed",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sbj_batch(n_runs: int = 100, perturbation: float = 0.10, base_seed: int = 42) -> None:
    """
    Run SBJ optimization n_runs times with randomized baseline values.

    Args:
        n_runs: Number of runs (default 100)
        perturbation: Perturbation fraction (default 0.10 = ±10%)
        base_seed: Base random seed; each run uses base_seed + run_id for reproducibility
    """
    base_dir = Path(__file__).resolve().parent
    base_yaml_path = base_dir / "SBJ.yaml"

    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Could not find input file: {base_yaml_path}")

    base_yaml = _load_yaml(base_yaml_path)

    # Requested behavior: wipe previous run* folders every time this script starts.
    _cleanup_previous_runs(base_dir)
    summary_rows = []

    for run_id in range(1, n_runs + 1):
        run_name = f"run{run_id:03d}"
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_seed = base_seed + (run_id - 1)
        run_yaml = _randomize_baselines(base_yaml, run_seed, perturbation)
        run_yaml_path = run_dir / "SBJ.yaml"
        _save_yaml(run_yaml_path, run_yaml)

        def_dict = {
            "setup_file": str(run_yaml_path),
            "run_mode": "build",
            "exec_mode": "serial",
            "mdo_name": "SBJ",
            "working_dir": str(run_dir),
        }

        mdao: MDO = main(def_dict)
        _bind_callable_blackboxes(mdao)

        # This writes Coordination_history.out in run_dir.
        # DMDO also creates run_dir/SBJ_post during setup.
        mdao.run(str(run_dir / "SBJ.out"), mode="serial")
        summary_rows.append(_collect_run_summary(run_dir, run_name, run_id))

        print(f"Completed {run_name} with seed={run_seed} (perturbation: ±{perturbation*100:.0f}%)")

    summary_file = base_dir / "SBJ_batch_summary.csv"
    _write_batch_summary(summary_file, summary_rows)
    print(f"\nSaved batch summary: {summary_file}")


if __name__ == "__main__":
    run_sbj_batch(n_runs=100, perturbation=0.10, base_seed=42)
