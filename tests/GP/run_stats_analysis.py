"""
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
# """

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR_PATTERN = re.compile(r"^run(\d+)$", re.IGNORECASE)


def _run_sort_key(path: Path) -> Tuple[int, str]:
    match = RUN_DIR_PATTERN.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    keep_cols = [c for c in df.columns if c and not c.lower().startswith("unnamed")]
    return df[keep_cols]


def _load_history(history_file: Path) -> pd.DataFrame:
    df = pd.read_csv(history_file, engine="python")
    df = _normalize_columns(df)
    required = ["Objective", "Max. inconsistency"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {history_file}")

    out = pd.DataFrame()
    out["Objective"] = pd.to_numeric(df["Objective"], errors="coerce")
    out["Max. inconsistency"] = pd.to_numeric(df["Max. inconsistency"], errors="coerce")
    out = out.dropna(how="all").reset_index(drop=True)
    return out


def _collect_run_histories(base_dir: Path) -> Dict[str, pd.DataFrame]:
    runs: Dict[str, pd.DataFrame] = {}
    run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and RUN_DIR_PATTERN.match(p.name)], key=_run_sort_key)

    for run_dir in run_dirs:
        history_file = run_dir / "Coordination_history.out"
        if not history_file.exists():
            continue
        try:
            runs[run_dir.name] = _load_history(history_file)
        except Exception as exc:
            print(f"Skipping {run_dir.name}: {exc}")

    return runs


def _build_run_summary(runs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for run_name, df in runs.items():
        if df.empty:
            rows.append(
                {
                    "run_name": run_name,
                    "iterations_total": 0,
                    "final_objective": np.nan,
                    "final_max_inconsistency": np.nan,
                }
            )
            continue

        rows.append(
            {
                "run_name": run_name,
                "iterations_total": int(len(df)),
                "final_objective": float(df["Objective"].iloc[-1]),
                "final_max_inconsistency": float(df["Max. inconsistency"].iloc[-1]),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("run_name", key=lambda s: s.map(lambda x: _run_sort_key(Path(x))[0])).reset_index(drop=True)
    return out


def _stack_trajectories(runs: Dict[str, pd.DataFrame], column: str) -> np.ndarray:
    if not runs:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(df) for df in runs.values())
    matrix = np.full((len(runs), max_len), np.nan, dtype=float)

    for i, run_name in enumerate(sorted(runs, key=lambda x: _run_sort_key(Path(x))[0])):
        values = runs[run_name][column].to_numpy(dtype=float)
        matrix[i, : len(values)] = values

    return matrix


def _trajectory_stats(matrix: np.ndarray) -> pd.DataFrame:
    if matrix.size == 0:
        return pd.DataFrame(columns=["iteration", "n_runs", "mean", "median", "q25", "q75", "min", "max"])

    iteration = np.arange(matrix.shape[1], dtype=int)
    n_runs = np.sum(~np.isnan(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)
    median = np.nanmedian(matrix, axis=0)
    q05 = np.nanpercentile(matrix, 5, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    q95 = np.nanpercentile(matrix, 95, axis=0)
    minv = np.nanmin(matrix, axis=0)
    maxv = np.nanmax(matrix, axis=0)

    return pd.DataFrame(
        {
            "iteration": iteration,
            "n_runs": n_runs,
            "mean": mean,
            "median": median,
            "q05": q05,
            "q25": q25,
            "q75": q75,
            "q95": q95,
            "min": minv,
            "max": maxv,
        }
    )


def _plot_summary_distributions(run_summary: pd.DataFrame, output_dir: Path) -> None:
    if run_summary.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(run_summary["iterations_total"].dropna(), bins=20, color="#4c78a8", edgecolor="black")
    axes[0].set_title("Iterations Consumed")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Number of Runs")

    axes[1].hist(run_summary["final_objective"].dropna(), bins=20, color="#f58518", edgecolor="black")
    axes[1].set_title("Final Objective")
    axes[1].set_xlabel("Objective")
    axes[1].set_ylabel("Number of Runs")

    axes[2].hist(run_summary["final_max_inconsistency"].dropna(), bins=20, color="#54a24b", edgecolor="black")
    axes[2].set_title("Final Max. Inconsistency")
    axes[2].set_xlabel("Max. inconsistency")
    axes[2].set_ylabel("Number of Runs")

    fig.tight_layout()
    fig.savefig(output_dir / "run_summary_distributions.png", dpi=160)
    plt.close(fig)


def _plot_trajectory(
    matrix: np.ndarray,
    stats_df: pd.DataFrame,
    output_path: Path,
    title: str,
    y_label: str,
    use_log_scale: bool = False,
) -> None:
    if matrix.size == 0 or stats_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = stats_df["iteration"].to_numpy()

    for i in range(matrix.shape[0]):
        ax.plot(x, matrix[i, :], color="#9ecae1", linewidth=0.8, alpha=0.25)

    ax.plot(x, stats_df["median"].to_numpy(), color="#1f77b4", linewidth=2.0, label="Median")
    ax.plot(x, stats_df["mean"].to_numpy(), color="#d62728", linewidth=1.8, linestyle="--", label="Mean")
    ax.fill_between(
        x,
        stats_df["q05"].to_numpy(),
        stats_df["q95"].to_numpy(),
        color="#1f77b4",
        alpha=0.2,
        label="IQR (5%-95%)",
    )
    ax.set_yscale("log")

    if use_log_scale:
        positive = np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else None
        if positive is not None:
            ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_active_runs(objective_stats: pd.DataFrame, output_path: Path) -> None:
    if objective_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(
        objective_stats["iteration"].to_numpy(),
        objective_stats["n_runs"].to_numpy(),
        where="post",
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.set_title("Active Runs Per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Runs Contributing")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_analysis(base_dir: Path) -> None:
    runs = _collect_run_histories(base_dir)
    if not runs:
        raise RuntimeError(f"No valid run*/Coordination_history.out files found in {base_dir}")

    output_dir = base_dir / "run_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = _build_run_summary(runs)
    objective_matrix = _stack_trajectories(runs, "Objective")
    inconsistency_matrix = _stack_trajectories(runs, "Max. inconsistency")

    objective_stats = _trajectory_stats(objective_matrix)
    inconsistency_stats = _trajectory_stats(inconsistency_matrix)

    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    objective_stats.to_csv(output_dir / "objective_trajectory_stats.csv", index=False)
    inconsistency_stats.to_csv(output_dir / "inconsistency_trajectory_stats.csv", index=False)

    _plot_summary_distributions(run_summary, output_dir)
    _plot_trajectory(
        objective_matrix,
        objective_stats,
        output_dir / "objective_trajectory.png",
        title="Objective Trajectory Across Runs",
        y_label="Objective",
        use_log_scale=False,
    )
    _plot_trajectory(
        inconsistency_matrix,
        inconsistency_stats,
        output_dir / "max_inconsistency_trajectory.png",
        title="Max. Inconsistency Trajectory Across Runs",
        y_label="Max. inconsistency",
        use_log_scale=True,
    )
    _plot_active_runs(objective_stats, output_dir / "active_runs_per_iteration.png")

    print(f"Analyzed {len(runs)} runs")
    print(f"Saved outputs in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GP run*/Coordination_history.out files")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing run* folders",
    )
    args = parser.parse_args()

    run_analysis(args.base_dir)


if __name__ == "__main__":
    main()
