"""Benchmark figures and summary table of the documentation, from the recorded regression traces.

The traces in ``tests/golden/traces`` are complete histories (objective and maximum inconsistency at every outer
iteration, final multipliers and design) of the regression problems, recorded with fixed seeds. The test suite checks
that the current code reproduces them exactly, so these figures can be regenerated without running any optimization:

    python docs_src/scripts/plot_benchmarks.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRACES = os.path.join(ROOT, "tests", "golden", "traces")
OUT = os.path.abspath(os.path.join(HERE, "..", "_static"))

# Best known solutions: Sellar and speed reducer from Tedford and Martins (2010), geometric programming from Talgorn et
# al. (2017); none is published for the basic problem.
SELLAR_OPT = 3.18339395045
SPEED_REDUCER_OPT = 2994.355026
GP_OPT = 17.59

# name: (label, problem, subproblems, inner solver, outer budget, reference optimum or None)
CASES = {
  "basic_poll": ("Basic, POLL", "Basic two-discipline problem", 2, "OMADS poll", 50, None),
  "sellar_scipy": ("Sellar, SLSQP", "Sellar", 2, "SciPy SLSQP", 100, SELLAR_OPT),
  "sellar_poll": ("Sellar, POLL", "Sellar", 2, "OMADS poll", 150, SELLAR_OPT),
  "sellar_mads": ("Sellar, MADS", "Sellar", 2, "OMADS MADS", 150, SELLAR_OPT),
  "speed_reducer_scipy": ("Speed reducer, SLSQP", "Golinski speed reducer", 4, "SciPy SLSQP", 50, SPEED_REDUCER_OPT),
  "gp_mads": ("Geometric programming, MADS", "Geometric programming", 3, "OMADS MADS", 100, GP_OPT),
}
COLORS = ["#1d3557", "#2a9d8f", "#c99a06", "#e76f51", "#6d597a", "#457b9d"]


def load(name):
  with open(os.path.join(TRACES, f"{name}.json")) as fh:
    return json.load(fh)


def style(ax):
  ax.grid(True, which="major", color="#d9d9d9", linewidth=0.6)
  ax.grid(True, which="minor", color="#efefef", linewidth=0.4)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)


def summary():
  rows = []
  for name, (label, problem, nsp, solver, budget, fref) in CASES.items():
    d = load(name)
    f = np.asarray(d["objective_history"], dtype=float)
    q = np.asarray(d["qmax_history"], dtype=float)
    first = next((i for i, v in enumerate(q) if v < 1e-3), None)
    gap = f"{100 * (f[-1] - fref) / fref:+.2f} %" if fref else "n/a"
    rows.append([label, nsp, solver, d["iterations"], f"{f[-1]:.6g}", f"{fref:.6g}" if fref else "n/a", gap,
                 f"{q[-1]:.2e}", "n/a" if first is None else first])
  path = os.path.join(OUT, "bench_summary.csv")
  with open(path, "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows(rows)
  return path


def figure_consistency():
  fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
  for color, (name, meta) in zip(COLORS, CASES.items()):
    q = np.asarray(load(name)["qmax_history"], dtype=float)
    ax.semilogy(np.arange(len(q)), q, color=color, linewidth=1.6, label=meta[0])
  ax.axhline(1e-3, color="#888888", linestyle="--", linewidth=0.8)
  ax.text(1.0, 1e-3, r" $10^{-3}$", color="#666666", fontsize=8, va="center", transform=ax.get_yaxis_transform())
  ax.set_xlabel("outer iteration")
  ax.set_ylabel(r"max. inconsistency $\|\mathbf{q}\|_\infty$")
  style(ax)
  ax.legend(frameon=False, fontsize=8, ncol=2)
  fig.tight_layout()
  path = os.path.join(OUT, "bench_consistency.svg")
  fig.savefig(path, facecolor="white")
  plt.close(fig)
  return path


def figure_sellar():
  fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=150)
  for color, name in zip(COLORS[1:4], ("sellar_scipy", "sellar_poll", "sellar_mads")):
    d = load(name)
    f = np.asarray(d["objective_history"], dtype=float)
    q = np.asarray(d["qmax_history"], dtype=float)
    it = np.arange(len(f))
    a1.plot(it, f, color=color, linewidth=1.6, label=CASES[name][0].split(", ")[1])
    a2.semilogy(it, q, color=color, linewidth=1.6)
  a1.axhline(SELLAR_OPT, color="#222222", linestyle="--", linewidth=0.9)
  a1.text(len(f) * 0.55, SELLAR_OPT * 0.985, r"$f^\ast = 3.18339$", fontsize=8, va="top")
  a1.set_ylim(2.6, 4.2)
  a1.set_xlabel("outer iteration")
  a1.set_ylabel("objective of the master subproblem")
  a2.set_xlabel("outer iteration")
  a2.set_ylabel(r"$\|\mathbf{q}\|_\infty$")
  for ax in (a1, a2):
    style(ax)
  a1.legend(frameon=False, fontsize=8, title="inner solver", title_fontsize=8)
  fig.tight_layout()
  path = os.path.join(OUT, "bench_sellar.svg")
  fig.savefig(path, facecolor="white")
  plt.close(fig)
  return path


def figure_objectives():
  names = ("basic_poll", "speed_reducer_scipy", "gp_mads")
  fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=150)
  for ax, color, name in zip(axes, (COLORS[0], COLORS[4], COLORS[5]), names):
    f = np.asarray(load(name)["objective_history"], dtype=float)
    ax.plot(np.arange(len(f)), f, color=color, linewidth=1.6)
    ax.set_title(CASES[name][0], fontsize=9)
    ax.set_xlabel("outer iteration")
    style(ax)
  axes[0].set_ylabel("objective of the master subproblem")
  fig.tight_layout()
  path = os.path.join(OUT, "bench_objectives.svg")
  fig.savefig(path, facecolor="white")
  plt.close(fig)
  return path


if __name__ == "__main__":
  for path in (summary(), figure_consistency(), figure_sellar(), figure_objectives()):
    print(os.path.relpath(path, ROOT))
