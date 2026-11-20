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
"""Benchmark of the four ATC formulations of the supersonic business jet (Tosserams et al. 2010, Sec. 4.3).

Each formulation (Fig. 6a-d, ``SBJ_<x>.yaml``) is solved from ``--n`` starting points drawn uniformly within
+-10 % of the reference design of Table 1, with the paper's settings:

- sequential (Gauss-Seidel) alternating direction method of multipliers: one inner iteration per outer iteration;
- penalty update beta = 1.03, gamma = 0.75 (Eq. 8, per coupling), termination tolerance eps = 1e-3 (Eqs. 9-10);
- initial weights from an objective estimate (f_hat = 10 x 10^3 lb, w0 = 0.001, alpha = 0.1; Ref. [11] of the paper);
- inconsistencies scaled by the reference values (the paper scales every variable so that its reference is 1);
- SQP subproblem solvers (SciPy SLSQP in place of MATLAB fmincon).

Every formulation uses the same starting points (the same seed per run index). An all-in-one (MDF) SLSQP solution
from the same design-variable starts gives the reference column. Runs are distributed over ``--workers`` processes;
a run that exceeds ``--timeout`` seconds is terminated and counted as not converged.

Usage:
  python bm_SBJ.py [--forms a b c d] [--n 100] [--workers 50] [--timeout 2400] [--out DIR] [--seed 0]
  python bm_SBJ.py --summarize --out DIR      (tables from finished runs only)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PERTURBATION = 0.10
CONVERGED = "below stopping threshold"

# Coordinator / termination settings of Sec. 4.3.
PAPER_COORD = {"beta": 1.03, "gamma": 0.75, "M_update_scheme": "normal", "scaling": "reference", "mode": "serial",
               "single_multiplier_update": True, "w_init": 0.001, "w_init_f_hat": 1.0e4, "w_init_alpha": 0.1,
               "budget": 1000}
PAPER_EPS = 1.0e-3
SLSQP = {"method": "SLSQP", "is_con": True, "normalize": True, "cache_constraints": True, "restore_feasibility": True,
         "scale_objective": True, "tol": 1.0e-8, "options": {"maxiter": 100, "ftol": 1.0e-9, "eps": 1.0e-7, "disp": False}}


# ------------------------------------------------------------------------------------------------ problem setup
def _groups(variables: dict) -> dict:
  """Variable key -> (name, occurrence within its subproblem): every copy of a quantity gets the same start."""
  seen, out = {}, {}
  for key, row in variables.items():
    name, sp = row[0], row[1]
    n = seen.get((name, sp), 0)
    seen[(name, sp)] = n + 1
    out[key] = (name, n)
  return out


def random_factors(seed: int, keys) -> dict:
  """1 + U(-0.1, 0.1) per design quantity, drawn in a fixed order so all formulations share the start."""
  rng = np.random.RandomState(seed)
  return {k: 1.0 + rng.uniform(-PERTURBATION, PERTURBATION) for k in sorted(keys)}


def all_quantities() -> set:
  keys = set()
  for form in "abcd":
    keys |= set(_groups(yaml.safe_load(open(HERE / f"SBJ_{form}.yaml"))["variables"]).values())
  return keys


def paper_config(form: str, factors: dict) -> dict:
  cfg = yaml.safe_load(open(HERE / f"SBJ_{form}.yaml"))
  groups = _groups(cfg["variables"])
  for key, row in cfg["variables"].items():
    lb, bl, ub = row[4], row[5], row[6]
    if row[3] == "dummy" or bl == 0.0:          # dummies and slacks start at their (zero) baseline
      continue
    row[5] = float(np.clip(bl * factors[groups[key]], lb, ub))
  cfg["coord"]["c1"].update(PAPER_COORD)
  for sp in cfg["subproblem"].values():
    maxiter = 200 if sp["nv"] > 20 else 100
    sp["solver"] = "scipy"
    sp["scipy"] = copy.deepcopy(SLSQP)
    sp["scipy"]["options"]["maxiter"] = maxiter
    sp["display"] = False
  cfg["MDO"]["inc_stop"] = PAPER_EPS
  cfg["MDO"]["display"] = True
  return cfg


# ------------------------------------------------------------------------------------------------ one ATC run
def design_from_master(master_vars) -> dict:
  def val(name, sp=None):
    return [v.value for v in master_vars if v.name == name and (sp is None or v.sp_index == sp)]
  struc = [v.sp_index for v in master_vars if v.name == "lambda"][0]
  z = [val(n, struc)[0] for n in ("tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht")]
  return dict(z=z, T=val("T")[0], LAMBDAht=val("LAMBDAht")[0], Lw=val("Lw")[0], Lht=val("Lht")[0],
              t=val("t"), ts=val("ts"), lam=val("lambda")[0])


def worker(yaml_path: str, wd: str) -> None:
  """Solve one problem definition and write result.json (run in its own process)."""
  warnings.filterwarnings("ignore")
  sys.path.insert(0, str(HERE))
  import run_SBJ as R
  from DMDO import Session
  from DMDO.subproblem.subproblem import SubProblem
  # No per-evaluation inner-loop files (SP_<i>/SP_<i>_<iter>.out): a few MB per solve, hundreds of GB per campaign.
  prepare = SubProblem.prepare_post

  def prepare_without_inner_files(self, file):
    prepare(self, file)
    if self.Il_file and os.path.exists(self.Il_file):
      os.remove(self.Il_file)
    self.Il_file = None
  SubProblem.prepare_post = prepare_without_inner_files
  t0 = time.time()
  session = Session.from_yaml(yaml_path, working_dir=wd, blackboxes=R.BLACKBOXES)
  res = session.run()
  d = design_from_master(session.mdo.Coordinator.master_vars)
  chk = R.mda_check(d["z"], d["T"], d["LAMBDAht"], d["Lw"], d["Lht"], d["t"], d["ts"], d["lam"])
  out = dict(status="converged" if CONVERGED in str(res.stop_reason) else "not_converged",
             stop_reason=str(res.stop_reason), iterations=int(res.iterations), max_inconsistency=float(res.max_inconsistency),
             objective=float(res.objective), wall=time.time() - t0, design=d,
             mda={k: float(v) for k, v in chk.items()})
  json.dump(out, open(os.path.join(wd, "result.json"), "w"), indent=1)


def _last_history(wd: Path) -> dict:
  files = list(wd.rglob("Coordination_history.out"))
  if not files:
    return {}
  rows = [r.split(",") for r in files[0].read_text().splitlines()[1:] if r.strip()]
  if not rows:
    return {}
  return dict(iterations=len(rows), max_inconsistency=float(rows[-1][2]), objective=float(rows[-1][5]))


def run_task(form: str, run_id: int, out: Path, timeout: float) -> dict:
  wd = out / form / f"run{run_id:03d}"
  res_file = wd / "result.json"
  if res_file.exists():
    return dict(form=form, run=run_id, **json.load(open(res_file)))
  yaml_path = wd / f"SBJ_{form}.yaml"
  env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
  t0 = time.time()
  with open(wd / "run.log", "w") as log:
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(yaml_path), str(wd)],
                            stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE), env=env)
    try:
      code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait()
      out_d = dict(status="timeout", stop_reason=f"terminated after {timeout:.0f} s", wall=time.time() - t0,
                   **_last_history(wd))
      json.dump(out_d, open(res_file, "w"), indent=1)
      return dict(form=form, run=run_id, **out_d)
  if code != 0 or not res_file.exists():
    out_d = dict(status="error", stop_reason=f"exit code {code}", wall=time.time() - t0, **_last_history(wd))
    json.dump(out_d, open(res_file, "w"), indent=1)
    return dict(form=form, run=run_id, **out_d)
  return dict(form=form, run=run_id, **json.load(open(res_file)))


# ------------------------------------------------------------------------------------------------ all-in-one
_AIO_NAMES = ["tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "T", "LAMBDAht", "Lw", "Lht"]


def aio_task(run_id: int, factors: dict) -> dict:
  """All-in-one (MDF) SLSQP solution from the run's design-variable start."""
  warnings.filterwarnings("ignore")
  from scipy.optimize import minimize
  sys.path.insert(0, str(HERE))
  import run_SBJ as R
  lb = np.array([.01, 2.5, 40, 200, 50, 2.5, .1, 40, .01, 1] + [.1] * 9 + [.1] * 9 + [.1])
  ub = np.array([.1, 8, 70, 800, 148.9, 8.5, 1, 70, .2, 3.5] + [4] * 9 + [9] * 9 + [.4])
  ref = np.array([.05, 3, 60, 500, 100, 5.5, .6, 45, .15, 1.5] + [3] * 9 + [6] * 9 + [.3])
  f = [factors[(n, 0)] for n in _AIO_NAMES] + [factors[("t", i)] for i in range(9)] + \
      [factors[("ts", i)] for i in range(9)] + [factors[("lambda", 0)]]
  x0 = np.clip(ref * np.array(f), lb, ub)
  cache = {}

  def ev(z):
    key = z.tobytes()
    if key not in cache:
      x = lb + np.clip(z, 0, 1) * (ub - lb)
      cache[key] = R.mda_check(x[:6], x[6], x[7], x[8], x[9], list(x[10:19]), list(x[19:28]), x[28], tol=1e-10,
                               return_g=True)
    return cache[key]

  def con(z):
    return -np.asarray(ev(z)["g"], dtype=float)

  t0 = time.time()
  z0 = (x0 - lb) / (ub - lb)
  r = minimize(lambda z: ev(z)["Wt"] / 1e4, z0, method="SLSQP", bounds=[(0, 1)] * len(z0),
               constraints=[{"type": "ineq", "fun": con}], options=dict(maxiter=300, ftol=1e-9))
  chk = ev(r.x)
  return dict(form="aio", run=run_id, status="converged" if r.success else "not_converged", stop_reason=str(r.message),
              iterations=int(r.nit), wall=time.time() - t0, mda={k: float(v) for k, v in chk.items() if k != "g"})


# ------------------------------------------------------------------------------------------------ tables
def table(results: list, forms: list, n: int) -> str:
  """Table 2 of the paper: converged runs; optimal weight, max. constraint violation and iterations (min/mean/max)
  over the converged runs only."""
  head = ["Formulation"] + [("All-in-one" if f == "aio" else f"({f})") for f in forms]
  coord = ["Coordination"] + [("MDF, SLSQP" if f == "aio" else "Sequential") for f in forms]
  lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), "| " + " | ".join(coord) + " |"]
  stats = {}
  for f in forms:
    conv = [r for r in results if r["form"] == f and r.get("status") == "converged" and "mda" in r]
    wt = np.array([r["mda"]["Wt"] / 1e3 for r in conv])
    cv = np.array([max(0.0, r["mda"]["gmax"]) for r in conv])
    it = np.array([r["iterations"] for r in conv])
    stats[f] = (len(conv), sum(1 for r in results if r["form"] == f), wt, cv, it)
  lines.append("| Converged runs | " + " | ".join(f"{stats[f][0]}/{n}" for f in forms) + " |")

  def row(label, idx, fmt, skip_aio=False):
    for k, name in enumerate(("Min", "Mean", "Max")):
      cells = []
      for f in forms:
        arr = stats[f][idx]
        if (skip_aio and f == "aio") or arr.size == 0:
          cells.append("-")
        else:
          cells.append(fmt.format((arr.min(), arr.mean(), arr.max())[k]))
      lines.append(f"| {label if k == 0 else ''} {name} | " + " | ".join(cells) + " |")

  row("Optimal weight (x10^3 lb)", 2, "{:.1f}")
  row("Maximum constraint violation", 3, "{:.3f}")
  row("Iterations (total)", 4, "{:.0f}", skip_aio=True)
  return "\n".join(lines)


def summarize(out: Path, forms: list, n: int) -> str:
  results = []
  for f in forms:
    if f == "aio":
      p = out / "aio_results.json"
      results += json.load(open(p)) if p.exists() else []
      continue
    for i in range(1, n + 1):
      p = out / f / f"run{i:03d}" / "result.json"
      if p.exists():
        results.append(dict(form=f, run=i, **json.load(open(p))))
  with open(out / "results.csv", "w") as fh:
    fh.write("form,run,status,iterations,max_inconsistency,Wt_mda,gmax_mda,wall_s,stop_reason\n")
    for r in results:
      m = r.get("mda", {})
      fh.write(f'{r["form"]},{r["run"]},{r.get("status")},{r.get("iterations", "")},{r.get("max_inconsistency", "")},'
               f'{m.get("Wt", "")},{m.get("gmax", "")},{r.get("wall", 0):.0f},"{r.get("stop_reason", "")}"\n')
  md = table(results, forms, n)
  extra = []
  for f in forms:
    rs = [r for r in results if r["form"] == f]
    if rs and f != "aio":
      extra.append(f"({f}): " + ", ".join(f"{s} {sum(1 for r in rs if r.get('status') == s)}"
                                          for s in ("converged", "not_converged", "timeout", "error")))
  text = md + "\n\n" + "\n".join(extra) + "\n"
  (out / "summary.md").write_text(text)
  return text


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--forms", nargs="+", default=list("abcd"))
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--workers", type=int, default=50)
  ap.add_argument("--timeout", type=float, default=2400.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=str(HERE / "SBJ_BM_paper"))
  ap.add_argument("--summarize", action="store_true")
  ap.add_argument("--no-aio", action="store_true")
  ap.add_argument("--worker", nargs=2, metavar=("YAML", "WD"))
  args = ap.parse_args()
  if args.worker:
    worker(*args.worker)
    return
  out = Path(args.out)
  forms = (["aio"] if not args.no_aio else []) + args.forms
  if args.summarize:
    print(summarize(out, forms, args.n))
    return
  keys = all_quantities()
  factors = {i: random_factors(args.seed + i, keys) for i in range(1, args.n + 1)}
  for f in args.forms:
    for i in range(1, args.n + 1):
      wd = out / f / f"run{i:03d}"
      wd.mkdir(parents=True, exist_ok=True)
      if not (wd / f"SBJ_{f}.yaml").exists():
        yaml.safe_dump(paper_config(f, factors[i]), open(wd / f"SBJ_{f}.yaml", "w"), sort_keys=False)
  if not args.no_aio and not (out / "aio_results.json").exists():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(args.workers, os.cpu_count() or 1)) as ex:
      aio = list(ex.map(aio_task, range(1, args.n + 1), [factors[i] for i in range(1, args.n + 1)]))
    json.dump(aio, open(out / "aio_results.json", "w"), indent=1)
    print(f"all-in-one: {sum(r['status'] == 'converged' for r in aio)}/{args.n} converged", flush=True)
  # Interleave the formulations so that all of them progress together.
  tasks = [(f, i) for i in range(1, args.n + 1) for f in args.forms]
  t0 = time.time()
  done = 0
  with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(run_task, f, i, out, args.timeout): (f, i) for f, i in tasks}
    for fut in as_completed(futs):
      f, i = futs[fut]
      done += 1
      try:
        r = fut.result()
        print(f"[{done}/{len(tasks)} {time.time() - t0:7.0f}s] ({f}) run{i:03d}: {r.get('status')} "
              f"it={r.get('iterations')} q={r.get('max_inconsistency', float('nan')):.1e} "
              f"Wt={r.get('mda', {}).get('Wt', float('nan')):.0f}", flush=True)
      except Exception as exc:  # noqa: BLE001
        print(f"[{done}/{len(tasks)}] ({f}) run{i:03d}: driver error {exc!r}", flush=True)
  print(summarize(out, forms, args.n))


if __name__ == "__main__":
  main()
