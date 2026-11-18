"""Supersonic business jet, non-hierarchical ATC formulation (b) of Tosserams et al. (2010), Fig. 6(b).

Starts from the Table 1 reference design; every subproblem is solved with SLSQP (see SBJ.yaml). The published
optimum (Table 1, MDF) is Wt = 34.3e3 lb. At the end the DMDO design is re-analysed with a full MDA and compared
with Table 1.
Usage: python run_SBJ.py [a|b|c|d] [working_dir]   (formulation of Fig. 6, default b; SBJ_<x>.yaml)
"""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from DMDO import Session  # noqa: E402
from SSBJ_Aircraft import SSBJ_Aircraft  # noqa: E402
from SSBJ_Aerodynamics import SSBJAerodynamics  # noqa: E402
from SSBJ_Propulsion import SSBJPropulsion  # noqa: E402
from SSBJ_Structures import WingDesignAnalyzer  # noqa: E402
from OMADS._heuristics._exploration import EfficientExploration  # noqa: E402

# OMADS 2612 runs a quadratic-model search before every MADS search step (no option to turn it off). On the
# structures subproblem (27 variables, 72 constraints) each model fit takes minutes. Skip it for large subproblems
# so they use MADS with the Latin-hypercube sampling search only (as OMADS 2401 did in March).
MODEL_SEARCH_MAX_DIM = 5
_quad_model_points = EfficientExploration.generate_quad_model_points


def _quad_model_points_small_only(self, *args, **kwargs):
  if self.dim >= MODEL_SEARCH_MAX_DIM:
    self.candidate_points_set = []
    return None
  return _quad_model_points(self, *args, **kwargs)


EfficientExploration.generate_quad_model_points = _quad_model_points_small_only

def SBJ_aircraft_analysis(x):
  ssbj_aircraft = SSBJ_Aircraft(We=x[1], Ws=x[3], Wf=x[4], SFCp=x[0], LDr=x[2])
  return ssbj_aircraft.SBJ_aircraft_analysis()

def SBJ_propulsion_analysis(x):
  ssbj_prop = SSBJPropulsion(D=x[0], T=x[1])
  return ssbj_prop.SBJ_propulsion_analysis()

def SBJ_aerodynamics_analysis(x):
  ssbj_prop = SSBJAerodynamics(Wt=x[0], ESFp=x[1], theta=x[2], tc=x[3], ARw=x[4], LAMBDAw=x[5], Sref=x[6], Sht=x[7], \
                               ARht=x[8], LAMBDAht=x[9], Lw=x[10], Lht=x[11])
  return ssbj_prop.SBJ_aerodynamics_analysis()

def SBJ_structure_analysis(x):
  ssbj_structure = WingDesignAnalyzer(Lift=x[0], tc=x[1],ARw=x[2], LAMBDAw=x[3], Sref=x[4], Sht=x[5], ARht=x[6], \
                                      lambdatr=x[7], t=x[8:17], ts=x[17:])
  return ssbj_structure.SBJ_structure_analysis()

def SBJ_aircraft_opt(x, y, *args):
  ssbj_aircraft = SSBJ_Aircraft(We=x[1], Ws=x[3], Wf=x[4], SFCp=x[0], LDr=x[2])
  return ssbj_aircraft.SBJ_aircraft_opt(Wt=y[0], range=y[1])

def SBJ_propulsion_opt(x, y, *args):
  ssbj_prop = SSBJPropulsion(D=x[0], T=x[1])
  return ssbj_prop.SBJ_propulsion_opt(Temp_E=y[3], Throttle_uA=y[4])

def SBJ_aerodynamics_opt(x, y, *args):
  ssbj_prop = SSBJAerodynamics(Wt=x[0], ESFp=x[1], theta=x[2], tc=x[3], ARw=x[4], LAMBDAw=x[5], Sref=x[6], Sht=x[7], \
                               ARht=x[8], LAMBDAht=x[9], Lw=x[10], Lht=x[11])
  return ssbj_prop.SBJ_aerodynamics_opt(Pg=y[3], CLo=[y[4], y[5]])

def SBJ_structure_opt(x, y, *args):
  ssbj_structure = WingDesignAnalyzer(Lift=x[0], tc=x[1],ARw=x[2], LAMBDAw=x[3], Sref=x[4], Sht=x[5], ARht=x[6], \
                                      lambdatr=x[7], t=x[8:17], ts=x[17:])
  return ssbj_structure.SBJ_structure_opt()


# Formulations (c) and (d): aircraft without the range constraint, and the system-wide functions.
def SBJ_aircraft_weight_analysis(x):
  """Wt = We + Ws + Wf (x = [We, Ws, Wf])."""
  return [x[0] + x[1] + x[2]]

def SBJ_aircraft_weight_opt(x, y, *args):
  return [y[0], []]

def SBJ_aerodynamics_weight_opt(x, y, *args):
  """Formulation (d): aerodynamics owns Wt (x[0]) and carries the objective min Wt."""
  return [x[0], SBJ_aerodynamics_opt(x, y)[1]]

def SBJ_range_constraint(x):
  """System-wide g_aircraft = 1 - range / 2000 <= 0 with x = [SFC, We, L/D, Ws, Wf]."""
  _, rng = SSBJ_Aircraft(SFCp=x[0], We=x[1], LDr=x[2], Ws=x[3], Wf=x[4]).SBJ_aircraft_analysis()
  return 1.0 - rng / 2000.0

def SBJ_weight_balance(x):
  """System-wide h_aircraft = Wt - We - Ws - Wf = 0 with x = [Wt, We, Ws, Wf]."""
  return x[0] - x[1] - x[2] - x[3]


BLACKBOXES = {name: globals()[name] for name in (
  "SBJ_aircraft_analysis", "SBJ_propulsion_analysis", "SBJ_aerodynamics_analysis", "SBJ_structure_analysis",
  "SBJ_aircraft_opt", "SBJ_propulsion_opt", "SBJ_aerodynamics_opt", "SBJ_structure_opt",
  "SBJ_aircraft_weight_analysis", "SBJ_aircraft_weight_opt", "SBJ_aerodynamics_weight_opt",
  "SBJ_range_constraint", "SBJ_weight_balance")}

def mda_check(z, T, LAMht, Lw, Lht, t, ts, lam, tol=1e-12, max_iter=500):
  """Gauss-Seidel MDA of the design (z = [tc, ARw, LAMw, Sref, Sht, ARht]): coupling values and all constraints."""
  tc, ARw, LAMw, Sref, Sht, ARht = z
  Wt, theta, ESF = 25000., 10., 1.
  for _ in range(max_iter):
    aero = SSBJAerodynamics(Wt=Wt, ESFp=ESF, theta=theta, tc=tc, ARw=ARw, LAMBDAw=LAMw, Sref=Sref, Sht=Sht,
                            ARht=ARht, LAMBDAht=LAMht, Lw=Lw, Lht=Lht)
    L, D, LD, Pg, CL0, CL1 = aero.SBJ_aerodynamics_analysis()
    struc = WingDesignAnalyzer(Lift=L, tc=tc, ARw=ARw, LAMBDAw=LAMw, Sref=Sref, Sht=Sht, ARht=ARht, lambdatr=lam,
                               t=list(t), ts=list(ts))
    Ws, Wf, th = struc.SBJ_structure_analysis()
    prop = SSBJPropulsion(D=D, T=T)
    SFC, We, ESFn, TempE, ThrUA = prop.SBJ_propulsion_analysis()
    Wtn = We + Wf + Ws
    err = abs(Wtn - Wt) / Wt + abs(th - theta) / max(1., abs(theta)) + abs(ESFn - ESF)
    Wt, theta, ESF = 0.5 * (Wt + Wtn), 0.5 * (theta + th), 0.5 * (ESF + ESFn)
    if err < tol:
      break
  Wt, rng = SSBJ_Aircraft(We=We, Ws=Ws, Wf=Wf, SFCp=SFC, LDr=LD).SBJ_aircraft_analysis()
  g = (list(struc.calculate_constraints()) + list(aero.calculate_constraints(Pg, [CL0, CL1]))
       + list(prop.calculate_constraints(TempE, ThrUA)) + [1. - rng / 2000.])
  return dict(Wt=Wt, Ws=Ws, Wf=Wf, We=We, theta=th, L=L, D=D, LD=LD, ESF=ESFn, SFC=SFC, range=rng, gmax=max(g))


# Tosserams et al. (2010), Table 1 (MDF optimum)
PAPER = dict(tc=0.0641, ARw=2.5, LAMBDAw=70., Sref=667., Sht=99.7, ARht=2.5, T=0.196, LAMBDAht=70., Lw=0.01, Lht=3.5,
             t=[0.97, 0.52, 0.21, 4.00, 3.66, 0.91, 0.97, 0.52, 0.21],
             ts=[2.17, 1.48, 0.76, 4.40, 4.02, 1.01, 2.17, 1.48, 0.76], lam=0.10,
             Wt=34300., Ws=17500., Wf=10300., We=6490., theta=19.1, L=34300., D=4800., LD=7.03, ESF=0.75, SFC=1.0)


def summarize(session):
  """Print the final design next to Table 1 and verify it with a full MDA."""
  mv = session.mdo.Coordinator.master_vars
  def val(name, sp=None):
    return [v.value for v in mv if v.name == name and (sp is None or v.sp_index == sp)]
  # Subproblem numbering differs between formulations: take z from the structures copy (the owner of lambda).
  struc = [v.sp_index for v in mv if v.name == "lambda"][0]
  z = [val(n, struc)[0] for n in ("tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht")]
  x = dict(T=val("T")[0], LAMBDAht=val("LAMBDAht")[0], Lw=val("Lw")[0], Lht=val("Lht")[0],
           t=val("t"), ts=val("ts"), lam=val("lambda")[0])
  print("\nDesign variable      DMDO      Table 1")
  for n, v in zip(("tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht"), z):
    print(f"  {n:<12} {v:>12.4g} {PAPER[n]:>10.4g}")
  for n in ("T", "LAMBDAht", "Lw", "Lht", "lam"):
    print(f"  {n:<12} {x[n]:>12.4g} {PAPER[n]:>10.4g}")
  print("  t   ", [round(a, 2) for a in x["t"]], "\n      ", PAPER["t"])
  print("  ts  ", [round(a, 2) for a in x["ts"]], "\n      ", PAPER["ts"])
  chk = mda_check(z, x["T"], x["LAMBDAht"], x["Lw"], x["Lht"], x["t"], x["ts"], x["lam"])
  print("\nMDA of the DMDO design (coupling values from the disciplines, not the copies):")
  for n in ("Wt", "Ws", "Wf", "We", "theta", "L", "D", "LD", "ESF", "SFC"):
    print(f"  {n:<12} {chk[n]:>12.5g} {PAPER[n]:>10.4g}")
  print(f"  range        {chk['range']:>12.5g}  (>= 2000 nm)")
  print(f"  max g        {chk['gmax']:>12.3e}  (<= 0)")
  return chk


if __name__ == "__main__":
  # python run_SBJ.py [a|b|c|d] [working_dir]: formulation of Fig. 6 (default b).
  args = sys.argv[1:]
  form = args.pop(0).lower() if args and args[0].lower() in ("a", "b", "c", "d") else "b"
  wd = args[0] if args else os.path.join(HERE, f"SBJ_BM_{form}")
  os.makedirs(wd, exist_ok=True)
  t0 = time.time()
  session = Session.from_yaml(os.path.join(HERE, f"SBJ_{form}.yaml"), working_dir=wd, blackboxes=BLACKBOXES)
  result = session.run()
  print(result.status, result.stop_reason, result.iterations, result.objective, result.max_inconsistency)
  print(f"WALL {time.time() - t0:.1f}s")
  summarize(session)
