[![pages-build-deployment](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/pages/pages-build-deployment/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/DNDO/actions/workflows/pages/pages-build-deployment)
[![lx-build-and-pytest](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/lx-build-and-tests.yml/badge.svg)](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/lx-build-and-tests.yml)
[![win-build-and-pytest](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/win-build-and-pytest.yml/badge.svg)](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/win-build-and-pytest.yml)
[![macos-build-and-pytest](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/macos-build-and-pytest.yml/badge.svg)](https://github.com/Ahmed-Bayoumy/DMDO/actions/workflows/macos-build-and-pytest.yml)

<img width="870" alt="DMDO_logo" src="DMDO_logo.png">

# DMDO

DMDO is a python implementation for the distributed multidisciplinary design optimization method called nonhierarchical analytical target cascading (NHATC)

For technical and code documentation, please visit [DMDO Webpage](https://ahmed-bayoumy.github.io/DMDO/).

---

**Version 2604.0**

---

## License & copyright

© Ahmed H. Bayoumy 

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).

See the LICENSE file for details.

---

## Citation

If you use this code, please cite it as shown below.

```pycon
   @software{DMDO_AB,
   author       = {Bayoumy, A.},
   title        = {DMDO},
   year         = 2022,
   publisher    = {Github},
   version      = {1.3.0},
   url          = {https://github.com/Ahmed-Bayoumy/DMDO}
   }
```

## How to use DMDO package

After installing the `DMDO` package using the `pip` command, the functions and classes of `DMDO` module can be imported directly to the python script as follows:

```pycon
from DMDO import *
```

## Using DMDO from other code: the Session API

`DMDO.Session` builds a problem and drives its coordination loop. It is the entry point for scripts and for other
packages; `main({...})` and the YAML files keep working and use the same building blocks (`load_setup`, `build_mdo`).

```python
from DMDO import Session, load_setup, solve_task

data = load_setup("problem.yaml")                 # a plain dict: edit it before building if needed
session = Session.from_dict(data, working_dir="runs", mdo_name="wing", blackboxes={"A1": aero, "A2": structures})

result = session.run()                            # 1. to the end
# or
while (record := session.step()) is not None:     # 2. one outer iteration at a time (IterationRecord)
    if record.max_inconsistency < 1e-4:
        session.request_stop()
# or
while not session.done:                           # 3. ask/tell: you solve the subproblems, anywhere
    tasks = session.ask()                         #    independent SubproblemTasks (Jacobi: all subproblems)
    session.tell([solve_task(t) for t in tasks])  #    results in any order

result = session.result()                         # RunResult: status, stop_reason, objective, variables, history
result.to_dict()                                  # JSON-ready
```

`Session.from_yaml(path, ...)` and `Session.from_checkpoint("coord_chk.pkl")` are the other constructors; `mode`,
`sweep`, `executor` and `observers` are accepted as keyword options. The `examples/` folder has runnable scripts and
adapter sketches for OpenMDAO and GEMSEO in both directions (their model inside DMDO, and DMDO inside their model).

## Extending DMDO and connecting other packages

DMDO talks to analyses and optimizers through small, framework-neutral interfaces (`DMDO.ports`). Nothing in DMDO
depends on a specific external package; an adapter only has to satisfy the interface.

**Disciplines.** Any object with a `compute(inputs: dict) -> dict` method (inputs and outputs keyed by variable name)
can be used wherever a blackbox callable is accepted, e.g. a wrapper around an OpenMDAO `Problem` or a GEMSEO
discipline:

```python
class MyDiscipline:
    def compute(self, inputs):
        return {"y1": inputs["x1"] ** 2}
```

**Naming blackboxes from YAML.** `blackbox:` / `opt:` names in a problem definition are resolved, in order, from the
`blackboxes` mapping passed to `main`, from the global registry, or as an import path:

```python
from DMDO import main, register_blackbox
register_blackbox("A1", my_analysis)                      # global registry
main({"setup_file": "problem.yaml", "mdo_name": "run", "working_dir": ".",
      "blackboxes": {"opt1": my_objective}})              # per-run mapping
# YAML: blackbox: "my_package.analyses:aero"             # import path
```

**Subproblem solvers.** Built-in strategies are `mads` (alias `OMADS`), `poll`, `search` and `scipy`. A new solver
subclasses `DMDO.BaseSolver`, implements `solve(request: SolverRequest) -> SolverOutcome`, and is registered either
in code or as a plugin through the `dmdo.solvers` entry-point group:

```python
from DMDO import BaseSolver, SolverOutcome, solvers

@solvers.register("my_solver")
class MySolver(BaseSolver):
    def solve(self, request):
        ...  # minimise request.objective(x) within request.lb/request.ub
        return SolverOutcome(x=x_best, fmin=f_best, hmin=h_best)
```

```toml
# pyproject.toml of a plugin package
[project.entry-points."dmdo.solvers"]
my_solver = "my_pkg.solvers:MySolver"
```

**Serial and parallel execution.** Two choices are independent of each other:

- the *sweep* decides what each subproblem sees within an outer iteration: `gauss_seidel` (each solve uses the
  targets updated by the previous ones) or `jacobi` (all solves start from the same snapshot; results are merged in
  subproblem order, so the outcome does not depend on which worker finishes first);
- the *executor* decides where the solves run: `inprocess`, `process` (a pool of worker processes), or any object with
  `map(fn, tasks)` / `close()` — e.g. a client for remote workers or a cluster.

```python
MDAO.run(file, mode="serial")                         # Gauss-Seidel, in this process (default)
MDAO.run(file, mode="parallel")                       # Jacobi, one worker process per subproblem
MDAO.run(file, sweep="jacobi", executor=my_cluster)   # Jacobi on remote workers
```

Each solve is shipped as a `SubproblemTask` and solved by the module-level `DMDO.solve_task`, so a remote worker only
needs DMDO and the problem's blackboxes installed. Executors can also be registered by name
(`DMDO.executors.register`) or through the `dmdo.executors` entry-point group.

**Observing a run.** The coordination loop calls `DMDO.RunObserver` hooks in this order: `on_run_start`, then per
outer iteration `on_iteration_start`, `on_sweep_end` (subproblems solved, inconsistencies computed), `on_iteration_end`
(an `IterationRecord` with the objective, max. inconsistency, max. penalty weight, variables change and worst link) and
`on_multipliers_updated`, and finally `on_run_end`. DMDO's own outputs are observers too (`CoordinationHistoryRecorder`
writes `Coordination_history.out`, `ConsoleReporter` the progress lines, `CheckpointWriter` `coord_chk.pkl`, and with
the AI regularizer `AIRegularizerHooks` and `RollbackGuard`). Extra observers, e.g. to stream progress to a dashboard
or a monitoring service, are passed to `run` and are called after DMDO's own; any object defining some of the hooks
works:

```python
from DMDO import RunObserver

class Progress(RunObserver):
    def on_iteration_end(self, run, record):
        print(record.iteration, record.objective, record.max_inconsistency)

MDAO.run(file, observers=[Progress()])
```

**Coordination policies.** How the ADMM penalty update is adapted is a separate strategy, chosen per coordinator:
`none` (plain ADMM) or `spike` (the AI regularizer; the default when `is_AI_regularizer` is set). The coordinator
option `adaptation_policy` selects another one by name or import path. A policy subclasses
`DMDO.CoordinationPolicy`, implements `compute_scales(coord, iteration) -> (beta_scale, gamma_scale, budget_scale,
event)` and is registered with `DMDO.policies.register("name", MyPolicy)` or through the `dmdo.policies` entry-point
group.

## Running DMDO as a service (optional)

`DMDO.service` exposes sessions over HTTP and sends subproblem solves to remote workers, for setups where the driver,
the coordination and the analyses run on different machines or in different languages. It uses only the Python
standard library and is not imported by `import DMDO`.

```commandline
dmdo serve  --host 0.0.0.0 --port 8400 --token <secret>                   # the service
dmdo worker --connect http://dmdo-host:8400 --tags aero --token <secret>  # where the aero code lives
dmdo worker --connect http://dmdo-host:8400 --tags structures --token <secret>
```

```python
from DMDO.service import Client

client = Client("http://dmdo-host:8400", token="<secret>")
session = client.create_session(setup_file="problem.yaml", mdo_name="wing",
                                blackboxes={"A1": "my_pkg.aero:A1", "A2": "my_pkg.structures:A2"},
                                sweep="jacobi", placement={1: "aero", 2: "structures"})
session.run()                       # or session.step() for one outer iteration
for event in session.events():      # server-sent events: iteration_end records, then run_end
    print(event)
print(session.result())
```

Endpoints: `POST /sessions`, `GET /sessions/{id}`, `POST /sessions/{id}/run|step|stop`,
`GET /sessions/{id}/result|history|events`, `DELETE /sessions/{id}`, plus `/work` for workers (see
`DMDO/service/server.py`). Workers run the same `solve_task` as local process pools, and with the Jacobi sweep the
results do not depend on which worker finishes first. `python -m DMDO.service` works without the `dmdo` script.

**Security.** Jobs and results are pickled, so the service, its workers and every client that holds the token must
trust each other. The service listens on `127.0.0.1` by default and refuses other interfaces without a token. Put TLS
(a reverse proxy) in front of it when traffic leaves a trusted network.

## How to run DMDO from terminal
After installing the libraries listed in the `requirements.txt`, `DMDO/DMDO.py` can be called directly from a 
terminal window under the src directory. The path of the JSON template, which contains the problem input parameters, should be entered as an input argument to the `DMDO.py` call. 

```commandline
python ./src/DMDO/DMDO.py ./tests/test_files/Basic_MDO.yaml
```

The generated `Coordination_history.out` file now includes two AI spike-tracking columns when the AI regularizer is enabled:

- `alarm_detected`: the raw detector alarm for the current update
- `spike_detected`: the stricter expected-spike flag returned by the coordinator helper after decisor and non-improvement gating






