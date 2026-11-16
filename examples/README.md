# DMDO examples

Run the scripts from this directory (`cd examples`). They use the basic two-subproblem problem in
`basic_problem.yaml`, with analyses written as `DMDO.Discipline` objects in `basic_disciplines.py`.

| Script | Shows |
|---|---|
| `session_basics.py` | `Session.run()`, `step()` with an observer and an early stop, and `ask()`/`tell()` |
| `ask_tell_process_pool.py` | ask/tell with your own scheduler (a `concurrent.futures` process pool) |
| `service_demo.py` | the HTTP service: a session driven over HTTP, solved by a remote worker |
