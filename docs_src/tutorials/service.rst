The HTTP service and remote workers
###################################

``DMDO.service`` runs DMDO as a service: clients create and drive sessions over HTTP, follow their progress as a
live event stream, and the subproblem solves are distributed to remote workers, each pinned to the machine that
holds its analysis code. It uses only the Python standard library and is not imported by ``import DMDO``.

When to use it
==============

* the analyses live on different machines (a CFD code on a cluster, a structural model on a licensed workstation);
* the driver is not Python, or runs in another process or container;
* several people or tools need to follow or steer the same run.

For everything else, :class:`~DMDO.Session` in-process is simpler.

Starting the service and the workers
====================================

.. code-block:: console

   $ dmdo serve --host 0.0.0.0 --port 8400 --token <secret>

   $ dmdo worker --connect http://dmdo-host:8400 --tags aero --token <secret> --path /opt/aero_models
   $ dmdo worker --connect http://dmdo-host:8400 --tags structures --token <secret>

``python -m DMDO.service serve`` and ``python -m DMDO.service worker`` are equivalent. ``--path`` adds a directory
to the worker's import path, where the problem's analysis modules are. Without ``--host`` the service listens on
``127.0.0.1`` only; it refuses to listen on any other interface without a token.

Driving a session from Python
=============================

.. code-block:: python

   from DMDO.service import Client

   client = Client("http://dmdo-host:8400", token="<secret>")
   session = client.create_session(
       setup_file="problem.yaml",                    # a path the service can read, or setup={...} (a dict)
       mdo_name="wing",
       blackboxes={"A1": "my_pkg.aero:A1",           # import paths, resolved by the service and the workers
                   "A2": "my_pkg.structures:A2"},
       sweep="jacobi",
       placement={1: "aero", 2: "structures"},       # subproblem index -> worker tag
   )

   session.run()                                     # runs in the service; returns immediately
   for event in session.events():                    # iteration_end records, then run_end
       print(event)
   print(session.result())

``session.step()`` runs a single outer iteration instead and returns its record; ``session.stop()`` ends the run
after the current iteration; ``session.history()`` returns all iteration records.

Driving it from any language
============================

The service speaks plain HTTP and JSON (send ``Authorization: Bearer <token>`` with every request):

.. code-block:: text

   $ curl -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
          -d '{"setup_file": "problem.yaml", "mdo_name": "wing",
               "blackboxes": {"A1": "my_pkg.aero:A1", "A2": "my_pkg.structures:A2"},
               "options": {"sweep": "jacobi", "placement": {"1": "aero", "2": "structures"}}}' \
          http://dmdo-host:8400/sessions
   {"session_id": "3f9c2a71b0de", "status": "created", ...}

   $ curl -X POST -H "Authorization: Bearer $T" http://dmdo-host:8400/sessions/3f9c2a71b0de/step
   $ curl -N -H "Authorization: Bearer $T" http://dmdo-host:8400/sessions/3f9c2a71b0de/events

.. list-table:: Endpoints
   :header-rows: 1
   :widths: 10 34 56

   * - Method
     - Path
     - Action
   * - GET
     - ``/health``
     - service status, number of sessions and job counters
   * - GET / POST
     - ``/sessions``
     - list sessions / create one
   * - GET / DELETE
     - ``/sessions/{id}``
     - status / remove an idle or finished session
   * - POST
     - ``/sessions/{id}/run``, ``/step``, ``/stop``
     - run to the end in the background, run one outer iteration, stop after the current one
   * - GET
     - ``/sessions/{id}/result``, ``/history``
     - the :class:`~DMDO.RunResult` and the iteration records as JSON
   * - GET
     - ``/sessions/{id}/events?since=n``
     - server-sent events: ``iteration_end`` per outer iteration, then ``run_end`` (or ``run_failed``,
       ``closed``)
   * - GET / POST
     - ``/work``, ``/work/{job}``
     - used by workers to claim jobs and post their results

A complete run on one machine
=============================

``examples/service_demo.py`` starts a service, two worker processes tagged ``aero`` and ``structures``, and a
client that runs the basic problem with each subproblem pinned to one worker:

.. literalinclude:: ../../examples/service_demo.py
   :language: python
   :lines: 13-

Output:

.. code-block:: text

   DMDO worker connected to http://127.0.0.1:63514 (tags: structures)
   DMDO worker connected to http://127.0.0.1:63514 (tags: aero)
   service at http://127.0.0.1:63514
   iteration 0: objective 3.000000  max|q| 1.100e-01
   iteration 1: objective 3.864116  max|q| 7.967e-02
   iteration 2: objective 4.289449  max|q| 5.994e-02
   iteration 3: objective 4.136221  max|q| 8.918e-02
   iteration 4: objective 3.065328  max|q| 1.254e-01
   iteration 5: objective 3.071269  max|q| 8.791e-02
   iteration 6: objective 3.548070  max|q| 6.247e-02
   iteration 7: objective 4.337015  max|q| 4.121e-02
   run_end {'stop_reason': 'Iteration budget exhausted', 'iterations': 8}
   status: finished | variables: {'u_1': 1.4966531250000001, 'v_1': 0.7853000000000002, 'a_1': 0.3115753094415026,
   'b_1': 1.7434863644393819, 'u_2': 1.8410000000000006, 'w_2': 7.835630000000001, 'a_2': 0.6735099999999999,
   'b_2': 2.155564158811199}

The design is identical to the one computed in process (:doc:`session`) and in a local process pool
(:doc:`parallel`): with the Jacobi sweep, where the subproblems run does not change the result.

Reliability
===========

* A worker that dies in the middle of a job can be covered with ``dmdo serve --lease 600``: a job whose result has not
  arrived after 600 s is offered to another worker.
* A job that raises on the worker is reported back with its traceback, and the session fails with that error.
* A service restart loses the sessions held in memory, but every run still writes its checkpoint
  (``coord_chk.pkl``); resume it with ``Session.from_checkpoint``.

Security
========

Jobs and results are pickled, so the service, its workers and every client that holds the token must trust each
other: anyone who can submit a job can run code on the workers. Keep the service inside a trusted network, always
set a token when it listens on a non-loopback interface, and put a TLS-terminating reverse proxy in front of it when
traffic leaves that network. Problem files given as ``setup_file`` are read from the service's file system.
