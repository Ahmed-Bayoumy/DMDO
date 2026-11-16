Sessions and problem building
=============================

.. code-block:: python

   from DMDO import Session, RunResult, load_setup, build_mdo, main

``Session``
-----------

.. autoclass:: DMDO.Session
   :members: from_yaml, from_dict, from_checkpoint, run, step, ask, tell, result, request_stop, add_observer, close,
             done, iteration, pending_tasks

.. autoclass:: DMDO.RunResult
   :members: to_dict

.. autoexception:: DMDO.SessionError

Building problems
-----------------

.. autofunction:: DMDO.load_setup

.. autofunction:: DMDO.build_mdo

.. function:: DMDO.main(options)

   Build and run a problem from a YAML file, as in earlier releases. ``options`` is a dictionary with the keys
   ``setup_file``, ``mdo_name``, ``working_dir``, and optionally ``run_mode`` (``"run"`` or ``"build"``),
   ``exec_mode`` (``"serial"`` or ``"parallel"``), ``is_resume`` with ``restart_file`` (resume from a checkpoint),
   and ``blackboxes`` (name -> callable or discipline). Returns the :class:`~DMDO.MDO` object.

   .. code-block:: python

      mdo = main({"setup_file": "problem.yaml", "mdo_name": "wing", "working_dir": "runs",
                  "blackboxes": {"A1": aero, "A2": structures}})
