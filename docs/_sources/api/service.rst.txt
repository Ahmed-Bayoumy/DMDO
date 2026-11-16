Service layer
=============

.. code-block:: python

   from DMDO.service import Client, start_service, serve, run_worker, RemoteExecutor, TaskQueue

Client
------

.. autoclass:: DMDO.service.Client
   :members: health, create_session, session, sessions

.. autoclass:: DMDO.service.RemoteSession
   :members: status, run, step, stop, result, history, events, delete

.. autoexception:: DMDO.service.ServiceError

Service
-------

.. autofunction:: DMDO.service.serve

.. autofunction:: DMDO.service.start_service

.. autoclass:: DMDO.service.ServiceHandle
   :members: url, shutdown

.. autoclass:: DMDO.service.DMDOService

Workers and remote execution
----------------------------

.. autofunction:: DMDO.service.run_worker

.. autoclass:: DMDO.service.RemoteExecutor
   :members: map, close

.. autoclass:: DMDO.service.TaskQueue
   :members: put, wait, claim, complete, cancel, stats

.. autoexception:: DMDO.service.RemoteTaskError
