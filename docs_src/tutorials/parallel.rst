Serial, parallel and custom execution
#####################################

Two independent choices decide how the subproblems of an outer iteration are solved:

the **sweep**
   which coordinator state each subproblem sees: ``gauss_seidel`` (each solve uses the copies updated by the
   previous solves of the same outer iteration) or ``jacobi`` (all solves start from the same snapshot);
the **executor**
   where the solves run: ``inprocess``, ``process`` (a pool of worker processes), a remote service, or any object
   with ``map(fn, tasks)`` and ``close()``.

``mode`` picks a common combination, and ``sweep=`` and ``executor=`` override it:

.. code-block:: python

   MDAO.run(file, mode="serial")                          # Gauss–Seidel, in this process (default)
   MDAO.run(file, mode="parallel")                        # Jacobi, one worker process per subproblem
   MDAO.run(file, sweep="jacobi", executor="inprocess")   # Jacobi without worker processes
   MDAO.run(file, sweep="jacobi", executor=my_executor)   # Jacobi on your own back end

The same options are accepted by :class:`~DMDO.Session` (``Session.from_yaml(..., mode="parallel")``).

Which sweep?
============

Gauss–Seidel propagates information within an outer iteration and usually needs fewer outer iterations. Jacobi
solves all subproblems concurrently, so an outer iteration takes as long as the slowest subproblem instead of the
sum of all of them; it pays off when the subproblems are expensive and of similar cost. The two sweeps follow
different paths and in general end at different, equally consistent designs.

Same answer, whatever the execution order
=========================================

In a Jacobi sweep every subproblem starts from the same snapshot, and the results are merged in subproblem order,
each subproblem contributing only the copies it owns. The outcome therefore does not depend on which worker finishes
first. The regression suite checks that a Jacobi run gives bit-identical results when its subproblems are solved in
process, in reversed order, with the results returned out of order, and in a process pool.

Solving the tasks in your own process pool
==========================================

``examples/ask_tell_process_pool.py`` uses :class:`~DMDO.Session`'s ask/tell interface with a
:class:`concurrent.futures.ProcessPoolExecutor`; any other scheduler (Dask, Ray, a job queue, another framework's
parallel evaluator) is used the same way:

.. literalinclude:: ../../examples/ask_tell_process_pool.py
   :language: python
   :lines: 8-

Output:

.. code-block:: text

   iteration 0: max|q| = 1.100e-01
   iteration 1: max|q| = 7.967e-02
   iteration 2: max|q| = 5.994e-02
   iteration 3: max|q| = 8.918e-02
   iteration 4: max|q| = 1.254e-01
   iteration 5: max|q| = 8.791e-02
   iteration 6: max|q| = 6.247e-02
   iteration 7: max|q| = 4.121e-02
   {'u_1': 1.4966531250000001, 'v_1': 0.7853000000000002, 'a_1': 0.3115753094415026, 'b_1': 1.7434863644393819,
    'u_2': 1.8410000000000006, 'w_2': 7.835630000000001, 'a_2': 0.6735099999999999, 'b_2': 2.155564158811199}

The final design is exactly the one of the in-process Jacobi run of :doc:`session` (objective 4.337014798880885)
and of the remote workers of :doc:`service`: three different execution back ends, one answer.

.. important::

   Worker processes import the analyses and objectives by reference, so they must be defined at module level in an
   importable module (here ``basic_disciplines.py``), and the script needs the ``if __name__ == "__main__":``
   guard on Windows and macOS.

A custom executor
=================

An executor is any object with ``map(fn, tasks)`` returning the results in task order, and ``close()``. Register a
factory under a name to select it with ``executor="<name>"``:

.. code-block:: python

   from DMDO import InProcessExecutor, executors

   class CountingExecutor(InProcessExecutor):
       """Solves tasks in this process and counts them."""

       def __init__(self, max_workers=None):
           self.solved = 0

       def map(self, fn, tasks):
           self.solved += len(tasks)
           return super().map(fn, tasks)

   executors.register("counting", CountingExecutor)

   session = Session.from_yaml("problem.yaml", executor="counting", blackboxes={...})

A factory is called as ``factory(max_workers=n)``. Executors given as instances are not closed by DMDO; executors
created from a name are closed at the end of the run.
