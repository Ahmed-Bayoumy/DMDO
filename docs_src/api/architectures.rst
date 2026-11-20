MDO architectures and MDA solvers
=================================

The architecture of a run is selected by the ``MDO.architecture`` key (:doc:`../user_guide/architectures`).

Architecture selection
----------------------

.. autoclass:: DMDO.ArchitectureSpec
   :members: path, label, is_monolithic, is_mdf, is_atc, is_nhatc

.. autofunction:: DMDO.parse_architecture

.. autoexception:: DMDO.ArchitectureError

.. autodata:: DMDO.architectures.DEFAULT_ARCHITECTURE
   :no-value:

Hierarchical ATC
----------------

.. automodule:: DMDO.architectures.atc
   :no-members:

.. autofunction:: DMDO.architectures.atc.build_hierarchy

.. autoclass:: DMDO.architectures.atc.Hierarchy
   :members: describe

.. autoexception:: DMDO.architectures.atc.HierarchyError

Monolithic MDF
--------------

.. automodule:: DMDO.architectures.mdf
   :no-members:

.. autofunction:: DMDO.architectures.mdf.run_mdf

.. autoclass:: DMDO.architectures.mdf.MDFResult

MDA feedback detection and solvers
----------------------------------

.. automodule:: DMDO.analysis.mda_solvers
   :no-members:

.. autofunction:: DMDO.analysis.mda_solvers.coupling_components

.. autoclass:: DMDO.analysis.mda_solvers.Component

.. autoclass:: DMDO.analysis.mda_solvers.MDASolver
   :members: solve

.. autoclass:: DMDO.analysis.mda_solvers.GaussSeidelSolver

.. autoclass:: DMDO.analysis.mda_solvers.JacobiSolver

.. autoclass:: DMDO.analysis.mda_solvers.NewtonSolver

.. autoclass:: DMDO.analysis.mda_solvers.MDAResult

.. autofunction:: DMDO.analysis.mda_solvers.create_mda_solver

.. automethod:: DMDO.MDA.run_coupled

.. autodata:: DMDO.registry.mda_solvers
   :no-value:
