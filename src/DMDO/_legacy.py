"""Load the legacy module aliases (``DMDO.SP``, ``DMDO.coordinator``, ...) of the pre-subpackage layout.

Each legacy path is a thin module that aliases its new location (e.g. ``DMDO.SP`` -> ``DMDO.subproblem.subproblem``),
so old imports and checkpoints pickled by earlier versions keep working.

They are imported here, before ``DMDO/__init__.py`` binds its public names, to reproduce the old import order: the
package attributes ``DMDO.MDO``, ``DMDO.DA``, ``DMDO.MDA`` (classes) and ``DMDO.coordinator`` (protocol) then take
precedence over the same-named legacy modules. Importing a legacy alias for the first time later, for instance while
unpickling an old checkpoint, would otherwise rebind the package attribute to the module.
"""
import importlib

LEGACY_MODULES = ("_globals", "variables", "_protocols", "_common", "_decisor", "_detector", "coordinator", "DA",
                  "MDA", "SP", "MDO", "preprocess", "DMDO")

for _name in LEGACY_MODULES:
  importlib.import_module(f"{__package__}.{_name}")
