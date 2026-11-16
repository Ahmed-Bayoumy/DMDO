"""Package layout guards: legacy import paths keep working, and subpackages only import downwards."""
import ast
import importlib
import os
import pickle

import pytest

import DMDO

LEGACY = {
  "DMDO._globals": "DMDO.core.enums",
  "DMDO.variables": "DMDO.core.variables",
  "DMDO._protocols": "DMDO.core.protocols",
  "DMDO._common": "DMDO.utils.log",
  "DMDO.fix_windows_encoding": "DMDO.utils.fix_windows_encoding",
  "DMDO._decisor": "DMDO.ai.decisor",
  "DMDO._detector": "DMDO.ai.detector",
  "DMDO.coordinator": "DMDO.coordination.admm",
  "DMDO.DA": "DMDO.analysis.da",
  "DMDO.MDA": "DMDO.analysis.mda",
  "DMDO.SP": "DMDO.subproblem.subproblem",
  "DMDO.MDO": "DMDO.workflow.mdo",
  "DMDO.preprocess": "DMDO.app.builder",
  "DMDO.DMDO": "DMDO.app.main",
}


@pytest.mark.parametrize("legacy, new", sorted(LEGACY.items()))
def test_legacy_module_paths_alias_the_new_modules(legacy, new):
  assert importlib.import_module(legacy) is importlib.import_module(new)


def test_legacy_private_names_still_importable():
  from DMDO.SP import SubProblem, _seed_all_rngs  # noqa: F401
  from DMDO._detector import HybridAnomalyDetector, _rls_step  # noqa: F401
  from DMDO.coordinator import ADMM  # noqa: F401


def test_package_level_classes_are_not_shadowed_by_legacy_modules():
  importlib.import_module("DMDO.MDO")
  importlib.import_module("DMDO.DA")
  importlib.import_module("DMDO.MDA")
  assert isinstance(DMDO.MDO, type) and DMDO.MDO.__name__ == "MDO"
  assert isinstance(DMDO.DA, type) and isinstance(DMDO.MDA, type)


@pytest.mark.parametrize("legacy_module, name", [("DMDO.SP", "SubProblem"), ("DMDO.MDO", "MDO"),
                                                 ("DMDO.coordinator", "ADMM"), ("DMDO.variables", "variableData")])
def test_checkpoints_pickled_with_legacy_paths_resolve(legacy_module, name):
  # A pickle written by an earlier DMDO version references classes by their old module path.
  legacy_ref = f"c{legacy_module}\n{name}\n.".encode()
  assert pickle.loads(legacy_ref) is getattr(DMDO, name)


# --------------------------------------------------------------------------------------------- layering
# A module may import from its own subpackage or from a strictly lower layer.
LAYERS = {
  "core": 0,
  "utils": 1, "ports": 1, "registry": 1,
  "ai": 2,
  "solvers": 3, "analysis": 3, "coordination": 3,
  "subproblem": 4,
  "workflow": 5,
  "app": 6, "dashboard": 6,
}
PKG_DIR = os.path.dirname(DMDO.__file__)


def _internal_imports(path, own_parts):
  tree = ast.parse(open(path, encoding="utf-8").read())
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
      if node.level:
        base = own_parts[: len(own_parts) - node.level]
        target = base + (node.module.split(".") if node.module else [])
        if not node.module:  # "from . import x" / "from .. import x"
          for alias in node.names:
            yield target + [alias.name]
          continue
        yield target
      elif node.module and node.module.split(".")[0] == "DMDO":
        yield node.module.split(".")[1:]


def _layered_modules():
  for sub in LAYERS:
    d = os.path.join(PKG_DIR, sub)
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".py")] if os.path.isdir(d) \
      else [os.path.join(PKG_DIR, sub + ".py")]
    for path in files:
      rel = os.path.relpath(path, PKG_DIR)[:-3].replace(os.sep, "/").split("/")
      yield path, rel


_MODULES = list(_layered_modules())


@pytest.mark.parametrize("path, parts", _MODULES, ids=["/".join(parts) for _, parts in _MODULES])
def test_subpackages_only_import_downwards(path, parts):
  own = parts[0]
  # relative imports resolve against the module's package (its parts minus the module name)
  for target in _internal_imports(path, parts):
    if not target or target[0] == own or target[0] not in LAYERS:
      continue
    assert LAYERS[target[0]] < LAYERS[own], (
      f"{'/'.join(parts)} (layer {LAYERS[own]}) imports DMDO.{'.'.join(target)} (layer {LAYERS[target[0]]})")
