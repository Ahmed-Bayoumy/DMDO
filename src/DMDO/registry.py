"""Name -> component registries (solvers, blackboxes).

A registry maps a case-insensitive name to a component. Components come from three sources, tried in order:

1. explicit registration: ``solvers.register("my_solver", MySolver)`` or the ``@solvers.register("my_solver")``
   decorator;
2. installed plugins: any distribution that declares an entry point in the registry's group
   (``dmdo.solvers`` / ``dmdo.blackboxes``) is discovered on first lookup, e.g. in a plugin's ``pyproject.toml``::

     [project.entry-points."dmdo.solvers"]
     my_solver = "my_pkg.solvers:MySolver"

3. import paths: a name of the form ``"package.module:attribute"`` is imported directly.

This is the extension point that lets other packages plug components into DMDO without DMDO knowing about them.
"""
from __future__ import annotations

import importlib
from importlib import metadata
from typing import Callable, Dict, Generic, Iterable, List, Optional, TypeVar

T = TypeVar("T")

__all__ = ["Registry", "RegistryError", "solvers", "blackboxes", "register_blackbox"]


class RegistryError(KeyError):
  """Raised when a name cannot be resolved by a registry."""


def import_from_path(path: str):
  """Import ``"package.module:attr"`` (or ``"package.module.attr"``) and return the attribute."""
  if ":" in path:
    module_name, _, attr_path = path.partition(":")
  else:
    module_name, _, attr_path = path.rpartition(".")
  if not module_name or not attr_path:
    raise ImportError(f"'{path}' is not an import path of the form 'package.module:attribute'")
  obj = importlib.import_module(module_name)
  for attr in attr_path.split("."):
    obj = getattr(obj, attr)
  return obj


class Registry(Generic[T]):
  """A case-insensitive name -> component registry with aliases and plugin discovery."""

  def __init__(self, kind: str, entry_point_group: Optional[str] = None) -> None:
    self.kind = kind
    self.entry_point_group = entry_point_group
    self._items: Dict[str, T] = {}
    self._aliases: Dict[str, str] = {}
    self._plugins_loaded = False

  @staticmethod
  def _key(name: str) -> str:
    return str(name).strip().lower()

  def register(self, name: str, component: Optional[T] = None, *, aliases: Iterable[str] = (),
               override: bool = False):
    """Register ``component`` under ``name``; usable as a decorator when ``component`` is omitted."""
    def _do(obj: T) -> T:
      key = self._key(name)
      if not override and key in self._items and self._items[key] is not obj:
        raise ValueError(f"A {self.kind} named '{name}' is already registered; pass override=True to replace it")
      self._items[key] = obj
      self._aliases.pop(key, None)
      for alias in aliases:
        self._aliases[self._key(alias)] = key
      return obj

    if component is None:
      return _do
    return _do(component)

  def unregister(self, name: str) -> None:
    key = self._key(name)
    self._items.pop(key, None)
    self._aliases = {a: k for a, k in self._aliases.items() if k != key and a != key}

  def _load_plugins(self) -> None:
    if self._plugins_loaded or self.entry_point_group is None:
      return
    self._plugins_loaded = True
    for ep in metadata.entry_points(group=self.entry_point_group):
      key = self._key(ep.name)
      if key not in self._items and key not in self._aliases:
        self._items[key] = _LazyEntryPoint(ep)  # type: ignore[assignment]

  def _canonical(self, name: str) -> Optional[str]:
    key = self._key(name)
    key = self._aliases.get(key, key)
    if key in self._items:
      return key
    self._load_plugins()
    key = self._aliases.get(key, key)
    return key if key in self._items else None

  def canonical_name(self, name: str) -> Optional[str]:
    """The registered name ``name`` resolves to (following aliases), or None."""
    return self._canonical(name)

  def __contains__(self, name: object) -> bool:
    if not isinstance(name, str):
      return False
    if self._canonical(name) is not None:
      return True
    if ":" in name:
      try:
        import_from_path(name)
        return True
      except (ImportError, AttributeError):
        return False
    return False

  def get(self, name: str) -> T:
    key = self._canonical(name)
    if key is not None:
      item = self._items[key]
      if isinstance(item, _LazyEntryPoint):
        item = item.load()
        self._items[key] = item
      return item
    if isinstance(name, str) and ":" in name:
      try:
        return import_from_path(name)
      except (ImportError, AttributeError) as exc:
        raise RegistryError(f"Could not import {self.kind} '{name}': {exc}") from exc
    raise RegistryError(f"Unknown {self.kind} '{name}'. Registered: {', '.join(self.names()) or '(none)'}")

  def names(self) -> List[str]:
    self._load_plugins()
    return sorted(self._items)


class _LazyEntryPoint:
  """Defers importing a plugin until it is first used."""

  def __init__(self, ep) -> None:
    self.ep = ep

  def load(self):
    return self.ep.load()


#: Subproblem solver strategies (classes implementing :class:`DMDO.ports.SubproblemSolver`).
solvers: Registry[Callable] = Registry("solver", "dmdo.solvers")

#: Analysis / optimization blackboxes referenced by name from a problem definition (YAML ``blackbox:`` and
#: ``opt:`` keys). Values are callables or :class:`DMDO.ports.Discipline` objects.
blackboxes: Registry[Callable] = Registry("blackbox", "dmdo.blackboxes")


def register_blackbox(name: str, component: Optional[Callable] = None, *, override: bool = True):
  """Register a callable or discipline so problem definitions can refer to it by ``name``."""
  return blackboxes.register(name, component, override=override)
