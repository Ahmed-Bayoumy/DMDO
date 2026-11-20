# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#
"""The ``MDO.architecture`` key of a problem definition.

The key holds only a path of nested YAML keys, no options::

  architecture:           architecture:            architecture:
    MON: MDF                DIST:                    DIST:
                              IDF: ATC                 IDF: NHATC

Options stay in the blocks that already exist (``MDA.<id>.solver``, the subproblems, the ``coord`` blocks).

Any value that is not a path DMDO knows (for example the plain string ``IDF`` of older problem files) falls back to
the default ``DIST: IDF: NHATC``, which runs exactly as before, and the builder logs one warning.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

MON = "MON"
DIST = "DIST"
MDF = "MDF"
IDF = "IDF"
ATC = "ATC"
NHATC = "NHATC"
BLISS = "BLISS"


@dataclass(frozen=True)
class ArchitectureSpec:
  """A parsed ``MDO.architecture`` path."""
  family: str
  formulation: str
  method: Optional[str] = None

  @property
  def path(self) -> Tuple[str, ...]:
    return tuple(p for p in (self.family, self.formulation, self.method) if p is not None)

  @property
  def label(self) -> str:
    return ": ".join(self.path)

  @property
  def is_monolithic(self) -> bool:
    return self.family == MON

  @property
  def is_mdf(self) -> bool:
    return self.path == (MON, MDF)

  @property
  def is_atc(self) -> bool:
    return self.path == (DIST, IDF, ATC)

  @property
  def is_nhatc(self) -> bool:
    return self.path == (DIST, IDF, NHATC)

  def __str__(self) -> str:
    return self.label


#: The default architecture: distributed NHATC coordination, DMDO's original behaviour.
DEFAULT_ARCHITECTURE = ArchitectureSpec(DIST, IDF, NHATC)

SUPPORTED = (ArchitectureSpec(MON, MDF), ArchitectureSpec(DIST, IDF, ATC), DEFAULT_ARCHITECTURE)

# Recognised architectures that are deliberately out of scope; naming one is an error rather than a silent fallback.
_NOT_SUPPORTED = {(MON, IDF): "monolithic IDF", (BLISS,): "BLISS"}


class ArchitectureError(ValueError):
  """The ``MDO.architecture`` key names an architecture DMDO does not support."""


def _path_of(value: Any) -> Optional[List[str]]:
  """Flatten nested single-key mappings ending in a string (or an empty leaf) into a list of upper-case tokens;
  ``None`` if the value has another shape."""
  path: List[str] = []
  while True:
    if isinstance(value, str):
      token = value.strip()
      if not token:
        return path or None
      return path + [token.upper()]
    if value is None:
      return path or None
    if isinstance(value, dict) and len(value) == 1:
      (key, value), = value.items()
      if not isinstance(key, str):
        return None
      path.append(key.strip().upper())
      continue
    return None


def _supported_text() -> str:
  return ", ".join(s.label for s in SUPPORTED)


def parse_architecture(value: Any) -> Tuple[ArchitectureSpec, Optional[str]]:
  """Parse the ``MDO.architecture`` value.

  :returns: ``(spec, warning)``. ``warning`` is a message to log once when the value is not a known path and the
    default (NHATC) was selected; it is ``None`` when the value was understood or the key was absent.
  :raises ArchitectureError: the value names an architecture that is recognised but not supported (monolithic IDF,
    BLISS).
  """
  if value is None:
    return DEFAULT_ARCHITECTURE, None
  path = _path_of(value)
  if path is not None:
    for spec in SUPPORTED:
      if tuple(path) == spec.path:
        return spec, None
    for key, name in _NOT_SUPPORTED.items():
      if tuple(path) == key or (len(key) == 1 and key[0] in path):
        raise ArchitectureError(f"MDO.architecture {': '.join(path)} ({name}) is not supported. "
                                f"Supported: {_supported_text()}.")
  return DEFAULT_ARCHITECTURE, (
    f"MDO.architecture {value!r} is not a recognised architecture path; using the default "
    f"{DEFAULT_ARCHITECTURE.label}. Supported: {_supported_text()}.")
