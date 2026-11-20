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
"""Internal helper: UTF-8 console streams on Windows.

OMADS progress bars print block characters (``█``), which raise ``UnicodeEncodeError`` when stdout/stderr use the
Windows ANSI code page (e.g. when output is redirected). Not part of the public API.

Importing this module has no side effects. :func:`fix_encoding` is called only where DMDO owns the process (its worker
processes and the command-line entry point), never on ``import DMDO``, so a host application's streams are left alone.
"""
import os
import sys


def fix_encoding() -> bool:
  """Reconfigure ``sys.stdout``/``sys.stderr`` to UTF-8 on Windows; a no-op elsewhere.

  Returns ``False`` if the streams could not be reconfigured (``PYTHONIOENCODING`` is then set for child processes).
  """
  if sys.platform != "win32":
    return True
  try:
    for stream in (sys.stdout, sys.stderr):
      if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    return True
  except Exception as e:  # pragma: no cover - depends on the host's stream objects
    os.environ["PYTHONIOENCODING"] = "utf-8"
    print(f"Warning: could not reconfigure the console encoding ({e}); set PYTHONIOENCODING=utf-8 instead.")
    return False
