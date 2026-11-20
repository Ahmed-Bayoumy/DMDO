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
"""
Example of fixing Windows console encoding for Unicode support.
Add the code from this file to the top of your scripts that use OMADS.
"""

import sys

# ============================================================================
# FIX WINDOWS CONSOLE ENCODING - Add this to your scripts
# ============================================================================

# Method 1: Reconfigure stdout to use UTF-8 with error handling
if sys.platform == 'win32':
    # Reconfigure stdout to UTF-8 with backslashreplace for unsupported chars
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

# ============================================================================
# END OF FIX
# ============================================================================

# Test the fix
if __name__ == "__main__":
    print(f"Current encoding: {sys.stdout.encoding}")
    print("Testing Unicode block character: █")
    print("Testing progress bar: |████░░░░| 50%")
    print("✓ If you see blocks above, the fix worked!")
