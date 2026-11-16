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
