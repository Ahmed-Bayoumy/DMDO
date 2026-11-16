"""
Windows console encoding fix for Unicode support.

This module automatically fixes Windows console encoding issues with Unicode
characters (like the block character █ used in OMADS progress bars).

Simply import this module at the start of your script:
    import fix_windows_encoding

Or call the function explicitly:
    from fix_windows_encoding import fix_encoding
    fix_encoding()
"""

import sys
import platform


def fix_encoding():
    """
    Fix Windows console encoding to support Unicode characters.
    
    This reconfigures sys.stdout and sys.stderr to use UTF-8 encoding
    on Windows systems, which allows proper display of Unicode block
    characters used by OMADS progress bars.
    
    Safe to call on any platform - only affects Windows systems.
    """
    if sys.platform == 'win32' or platform.system() == 'Windows':
        try:
            # Reconfigure stdout and stderr to use UTF-8
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
            return True
        except Exception as e:
            # Fallback - set environment variable for future streams
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            print(f"Warning: Could not reconfigure encoding ({e}), set PYTHONIOENCODING instead.")
            return False
    return True


# Automatically fix encoding when this module is imported
fix_encoding()


if __name__ == "__main__":
    # Test the fix
    print(f"Platform: {platform.system()}")
    print(f"stdout encoding: {sys.stdout.encoding}")
    print(f"stderr encoding: {sys.stderr.encoding}")
    print("\nTesting Unicode characters:")
    print("  Block character: █")
    print("  Progress bar: |████████░░| 80%")
    print("  Checkmark: ✓")
    print("  Cross: ✗")
    print("\n✓ If you see the characters above correctly, the fix is working!")
