"""Frozen-build entry point.

PyInstaller runs its target as a top-level script, which breaks the relative
imports in ``audio_overlap_removal/__main__.py``; this wrapper avoids that.
"""

from audio_overlap_removal.cli import main

if __name__ == "__main__":
    main()
