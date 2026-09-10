"""Single source of the package version.

Kept import-free so ``setuptools`` can read it statically and so a frozen
onedir build, where ``importlib.metadata`` has no distribution to consult,
still reports a real version.
"""

__version__ = "0.1.0"
