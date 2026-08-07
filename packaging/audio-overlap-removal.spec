# PyInstaller spec for the standalone CLI. One directory, no bundled FFmpeg.
#
# Build with:  pyinstaller --noconfirm packaging/audio-overlap-removal.spec
# or, for the archive and checksum as well:  python packaging/build.py

import os

# Trimming scipy subpackages is not an option: `import scipy.signal` eagerly
# pulls in optimize, sparse, special, linalg, stats and the rest, so all of
# them are genuinely loaded at runtime. Measured, not assumed.
#
# ssl is different: nothing here opens a socket, so dropping it is safe.
#
# _hashlib is deliberately NOT excluded. It looks like another 5 MB of dead
# OpenSSL, and the application really never uses it, but on Linux PyInstaller
# adds a pkg_resources runtime hook that imports it before main() runs, and
# the build dies at startup. Platform-dependent excludes are not worth 4%.
#
# Re-measure before adding anything to this list, on every platform.
EXCLUDES = [
    "IPython",
    "PIL",
    "matplotlib",
    "pandas",
    "pytest",
    "ssl",
    "tkinter",
]

project_root = os.path.dirname(SPECPATH)

analysis = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[project_root],
    excludes=EXCLUDES,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name="audio-overlap-removal",
    console=True,
    # UPX corrupts some numpy/scipy shared libraries and markedly raises
    # antivirus false positives.
    upx=False,
)

COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="audio-overlap-removal",
)
