"""Build the standalone one-directory distribution and archive it.

Run on the target platform; PyInstaller cannot cross-compile:

    python packaging/build.py
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

NAME = "audio-overlap-removal"
ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / f"{NAME}.spec"
DIST = ROOT / "dist"
SIZE_CEILING = 400_000_000


def _version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("Could not read version from pyproject.toml.")
    return match.group(1)


def _platform_tag() -> str:
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64"}.get(
        machine, machine or "unknown"
    )
    return f"{system}-{arch}"


def _run_pyinstaller(tag: str) -> Path:
    # Per-platform work and dist paths: the same checkout is built from Windows,
    # a Linux container and macOS, and a shared cache would mix their objects.
    # Run PyInstaller through this interpreter so the build always analyses the
    # environment the script was started from, whether or not it is activated.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--workpath",
            str(ROOT / "build" / tag),
            "--distpath",
            str(DIST / tag),
            str(SPEC),
        ],
        cwd=ROOT,
        check=True,
    )
    bundle = DIST / tag / NAME
    if not bundle.is_dir():
        raise SystemExit(f"PyInstaller did not produce {bundle}.")
    return bundle


def _smoke_test(bundle: Path) -> None:
    """--help exercises the full import chain, so a broken exclude shows up here."""
    executable = bundle / (f"{NAME}.exe" if sys.platform == "win32" else NAME)
    subprocess.run([str(executable), "--help"], check=True, stdout=subprocess.DEVNULL)


def _archive(bundle: Path, stem: str) -> Path:
    if sys.platform == "win32":
        # Plain zip: Windows has no executable bit to lose.
        archive = DIST / f"{stem}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(bundle.rglob("*")):
                handle.write(path, Path(stem) / path.relative_to(bundle))
        return archive

    if sys.platform == "darwin":
        # ditto is what Apple documents for shipping signed software, and its
        # zip keeps the symlinks, permissions and signatures that the embedded
        # Python.framework needs. A tar.gz would carry those too, but third
        # party unarchivers choke on it -- The Unarchiver fails on the plain
        # directory entries under numpy's dist-info -- while Archive Utility,
        # the handler most users actually have, unpacks a zip correctly.
        archive = DIST / f"{stem}.zip"
        archive.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as staging:
            # --keepParent names the top-level directory after its source, so
            # the copy has to carry the name users should end up with.
            named = Path(staging) / stem
            subprocess.run(["ditto", str(bundle), str(named)], check=True)
            subprocess.run(
                ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
                 str(named), str(archive)],
                check=True,
            )
        return archive

    archive = DIST / f"{stem}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=stem)
    return archive


def _check_size(bundle: Path) -> int:
    """Guard against building from a polluted environment.

    PyInstaller analyses whatever is installed alongside the package, so a
    shared or base environment can silently drag an entire ML stack into the
    bundle. numpy, scipy and libsndfile alone land well under this ceiling.
    """
    total = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    if total > SIZE_CEILING:
        raise SystemExit(
            f"Bundle is {total / 1e6:.0f} MB, far above the {SIZE_CEILING / 1e6:.0f} MB "
            "expected for this project. Build from a clean virtual environment "
            "holding only this package and its dependencies."
        )
    return total


def main() -> None:
    tag = _platform_tag()
    bundle = _run_pyinstaller(tag)
    unpacked = _check_size(bundle)
    _smoke_test(bundle)
    stem = f"{NAME}-{_version()}-{tag}"
    archive = _archive(bundle, stem)

    print(f"\n{archive.name}")
    print(f"  archive   {archive.stat().st_size / 1e6:.1f} MB")
    print(f"  unpacked  {unpacked / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
