#!/usr/bin/env bash
# Build the portable Linux distribution inside the manylinux build image.
#
#   docker build -t audio-overlap-removal-build:manylinux \
#     -f packaging/Dockerfile.manylinux packaging
#   docker run --rm -v "$PWD:/src" -w /src \
#     audio-overlap-removal-build:manylinux bash packaging/build-manylinux.sh
#
# See Dockerfile.manylinux for why the interpreter is the distribution's
# python3.11 rather than one of the /opt/python builds.
set -euo pipefail

PYTHON=${PYTHON:-/usr/bin/python3.11}
VENV=${VENV:-/tmp/aor-build-venv}

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e ".[build]"
exec "$VENV/bin/python" packaging/build.py
