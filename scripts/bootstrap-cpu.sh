#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -e .venv ]; then
    echo "Refusing to overwrite an existing .venv" >&2
    exit 1
fi

python="${PYTHON:-python3}"
"$python" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Python 3.11 or 3.12 is required"'
if "$python" -c 'import ensurepip' >/dev/null 2>&1; then
    "$python" -m venv .venv
else
    if [ -z "${PIP_BOOTSTRAP_PYTHON:-}" ]; then
        echo "This Python lacks ensurepip; set PIP_BOOTSTRAP_PYTHON to a Python with pip" >&2
        exit 1
    fi
    "$python" -m venv --without-pip .venv
    "$PIP_BOOTSTRAP_PYTHON" -m pip --python "$PWD/.venv/bin/python" install --no-cache-dir pip
fi
.venv/bin/python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
.venv/bin/python -m pip install --no-cache-dir -e '.[test]'
