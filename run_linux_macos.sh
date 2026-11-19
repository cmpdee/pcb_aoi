#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -f requirements.txt ]; then
  echo "[ERROR] Missing requirements.txt. Please extract the full release ZIP." >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python pcb_aoi.py
