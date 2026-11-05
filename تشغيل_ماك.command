#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "اول مرة: تجهيز البرنامج..."
  python3 -m venv .venv
  .venv/bin/python -m pip install -q -r requirements.txt
fi
.venv/bin/python scanner.py --serve --every 30
