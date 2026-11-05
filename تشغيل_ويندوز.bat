@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo اول مرة: تجهيز البرنامج...
  python -m venv .venv
  .venv\Scripts\python -m pip install -q -r requirements.txt
)
.venv\Scripts\python scanner.py --serve --every 30
pause
