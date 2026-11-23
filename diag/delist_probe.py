"""فحص رادار الشطب على بيانات SEC الحقيقية بدون تيليجرام (تشخيص)."""
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "delist_radar/delist_scanner.py", "--out", "diag_out/delisting.html", "--no-telegram"]))
