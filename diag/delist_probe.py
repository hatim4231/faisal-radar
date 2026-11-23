"""فحص رادار الشطب على بيانات SEC الحقيقية بدون تيليجرام (تشخيص) + عيّنة نصوص للتحقق."""
import subprocess, sys, json, re
rc = subprocess.call([sys.executable, "delist_radar/delist_scanner.py", "--out", "diag_out/delisting.html", "--no-telegram"])
P = json.load(open("output/sig_delist.json", encoding="utf-8"))
for sev in (3, 2, 1, 0):
    L = [r for r in P["stocks"] if r["sev"] == sev]
    print(f"\n===== sev {sev}: {len(L)} =====")
    for r in L[:12]:
        e = next(x for x in r["events"] if x["sev"] == max(y["sev"] for y in r["events"])) if sev else r["events"][0]
        print(f"{r['symbol']} ${r.get('price')} {e['form']} {e['date']} reasons={[a for _, a in e.get('reasons') or []][:2]} dl={e.get('deadline')}")
        print("    ", (e.get("snippet") or "")[:300])
sys.exit(rc)
