import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sec_tools as st
for t in sys.argv[1].split(","):
    r = st.check(t)
    print(f"\n#### {t}: level={r and r['level']}")
    for h in (r or {}).get("hits", []):
        print(f"   {h['date']}  {h['form']:16s} sev{h['sev']}  {h['label']}  (قبل {h['days_ago']} يوم)")
