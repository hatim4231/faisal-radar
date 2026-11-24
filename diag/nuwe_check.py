"""فحص NUWE: هل أعلنت نتائج اجتماع المساهمين؟ (ملفات SEC: 8-K بند 5.07، DEF 14A، PRE 14A، DEFA14A، 8-K تأجيل)."""
import sys, re, html
sys.path.insert(0, ".")
import sec_tools
from delist_radar.delist_scanner import fetch_text
T = "NUWE"
cik = sec_tools._cik(T)
print("CIK", cik)
js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
print("name", js.get("name"))
rec = js["filings"]["recent"]
rows = list(zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"], rec.get("items", [""] * len(rec["form"]))))
print("\n=== آخر 40 ملف ===")
for f, d, a, doc, it in rows[:40]:
    print(d, f, it, doc)
KEYS = r"(annual|special) meeting|adjourn|postpon|quorum|vote[sd]?|approved|reverse (stock )?split|proposal"
print("\n=== تفاصيل ملفات الاجتماع ===")
for f, d, a, doc, it in rows[:60]:
    if d < "2026-07-01":
        break
    if f in ("DEF 14A", "PRE 14A", "DEFA14A", "DEFM14A", "PRER14A", "DEFR14A") or (f.startswith("8-K") and ("5.07" in it or "8.01" in it or "5.03" in it)):
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{doc}"
        t = fetch_text(url)
        print(f"\n--- {d} {f} items={it} {url}")
        for m in re.finditer(r"[^.]{0,220}(?:" + KEYS + r")[^.]{0,260}\.", t, re.I):
            print("   •", m.group(0).strip()[:480])
            if m.start() > 40000:
                break
