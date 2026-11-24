"""NUWE: وش تقول ملفات SEC عن اجتماع/تواصل مع الـ FDA (آخر 8 شهور) + نص 8-K 17 نوفمبر ومرفقاته."""
import sys, re
sys.path.insert(0, ".")
import sec_tools
from delist_radar.delist_scanner import fetch_text
cik = sec_tools._cik("NUWE")
rec = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
PAT = r"[^.]{0,300}\b(?:FDA|Food and Drug|510\(k\)|De Novo|Pre-?Sub|Q-?Sub|breakthrough|clearance|indication)[^.]{0,350}\."
for f, d, a, doc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"]):
    if d < "2026-04-01":
        break
    if not (f.startswith("8-K") or f.startswith("10-Q") or f.startswith("10-K") or f.startswith("424B")):
        continue
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/"
    urls = [base + doc]
    try:   # المرفقات (البيان الصحفي EX-99)
        idx = sec_tools._get(base + "index.json")
        urls += [base + x["name"] for x in idx["directory"]["item"] if re.search(r"ex.?99", x["name"], re.I)]
    except Exception as e:
        pass
    for u in urls:
        t = fetch_text(u)
        hits = [m.group(0).strip() for m in re.finditer(PAT, t, re.I)]
        seen = set(); hits = [h for h in hits if not (h[:80] in seen or seen.add(h[:80]))]
        print(f"\n=== {d} {f} {u.rsplit('/',1)[-1]} · FDA ذُكرت {len(hits)} مرة")
        if u.rsplit('/',1)[-1] != doc or f.startswith("8-K"):
            print("   [بداية النص]", t[:900])
        for h in hits[:12]:
            print("   •", h[:600])
