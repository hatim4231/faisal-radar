import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sec_tools as st
for t in sys.argv[1].split(","):
    cik = st._cik(t)
    print("\n####", t, "CIK", cik)
    if not cik:
        continue
    js = st._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    print("name:", js.get("name"), "| category:", js.get("category"))
    rec = js["filings"]["recent"]
    rows = list(zip(rec["form"], rec["filingDate"], rec.get("primaryDocDescription", [""] * 9999), rec["accessionNumber"], rec["primaryDocument"]))
    for f, d, desc, acc, doc in rows[:30]:
        print(f"  {d}  {f:10s} {desc}")
    print("  check():", json.dumps(st.check(t), ensure_ascii=False)[:300])
    # بحث نص الملفات (EDGAR full-text search)
    q = '"registered direct" OR "private placement" OR "securities purchase agreement" OR "public offering" OR "warrant" OR "convertible"'
    url = ("https://efts.sec.gov/LATEST/search-index?q=" + __import__("urllib.parse").parse.quote(q) +
           f"&ciks={cik:010d}&forms=6-K,8-K&dateRange=custom&startdt=2026-08-01&enddt=2026-11-06")
    try:
        r = st._get(url)
        hits = r.get("hits", {}).get("hits", [])
        print("  FTS hits:", r.get("hits", {}).get("total"))
        for h in hits[:10]:
            s = h.get("_source", {})
            print("   FTS", s.get("file_date"), s.get("form"), s.get("file_description") or "", "|", (s.get("display_names") or [""])[0][:40], "|", h.get("_id"))
    except Exception as e:
        print("  FTS err", repr(e)[:200])
    # جرب أول 6-K / 8-K: نصه
    for f, d, desc, acc, doc in rows[:15]:
        if f in ("6-K", "8-K"):
            try:
                import requests
                u = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
                txt = requests.get(u, headers={"User-Agent": st.USER_AGENT}, timeout=20).text
                import re
                plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", txt))
                kws = [k for k in ("registered direct", "private placement", "securities purchase agreement", "public offering",
                                   "warrant", "convertible", "at-the-market", "equity line", "pre-funded") if k in plain.lower()]
                print(f"   DOC {d} {f}: {kws} :: {plain[300:520]}")
            except Exception as e:
                print("   DOC err", repr(e)[:120])
