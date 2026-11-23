"""فحص استعلامات بحث نص SEC لرادار الشطب (تشخيص)."""
import sys, json
from urllib.parse import quote
sys.path.insert(0, ".")
import sec_tools
S, E = "2026-10-09", "2026-11-23"
Q = [('"Item 3.01"', "8-K"), ('"3.01"', "8-K"), ('"Notice of Delisting"', "8-K"), ('"Listing Rule 5550"', "8-K"), ('"Listing Rule 5550"', ""),
     ('"minimum bid price"', "8-K"), ('"minimum bid price"', "6-K"), ('"deficiency letter"', ""), ('"delisting determination"', ""),
     ('"Hearings Panel"', "6-K"), ('nasdaq', "8-K")]
for q, f in Q:
    url = f"https://efts.sec.gov/LATEST/search-index?q={quote(q)}&dateRange=custom&startdt={S}&enddt={E}" + (f"&forms={f}" if f else "")
    try:
        js = sec_tools._get(url)
        h = js.get("hits", {})
        print(q, f or "ALL", "total", h.get("total"), "page", len(h.get("hits", [])))
        for x in h.get("hits", [])[:2]:
            s = x["_source"]; print("   ", x["_id"], s.get("form"), s.get("file_date"), s.get("display_names"), s.get("items"))
    except Exception as e:
        print(q, f, "ERR", repr(e)[:200])
# الصيغة البديلة (واجهة الموقع)
for url in ["https://efts.sec.gov/LATEST/search-index?q=%22Item%203.01%22&forms=8-K",
            "https://efts.sec.gov/LATEST/search-index?q=%22minimum%20bid%20price%22&forms=8-K&startdt=2026-10-09&enddt=2026-11-23"]:
    try:
        js = sec_tools._get(url); print("ALT", url[-60:], js.get("hits", {}).get("total"))
    except Exception as e:
        print("ALT ERR", repr(e)[:200])
