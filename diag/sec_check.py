import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sec_tools as st, scanner
for t in sys.argv[1].split(","):
    cik = st._cik(t)
    print("\n####", t, "CIK", cik)
    if not cik:
        continue
    js = st._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    print("name:", js.get("name"), "| category:", js.get("category"), "| fiscal:", js.get("stateOfIncorporation"))
    rec = js["filings"]["recent"]
    for f, d, desc in list(zip(rec["form"], rec["filingDate"], rec.get("primaryDocDescription", [""] * 999)))[:40]:
        print(f"  {d}  {f:10s} {desc}")
    print("  check():", json.dumps(st.check(t), ensure_ascii=False)[:400])
    try:
        import yfinance as yf
        for n in scanner.parse_news(yf.Ticker(t).news)[:8]:
            print("  NEWS", n["time"][:10] if n.get("time") else "", n["title"])
    except Exception as e:
        print("  news err", e)
