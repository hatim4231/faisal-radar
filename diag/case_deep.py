"""
دراسة سهم عميقة (DATA_RUN = الرمز) — تشخيص فقط، ما يغير شي:
1) الأسعار: يومي آخر 60 جلسة + التقسيمات سنتين + شموع الساعة آخر 6 أيام (بري/جلسة/أفتر) + شموع 15 دقيقة لآخر جلستين.
2) الأخبار: عناوين ياهو + نص الخبر.
3) SEC: كل الملفات آخر 12 شهر + نص البيانات الصحفية (8-K/6-K EX-99) آخر 30 يوم + غلاف ملفات الطرح/إعادة البيع آخر 120 يوم.
4) معلومات الشركة (فلووت، كاش، إيرادات، ملاك) + الشورت + HRT + أحداث التخفيف.
5) تقييم الرادارات (case_symbol) على آخر 6 جلسات.
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import pandas as pd
import requests
import alpaca_data as A
import scanner
import sec_tools
import news_tools

T = (os.environ.get("DATA_RUN") or "NUWE").strip().upper()
NOW = datetime.now(timezone.utc)


def sec_text(url: str, n: int = 3000) -> str:
    r = requests.get(url, headers={"User-Agent": sec_tools.USER_AGENT, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    r.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    txt = re.sub(r"[ \t\xa0]+", " ", soup.get_text("\n"))
    txt = re.sub(r"\n\s*\n+", "\n", txt).strip()
    return txt[:n]


def prices():
    d = A.daily([T], days=800).get(T)
    if d is None or d.empty:
        print("ما فيه بيانات يومية من Alpaca")
        return None
    sp = d[d["Stock Splits"].fillna(0) != 0]
    print("التقسيمات (سنتين):", [(str(ix.date()), float(r["Stock Splits"])) for ix, r in sp.iterrows()] or "ما فيه")
    print("\nيومي آخر 60 جلسة (فتح/أعلى/أدنى/إغلاق/فوليوم):")
    for ix, r in d.tail(60).iterrows():
        print(f"  {ix.date()}  {r.Open:.3f} / {r.High:.3f} / {r.Low:.3f} / {r.Close:.3f} / {r.Volume:,.0f}")
    hi52, lo52 = d.tail(252)["High"].max(), d.tail(252)["Low"].min()
    print(f"قمة/قاع سنة: {hi52:.3f} / {lo52:.3f} · متوسط فوليوم 20 جلسة {d.tail(20)['Volume'].mean():,.0f}")
    start = NOW - timedelta(days=7)
    hb = A.bars([T], "1Hour", start).get(T)
    if hb is not None and len(hb):
        ses = A.session_of(hb.index)
        print("\nشموع الساعة آخر 6 أيام (نيويورك · الجلسة · فتح/أعلى/أدنى/إغلاق · فوليوم):")
        for ix, r in hb.iterrows():
            print(f"  {ix.strftime('%m-%d %H:%M')} {ses.loc[ix]:>3}  {r.Open:.3f}/{r.High:.3f}/{r.Low:.3f}/{r.Close:.3f}  {r.Volume:,.0f}")
    mb = A.bars([T], "15Min", NOW - timedelta(days=3)).get(T)
    if mb is not None and len(mb):
        ses = A.session_of(mb.index)
        print("\nشموع 15 دقيقة آخر 3 أيام (بس اللي فيها فوليوم):")
        for ix, r in mb.iterrows():
            if r.Volume > 0:
                print(f"  {ix.strftime('%m-%d %H:%M')} {ses.loc[ix]:>3}  {r.Open:.3f}/{r.High:.3f}/{r.Low:.3f}/{r.Close:.3f}  {r.Volume:,.0f}")
    return d


def news():
    import yfinance as yf
    try:
        items = scanner.parse_news(yf.Ticker(T).news)
    except Exception as e:
        print("الأخبار خطأ:", repr(e)[:120])
        return
    print(f"\nالأخبار من ياهو ({len(items)}):")
    for k, n in enumerate(items[:10]):
        s = news_tools.sentiment(n.get("title", ""), n.get("summary", "") or "")
        print(f"  [{n.get('time')}] {n.get('source', '')} · {s['label']} · {n.get('title')}")
        if k < 4 and n.get("url"):
            body = news_tools.fetch_article(n.get("url"))
            if body:
                print("     نص:", body[:1800].replace("\n", " ⏎ "))


def sec():
    cik = sec_tools._cik(T)
    print(f"\nSEC: CIK {cik}")
    if not cik:
        return
    js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    rec = js.get("filings", {}).get("recent", {})
    rows = list(zip(rec.get("form", []), rec.get("filingDate", []), rec.get("accessionNumber", []),
                    rec.get("primaryDocument", []), rec.get("primaryDocDescription", []), rec.get("items", [""] * 9999)))
    cut12 = (NOW - timedelta(days=365)).date().isoformat()
    print("كل الملفات آخر 12 شهر (النوع · التاريخ · الوصف · البنود):")
    for f, dt, acc, doc, desc, items in rows:
        if dt >= cut12:
            print(f"  {f:10s} {dt}  {desc or ''}  {items or ''}")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}"
    cut30 = (NOW - timedelta(days=30)).date().isoformat()
    cut120 = (NOW - timedelta(days=120)).date().isoformat()
    shown = 0
    for f, dt, acc, doc, desc, items in rows:
        a = acc.replace("-", "")
        try:
            if f in ("8-K", "6-K") and dt >= cut30 and shown < 6:
                idx = sec_tools._get(f"{base}/{a}/index.json")
                ex = [x["name"] for x in idx.get("directory", {}).get("item", []) if re.search(r"ex-?99|ex99|exhibit99|press", x["name"], re.I)]
                for name in ex[:2]:
                    print(f"\n── {f} {dt} بند {items} · {name}:\n{sec_text(f'{base}/{a}/{name}', 3500)}")
                if not ex:
                    print(f"\n── {f} {dt} بند {items} (بدون EX-99) · {doc}:\n{sec_text(f'{base}/{a}/{doc}', 2500)}")
                shown += 1
            elif re.match(r"^(424B|S-1|S-3|F-1|F-3)", f) and dt >= cut120:
                print(f"\n── {f} {dt} · {doc} (الغلاف):\n{sec_text(f'{base}/{a}/{doc}', 3000)}")
        except Exception as e:
            print("   ملف ما انقرا:", f, dt, repr(e)[:100])


def info():
    infos = scanner.get_info([T], scanner.load_cfg())
    inf = infos.get(T) or {}
    keys = ("shortName", "longName", "country", "sector", "industry", "marketCap", "floatShares", "sharesOutstanding",
            "impliedSharesOutstanding", "heldPercentInsiders", "heldPercentInstitutions", "sharesShort", "sharesShortPriorMonth",
            "dateShortInterest", "shortPercentOfFloat", "averageVolume", "averageVolume10days", "totalCash", "totalDebt",
            "totalRevenue", "revenueGrowth", "grossMargins", "operatingCashflow", "freeCashflow", "ebitda", "fullTimeEmployees",
            "lastSplitFactor", "lastSplitDate")
    print("\nمعلومات ياهو:")
    for k in keys:
        if inf.get(k) is not None:
            print(f"  {k}: {inf.get(k)}")
    try:
        import yfinance as yf
        tk = yf.Ticker(T)
        for name in ("quarterly_cashflow", "quarterly_balance_sheet", "quarterly_income_stmt"):
            df = getattr(tk, name)
            if df is not None and not df.empty:
                want = [r for r in df.index if re.search(r"Cash|Revenue|Net Income|Operating|Share Issued|Ordinary Shares", str(r))][:12]
                print(f"\n{name}:\n", df.loc[want].iloc[:, :4].to_string()[:2500])
    except Exception as e:
        print("القوائم المالية خطأ:", repr(e)[:120])
    try:
        ib = scanner.load_ib_short()
        print("\nIB:", ib.get(T), "· IB فيه", len(ib), "سهم · الشورت المستخدم:", scanner.short_of(T, ib, inf))
    except Exception as e:
        print("IB خطأ:", repr(e)[:120])
    try:
        row = {"symbol": T, "name": inf.get("shortName"), "news": []}
        sec_tools.annotate([row], print)
        ev = sec_tools.dilution_events(row)
        print("أحداث التخفيف (90 يوم):", [(e["date"], " + ".join(e["what"])) for e in ev])
    except Exception as e:
        print("SEC annotate خطأ:", repr(e)[:120])


if __name__ == "__main__":
    print("السهم:", T, "· الوقت UTC:", NOW.strftime("%Y-%m-%d %H:%M"))
    for fn in (prices, news, sec, info):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} تعطل:", repr(e)[:200])
    print("\n================ تقييم الرادارات (case_symbol) ================")
    try:
        import case_symbol
        case_symbol.main()
    except Exception as e:
        print("case_symbol تعطل:", repr(e)[:200])
