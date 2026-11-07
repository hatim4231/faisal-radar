"""
دراسة حالة NUWE قبل إعلان الأرباح (تشخيص فقط).
1) كل ملفات SEC من 2025 (النوع، التاريخ، بنود 8-K) + نصوص مهمة (نقد، أسهم قائمة، Warrants، ATM، going concern، إعادة تسعير).
2) أسعار Alpaca: يومي سنتين + آخر أسبوعين بالساعة (فيها البري والأفتر).
3) ردة فعل السهم في إعلانات الأرباح السابقة (8-K بند 2.02): الافتتاح/الإغلاق/أعلى/أدنى يوم الإعلان، وبعد 5 جلسات.
4) الشورت المتاح من IB.
"""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
import alpaca_data as A
import scanner
import sec_tools

SYM = "NUWE"
H = {"User-Agent": sec_tools.USER_AGENT}


def get(url):
    r = requests.get(url, headers=H, timeout=40)
    r.raise_for_status()
    return r


def text_of(url):
    html = get(url).text
    t = re.sub(r"<[^>]+>", " ", html)
    t = re.sub(r"&nbsp;|&#160;", " ", t)
    t = re.sub(r"&amp;", "&", t)
    return re.sub(r"\s+", " ", t)


def snippets(t, pats, width=260, maxn=4):
    out = []
    for p in pats:
        for m in list(re.finditer(p, t, re.I))[:maxn]:
            s = max(0, m.start() - width)
            out.append(f"[{p}] …{t[s:m.end() + width]}…")
    return out


def main():
    cik = sec_tools._cik(SYM)
    print("CIK", cik)
    js = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = js["filings"]["recent"]
    rows = []
    for i in range(len(rec["form"])):
        if rec["filingDate"][i] < "2025-06-01":
            continue
        rows.append({"form": rec["form"][i], "date": rec["filingDate"][i], "items": rec.get("items", [""] * 999)[i],
                     "acc": rec["accessionNumber"][i], "doc": rec["primaryDocument"][i], "desc": rec["primaryDocDescription"][i]})
    print("\n==== ملفات SEC من يونيو 2025 ====")
    for r in rows:
        print(f"{r['date']} {r['form']:<10} items={r['items']:<14} {r['desc']} {r['doc']}")

    def url(r):
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{r['acc'].replace('-', '')}/{r['doc']}"

    # آخر 10-Q
    q = [r for r in rows if r["form"] == "10-Q"]
    if q:
        t = text_of(url(q[0]))
        print(f"\n==== آخر 10-Q ({q[0]['date']}) مقتطفات ====")
        for s in snippets(t, [r"going concern", r"cash and cash equivalents (were|of)", r"shares of common stock outstanding",
                              r"substantial doubt", r"at-the-market", r"Series [A-Z] Warrants?", r"reset", r"runway",
                              r"stockholders.{0,3} equity"], 220, 2):
            print("-", s[:700])
    # آخر 8-K و 424B و S-1/S-3 من يونيو 2026
    print("\n==== ملفات آخر 5 شهور (مقتطفات) ====")
    for r in rows:
        if r["date"] < "2026-06-01" or r["form"] not in ("8-K", "424B5", "424B4", "424B3", "S-1", "S-1/A", "S-3", "DEF 14A", "EFFECT", "6-K"):
            continue
        try:
            t = text_of(url(r))
        except Exception as e:
            print(r["date"], r["form"], "خطأ", e); continue
        print(f"\n--- {r['date']} {r['form']} items={r['items']} ({len(t)} حرف)")
        print(t[:900])
        for s in snippets(t, [r"exercise price", r"reset", r"shares of (our )?common stock outstanding", r"gross proceeds",
                              r"Nasdaq", r"warrants? to purchase", r"stockholders.{0,3} equity", r"at[- ]the[- ]market"], 200, 2):
            print("  -", s[:520])
    # ملحقات 8-K للأرباح (EX-99.1)
    print("\n==== إعلانات الأرباح السابقة (8-K بند 2.02) ====")
    earn = [r for r in rows if r["form"] == "8-K" and "2.02" in (r["items"] or "")]
    for r in earn:
        print(r["date"], r["items"])
    # الأسعار
    now = datetime.now(timezone.utc)
    d = A.bars([SYM], "1Day", now - timedelta(days=760)).get(SYM)
    spl = A.splits([SYM], now - timedelta(days=760)).get(SYM, [])
    print("\n==== التقسيمات (Alpaca) ====", spl)
    if d is None or d.empty:
        print("ما فيه أسعار يومية"); return
    d.index = d.index.tz_localize(None).normalize()
    print("\n==== آخر 30 جلسة (معدّل للتقسيم) ====")
    print(d[["Open", "High", "Low", "Close", "Volume"]].tail(30).round(3).to_string())
    print("\n==== ردة فعل السهم يوم إعلان الأرباح ====")
    for r in earn:
        day = pd.Timestamp(r["date"])
        pos = d.index.searchsorted(day)
        if pos >= len(d) or pos < 1:
            continue
        # الإعلان قبل الافتتاح عادة؛ لو تاريخ الملف بعد يوم التداول نعرضه بس
        pc = float(d["Close"].iloc[pos - 1])
        x = d.iloc[pos]
        after5 = float(d["Close"].iloc[min(pos + 5, len(d) - 1)])
        print(f"{r['date']}: إغلاق قبل {pc:.3f} · افتتاح {x['Open'] / pc - 1:+.1%} · أعلى {x['High'] / pc - 1:+.1%} · "
              f"أدنى {x['Low'] / pc - 1:+.1%} · إغلاق {x['Close'] / pc - 1:+.1%} · بعد 5 جلسات {after5 / pc - 1:+.1%} · فوليوم {int(x['Volume']):,}")
    # آخر أسبوعين بالساعة
    h = A.bars([SYM], "1Hour", now - timedelta(days=16)).get(SYM)
    if h is not None and not h.empty:
        h["sess"] = A.session_of(h.index).values
        print("\n==== آخر أسبوعين بالساعة (فيها البري/الأفتر) ====")
        print(h[["Open", "High", "Low", "Close", "Volume", "sess"]].round(3).to_string())
    # الشورت المتاح
    try:
        ib = scanner.load_ib_short()
        print("\n==== IB ====", ib.get(SYM))
    except Exception as e:
        print("IB خطأ", e)
    try:
        inf = scanner.get_info([SYM], scanner.load_cfg()).get(SYM) or {}
        print("\n==== ياهو (معلومات الشركة) ====", {k: inf.get(k) for k in ("marketCap", "floatShares", "sharesOutstanding", "sharesShort",
                                                                         "shortPercentOfFloat", "heldPercentInsiders", "heldPercentInstitutions")})
    except Exception as e:
        print("info خطأ", e)
    try:
        import yfinance as yf
        for n in (yf.Ticker(SYM).news or [])[:15]:
            c = n.get("content") or n
            print("خبر:", c.get("pubDate") or c.get("providerPublishTime"), "|", c.get("title"))
    except Exception as e:
        print("news خطأ", e)


if __name__ == "__main__":
    main()
