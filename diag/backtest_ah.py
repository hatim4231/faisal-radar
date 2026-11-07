"""
اختبار تاريخي لطريقة الأفتر ماركت (تشخيص فقط) — آخر ~60 يوم بشموع الساعة (فيها البري والأفتر).

الإشارة: يوم D، السهم بين 0.50 و 5$ عند الإغلاق،
  - قفز في الأفتر: أعلى سعر في الأفتر ≥ الإغلاق +15%
  - حافظ على الصعود: آخر سعر في الأفتر محافظ على 70%+ من القفزة (ما رجع لسعر انطلاقه)
  - فوليوم الأفتر ≥ 50 ألف سهم
الدخول (طريقتين):
  A) آخر سعر في الأفتر (كأنك دخلت آخر الأفتر)
  B) الدعم في البري: إذا نزل البري لفواب الأفتر (أو أقل) ندخل عنده، وإلا ما فيه صفقة
النتيجة في يوم D+1 (بري + جلسة) بشموع الساعة: هل وصل +10% / +15% / +30% قبل −10%؟ (نفس الشمعة = وقف، الأحوط)
مقارنة: كل قفزات الأفتر (بدون شرط الثبات) + أسهم عادية.
وكمان: كم مرة طلع ملف طرح (424B) في SEC يوم D+1 أو D+2.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scanner
import sec_tools


def sessions(d: pd.DataFrame):
    d = d.dropna(subset=["Close"]).copy()
    idx = d.index.tz_convert("America/New_York") if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert("America/New_York")
    d.index = idx
    hm = idx.hour * 60 + idx.minute
    d["sess"] = np.where(hm < 570, "pm", np.where(hm < 960, "rth", "ah"))
    d["day"] = idx.date
    return d


def outcome(bars: pd.DataFrame, entry: float, tgt: float, stop: float = 0.10):
    for _, r in bars.iterrows():
        if r["Low"] <= entry * (1 - stop):
            return "fail"
        if r["High"] >= entry * (1 + tgt):
            return "hit"
    return "none"


DIAG = {"days": 0, "ah_bars": 0, "ah_zero_vol": 0, "pm_days": 0, "jumps15": 0}


def scan_ticker(t: str, d: pd.DataFrame):
    out = []
    d = sessions(d)
    days = sorted(d["day"].unique())
    for i in range(len(days) - 1):
        D, N = days[i], days[i + 1]
        x = d[d["day"] == D]
        rth, ah = x[x.sess == "rth"], x[x.sess == "ah"]
        if rth.empty or ah.empty:
            continue
        close = float(rth["Close"].iloc[-1])
        if not (0.5 <= close <= 5):
            continue
        ah_hi = float(ah["High"].max())
        ah_last = float(ah["Close"].iloc[-1])
        ah_vol = float(ah["Volume"].sum())
        jump = ah_hi / close - 1
        hold = (ah_last - close) / (ah_hi - close) if ah_hi > close else 0
        ah_vwap = float(((ah["High"] + ah["Low"] + ah["Close"]) / 3 * ah["Volume"]).sum() / ah_vol) if ah_vol else ah_last
        nxt = d[d["day"] == N]
        pm, rth2 = nxt[nxt.sess == "pm"], nxt[nxt.sess == "rth"]
        nb = pd.concat([pm, rth2])
        if nb.empty:
            continue
        rec = {"t": t, "date": str(D), "jump": jump, "hold": hold, "ah_vol": ah_vol,
               "ah_last_vs_close": ah_last / close - 1,
               "maxup": float(nb["High"].max()) / ah_last - 1, "dd": float(nb["Low"].min()) / ah_last - 1,
               "close_next": float(rth2["Close"].iloc[-1]) / ah_last - 1 if not rth2.empty else np.nan}
        DIAG["days"] += 1
        DIAG["ah_bars"] += len(ah)
        DIAG["ah_zero_vol"] += int((ah["Volume"] == 0).sum())
        DIAG["pm_days"] += int(not pm.empty)
        if jump >= 0.15:
            DIAG["jumps15"] += 1
        rth_vol = float(rth["Volume"].sum())
        if jump >= 0.15 and rth_vol >= 100_000:
            kind = "AH_HOLD" if hold >= 0.7 else "AH_FADE"
            rec["sig"] = kind
            for tg in (0.10, 0.15, 0.30):
                rec[f"A{int(tg * 100)}"] = outcome(nb, ah_last, tg)
            # B) الدخول عند فواب الأفتر في البري
            dip = pm[pm["Low"] <= ah_vwap]
            if not dip.empty and ah_vwap < ah_last:
                after = nb.loc[dip.index[0]:]
                for tg in (0.10, 0.15, 0.30):
                    rec[f"B{int(tg * 100)}"] = outcome(after, ah_vwap, tg)
            out.append(rec)
        elif abs(jump) < 0.05 and np.random.random() < 0.05:
            rec["sig"] = "NORMAL"
            for tg in (0.10, 0.15, 0.30):
                rec[f"A{int(tg * 100)}"] = outcome(nb, ah_last, tg)
            out.append(rec)
    return out


def main():
    import yfinance as yf
    np.random.seed(1)
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU")]
    print("tickers", len(tickers), flush=True)
    rows = []
    for i in range(0, len(tickers), 60):
        batch = tickers[i:i + 60]
        try:
            h = yf.download(batch, period="60d", interval="1h", prepost=True, group_by="ticker",
                            auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print("err", repr(e)[:100]); continue
        for t in batch:
            try:
                d = h[t] if isinstance(h.columns, pd.MultiIndex) else h
                if d.dropna(subset=["Close"]).empty:
                    continue
                rows += scan_ticker(t, d)
            except Exception:
                pass
        print(i, len(rows), flush=True)
    print("== تشخيص البيانات:", DIAG, flush=True)
    df = pd.DataFrame(rows)
    # ملفات طرح بعد القفزة (للإشارات فقط)
    sig = df[df.sig.isin(["AH_HOLD", "AH_FADE"])].copy()

    def offering(r):
        try:
            cik = sec_tools._cik(r.t)
            js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            rec = js["filings"]["recent"]
            d0 = pd.Timestamp(r.date)
            for f, fd in zip(rec["form"], rec["filingDate"]):
                if f.startswith("424B") and f != "424B3" and 0 <= (pd.Timestamp(fd) - d0).days <= 3:
                    return True
        except Exception:
            return None
        return False
    with ThreadPoolExecutor(4) as ex:
        sig["offer"] = list(ex.map(offering, [r for r in sig.itertuples()]))
    df = df.merge(sig[["t", "date", "offer"]], on=["t", "date"], how="left")

    def summarize(sub, name):
        n = len(sub)
        if not n:
            return
        line = {"n": n}
        for k in ("A10", "A15", "A30", "B10", "B15", "B30"):
            if k in sub:
                v = sub[k].dropna()
                if len(v):
                    line[k + "_hit%"] = round((v == "hit").mean() * 100, 1)
                    line[k + "_n"] = len(v)
        line["A10_stop%"] = round((sub["A10"] == "fail").mean() * 100, 1)
        line["maxup_med%"] = round(sub["maxup"].median() * 100, 1)
        line["close_next_med%"] = round(sub["close_next"].median() * 100, 1)
        print(f"\n## {name}\n   " + " · ".join(f"{k}={v}" for k, v in line.items()))

    print("\n==== النتائج (الدخول A = آخر سعر في الأفتر · B = فواب الأفتر في البري) ====")
    for s in ("AH_HOLD", "AH_FADE", "NORMAL"):
        summarize(df[df.sig == s], s)
    h = df[df.sig == "AH_HOLD"]
    summarize(h[h.offer == True], "AH_HOLD + طرح خلال 3 أيام")  # noqa: E712
    summarize(h[h.offer == False], "AH_HOLD بدون طرح")  # noqa: E712
    summarize(h[h.jump >= 0.30], "AH_HOLD والقفزة 30%+")
    summarize(h[h.ah_vol >= 500_000], "AH_HOLD وفوليوم الأفتر 500K+")
    print("\nنسبة الإشارات اللي جاها طرح خلال 3 أيام:", round(h.offer.mean() * 100, 1), "%")
    for r in h.sort_values("maxup", ascending=False).head(15).itertuples():
        print(f"   {r.t} {r.date}: قفزة {r.jump * 100:.0f}% · ثبات {r.hold * 100:.0f}% · أقصى بعدها {r.maxup * 100:+.0f}% · إغلاق اليوم التالي {r.close_next * 100:+.0f}% · طرح={r.offer}")
    print("   ... أسوأ:")
    for r in h.sort_values("close_next").head(8).itertuples():
        print(f"   {r.t} {r.date}: إغلاق اليوم التالي {r.close_next * 100:+.0f}% · أقصى نزول {r.dd * 100:.0f}% · طرح={r.offer}")


if __name__ == "__main__":
    main()
