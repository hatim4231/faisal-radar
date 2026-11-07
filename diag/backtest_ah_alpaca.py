"""
اختبار تاريخي لطريقة الأفتر ماركت ببيانات Alpaca (سنة، فوليوم صحيح، شموع 5 دقائق للإشارات).

1) شموع ساعة لسنة لكل الأسهم ← نلقى أيام القفزة في الأفتر.
2) لكل إشارة: شموع 5 دقائق (الأفتر + البري + الجلسة اللي بعده) ← النتيجة بدقة.

الإشارة يوم D: الإغلاق 0.50-5$ · أعلى سعر في الأفتر ≥ الإغلاق +15% · محافظ على 70%+ من القفزة آخر الأفتر
· فوليوم الأفتر ≥ 50 ألف سهم.
الدخول: A) آخر سعر في الأفتر · B) فواب الأفتر إذا نزل له البري.
النتيجة يوم D+1: +10% / +15% / +30% قبل −10% (نفس الشمعة = وقف)، وأقصى صعود في البري وأول ساعة، وإغلاق اليوم.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import sec_tools

NY = "America/New_York"


def outcome(b, entry, tg, stop=0.10):
    for _, r in b.iterrows():
        if r["Low"] <= entry * (1 - stop):
            return "fail"
        if r["High"] >= entry * (1 + tg):
            return "hit"
    return "none"


def find_signals(sym, h):
    out = []
    h = h.copy()
    h["sess"] = A.session_of(h.index).values
    h["day"] = h.index.date
    days = sorted(h["day"].unique())
    for i in range(len(days) - 1):
        D = days[i]
        x = h[h.day == D]
        rth, ah = x[x.sess == "rth"], x[x.sess == "ah"]
        if rth.empty or ah.empty:
            continue
        close = float(rth["Close"].iloc[-1])
        if not (0.5 <= close <= 5) or rth["Volume"].sum() < 100_000:
            continue
        ah_hi, ah_last, ah_vol = float(ah["High"].max()), float(ah["Close"].iloc[-1]), float(ah["Volume"].sum())
        jump = ah_hi / close - 1
        if jump < 0.15 or ah_vol < 50_000:
            continue
        hold = (ah_last - close) / (ah_hi - close)
        out.append({"t": sym, "D": D, "N": days[i + 1], "close": close, "jump": jump, "hold": hold,
                    "ah_vol": ah_vol, "ah_dollar": ah_vol * ah_last, "rth_vol": float(rth["Volume"].sum())})
    return out


def detail(sig):
    """شموع 5 دقائق: أفتر يوم D + بري وجلسة يوم N."""
    s = pd.Timestamp(sig["D"]).tz_localize(NY) + pd.Timedelta(hours=16)
    e = pd.Timestamp(sig["N"]).tz_localize(NY) + pd.Timedelta(hours=16)
    try:
        b = A.bars([sig["t"]], "5Min", s.tz_convert("UTC").to_pydatetime(), e.tz_convert("UTC").to_pydatetime()).get(sig["t"])
    except Exception as ex:
        return {**sig, "err": repr(ex)[:80]}
    if b is None or b.empty:
        return {**sig, "err": "no 5m"}
    b["sess"] = A.session_of(b.index).values
    ah = b[(b.index.date == sig["D"]) & (b.sess == "ah")]
    nxt = b[b.index.date == sig["N"]]
    pm, rth = nxt[nxt.sess == "pm"], nxt[nxt.sess == "rth"]
    if ah.empty or nxt.empty:
        return {**sig, "err": "empty"}
    ah_last = float(ah["Close"].iloc[-1])
    v = ah["Volume"].sum()
    ah_vwap = float((ah["Close"] * ah["Volume"]).sum() / v) if v else ah_last
    # آخر ساعة أفتر: رنج ضيق؟
    last_hr = ah[ah.index >= ah.index[-1] - pd.Timedelta(minutes=55)]
    rng = float(last_hr["High"].max() / last_hr["Low"].min() - 1) if len(last_hr) else np.nan
    seq = pd.concat([pm, rth])
    first_hr = rth[rth.index < rth.index[0] + pd.Timedelta(hours=1)] if not rth.empty else rth
    r = {**sig, "ah_last": ah_last, "ah_vwap": ah_vwap, "ah_last_hr_range": rng,
         "pm_vol": float(pm["Volume"].sum()),
         "pm_max": float(pm["High"].max()) / ah_last - 1 if not pm.empty else np.nan,
         "open_hr_max": float(first_hr["High"].max()) / ah_last - 1 if not first_hr.empty else np.nan,
         "day_max": float(seq["High"].max()) / ah_last - 1, "day_min": float(seq["Low"].min()) / ah_last - 1,
         "close_next": float(rth["Close"].iloc[-1]) / ah_last - 1 if not rth.empty else np.nan}
    for tg in (0.10, 0.15, 0.30):
        r[f"A{int(tg * 100)}"] = outcome(seq, ah_last, tg)
    dip = pm[pm["Low"] <= ah_vwap]
    if ah_vwap < ah_last * 0.995 and not dip.empty:
        after = seq.loc[dip.index[0]:]
        for tg in (0.10, 0.15, 0.30):
            r[f"B{int(tg * 100)}"] = outcome(after, ah_vwap, tg)
    return r


def main():
    if not A.available():
        print("⛔ مفاتيح Alpaca مو موجودة في Secrets"); return
    # فحص المفاتيح أول
    try:
        t = A.bars(["AAPL"], "1Day", datetime.now(timezone.utc) - timedelta(days=10))
        print("✅ مفاتيح Alpaca شغالة · AAPL شموع:", len(t.get("AAPL", [])), flush=True)
    except Exception as e:
        k = os.environ.get("ALPACA_KEY_ID", "")
        print(f"⛔ Alpaca رفض المفاتيح: {repr(e)[:160]} · طول المفتاح {len(k)} يبدأ بـ {k[:2]!r} · "
              f"طول السر {len(os.environ.get('ALPACA_SECRET_KEY', ''))}", flush=True)
        sys.exit(1)
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    print("tickers", len(tickers), flush=True)
    start = datetime.now(timezone.utc) - timedelta(days=365)
    sigs, t0 = [], time.time()
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i + 40]
        try:
            hb = A.bars(batch, "1Hour", start)
        except Exception as e:
            print("err", repr(e)[:150], flush=True); continue
        for s, h in hb.items():
            sigs += find_signals(s, h)
        print(f"{i + len(batch)}/{len(tickers)} · إشارات {len(sigs)} · {round(time.time() - t0)}ث", flush=True)
    print("قفزات أفتر (15%+ وفوليوم 50K+):", len(sigs), flush=True)
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(detail, sigs))
    df = pd.DataFrame([r for r in rows if "err" not in r])
    if df.empty:
        print("ما فيه إشارات بالتفصيل"); return
    print("بالتفصيل (5 دقائق):", len(df), "· أخطاء:", sum("err" in r for r in rows), flush=True)

    def summ(sub, name):
        if not len(sub):
            print(f"\n## {name}: لا يوجد"); return
        L = {"n": len(sub)}
        for k in ("A10", "A15", "A30", "B10", "B15", "B30"):
            if k in sub:
                v = sub[k].dropna()
                if len(v):
                    L[k] = f"{(v == 'hit').mean() * 100:.0f}% (n={len(v)})"
        L["A10_وقف"] = f"{(sub['A10'] == 'fail').mean() * 100:.0f}%"
        L["أقصى_بري_وسيط"] = f"{sub['pm_max'].median() * 100:+.0f}%"
        L["أقصى_أول_ساعة_وسيط"] = f"{sub['open_hr_max'].median() * 100:+.0f}%"
        L["إغلاق_التالي_وسيط"] = f"{sub['close_next'].median() * 100:+.0f}%"
        L["طار30%"] = f"{(sub['day_max'] >= 0.3).mean() * 100:.0f}%"
        print(f"\n## {name}\n   " + " · ".join(f"{k}={v}" for k, v in L.items()))

    print("\n==== النتائج (A = دخول آخر الأفتر · B = دخول عند فواب الأفتر في البري) ====")
    hold = df[df.hold >= 0.7]
    summ(hold, "قفز وحافظ (70%+)")
    summ(df[df.hold < 0.7], "قفز وما حافظ")
    summ(hold[hold.jump >= 0.30], "حافظ + قفزة 30%+")
    summ(hold[hold.ah_vol >= 500_000], "حافظ + فوليوم أفتر 500K+")
    summ(hold[(hold.jump >= 0.30) & (hold.ah_vol >= 500_000)], "حافظ + قفزة 30%+ + فوليوم 500K+")
    summ(hold[hold.ah_last_hr_range <= 0.06], "حافظ + آخر ساعة رنج ضيق (≤6%)")
    summ(hold[hold.pm_vol >= hold.ah_vol * 0.5], "حافظ + فوليوم البري قوي")
    summ(hold[hold.close < 1], "حافظ + سعر تحت 1$")
    summ(hold[hold.close >= 1], "حافظ + سعر 1-5$")

    # طرح بعد القفزة
    def off(r):
        try:
            cik = sec_tools._cik(r.t)
            js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            rec = js["filings"]["recent"]
            for f, fd in zip(rec["form"], rec["filingDate"]):
                if f.startswith("424B") and f != "424B3" and 0 <= (pd.Timestamp(fd) - pd.Timestamp(r.D)).days <= 3:
                    return True
            return False
        except Exception:
            return None
    with ThreadPoolExecutor(4) as ex:
        hold = hold.assign(offer=list(ex.map(off, list(hold.itertuples()))))
    summ(hold[hold.offer == True], "حافظ + جاه طرح خلال 3 أيام")  # noqa: E712
    summ(hold[hold.offer == False], "حافظ + بدون طرح")  # noqa: E712
    print("\nأفضل:", ", ".join(f"{r.t}@{r.D}:+{r.day_max * 100:.0f}%" for r in hold.sort_values("day_max", ascending=False).head(12).itertuples()))
    print("أسوأ:", ", ".join(f"{r.t}@{r.D}:{r.close_next * 100:+.0f}%(طرح={r.offer})" for r in hold.sort_values("close_next").head(10).itertuples()))


if __name__ == "__main__":
    main()
