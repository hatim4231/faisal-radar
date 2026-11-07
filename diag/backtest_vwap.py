"""
اختبار تاريخي: الارتداد من VWAP أول الجلسة (سنة، بيانات Alpaca، شموع 5 دقائق).

الأسهم: 0.50-5$ · فجوة افتتاح 10%+ عن إغلاق أمس · فوليوم اليوم 500K+.
الإعداد: بعد الافتتاح السهم يطلع فوق VWAP بـ 3%+ ، ثم يرجع يلمس VWAP بين 9:35 و 10:30.
الدخول:
  T) لمسة: Limit عند VWAP (آخر قيمة قبل الشمعة).
  R) استرجاع: بعد اللمسة، أول شمعة 5 دقائق تسكر فوق VWAP (خلال 30 دقيقة) ← الدخول بإغلاقها.
المقارنة (نفس أيام الفجوة):
  X) عشوائي: إغلاق شمعة عشوائية بين 9:35 و 10:30.
النتيجة: الهدف قبل الوقف لين إغلاق اليوم (نفس الشمعة = وقف). ولو ما صار شي = الربح/الخسارة عند الإغلاق.
"""
import random
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

NY = "America/New_York"
COMBOS = [(5, 5), (10, 5), (10, 10)]
random.seed(7)


def run_out(b, entry, tg, st, skip_first=False):
    """b: الشموع من شمعة الدخول. skip_first: شمعة الدخول نفحص فيها الوقف بس (اللمسة نفسها)."""
    for k, (_, r) in enumerate(b.iterrows()):
        if r["Low"] <= entry * (1 - st / 100):
            return "fail", -st
        if k == 0 and skip_first:
            continue
        if r["High"] >= entry * (1 + tg / 100):
            return "hit", tg
    return "none", (float(b["Close"].iloc[-1]) / entry - 1) * 100


def gap_days(tickers):
    start = datetime.now(timezone.utc) - timedelta(days=372)
    out = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            d = A.bars(batch, "1Day", start)
        except Exception as e:
            print("err daily", repr(e)[:100], flush=True)
            continue
        for s, x in d.items():
            x = x.copy()
            x["pc"] = x["Close"].shift(1)
            g = x[(x["Open"] / x["pc"] - 1 >= 0.10) & (x["Open"].between(0.5, 5)) & (x["Volume"] >= 500_000)]
            for dt, r in g.iterrows():
                out.append({"t": s, "day": dt.tz_convert(NY).date() if dt.tzinfo else dt.date(),
                            "gap": float(r["Open"] / r["pc"] - 1), "open": float(r["Open"]),
                            "pc": float(r["pc"]), "vol": float(r["Volume"])})
    return out


def detail(g):
    s = pd.Timestamp(g["day"]).tz_localize(NY)
    try:
        b = A.bars([g["t"]], "5Min", (s + pd.Timedelta(hours=4)).tz_convert("UTC").to_pydatetime(),
                   (s + pd.Timedelta(hours=16)).tz_convert("UTC").to_pydatetime()).get(g["t"])
    except Exception as e:
        return {**g, "err": repr(e)[:60]}
    if b is None or b.empty:
        return {**g, "err": "empty"}
    hm = b.index.hour * 60 + b.index.minute
    pm, rth = b[hm < 570], b[(hm >= 570) & (hm < 960)].copy()
    if len(rth) < 30:
        return {**g, "err": "short"}
    px = rth["VWAP"] if "VWAP" in rth else (rth["High"] + rth["Low"] + rth["Close"]) / 3
    rth["vwap"] = (px * rth["Volume"]).cumsum() / rth["Volume"].cumsum().replace(0, np.nan)
    rth["vwap_prev"] = rth["vwap"].shift(1)
    r = {**g, "pm_vol": float(pm["Volume"].sum()), "pm_vol_ratio": float(pm["Volume"].sum() / max(g["vol"], 1))}
    tmin = rth.index.hour * 60 + rth.index.minute
    # عشوائي
    win = rth[(tmin >= 575) & (tmin <= 630)]
    k = random.randrange(len(win))
    e_x = float(win["Close"].iloc[k])
    after_x = rth.loc[win.index[k]:].iloc[1:]
    for tg, st in COMBOS:
        r[f"X{tg}_{st}"] = run_out(after_x, e_x, tg, st) if len(after_x) else ("none", 0)
    # الإعداد: طلع فوق VWAP 3%+ ثم لمسه
    ran = False
    touch_i = None
    for i in range(1, len(rth)):
        row = rth.iloc[i]
        t = tmin[i]
        vp = rth["vwap_prev"].iloc[i]
        if t > 630:
            break
        if rth["High"].iloc[:i].max() >= vp * 1.03:
            ran = True
        if ran and t >= 575 and row["Low"] <= vp:
            touch_i = i
            break
    r["setup"] = touch_i is not None
    if touch_i is None:
        return r
    vp = float(rth["vwap_prev"].iloc[touch_i])
    run_hi = float(rth["High"].iloc[:touch_i].max())
    r.update({"touch_min": int(tmin[touch_i] - 570), "run_pct": (run_hi / vp - 1) * 100,
              "open_vs_vwap": float(rth["Open"].iloc[0]) / vp - 1})
    after_t = rth.iloc[touch_i:]
    for tg, st in COMBOS:
        r[f"T{tg}_{st}"] = run_out(after_t, vp, tg, st, skip_first=True)
    # استرجاع: أول إغلاق فوق VWAP خلال 30 دقيقة بعد اللمسة
    for j in range(touch_i, min(touch_i + 7, len(rth))):
        if rth["Close"].iloc[j] > rth["vwap"].iloc[j]:
            e_r = float(rth["Close"].iloc[j])
            after_r = rth.iloc[j + 1:]
            if len(after_r):
                r["reclaim_min"] = int(tmin[j] - 570)
                for tg, st in COMBOS:
                    r[f"R{tg}_{st}"] = run_out(after_r, e_r, tg, st)
            break
    return r


def summ(df, name):
    if not len(df):
        print(f"\n## {name}: لا يوجد"); return
    print(f"\n## {name}  (n={len(df)})")
    for p in ("T", "R", "X"):
        parts = []
        for tg, st in COMBOS:
            c = f"{p}{tg}_{st}"
            if c not in df:
                continue
            v = df[c].dropna()
            if not len(v):
                continue
            res = v.map(lambda x: x[0])
            pnl = v.map(lambda x: x[1])
            parts.append(f"+{tg}/−{st}: نجح {(res == 'hit').mean() * 100:.0f}% وقف {(res == 'fail').mean() * 100:.0f}% "
                         f"متوسط {pnl.mean():+.2f}% (n={len(v)})")
        if parts:
            print(f"   {dict(T='لمسة VWAP', R='استرجاع VWAP', X='عشوائي')[p]}: " + " · ".join(parts))


def main():
    try:
        A.bars(["AAPL"], "1Day", datetime.now(timezone.utc) - timedelta(days=10))
        print("✅ مفاتيح Alpaca شغالة", flush=True)
    except Exception as e:
        print("⛔ Alpaca:", repr(e)[:150]); sys.exit(1)
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    print("tickers", len(tickers), flush=True)
    gaps = gap_days(tickers)
    print("أيام فجوة 10%+:", len(gaps), flush=True)
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(detail, gaps)):
            rows.append(r)
            if i % 300 == 0:
                print(f"  {i}/{len(gaps)} · {round(time.time() - t0)}ث", flush=True)
    df = pd.DataFrame([r for r in rows if "err" not in r])
    print("بالتفصيل:", len(df), "· أخطاء:", sum("err" in r for r in rows), flush=True)
    print("==== النتائج: الهدف/الوقف لين إغلاق نفس اليوم · المتوسط = ربح/خسارة الصفقة بالمتوسط ====")
    summ(df, "كل أيام الفجوة (العشوائي = المقارنة)")
    S = df[df["setup"]]
    print(f"\nالإعداد (طلع 3%+ فوق VWAP ثم لمسه قبل 10:30): {len(S)} من {len(df)}")
    summ(S, "الإعداد كامل")
    summ(S[S.gap < 0.2], "فجوة 10-20%")
    summ(S[(S.gap >= 0.2) & (S.gap < 0.4)], "فجوة 20-40%")
    summ(S[S.gap >= 0.4], "فجوة 40%+")
    summ(S[S.open < 1], "سعر تحت 1$")
    summ(S[S.open >= 1], "سعر 1-5$")
    summ(S[S.touch_min <= 20], "اللمسة أول 20 دقيقة")
    summ(S[S.touch_min > 20], "اللمسة بعد 20 دقيقة")
    summ(S[S.run_pct >= 8], "طلع 8%+ فوق VWAP قبل اللمسة")
    summ(S[S.run_pct < 8], "طلع 3-8% فوق VWAP")
    summ(S[S.open_vs_vwap > 0], "فتح فوق VWAP")
    q = S["pm_vol_ratio"].median() if len(S) else 0
    summ(S[S.pm_vol_ratio >= q], "فوليوم البري قوي (فوق الوسيط)")
    summ(S[S.vol >= 5_000_000], "فوليوم اليوم 5M+")


if __name__ == "__main__":
    main()
