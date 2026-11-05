"""بيانات وهمية لتجربة التقرير بدون إنترنت. الرموز DEMO* ليست أسهم حقيقية."""
from __future__ import annotations

import numpy as np
import pandas as pd

import faisal_rules as fr


def _series(segments, seed, start_price, vol_base=400_000, noise=0.025):
    """segments: [(عدد الأيام, التغير الكلي %), ...] -> OHLCV"""
    rng = np.random.default_rng(seed)
    closes = []
    p = start_price
    for days, total in segments:
        step = (1 + total / 100) ** (1 / days)
        for _ in range(days):
            p = p * step * (1 + rng.normal(0, noise))
            closes.append(max(0.05, p))
    closes = np.array(closes)
    n = len(closes)
    opens = np.r_[closes[0], closes[:-1]] * (1 + rng.normal(0, noise / 3, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, noise / 1.5, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, noise / 1.5, n)))
    vols = vol_base * np.exp(rng.normal(0, 0.35, n))
    idx = pd.bdate_range(end=pd.Timestamp("2026-11-04"), periods=n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": vols.astype(int), "Stock Splits": 0.0}, index=idx)


def _hammer_end(df, support):
    """آخر شمعتين: سحب سيولة تحت الدعم ثم استرداد + Hammer."""
    df = df.copy()
    i = df.index
    df.loc[i[-2], ["Open", "High", "Low", "Close"]] = [support * 1.02, support * 1.03, support * 0.955, support * 1.01]
    df.loc[i[-1], ["Open", "High", "Low", "Close"]] = [support * 1.01, support * 1.07, support * 0.99, support * 1.06]
    df.loc[i[-1], "Volume"] = int(df["Volume"].tail(20).mean() * 1.8)
    return df


def scenarios():
    out = []
    # 1) ارتكاز شبه جاهز: سبق انفجر، هبط طويل، قاعدة عند ~1.50، سحب سيولة + Hammer
    d = _series([(120, -10), (12, 160), (25, -45), (230, -50), (25, -4)], 1, 2.4, noise=0.02)
    sup = float(d["Low"].tail(25).min())
    d = _hammer_end(d, sup)
    out.append(("DEMOA", "Demo Irtikaz Ready", d, {"marketCap": 9e6, "floatShares": 3.2e6,
                "sharesOutstanding": 4e6, "sharesShort": 15000, "shortPercentOfFloat": 0.005}))
    # 2) قريب: يهبط ولسه فوق الدعم بـ 15%
    d = _series([(200, -40), (150, -60), (30, -8), (12, 18)], 2, 6.0)
    out.append(("DEMOB", "Demo Near Support", d, {"marketCap": 22e6, "floatShares": 11e6,
                "sharesOutstanding": 14e6, "sharesShort": 300000, "shortPercentOfFloat": 0.03}))
    # 3) لا تطارد: صعد 70% من القاع
    d = _series([(300, -70), (40, -10), (8, 75)], 3, 5.0)
    out.append(("DEMOC", "Demo Chase Warning", d, {"marketCap": 40e6, "floatShares": 18e6,
                "sharesOutstanding": 25e6, "sharesShort": 2.5e6, "shortPercentOfFloat": 0.14}))
    # 4) زخم اليوم: فجوة وفوليوم
    d = _series([(200, -50), (150, -20), (20, 5)], 4, 3.0)
    i = d.index[-1]
    pc = float(d["Close"].iloc[-2])
    d.loc[i, ["Open", "High", "Low", "Close"]] = [pc * 1.12, pc * 1.48, pc * 1.08, pc * 1.36]
    d.loc[i, "Volume"] = int(d["Volume"].tail(30).mean() * 9)
    out.append(("DEMOD", "Demo Momentum Today", d, {"marketCap": 15e6, "floatShares": 6e6,
                "sharesOutstanding": 8e6, "sharesShort": 40000, "shortPercentOfFloat": 0.007}))
    # 5) قروبات وهم
    d = _series([(250, -60), (120, -20)], 5, 4.0)
    for k in (-25, -17, -9, -4):
        i = d.index[k]
        c = float(d["Close"].iloc[k])
        d.loc[i, "High"] = c * 1.35
        d.loc[i, "Volume"] = int(d["Volume"].mean() * 6)
    out.append(("DEMOE", "Demo Pump Groups", d, {"marketCap": 30e6, "floatShares": 20e6,
                "sharesOutstanding": 30e6, "sharesShort": 1e6, "shortPercentOfFloat": 0.05}))
    # 6) تقسيم عكسي حديث + لا يزال يهبط
    d = _series([(300, -80), (40, -35)], 6, 3.0)
    d.loc[d.index[-14], "Stock Splits"] = 0.1
    out.append(("DEMOF", "Demo Reverse Split", d, {"marketCap": 5e6, "floatShares": 1.4e6,
                "sharesOutstanding": 1.6e6, "sharesShort": 120000, "shortPercentOfFloat": 0.09}))
    return out


def build(cfg) -> dict:
    rows = []
    for sym, name, d, info in scenarios():
        a = fr.analyze(d, info, cfg)
        a.update({"symbol": sym, "name": name + " (تجريبي)", "sector": "Demo", "country": "-"})
        a["news"] = [{"title": "خبر تجريبي — هذا تقرير ببيانات وهمية", "url": None,
                      "source": "Demo", "time": "2026-11-04T13:00:00Z"}]
        rows.append(a)
    rows.sort(key=lambda r: -r["irtikaz"]["score"])
    return {"generated_at": "تجريبي — بيانات وهمية", "source": "بيانات وهمية للتجربة فقط",
            "scanned": len(rows), "passed_filter": len(rows), "seconds": 0, "config": cfg,
            "stocks": rows, "demo": True}
