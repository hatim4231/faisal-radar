"""
⚓ كاشف «الارتكاز» (منشور نايف @Naifo9x) — نفس تعريفات الدراسة diag/irtikaz_naif.py بالضبط (فرع diag-flye).
الشروط القابلة للحساب من الشارت:
  rsi    RSI14 < 35
  stoch  تشبّع بيعي: Stochastic %K(14) < 20
  annual قاع سنوي (أدنى سعر 252 جلسة) انعمل خلال آخر 3-20 جلسة (ثبت 3 جلسات بدون قاع أدنى) والسعر ضمن 15% فوقه
  rs     تقسيم عكسي خلال سنة
  sweep  سحب سيولة خلال 10 جلسات (نزل تحت قاع 20 جلسة السابق وأغلق فوقه)
  ma     إغلاق ≥ متوسط 20 أو 30 أو 50 (معلومة — يناقض RSI<35، يجتمعون 0.1% من الأيام)
والشرطين الحيّين بالرادار (ما لهم تاريخ): القيمة السوقية 2-50 مليون · الشورت المتاح أقل من 20 ألف.
نتيجة الدراسة (2026-11-26): الأساسي (annual+rsi+stoch) +50%/20ج +2.5 ±1.7 مقابل المطابق لكن نزيف −30% +3.8 والعائد صفر؛
مع التقسيم العكسي طيران +6.2 ونزيف +8.1 والعائد سالب. ← قائمة مراقبة بتذاكر صغيرة، مو خطة ربح.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from faisal_rules import rsi

MCAP = (2e6, 50e6)
MAX_SHORT = 20_000


def detect(d: pd.DataFrame, splits: list[tuple] | None = None) -> dict | None:
    if d is None or len(d) < 260:
        return None
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    h, l, c = (d[k].to_numpy(float) for k in ("High", "Low", "Close"))
    C = pd.Series(c)
    r14 = float(rsi(C, 14).iloc[-1])
    ll, hh = l[-14:].min(), h[-14:].max()
    st = float((c[-1] - ll) / max(hh - ll, 1e-9) * 100)
    w = l[-252:]
    k = int(np.argmin(w))
    lo, since = float(w[k]), len(w) - 1 - k
    px = float(c[-1])
    annual = bool(3 <= since <= 20 and px <= lo * 1.15)
    low20prev = pd.Series(l).shift(1).rolling(20).min().to_numpy()
    sw = (l < low20prev) & (c > low20prev)
    sweep = bool(sw[-10:].any())
    mas = {n: float(C.rolling(n).mean().iloc[-1]) for n in (20, 30, 50)}
    ma_hit = [n for n, v in mas.items() if px >= v]
    rs = None
    if splits:
        last = pd.Timestamp(d.index[-1])
        last = last.tz_localize(None) if last.tz else last
        for dt, ratio in sorted(splits, key=lambda x: x[0], reverse=True):
            dt = pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tz else pd.Timestamp(dt)
            if ratio < 1 and 0 <= (last - dt).days <= 365:
                rs = {"date": str(dt.date()), "ratio": f"1:{round(1 / ratio)}", "days_ago": (last - dt).days}
                break
    core = bool(annual and r14 < 35 and st < 20)
    return {"price": round(px, 4), "rsi": round(r14, 1), "stoch": round(st, 1), "annual_low": round(lo, 4),
            "low_days_ago": int(since), "above_low_pct": round((px / lo - 1) * 100, 1), "sweep": sweep, "ma_above": ma_hit,
            "mas": {k: round(v, 4) for k, v in mas.items()}, "rs": rs,
            "conds": {"rsi": bool(r14 < 35), "stoch": bool(st < 20), "annual": annual, "rs": rs is not None, "sweep": sweep, "ma": bool(ma_hit)},
            "core": core, "stop": round(lo * 0.9, 4), "stop_pct": round((1 - lo * 0.9 / px) * 100, 1)}
