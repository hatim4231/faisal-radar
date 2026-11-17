"""شرط «تاريخ ارتفاع سابق» (طلب أبوفرح 2026-11-17): أي سهم يُضاف في رادارات فيصل (المطوّر/الأصلي/المدمج/الرهيب)
وعبدالله لازم يكون عنده ارتفاع سابق ≥ 50% خلال آخر 6 شهور — سلوك يثبت إنه يتحرك (قاعدة فيصل: اعرف سلوك السهم).

الارتفاع = من قاع إلى قمة لاحقة ضمن نافذة ~30 جلسة، وبدايته خلال آخر 6 شهور (~126 جلسة).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def has_prior_rise(d: pd.DataFrame, min_pct: float = 50, months: int = 6, window: int = 30) -> tuple[bool, dict]:
    """يرجع (هل عنده ارتفاع ≥min_pct% خلال آخر months شهور، تفاصيل أكبر ارتفاع)."""
    if d is None or len(d) < 20:
        return False, {}
    lookback = int(months * 21)                       # ~21 جلسة تداول بالشهر
    h = d["High"].to_numpy(float); l = d["Low"].to_numpy(float)
    dates = d.index
    n = len(h)
    start = max(0, n - lookback)
    best, best_i, best_j = 0.0, None, None
    for i in range(start, n):
        lo = l[i]
        if lo <= 0:
            continue
        seg = h[i:min(n, i + window)]
        if not len(seg):
            continue
        jm = int(np.argmax(seg))
        g = seg[jm] / lo - 1
        if g > best:
            best, best_i, best_j = g, i, i + jm
    ok = best >= min_pct / 100
    if best_i is None:
        return False, {}
    info = {"max_rise_pct": round(float(best * 100), 0),
            "from": round(float(l[best_i]), 4), "to": round(float(h[best_j]), 4),
            "from_date": str(dates[best_i])[:10], "date": str(dates[best_j])[:10]}
    return ok, info


def apply_gate(a: dict, d: pd.DataFrame, cfg: dict) -> bool:
    """يضيف a['prior_rise'] ويرجع True إذا نجح الشرط (أو الشرط مطفأ). a = قاموس تحليل أو صف."""
    BH = cfg.get("behavior", {}) or {}
    if not BH.get("require_prior_rise", True):
        a["prior_rise"] = None
        return True
    ok, info = has_prior_rise(d, BH.get("prior_rise_pct", 50), BH.get("prior_rise_months", 6))
    a["prior_rise"] = info or None
    return ok
