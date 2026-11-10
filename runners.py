#!/usr/bin/env python3
"""
🚀 كاشف «الطيارات» — من أمثلة المستخدم (OMH و QMMM، تغريدة @xzdixz)، بنفس شروط الاختبار التاريخي
(diag/backtest_runners2.py على فرع diag-flye) بالضبط:

  سهم انهار (أعلى سعر خلال سنة قبل القاع ≥ 3 أضعاف القاع) ← قاع (أدنى سعر 120 جلسة) عمره 15+ جلسة
  ← السعر 1.05-2 من القاع (ما طار للحين) ← قاع أعلى (أدنى سعر آخر 10 جلسات ≥ القاع +8% وبعده بـ 10+ جلسات)
  ← RSI عند القاع الأخير أعلى من RSI القاع الكبير بـ 5+ (دايفرجنس إيجابي)
  ← دعوم متصاعدة (أدنى سعر بكل 10 جلسات آخر 30 جلسة ما ينزل) ← سبق طار (يوم +50% خلال سنة)
  ← اختبر المقاومة مرتين، أو طار مرتين+ خلال سنة.

الحالات (الاختبار: يدبّل خلال 60 جلسة مقابل عشوائي بنفس التذبذب والسعر):
  🚀 جاهز  = V5 (V4 + اختبر المقاومة مرتين: 21.5% مقابل 15.3%) أو V6 (V4 + طار مرتين+: 21.7% مقابل 17.5%)
  🟡 قريب  = V4 (دعوم متصاعدة + قاع أعلى + RSI + سبق طار: 20.0% مقابل 16.7%)
  👀 مراقبة = ناقصه الدعوم المتصاعدة أو «سبق طار»
الخطة: الدخول قرب آخر قاع أعلى (Limit) · الوقف إغلاق تحت القاع الكبير −10% · الهدف +100% (والثاني +200%) · أقصى 60 جلسة.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CRASH_X = 3.0       # أعلى سعر خلال سنة قبل القاع ÷ القاع
MIN_AGE = 15        # جلسات من القاع
LIFT = (1.05, 2.0)  # السعر ÷ القاع
HL_MIN = 1.08       # آخر قاع ≥ القاع الكبير × 1.08
RSI_GAIN = 5        # RSI آخر قاع − RSI القاع الكبير
JUMP = 1.5          # يوم طيران: أعلى سعر ≥ إغلاق أمس × 1.5
HORIZON = 60

STATUS_AR = {"ready": "🚀 جاهز", "near": "🟡 قريب", "watch": "👀 مراقبة"}
LEVEL_AR = {"V5": "اكتملت + اختبر المقاومة مرتين", "V6": "اكتملت + طار مرتين خلال سنة",
            "V4": "اكتملت الأساسية", "V2": "ناقصه «سبق طار»", "V3": "ناقصه الدعوم المتصاعدة"}


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def detect(d: pd.DataFrame) -> dict | None:
    """يرجع تفاصيل الإشارة لآخر يوم في d، أو None إذا ما فيه إشارة (ولا حتى مراقبة)."""
    if d is None or len(d) < 60:
        return None
    c, h, l = (d[k].to_numpy(float) for k in ("Close", "High", "Low"))
    dates = d.index
    t = len(c) - 1
    r = rsi(c)
    w0 = max(0, t - 119)
    b = t - int(np.argmin(l[w0:t + 1][::-1]))          # آخر مرة لمس فيها القاع
    age = t - b
    pre0 = max(0, b - 250)
    if age < MIN_AGE or b - pre0 < 20:
        return None
    B = l[b]
    peak_i = pre0 + int(np.argmax(h[pre0:b]))
    crash = h[peak_i] / B
    lift = c[t] / B
    if crash < CRASH_X or not (LIFT[0] <= lift <= LIFT[1]):
        return None
    j = t - 9 + int(np.argmin(l[t - 9:t + 1]))
    R = l[j]
    hl = bool(R >= HL_MIN * B and j - b >= 10)
    div = bool(r[j] >= r[b] + RSI_GAIN)
    if not (hl and div):
        return None
    if t - 29 > b:
        m1, m2, m3 = l[t - 29:t - 19].min(), l[t - 19:t - 9].min(), l[t - 9:t + 1].min()
        rising = bool(m2 >= m1 * 0.99 and m3 >= m2 * 0.99 and m3 >= m1 * 1.03)
    else:
        m1 = m2 = m3 = None
        rising = False
    jump = np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)]
    js = [i for i in range(max(1, t - 250), t + 1) if jump[i] >= JUMP]
    nfr = len(js)
    top = float(h[b:t + 1].max())
    near = np.where(h[b:t + 1] >= 0.93 * top)[0]
    touches = 0 if len(near) == 0 else 1 + int((np.diff(near) >= 5).sum())
    v2 = rising
    v3 = nfr >= 1
    v4 = v2 and v3
    v5 = v4 and touches >= 2
    v6 = v2 and nfr >= 2
    if v5 or v6:
        status, level = "ready", ("V5" if v5 else "V6")
    elif v4:
        status, level = "near", "V4"
    elif v2 or v3:
        status, level = "watch", ("V2" if v2 else "V3")
    else:
        return None
    entry = float(c[t])
    stop = float(B * 0.9)
    stop_pct = 1 - stop / entry
    why = [f"انهار {(1 - 1 / crash) * 100:.0f}% (من {h[peak_i]:.3g} يوم {str(dates[peak_i])[:10]} إلى القاع {B:.3g})",
           f"قاع {B:.3g} يوم {str(dates[b])[:10]} وجالس فوقه {age} جلسة بدون قاع جديد",
           f"قاع أعلى {R:.3g} يوم {str(dates[j])[:10]} (+{(R / B - 1) * 100:.0f}% فوق القاع)",
           f"RSI عند القاع {r[b]:.0f} ← عند القاع الأخير {r[j]:.0f} (دايفرجنس إيجابي)"]
    if m1 is not None:
        why.append(f"قيعان كل 10 جلسات: {m1:.3g} ← {m2:.3g} ← {m3:.3g}" + (" (متصاعدة ✓)" if rising else " (مو متصاعدة)"))
    why.append(f"سبق طار {nfr} مرة خلال سنة" + (": " + "، ".join(f"{str(dates[i])[:10]} +{(jump[i] - 1) * 100:.0f}%" for i in js[-3:]) if js else ""))
    why.append(f"اختبر المقاومة {top:.3g} {touches} مرة")
    blocks = [(str(dates[i])[:10], _r(float(l[i:min(i + 5, t + 1)].min())),
               _r(float(r[i + int(np.argmin(l[i:min(i + 5, t + 1)]))]), 0)) for i in range(b, t + 1, 5)]
    return {
        "status": status, "level": level, "level_ar": LEVEL_AR[level],
        "conds": {"crash": True, "base": True, "hl_rsi": True, "rising": rising, "touches2": touches >= 2,
                  "fr": nfr >= 1, "serial": nfr >= 2, "not_flown": True},
        "crash_pct": _r((1 - 1 / crash) * 100, 1), "peak": _r(h[peak_i]), "peak_date": str(dates[peak_i])[:10],
        "bottom": _r(B), "bottom_date": str(dates[b])[:10], "age": int(age), "lift_pct": _r((lift - 1) * 100, 1),
        "last_low": _r(R), "last_low_date": str(dates[j])[:10], "hl_pct": _r((R / B - 1) * 100, 1),
        "rsi_bottom": _r(r[b], 1), "rsi_last_low": _r(r[j], 1), "rsi": _r(r[t], 1),
        "blocks": [_r(x) for x in (m1, m2, m3)] if m1 is not None else None, "rising": rising,
        "jumps": [{"date": str(dates[i])[:10], "pct": _r((jump[i] - 1) * 100, 0)} for i in js],
        "n_jumps": nfr, "top": _r(top), "touches": touches, "weekly": blocks,
        "plan": {"entry": _r(entry), "zone": [_r(R), _r(min(entry, R * 1.08))], "stop": _r(stop), "stop_pct": _r(stop_pct * 100, 1),
                 "t1": _r(entry * 2), "t2": _r(entry * 3), "horizon": HORIZON,
                 "size_pct": _r(min(10.0, 1.0 / max(stop_pct, 0.01)), 1)},
        "why": why,
    }
