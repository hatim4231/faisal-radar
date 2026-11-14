#!/usr/bin/env python3
"""🏹 كاشف «الصيد» — طريقة فيصل الإنسان: جمّع في القاعدة قبل الانطلاق، والسهم يطير خلال أيام.

مبني على «الحمض النووي» المثبّت باختبار 3 سنين مع المشطوبة (diag/dna_backtest.py):
  سهم بهالبصمة احتمال طيرانه ×2 خلال 20ج = 13.5% مقابل 4.1% للعشوائي (×3.3، ثابت كل سنة).
  ⚠ البيع على المتداول (الشراء الآلي يخسر؛ الأفضلية في الاختيار + تنفيذك ساعة القفزة).

البصمة (كلها معروفة بإغلاق اليوم — ما نشوف المستقبل):
  • طيّار سابق: طار ≥ +100% بيوم خلال آخر سنة (تكرار = أقوى)
  • منهار: ≥ 70% من قمة السنة (الأفضل 90%+)
  • قاعدة: قاع عمره 10+ جلسات، ما كسر قاع جديد آخر 3 جلسات، الإغلاق فوق أدنى 20ج
  • انكماش تذبذب: مدى آخر 5 جلسات < مدى آخر 20 (تهدئة = زنبرك)
  • بصمة تجميع: فوليوم الصعود ≥ الهبوط، أو OBV يطلع، أو قيعان أعلى
  • RSI مضغوط 20-50 (بدأ يلتف، مو ساقط ميت)
  • فلووت صغير (< 5M، الأفضل < 1M) = مكافأة

الحالات:
  🎯 جاهز للصيد = البصمة كاملة + «يتقوقع» قرب سقف القاعدة (أقرب للانطلاق)
  🟡 يتجمّع     = البصمة + بصمة تجميع واضحة (لسا بوسط القاعدة)
  👀 مبكر       = منهار + طيّار سابق بس القاعدة ما تأكدت
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_AGE = 10          # عمر القاعدة بالجلسات
JUMP_FR = 2.0         # يوم طيران سابق: أعلى ≥ إغلاق أمس × 2 (+100%)
FR_WIN = 252
CRASH_MIN = 0.70      # منهار ≥ 70% من قمة السنة

STATUS_AR = {"ready": "🎯 جاهز للصيد", "near": "🟡 يتجمّع", "watch": "👀 مبكر"}


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def _r(x, n=4):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), n)


def detect(d: pd.DataFrame, info: dict | None = None) -> dict | None:
    """تفاصيل إشارة الصيد لآخر يوم، أو None إذا ما فيه إعداد (ولا مبكر)."""
    info = info or {}
    if d is None or len(d) < 80:
        return None
    c, h, l, v = (d[k].to_numpy(float) for k in ("Close", "High", "Low", "Volume"))
    dates = d.index
    t = len(c) - 1
    price = float(c[t])
    r = rsi(c)

    # --- منهار من قمة السنة
    hi252 = float(h[max(0, t - 251):t + 1].max())
    crash = 1 - price / hi252 if hi252 > 0 else 0.0
    peak_i = max(0, t - 251) + int(np.argmax(h[max(0, t - 251):t + 1]))

    # --- طيّار سابق (أيام +100%)
    jump = np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)]
    js = [i for i in range(max(1, t - FR_WIN), t + 1) if jump[i] >= JUMP_FR]
    nfr = len(js)

    # لازم منهار + طيّار سابق (جوهر البصمة)
    if crash < CRASH_MIN or nfr < 1:
        return None

    # --- القاعدة: آخر قاع 120ج وعمره
    w0 = max(0, t - 119)
    b = t - int(np.argmin(l[w0:t + 1][::-1]))
    age = t - b
    support = float(l[b:t + 1].min())
    ceiling = float(h[b:t + 1].max())
    # ما كسر قاع جديد آخر 3 جلسات (ثبات)
    held = bool(l[max(b, t - 2):t + 1].min() >= support * 0.995)
    lo20 = float(l[max(0, t - 19):t + 1].min())
    above_lo20 = bool(price >= lo20 * 1.0)

    # --- انكماش التذبذب
    tr = np.r_[h[0] - l[0], np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]
    atr5 = float(tr[max(0, t - 4):t + 1].mean())
    atr20 = float(tr[max(0, t - 19):t + 1].mean())
    contract = bool(atr20 > 0 and atr5 <= atr20 * 0.95)

    # --- بصمة تجميع
    up = c > np.r_[c[0], c[:-1]]
    dn = c < np.r_[c[0], c[:-1]]
    upv = v[max(0, t - 9):t + 1][up[max(0, t - 9):t + 1]].sum()
    dnv = v[max(0, t - 9):t + 1][dn[max(0, t - 9):t + 1]].sum()
    upv_ratio = upv / max(dnv, 1)
    obv = np.cumsum(np.where(up, v, np.where(dn, -v, 0.0)))
    obv_up = bool(t >= 20 and obv[t] > obv[t - 20])
    if t - 29 > b:
        m1, m2, m3 = l[t - 29:t - 19].min(), l[t - 19:t - 9].min(), l[t - 9:t + 1].min()
        higher_lows = bool(m2 >= m1 * 0.99 and m3 >= m2 * 0.99)
    else:
        m1 = m2 = m3 = None
        higher_lows = False
    accum = bool(upv_ratio >= 1.0 or obv_up or higher_lows)

    # --- RSI مضغوط
    rsi_now = float(r[t])
    rsi_ok = bool(20 <= rsi_now <= 50)

    # --- التقوقع قرب سقف القاعدة (أقرب للانطلاق)
    pos = (price - support) / max(ceiling - support, 1e-9)   # 0 قاع .. 1 سقف
    coiling = bool(pos >= 0.5 and contract)

    # --- الفلووت
    flt = info.get("floatShares")
    if flt is not None and flt < 50_000:
        flt = None
    tiny_float = bool(flt and flt < 5e6)

    base_ok = bool(age >= MIN_AGE and held and above_lo20)
    if not base_ok:
        status = "watch"
    elif coiling and accum and rsi_ok:
        status = "ready"
    elif accum and rsi_ok:
        status = "near"
    else:
        status = "watch"

    # --- النقاط (للترتيب): كل ما قرب الانطلاق + البصمة أقوى
    score = 0.0
    score += min(crash, 0.99) * 25                      # الانهيار
    score += min(nfr, 4) * 6                             # تكرار الطيران
    score += 12 if accum else 0
    score += 10 if contract else 0
    score += 10 if rsi_ok else max(0, 10 - abs(rsi_now - 35) * 0.3)
    score += 15 * max(0, min(1, pos))                   # قرب السقف
    score += 8 if tiny_float else (4 if (flt and flt < 2e7) else 0)
    score += 6 if higher_lows else 0
    score = round(min(100.0, score), 1)

    # --- خطة التجميع
    mid = support + (ceiling - support) * 0.4
    tranches = [support * 1.01, (support + mid) / 2, min(price, mid)]   # 3 دفعات بأسفل القاعدة
    stop = support * 0.90
    stop_pct = (1 - stop / price) * 100 if price else None
    why = [
        f"انهار {crash*100:.0f}% من قمة السنة {hi252:.3g} ({str(dates[peak_i])[:10]})",
        f"سبق طار {nfr} مرة خلال سنة" + (": " + "، ".join(f"{str(dates[i])[:10]} +{(jump[i]-1)*100:.0f}%" for i in js[-3:]) if js else ""),
        f"قاعدة {support:.3g}–{ceiling:.3g} عمرها {age} جلسة" + (" (ثابتة، ما كسر قاع جديد)" if held else " (لسا تتكوّن)"),
        f"التذبذب " + ("انكمش ✓ (زنبرك)" if contract else "ما انكمش بعد"),
        f"بصمة تجميع: فوليوم صعود/هبوط {upv_ratio:.1f}x" + (" · OBV يطلع" if obv_up else "") + (" · قيعان أعلى" if higher_lows else ""),
        f"RSI {rsi_now:.0f} " + ("(مضغوط، مناسب)" if rsi_ok else "(خارج 20-50)"),
        f"موقع السعر بالقاعدة: {pos*100:.0f}% " + ("(متقوقع قرب السقف — قرب الانطلاق)" if coiling else "(لسا بالأسفل)"),
    ]
    if flt:
        why.append(f"فلووت {flt/1e6:.2f}M" + (" 🔥 ذري (سريع الانفجار)" if tiny_float else ""))

    return {
        "status": status, "status_ar": STATUS_AR[status], "score": score,
        "crash_pct": _r(crash * 100, 1), "peak": _r(hi252), "peak_date": str(dates[peak_i])[:10],
        "n_jumps": nfr, "jumps": [{"date": str(dates[i])[:10], "pct": _r((jump[i] - 1) * 100, 0)} for i in js[-5:]],
        "support": _r(support), "ceiling": _r(ceiling), "age": int(age), "pos_in_base_pct": _r(pos * 100, 0),
        "held": held, "contract": contract, "accum": accum, "upv_ratio": _r(upv_ratio, 2),
        "obv_up": obv_up, "higher_lows": higher_lows, "rsi": _r(rsi_now, 1), "coiling": coiling,
        "float": flt, "tiny_float": tiny_float,
        "weekly_lows": [_r(x) for x in (m1, m2, m3)] if m1 is not None else None,
        "plan": {
            "accumulate_zone": [_r(support), _r(mid)],
            "tranches": [_r(x) for x in tranches],
            "launch_trigger": _r(ceiling),
            "stop": _r(stop), "stop_note": "إغلاق تحت قاع القاعدة −10%",
            "stop_pct": _r(stop_pct, 1),
            "size_pct": _r(min(5.0, 1.0 / max((stop_pct or 20) / 100, 0.05)), 1),
            "hold": "3–7 أيام · البيع على المتداول (نص عند أول قفزة +15/+22% ثم trail)",
        },
        "why": why,
    }
