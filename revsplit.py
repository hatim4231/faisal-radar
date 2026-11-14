#!/usr/bin/env python3
"""🔄 كاشف «ما بعد التقسيم العكسي» — نمط أبوفرح (رادار مستقل).

النمط: سهم يسوي تقسيم عكسي ← ينزل لـ ≤60-65% من سعره بعد التقسيم ← يبني قاعدة ويجمّع أيام **بدون ما يكسر قاعه**
← ينفجر مضاربياً (قفزة داخل اليوم) ويصرّف نفس اليوم. المتداول يبيع بالقفزة، ويشتري **وقت التجميع**.

الميزة (دراسة 3 سنين، diag/revsplit_study.py + revsplit_accum.py): بعد التقسيم السهم فوق 1$ فمؤقتاً بعيد عن الشطب،
والارتفاع ≥70% من القاع يصير في ~44% (×3.4 عن العشوائي). الخطر: 35% فيهم هبوط طرح ≥30% بيوم — نحذّر منه.

الحالات:
  🎯 جاهز   = نزل ≤65% من سعر ما بعد التقسيم + قاعدة 10ج+ بدون كسر قاع + انكماش تذبذب + بصمة تجميع
  🟡 يتجمّع = نزل + قاعدة بدأت (ناقصها تأكيد)
  👀 مبكر   = قسّم حديثاً بس ما نزل كفاية أو لسا يصنع قيعان
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_DAYS_AGO = 260       # التقسيم خلال ~سنة
DROP_TO = 0.65           # نزل لـ ≤65% من سعر ما بعد التقسيم (≈ 60%)
BASE_MIN = 10            # جلسات القاعدة بدون كسر القاع
MAX_RISE = 0.50          # قرار أبوفرح: ارتفع ≥50% من القاع = انطلق، يُحذف من الرادار

STATUS_AR = {"ready": "🎯 جاهز", "near": "🟡 يتجمّع", "watch": "👀 مبكر"}


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def _r(x, n=4):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), n)


def detect(d: pd.DataFrame, info: dict | None = None) -> dict | None:
    """تفاصيل إشارة ما بعد التقسيم العكسي لآخر يوم، أو None إذا ما فيه تقسيم حديث."""
    info = info or {}
    if d is None or len(d) < 40 or "Stock Splits" not in d.columns:
        return None
    c, h, l, v = (d[k].to_numpy(float) for k in ("Close", "High", "Low", "Volume"))
    dates = d.index
    t = len(c) - 1
    price = float(c[t])

    sp = d["Stock Splits"].to_numpy(float)
    ri = np.flatnonzero((sp > 0) & (sp < 1))
    if not len(ri):
        return None
    rj = int(ri[-1])
    days_ago = t - rj
    if days_ago < 3 or days_ago > MAX_DAYS_AGO:
        return None
    ratio = sp[rj]
    p0 = float(h[rj:min(rj + 4, len(h))].max())          # سعر ما بعد التقسيم
    if p0 <= 0:
        return None
    drop_from_p0 = price / p0 - 1                          # كم نزل من سعر ما بعد التقسيم

    # القاعدة بعد الهبوط
    seg = slice(rj + 1, t + 1)
    cs, hs, ls = c[seg], h[seg], l[seg]
    if len(cs) < 6:
        return None
    low_i = int(np.argmin(cs))                            # قاع ما بعد التقسيم
    low = float(cs[low_i]); days_to_bottom = low_i + 1
    support = float(ls[low_i:].min()) if low_i < len(ls) else low
    ceiling = float(hs[low_i:].max()) if low_i < len(hs) else price
    age = len(cs) - low_i - 1                              # جلسات من القاع
    # أعلى سعر وصله بعد التقسيم + نسبة الارتفاع من قاع القاعدة (طلب أبوفرح)
    peak_after = float(hs.max()); peak_after_i = int(np.argmax(hs))
    peak_date = str(dates[rj + 1 + peak_after_i])[:10]
    peak_rise_pct = (peak_after / support - 1) * 100 if support > 0 else 0.0
    # قرار أبوفرح: لو ارتفع ≥50% من قاع القاعدة = انطلق، فات وقت التجميع ← احذفه
    if support > 0 and price >= support * (1 + MAX_RISE):
        return None
    held = bool(l[max(rj + 1 + low_i, t - 2):t + 1].min() >= support * 0.97)   # ما كسر قاعه آخر 3ج
    lo20 = float(l[max(0, t - 19):t + 1].min())
    above_lo20 = bool(price >= lo20 * 0.995)

    r = rsi(c); rsi_now = float(r[t])
    # انكماش التذبذب
    tr = np.r_[h[0] - l[0], np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]
    atr5 = float(tr[max(0, t - 4):t + 1].mean()); atr20 = float(tr[max(0, t - 19):t + 1].mean())
    contract = bool(atr20 > 0 and atr5 <= atr20 * 0.95)
    # بصمة تجميع
    up = c > np.r_[c[0], c[:-1]]; dn = c < np.r_[c[0], c[:-1]]
    upv = v[max(0, t - 9):t + 1][up[max(0, t - 9):t + 1]].sum()
    dnv = v[max(0, t - 9):t + 1][dn[max(0, t - 9):t + 1]].sum()
    upv_ratio = upv / max(dnv, 1)
    obv = np.cumsum(np.where(up, v, np.where(dn, -v, 0.0)))
    obv_up = bool(t >= 20 and obv[t] > obv[t - 20])
    accum = bool(upv_ratio >= 1.0 or obv_up)
    pos = (price - support) / max(ceiling - support, 1e-9)

    # عمق النزول من سعر ما بعد التقسيم (0..1). دراسة العمق (diag/revsplit_depth.py، 979 حدث):
    # نزول <35% انفجاره 4-7% (أسوأ من العشوائي) · 50-80% = المنطقة الذهبية (~20%) · 90%+ أغلبه يموت.
    dd = max(0.0, 1 - price / p0)
    dropped = bool(dd >= (1 - DROP_TO))                   # نزل ≥35% (المرشح)
    golden = bool(0.50 <= dd <= 0.85)                     # المنطقة الذهبية (أعلى انفجار)
    deep_ok = bool(dd >= 0.45)                            # عمق كافٍ لـ«جاهز»
    base_ok = bool(age >= BASE_MIN and held and above_lo20)
    # هبوط طرح سابق (≥30% بيوم) بعد التقسيم — إشارة خطر/أو واشر
    dd1 = cs / np.r_[c[rj], cs[:-1]] - 1
    had_offer_drop = bool((dd1 <= -0.30).any())

    if not dropped:
        status = "watch"                                  # قسّم بس ما نزل كفاية بعد
    elif base_ok and deep_ok and contract and accum:
        status = "ready"                                  # المنطقة الذهبية + قاعدة مؤكدة
    elif base_ok and accum:
        status = "near"
    else:
        status = "watch"

    score = 0.0
    score += min(dd, 0.80) * 45                           # كل ما نزل أعمق (حتى 80%) أفضل
    score += 10 if golden else 0                          # مكافأة المنطقة الذهبية 50-80%
    score -= 8 if dd >= 0.90 else 0                       # 90%+ خطر موت
    score += 15 if base_ok else 0
    score += 12 if accum else 0
    score += 8 if contract else 0
    score += 8 if (20 <= rsi_now <= 50) else 0
    score += 6 if ratio <= 0.1 else (3 if ratio <= 0.2 else 0)   # التقسيم الكبير (1:10+) أفضل
    score = round(max(0.0, min(100.0, score)), 1)

    flt = info.get("floatShares")
    if flt is not None and flt < 50_000:
        flt = None

    mid = support + (ceiling - support) * 0.4
    stop = support * 0.90
    stop_pct = (1 - stop / price) * 100 if price else None
    why = [
        f"🔄 تقسيم عكسي 1:{round(1/ratio)} قبل {days_ago} جلسة · سعر ما بعد التقسيم {p0:.3g}",
        f"نزل {drop_from_p0*100:.0f}% منه (الحين {price:.3g}" + (" — داخل منطقة التجميع ≤65% ✓" if dropped else " — ما نزل كفاية بعد") + ")",
        (f"🟢 المنطقة الذهبية: نزل {dd*100:.0f}% (50-80% = أعلى انفجار ~20% بالدراسة)" if golden else f"نزل {dd*100:.0f}% من سعر ما بعد التقسيم"),
        f"قاع ما بعد التقسيم {support:.3g} (بعد {days_to_bottom} جلسة من التقسيم) وجالس فوقه {age} جلسة" + (" بدون كسر ✓" if held else ""),
        f"📈 أعلى سعر وصله بعد التقسيم {peak_after:.3g} ({peak_date})",
        f"التذبذب " + ("انكمش ✓ (زنبرك)" if contract else "ما انكمش") + f" · بصمة تجميع فوليوم صعود/هبوط {upv_ratio:.1f}x" + (" · OBV يطلع" if obv_up else ""),
        f"RSI {rsi_now:.0f}",
    ]
    if had_offer_drop:
        why.append("⚠ صار هبوط يوم ≥30% بعد التقسيم (طرح؟) — تأكد من ملفات SEC قبل الدخول")

    return {
        "status": status, "status_ar": STATUS_AR[status], "score": score,
        "revsplit": {"ratio": f"1:{round(1/ratio)}", "days_ago": int(days_ago), "p0": _r(p0),
                     "drop_from_p0_pct": _r(drop_from_p0 * 100, 1)},
        "dropped": dropped, "post_split": bool(dropped and base_ok), "golden": golden, "drop_depth_pct": _r(dd * 100, 0),
        "peak_after": _r(peak_after), "peak_date": peak_date, "peak_rise_pct": _r(peak_rise_pct, 0),
        "support": _r(support), "ceiling": _r(ceiling), "age": int(age), "days_to_bottom": int(days_to_bottom),
        "pos_in_base_pct": _r(pos * 100, 0), "held": held, "contract": contract, "accum": accum,
        "upv_ratio": _r(upv_ratio, 2), "obv_up": obv_up, "rsi": _r(rsi_now, 1),
        "had_offer_drop": had_offer_drop, "float": flt, "tiny_float": bool(flt and flt < 5e6),
        "plan": {
            "accumulate_zone": [_r(support), _r(mid)],
            "tranches": [_r(support * 1.01), _r((support + mid) / 2), _r(min(price, mid))],
            "launch_trigger": _r(ceiling),
            "stop": _r(stop), "stop_note": "إغلاق تحت قاع ما بعد التقسيم −10%",
            "stop_pct": _r(stop_pct, 1),
            "size_pct": _r(min(5.0, 1.0 / max((stop_pct or 20) / 100, 0.05)), 1),
            "hold": "البيع على المتداول وقت القفزة المضاربية (نص عند +40% ثم trail) — القفزة تصرّف نفس اليوم",
        },
        "why": why,
    }
