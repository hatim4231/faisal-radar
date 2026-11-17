#!/usr/bin/env python3
"""🔥 كاشف «فيصل الرهيب» — نظرية الارتكاز البحتة (من دليل فيصل، طلب أبوفرح 2026-11-15).

الارتكاز (ص39 من الدليل): «سهم تحت المتابعة عند دعم حقيقي، وليس دخولاً فورياً».
الفلسفة (ص4): السلوك أولاً، الدعم ثانياً، الثبات ثالثاً، الدخول أخيراً. لا تطارد. السهم الجيد ≠ دخول جيد.

فلتر فيصل للارتكاز (ص6): سعر < 5$ · Micro/أقل من 300M · فلووت < 20M (قوي <5M) · متوسط فوليوم > 100K ·
  أداء الشهر أو الربع هابط (نبحث عن قيعان) · RVOL أي شيء (الارتكاز قد يكون نائماً).

الإعداد (ص13 قراءة التجميع): دعم حقيقي ملموس أكثر من مرة + تماسك + فوليوم البيع يقل + قيعان أعلى + انكماش الشموع.
  ليس كل تماسك تجميعاً (ص14): كسر الدعم بسهولة = ضعف · صعود فجائي فقط = خطر · لا وقف واضح = لا تدخل.
الخطة (ص15-16): منطقة طلب (دعم → +زون)، 3 أوامر Limit، دعم حرج (كسره = خروج)، جني ربح تدريجي عند المقاومة 1 ثم 2.

الحالات: 🔥 جاهز للارتكاز (عند الدعم + تجميع مؤكد + ثبات + مساحة صعود + مو ضخ) · 🟡 يتجمّع · 👀 مراقبة (مرشح ارتكاز لسا ما وصل الدعم).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import faisal_rules as FR   # نعيد استخدام أدوات فيصل (الدعم، المقاومات، القيعان الأعلى، الفوليوم، سحب السيولة، W...)
import prior_rise           # شرط أبوفرح: تاريخ ارتفاع سابق ≥50% خلال 6 شهور

STATUS_AR = {"ready": "🔥 جاهز للارتكاز", "near": "🟡 يتجمّع", "watch": "👀 مراقبة"}


def _r(x, n=4):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), n)


def detect(d: pd.DataFrame, info: dict | None, cfg: dict) -> dict | None:
    """يحلل سهم واحد على نظرية الارتكاز فقط. يرجع None إذا ما نجح فلتر فيصل أو ما فيه إعداد ارتكاز."""
    info = info or {}
    RC = cfg.get("rahib") or {}
    d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    d = d[d["Close"] > 0]
    if len(d) < 70:
        return None
    d["Volume"] = d["Volume"].fillna(0)
    c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); l = d["Low"].to_numpy(float)
    price = float(c[-1]); prev = float(c[-2]); n = len(c)

    # ===== فلتر فيصل للارتكاز (ص6) =====
    avg_vol = float(d["Volume"].iloc[-31:-1].mean()) if n > 31 else float(d["Volume"].mean())
    mcap = info.get("marketCap")
    flt = info.get("floatShares")
    if flt is not None and flt < 50_000:
        flt = None
    if not (RC.get("min_price", 0.1) <= price <= RC.get("max_price", 5.0)):
        return None
    if avg_vol < RC.get("min_avg_volume", 100_000):
        return None
    if mcap is not None and mcap > RC.get("max_market_cap", 300e6):
        return None
    if flt is not None and flt > RC.get("max_float", 20e6):
        return None
    # أداء الشهر أو الربع هابط (نبحث عن قيعان — ص6)
    perf_month = price / c[-22] - 1 if n > 22 else 0.0
    perf_quarter = price / c[-64] - 1 if n > 64 else 0.0
    down = bool(perf_month < 0 or perf_quarter < 0)
    if not down:
        return None   # فيصل يرفض المطاردة بعد الصعود — يبحث عن القيعان
    # شرط أبوفرح: تاريخ ارتفاع سابق ≥50% خلال 6 شهور (سلوك يثبت إنه يتحرك)
    pr_holder = {}
    if not prior_rise.apply_gate(pr_holder, d, cfg):
        return None
    prior = pr_holder.get("prior_rise")

    # ===== الإعداد: دعم حقيقي + تجميع + ثبات + مساحة + مو ضخ =====
    sup = FR.find_support(d); support = sup["level"]; touches = int(sup["touches"])
    dist = FR.pct(price, support)                       # البعد عن الدعم %
    # قراءة التجميع (ص13)
    hl = FR.higher_lows(d)                              # قيعان أعلى
    vol = FR.volume_read(d)                             # {up_vs_down, selling_drying}
    selling_drying = bool(vol.get("selling_drying"))
    up_vs_down = float(vol.get("up_vs_down") or 0)
    # انكماش الشموع (ATR5 < ATR20)
    tr = np.r_[h[0] - l[0], np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]
    atr5 = float(tr[-5:].mean()); atr20 = float(tr[-20:].mean())
    contract = bool(atr20 > 0 and atr5 <= atr20 * 0.95)
    # الثبات: ما كسر الدعم وبقي تحته (ص14: كسر الدعم بسهولة = ضعف)
    broke_stay = FR.broke_support_and_stayed(d, support)
    held = bool(l[-3:].min() >= support * 0.97 and not broke_stay)
    rsi_now = FR._r(FR.rsi(d["Close"]).iloc[-1], 1)

    # مو ضخ (ص10/37): مو شمعة خضراء عمودية، مو بعيد عن الدعم، صعد كم% من قاع 20
    up_from_low20 = FR.pct(price, float(l[-20:].min()))
    change_today = (price / prev - 1) * 100 if prev else 0
    pumps = FR.pump_candles(d, wick_pct=(cfg.get("behavior", {}) or {}).get("pump_upper_wick_pct", 15))
    not_pump = bool(change_today < 25 and (up_from_low20 is None or up_from_low20 < 30) and pumps < 2)

    # مساحة صعود: مقاومة/فجوة/شمعة ساقطة فوق = هدف (ص4 «هدف منطقي»)
    res = FR.resistances_above(d, price)
    fcs = [f for f in FR.falling_candles(d) if f["top"] > price]
    gaps = [g for g in FR.unfilled_gaps_down(d) if g["to"] > price]
    tops = sorted({round(x["level"], 4) for x in res})
    t1 = tops[0] if tops else None
    major = [f["top"] for f in fcs] + [g["to"] for g in gaps] + tops
    major = [x for x in major if x <= max(price, support) * 3]
    t2 = max(major) if major else (t1)
    upside_pct = FR.pct(t1, price) if t1 else None
    has_upside = bool(upside_pct is not None and upside_pct >= RC.get("min_upside", 10))

    # إشارات مساعدة (ص10): سحب سيولة · W · Former Runner · تقسيم عكسي · فجوات
    sweep = FR.liquidity_sweep(d, support)
    w = FR.w_pattern(d)
    fr_run = FR.former_runner(d)
    rsplits = FR.reverse_splits(d)
    last_rs = rsplits[-1] if rsplits else None

    # قوة التجميع (كم إشارة من إشارات ص13)
    accum_signs = sum([bool(hl), bool(contract), bool(selling_drying or up_vs_down >= 1.2), touches >= 2])
    at_support = bool(dist is not None and dist <= RC.get("support_near", 8))
    near_support = bool(dist is not None and dist <= RC.get("support_watch", 15))

    # ===== الحالات =====
    # دعم حقيقي (فيصل): ملموس أكثر من مرة، أو سحب سيولة واسترداد، أو قيعان أعلى — مو قاع جديد لسا نازل
    real_support = bool(touches >= 2 or sweep or hl)
    if broke_stay:
        status = "watch"                               # كسر الدعم وبقي تحته = ضعف (ص14)
    elif at_support and held and real_support and accum_signs >= 2 and not_pump and has_upside:
        status = "ready"                               # عند دعم حقيقي + تجميع مؤكد + ثبات + مساحة + مو ضخ
    elif near_support and real_support and accum_signs >= 1 and not_pump:
        status = "near"                                # قريب دعم حقيقي + بوادر تجميع
    else:
        status = "watch"                               # مرشح ارتكاز بس دعمه قاع جديد/ما وصل/ما تأكد

    # ===== خطة فيصل (ص15-16) =====
    entry_pct = RC.get("entry_zone_pct", 8)
    zone_hi = support * (1 + entry_pct / 100)
    # الوقف: تحت قاع السحب إن وُجد، وإلا تحت الدعم −10% (ص36: الارتكاز وقفه أوسع لكن محدد)
    sweep_low = sweep["sweep_low"] if sweep else None
    base_low = min([x for x in (support, sweep_low) if x])
    stop = base_low * (1 - RC.get("stop_below_pct", 10) / 100)
    stop_pct = (1 - stop / price) * 100 if price else None
    tranches = [support * 1.01, (support + zone_hi) / 2, min(price, zone_hi)]

    # نقاط الترتيب (أعلى = أقرب لإعداد ارتكاز نظيف)
    score = 0.0
    score += 25 if at_support else (12 if near_support else 0)
    score += accum_signs * 8
    score += 12 if has_upside else 0
    score += 8 if not_pump else 0
    score += 8 if touches >= 2 else 0
    score += 6 if sweep else 0
    score += 6 if (w and not w.get("broken")) else 0
    score += 6 if fr_run["max_pct"] >= 100 else 0
    score += 6 if (flt and flt < 5e6) else (3 if (flt and flt < 20e6) else 0)
    score = round(min(100.0, score), 1)

    # قائمة فحص فيصل (ص37)
    checklist = {
        "support_real": touches >= 2 or sup["kind"] != "قاع جديد",
        "accumulation": accum_signs >= 2,
        "sell_vol_down": selling_drying or up_vs_down >= 1.2,
        "higher_lows": bool(hl),
        "contraction": contract,
        "not_pump": not_pump,
        "has_stop": True,
        "has_upside": has_upside,
    }

    why = [
        f"نوع السلوك: ارتكاز (دعم حقيقي {FR.fmt_price(support)}، {sup['kind']}، لمسات {touches})",
        f"البعد عن الدعم {dist:.1f}% " + ("— داخل منطقة الطلب ✓" if at_support else "— لسا فوق منطقة الطلب"),
        f"أداء الشهر {perf_month*100:.0f}% · الربع {perf_quarter*100:.0f}% (هابط — فيصل يبحث عن قاع)",
        f"قراءة التجميع: قيعان أعلى {'✓' if hl else '✗'} · انكماش الشموع {'✓' if contract else '✗'} · فوليوم البيع يقل {'✓' if (selling_drying or up_vs_down>=1.2) else '✗'} ({up_vs_down:.1f}x)",
        f"مو ضخ: {'✓' if not_pump else '✗ (صعد/شمعة عمودية — فيصل يرفض المطاردة)'} · صعد {up_from_low20:.0f}% من قاع 20ج",
        (f"مساحة صعود: أول مقاومة {FR.fmt_price(t1)} (+{upside_pct:.0f}%)" if has_upside else "ما فيه هدف واضح فوق"),
        f"RSI {rsi_now}",
    ]
    if prior:
        why.append(f"📈 تاريخ ارتفاع سابق: +{prior['max_rise_pct']:.0f}% ({prior['from_date']}→{prior['date']}) — سلوك يثبت إنه يتحرك")
    if sweep:
        why.append(f"سحب سيولة إلى {FR.fmt_price(sweep['sweep_low'])} ثم رجع فوق الدعم ✓ (ادخل بعد الاسترداد)")
    if w and not w.get("broken"):
        why.append(f"نموذج W: قاعين {FR.fmt_price(w['low1'])}/{FR.fmt_price(w['low2'])} وخط عنق {FR.fmt_price(w['neck'])} — انتظر كسره وثبات فوقه")
    if fr_run["max_pct"] >= 100:
        why.append(f"Former Runner: سبق انفجر +{fr_run['max_pct']:.0f}% ({fr_run['date']})")
    if last_rs and last_rs.get("days_ago", 999) <= 120:
        why.append(f"تقسيم عكسي {last_rs['ratio']} قبل {last_rs['days_ago']} يوم (محفّز مضاربي، ليس سبب استثمار)")
    if broke_stay:
        why.append("⚠ كسر الدعم وبقي تحته = ضعف (ص14) — ما يترشح لين يسترد")

    tail = d.tail(100)
    ma20 = d["Close"].rolling(20).mean().tail(100); ma50 = d["Close"].rolling(50).mean().tail(100)
    return {
        "status": status, "status_ar": STATUS_AR[status], "score": score,
        "support": _r(support), "support_kind": sup["kind"], "touches": touches, "dist_support_pct": _r(dist, 1),
        "perf_month_pct": _r(perf_month * 100, 0), "perf_quarter_pct": _r(perf_quarter * 100, 0),
        "higher_lows": bool(hl), "contract": contract, "selling_drying": selling_drying, "up_vs_down": _r(up_vs_down, 2),
        "accum_signs": accum_signs, "not_pump": not_pump, "up_from_low20_pct": _r(up_from_low20, 0),
        "rsi": rsi_now, "sweep": bool(sweep), "w": bool(w and not w.get("broken")),
        "former_runner_pct": _r(fr_run["max_pct"], 0), "reverse_split": last_rs,
        "float": flt, "market_cap": mcap, "upside_pct": _r(upside_pct, 0), "has_upside": has_upside,
        "prior_rise": prior, "checklist": checklist,
        "plan": {
            "demand_zone": [_r(support), _r(zone_hi)],
            "tranches": [_r(x) for x in tranches],
            "critical_support": _r(support), "stop": _r(stop), "stop_pct": _r(stop_pct, 1),
            "t1": _r(t1), "t2": _r(t2),
            "note": "Limit داخل منطقة الطلب (3 أوامر) · لا تشتري من العرض · كسر الدعم الحرج = خروج · جني ربح تدريجي عند المقاومة 1 ثم 2",
        },
        "why": why,
        "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(x) for x in tail["Open"]],
                  "h": [_r(x) for x in tail["High"]], "l": [_r(x) for x in tail["Low"]],
                  "c": [_r(x) for x in tail["Close"]], "v": [int(x) for x in tail["Volume"]],
                  "ma20": [_r(x) for x in ma20], "ma50": [_r(x) for x in ma50]},
    }
