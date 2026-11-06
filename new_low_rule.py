"""
قاعدة "القاع الجديد" (طلب المستخدم — مثال VRAX):

إذا ذيل الشمعة اليومية الأخيرة نزل لأقل سعر من فترة طويلة (قاع جديد)،
فالسهم كسر دعمه ويدور على دعم ثاني. ما يترشح (مطابق / قريب) لين:
  - يمر عليه عدة جلسات بدون قاع أدنى (تماسك عند الدعم الجديد)، و
  - يختبر الدعم الجديد مرتين على الأقل ويصمد (يغلق فوقه).

طبقة تُطبَّق على نسختي رادار فيصل من scanner.py: تنزّل الحالة إلى "رادار" وتكتب السبب،
بدون تغيير نقاط النسخة أو ملف قواعدها.
"""
from __future__ import annotations

import pandas as pd

LOOKBACK = 60        # "أقل سعر من فترة": آخر 60 جلسة (~3 شهور)
MIN_HOLD = 3         # جلسات بدون قاع أدنى بعد القاع الجديد
MIN_TESTS = 2        # اختبارات الدعم الجديد
BAND = 0.03          # الاختبار = نزول ضمن 3% من القاع
REARM = 0.025        # اختبار جديد ينحسب بعد إغلاق 2.5%+ فوق القاع


def check(df: pd.DataFrame) -> dict | None:
    d = df.tail(LOOKBACK + 1)
    if len(d) < 20:
        return None
    lows, closes = d["Low"].values, d["Close"].values
    # القاع الجديد = أدنى سعر في آخر 60 جلسة، لازم يكون كاسر لقاع الفترة اللي قبله
    i_min = int(lows.argmin())
    new_low = float(lows[i_min])
    before = lows[:i_min]
    if len(before) < 15 or new_low >= before.min() * 0.995:
        return None  # ما فيه كسر لقاع سابق خلال الفترة
    prev_support = float(before.min())
    held = len(d) - 1 - i_min  # جلسات بعد القاع الجديد
    # اختبارات الدعم الجديد (من يوم القاع وبعده)
    tests, armed = 0, True
    for k in range(i_min, len(d)):
        if armed and lows[k] <= new_low * (1 + BAND) and closes[k] >= new_low:
            tests += 1
            armed = False
        if closes[k] >= new_low * (1 + REARM):
            armed = True
    last_is_low = held == 0
    pending = last_is_low or held < MIN_HOLD or tests < MIN_TESTS
    if last_is_low:
        why = (f"ذيل شمعة آخر يوم ({str(d.index[-1])[:10]}) نزل لـ {new_low:.2f} = أقل سعر من {LOOKBACK} جلسة "
               f"(كسر الدعم السابق {prev_support:.2f}) — انتظر تماسك عند الدعم الجديد واختباره")
    elif pending:
        need = []
        if held < MIN_HOLD:
            need.append(f"{MIN_HOLD - held} جلسة تماسك بدون قاع أدنى")
        if tests < MIN_TESTS:
            need.append(f"{MIN_TESTS - tests} اختبار إضافي للدعم الجديد")
        why = (f"قاع جديد {new_low:.2f} قبل {held} جلسة (كسر {prev_support:.2f}) — باقي: " + " + ".join(need))
    else:
        why = f"الدعم الجديد {new_low:.2f} صمد {held} جلسة واختُبر {tests} مرات ✓"
    return {"new_low": round(new_low, 4), "date": str(d.index[i_min])[:10], "prev_support": round(prev_support, 4),
            "held": held, "tests": tests, "last_is_low": last_is_low, "pending": pending, "why": why}


def apply(a: dict, df: pd.DataFrame) -> None:
    nl = check(df)
    if not nl:
        return
    a["new_low"] = nl
    I = a["irtikaz"]
    if nl["pending"]:
        a["tags"].append("⚠ قاع جديد")
        if I["status"] in ("ready", "near"):
            I["status_before_new_low"] = I["status"]
            I["status"] = "radar"
        I["adjustments"].append({"name": "قاع جديد — ينتظر تماسك الدعم الثاني", "pts": 0, "detail": nl["why"]})
        I["readiness"].insert(0, "يتماسك عند الدعم الجديد " + f"{nl['new_low']:.2f}" + " ويختبره مرتين ويصمد")
    else:
        I["adjustments"].append({"name": "الدعم الثاني متماسك", "pts": 0, "detail": nl["why"]})
