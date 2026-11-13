"""⛔ حارس الشطب — طلب المستخدم 2026-11-13: «أهم شي أنه السهم ماينشطب».

مهم (صدق): جرّبنا نبني نموذج إحصائي للشطب (`diag/backtest_delist.py`) لكن بيانات Alpaca المجانية **ما تغطي الأسهم البيني
المشطوبة** بشكل كافٍ (6 حالات شطب واضحة بـ7 سنين — Alpaca يحذفها بدل ما يعلّمها «موقوفة»). فالنموذج مو موثوق.
الأصدق: **قواعد ناسداك الحقيقية للشطب** — وهي اللي تتحقق فعلاً وتفهمها إنت:
  - إغلاق تحت 1$ لـ 30 جلسة متتالية ← إنذار عدم امتثال، مهلة 180 يوم (`compliance_rule`).
  - تقسيم عكسي خلال سنة + رجع تحت 1$ ← ما له مهلة، معرّض للشطب.
  - انهيار 90%+ من قمة السنة = موت شبه مؤكد (قياس: 83% من هالحالات انشطبت بالاختبار الوهمي).
تحذير فقط — **ما يستبعد** (قرار المستخدم). المستوى: 0 آمن · 1 متوسط · 2 عالي."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assess(daily: pd.DataFrame) -> dict:
    """يرجع {level, reasons, u1_streak, dd252, compliance}. daily = شارت يومي + عمود Stock Splits."""
    import compliance_rule as CR
    c = daily["Close"].to_numpy(float)
    n = len(c)
    px = float(c[-1])
    # عداد الأيام المتتالية تحت 1$ (عداد ناسداك)
    streak = 0
    for v in c[::-1]:
        if v < 1:
            streak += 1
        else:
            break
    hi252 = float(np.nanmax(c[-252:])) if n >= 20 else px
    dd = px / hi252 - 1 if hi252 > 0 else 0.0
    comp = None
    try:
        comp = CR.check(daily)
    except Exception:
        comp = None
    reasons, level = [], 0
    if comp:
        reasons.append(comp.get("why", ""))
        level = {"block": 2, "deadline": 2, "notice": 2, "info": 1}.get(comp.get("level"), 1)
    if dd <= -0.90:
        reasons.append(f"💀 منهار {abs(dd) * 100:.0f}% من قمة السنة — خطر موت/شطب عالي جداً")
        level = 2
    elif dd <= -0.80 and px < 1.5:
        reasons.append(f"منهار {abs(dd) * 100:.0f}% + سعر {px:.2f}$")
        level = max(level, 1)
    # تقسيم عكسي حديث (علامة شركة تهرب من الشطب)
    if "Stock Splits" in daily.columns:
        s = daily["Stock Splits"].fillna(0)
        rs = s[(s > 0) & (s < 1)]
        if len(rs):
            days_ago = (daily.index[-1] - rs.index[-1]).days
            if days_ago <= 365 and not any("تقسيم عكسي" in r for r in reasons):
                reasons.append(f"✂️ تقسيم عكسي 1:{round(1 / float(rs.iloc[-1]))} قبل {days_ago} يوم" + (" + تحت 1$" if px < 1 else ""))
                if px < 1:
                    level = 2
    reasons = [r for r in reasons if r]
    return {"level": level, "reasons": reasons, "u1_streak": streak, "dd252": round(dd, 3),
            "price": round(px, 4), "compliance": comp}
