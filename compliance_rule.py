"""
طبقة امتثال ناسداك (قرار المستخدم: نوسّع المجال لين 0.50$ مع الانتباه للامتثال).
تُطبق على رادارات فيصل (النسختين) وعبدالله — مو المطّلعين.

قواعد ناسداك (الحد الأدنى للسعر 1$):
  - 30 يوم تداول متتالية إغلاق تحت 1$  ← إنذار عدم امتثال.
  - المهلة 180 يوم تقويمي يرجع فيها فوق 1$ لمدة 10 أيام متتالية (وأحياناً مهلة ثانية).
  - إذا سوّت الشركة تقسيم عكسي خلال آخر سنة ونزلت تحت 1$ مرة ثانية ← ما فيه مهلة، إجراءات شطب مباشرة.
  - غالباً الحل تقسيم عكسي، وغالباً يجي معه طرح أسهم.

التصرف:
  - تحت 1$ أقل من 30 جلسة       ← وسم معلوماتي
  - 30 جلسة أو أكثر             ← ⚠ إنذار امتثال غالباً + كم باقي على المهلة
  - باقي على المهلة أقل من 45 يوم ← أقصى حالة "قريب" (ما يصير مطابق)
  - تقسيم عكسي خلال سنة + تحت 1$ ← ⛔ ما يترشح أبداً (أقصى حالة "رادار")
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

NOTICE_SESSIONS = 30
CURE_DAYS = 180
NEAR_DEADLINE_DAYS = 45


def check(df: pd.DataFrame) -> dict | None:
    c = df["Close"].values
    price = float(c[-1])
    streak = 0
    for v in c[::-1]:
        if v < 1:
            streak += 1
        else:
            break
    last = df.index[-1]
    rs_recent = None
    if "Stock Splits" in df.columns:
        s = df["Stock Splits"].fillna(0)
        rs = s[(s > 0) & (s < 1)]
        rs = rs[rs.index >= last - timedelta(days=365)]
        if not rs.empty:
            rs_recent = {"date": str(rs.index[-1].date()), "ratio": f"1:{round(1 / float(rs.iloc[-1]))}"}
    if price >= 1 and streak == 0:
        return None
    out = {"below_streak": streak, "price": round(price, 4), "rs_recent": rs_recent,
           "notice_likely": streak >= NOTICE_SESSIONS, "days_left": None, "deadline": None,
           "level": "info", "cap": None}
    if streak >= NOTICE_SESSIONS:
        notice_day = df.index[-(streak - NOTICE_SESSIONS + 1)]  # تقريباً يوم الجلسة 30
        deadline = notice_day + timedelta(days=CURE_DAYS)
        out["deadline"] = str(deadline.date())
        out["days_left"] = int((deadline - last).days)
    if rs_recent and price < 1:
        out["level"], out["cap"] = "block", "radar"
        out["why"] = (f"⛔ سوّى تقسيم عكسي {rs_recent['ratio']} ({rs_recent['date']}) خلال آخر سنة ورجع تحت 1$ — "
                      f"حسب قواعد ناسداك الجديدة ما له مهلة، معرّض للشطب. ما يترشح.")
    elif out["days_left"] is not None and out["days_left"] < NEAR_DEADLINE_DAYS:
        out["level"], out["cap"] = "deadline", "near"
        out["why"] = (f"⚠ تحت 1$ من {streak} جلسة — المهلة تقريباً تنتهي {out['deadline']} "
                      f"(باقي {max(out['days_left'], 0)} يوم). احتمال تقسيم عكسي قريب (وغالباً معه طرح)")
    elif out["notice_likely"]:
        out["level"] = "notice"
        out["why"] = (f"⚠ تحت 1$ من {streak} جلسة — غالباً عنده إنذار عدم امتثال، "
                      f"المهلة تقريباً لين {out['deadline']} (باقي {out['days_left']} يوم)")
    else:
        out["why"] = f"تحت 1$ من {streak} جلسة (الإنذار بعد {NOTICE_SESSIONS} جلسة متتالية)"
    return out


def _cap(status: str, cap: str | None) -> str:
    if not cap:
        return status
    rank = {"ready": 0, "near": 1, "radar": 2, "no": 3, "moved": 3, "invalid": 4}
    return status if rank.get(status, 9) >= rank[cap] else cap


def apply_faisal(a: dict, df: pd.DataFrame) -> None:
    c = check(df)
    if not c:
        return
    a["compliance"] = c
    I = a["irtikaz"]
    new = _cap(I["status"], c["cap"])
    if new != I["status"]:
        I["status_before_compliance"] = I["status"]
        I["status"] = new
    if c["level"] in ("block", "deadline", "notice"):
        a["tags"].append("⛔ امتثال ناسداك" if c["level"] == "block" else "⚠ امتثال ناسداك")
        I["adjustments"].append({"name": "امتثال ناسداك", "pts": 0, "detail": c["why"]})


def apply_abdullah(a: dict, df: pd.DataFrame) -> None:
    c = check(df)
    if not c:
        return
    a["compliance"] = c
    new = _cap(a["status"], c["cap"])
    if new != a["status"]:
        a["status_before_compliance"] = a["status"]
        a["status"] = new
    if c["level"] in ("block", "deadline", "notice"):
        a.setdefault("flags", []).insert(0, c["why"])
    else:
        a.setdefault("flags", []).append(c["why"])
