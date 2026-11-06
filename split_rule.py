"""
قاعدة منطقة التقسيم (من نصيحة عبدالإله @z_v11v):

  - أعلى ارتفاع وصله السهم يوم التقسيم العكسي، مثلاً 4.50
  - اقسمه على 2 ← 2.25 (المنطقة الأولى)
  - نقص 20% (مسح سيولة) ← 1.80 — أحياناً تصير وأحياناً لا، لكن احتياط
  - إذا نزل السهم بين 1.80 و 2.25: ركّز عليه مع الشروط الأخرى

طبقة إضافية تُطبَّق على نسختي رادار فيصل (المطوّر والأصلي) من scanner.py،
ولا تغيّر نقاط أو تصنيف أي نسخة: تضيف وسم + معلومات المنطقة فقط.
"""
from __future__ import annotations

import pandas as pd

MAX_DAYS = 180       # نطبقها على التقسيمات خلال آخر 6 شهور
SWEEP_PCT = 20       # مسح السيولة تحت نصف القمة
NEAR_PCT = 5         # هامش قريب من المنطقة


def split_zone(df: pd.DataFrame) -> dict | None:
    if "Stock Splits" not in df.columns or df.empty:
        return None
    s = df["Stock Splits"].fillna(0)
    rs = s[(s > 0) & (s < 1)]
    if rs.empty:
        return None
    dt = rs.index[-1]
    days = int((df.index[-1] - dt).days)
    if days > MAX_DAYS:
        return None
    # البيانات معدّلة للتقسيم، فأعلى سعر يوم التقسيم = بعد التقسيم مباشرة
    split_high = float(df.loc[dt, "High"])
    half = split_high / 2
    sweep = half * (1 - SWEEP_PCT / 100)
    price = float(df["Close"].iloc[-1])
    after = df.loc[dt:]
    low_since = float(after["Low"].min())
    touched = bool(((after["Low"] <= half * (1 + NEAR_PCT / 100)) & (after["Low"] >= sweep * 0.95)).any())

    if sweep * 0.97 <= price <= half * (1 + NEAR_PCT / 100):
        status, label = "in", "🎯 داخل منطقة التقسيم"
        note = f"السعر {price:.2f} بين {sweep:.2f} و {half:.2f} — ركّز عليه مع باقي الشروط"
    elif price > half * (1 + NEAR_PCT / 100):
        status, label = "above", "فوق منطقة التقسيم"
        note = f"باقي {(price / half - 1) * 100:.1f}% لين يوصل {half:.2f} (ونطاق المسح لين {sweep:.2f})"
    else:
        status, label = "below", "كسر منطقة المسح"
        note = f"السعر تحت {sweep:.2f} — نزل أكثر من مسح السيولة المتوقع، انتبه"
    return {
        "date": str(dt.date()), "ratio": f"1:{round(1 / float(rs.iloc[-1]))}", "days_ago": days,
        "split_high": round(split_high, 4), "half": round(half, 4), "sweep": round(sweep, 4),
        "status": status, "label": label, "note": note,
        "touched": touched, "low_since": round(low_since, 4),
        "window_20_50": 20 <= days <= 50,
    }


def apply(a: dict, df: pd.DataFrame) -> None:
    """يضيف منطقة التقسيم لنتيجة السهم (بدون تغيير النقاط أو الحالة)."""
    z = split_zone(df)
    if not z:
        return
    a["split_zone"] = z
    if z["status"] == "in":
        a["tags"].append("🎯 منطقة التقسيم")
