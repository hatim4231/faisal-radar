"""
🔍 رادار التجميع — قواعد كشف تجميع المضارب (أسلوب وايكوف + روح طريقة فيصل).

المراحل: تعب (هبوط) ← قاعدة عرضية يمتص فيها المضارب العرض ← هزة/سحب سيولة (Spring) ← اختبار هادي بفوليوم ناشف ← انطلاق.

البصمات المقاسة على الشارت اليومي داخل القاعدة:
  1) OBV/تراكم السيولة يطلع والسعر واقف (أقوى علامة)          20
  2) انكماش التذبذب (النطاق يضيق)                            15
  3) أيام امتصاص: فوليوم عالي + شمعة صغيرة + إغلاق فوق        10
  4) الإغلاق في أعلى الشمعة (CLV)                            10
  5) فوليوم الصعود أكبر من الهبوط                            10
  6) دوران الفلووت: تداول ≥ 1× الفلووت والسعر ما تحرك          10
  7) هزة/سحب سيولة تحت الدعم ورجوع فوقه                      10
  8) اختبار هادي: آخر أيام بفوليوم ناشف قرب الدعم             10
  9) هبوط قبل القاعدة (تعب)                                   5
  + مكافآت: الشورت المتاح يقل (+5)، فلووت أقل من 10M (+3)

ما يشمل الأسهم اللي انطلقت (السعر فوق سقف القاعدة بـ 8%+).
الوقف (قرار المستخدم): أقل سعر في القاعدة − 10%.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _r(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def fp(p):
    return "-" if p is None else (f"{p:.3f}" if p < 1 else f"{p:.2f}")


def find_base(df: pd.DataFrame, A: dict):
    """أطول نافذة أخيرة (10-60 جلسة) نطاقها (بالإغلاقات) ضمن base_max_range%. الذيول مسموحة (سحب سيولة)."""
    c = df["Close"].values
    best = None
    for k in range(A["base_min"], min(A["base_max"], len(df) - 70) + 1):
        seg = c[-k:]
        if seg.max() / seg.min() > 1 + A["base_max_range_pct"] / 100:
            break
        best = k
    return best


def slope_norm(y: np.ndarray) -> float:
    """ميل الانحدار الخطي ÷ متوسط القيمة المطلقة (عشان نقارن بين الأسهم)."""
    x = np.arange(len(y))
    if len(y) < 3:
        return 0.0
    m = np.polyfit(x, y, 1)[0]
    scale = np.mean(np.abs(y)) or 1.0
    return float(m * len(y) / scale)


def analyze(df: pd.DataFrame, info: dict | None, cfg: dict, short_readings: list | None = None) -> dict | None:
    A = cfg["accumulation"]
    info = info or {}
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df[df["Close"] > 0]
    if len(df) < 120:
        return None
    df["Volume"] = df["Volume"].fillna(0)
    k = find_base(df, A)
    if not k:
        return None
    b = df.tail(k)
    pre = df.iloc[-(k + 60):-k]
    price = float(df["Close"].iloc[-1])
    hi, lo, cl, op, vol = (b[x].values.astype(float) for x in ("High", "Low", "Close", "Open", "Volume"))
    support = float(np.percentile(lo, 15))
    base_top = float(np.percentile(hi, 90))
    base_low = float(lo.min())
    avg50 = float(df["Volume"].iloc[-(k + 50):-k].mean() or df["Volume"].tail(50).mean() or 1)
    base_avg = float(vol.mean() or 1)
    prev_close = df["Close"].shift(1).tail(k).values

    # انطلق؟
    if price > base_top * (1 + A["breakout_pct"] / 100):
        return {"status": "moved", "base_len": k}

    checks, pts = [], 0

    def add(key, title, w, got, detail, remaining=""):
        nonlocal pts
        got = max(0.0, min(w, got))
        pts += got
        checks.append({"key": key, "title": title, "weight": w, "got": round(got, 1), "ok": got >= w * 0.7,
                       "detail": detail, "remaining": remaining, "progress": round(got / w, 2)})

    # 1) OBV يطلع والسعر واقف
    direction = np.sign(cl - prev_close)
    obv = np.cumsum(direction * vol)
    clv = np.where(hi > lo, ((cl - lo) - (hi - cl)) / (hi - lo), 0.0)
    ad = np.cumsum(clv * vol)
    obv_s = slope_norm(obv - obv.min() + base_avg)
    ad_s = slope_norm(ad - ad.min() + base_avg)
    px_s = slope_norm(cl)
    flow = (obv_s + ad_s) / 2
    if flow > 0.3 and abs(px_s) < 0.12:
        add("obv", "السيولة داخلة والسعر واقف (OBV/A-D)", 20, 20, f"تراكم السيولة يطلع (+{flow:.2f}) والسعر عرضي ({px_s:+.2f})")
    elif flow > 0.15:
        add("obv", "السيولة داخلة والسعر واقف (OBV/A-D)", 20, 12, f"تراكم السيولة يطلع شوي (+{flow:.2f})", "يحتاج دخول سيولة أوضح")
    else:
        add("obv", "السيولة داخلة والسعر واقف (OBV/A-D)", 20, 0, f"تراكم السيولة ما يطلع ({flow:+.2f})", "ما فيه بصمة شراء خفي")

    # 2) انكماش التذبذب
    rng = (hi - lo) / np.where(cl > 0, cl, 1)
    half = max(3, k // 2)
    first, last = float(np.median(rng[:half])), float(np.median(rng[-half:]))
    ratio = last / first if first else 1
    add("contract", "التذبذب ينكمش (النطاق يضيق)", 15, 15 if ratio < 0.7 else 9 if ratio < 0.9 else 0,
        f"مدى الشمعة آخر القاعدة {last * 100:.1f}% مقابل {first * 100:.1f}% بأولها", "" if ratio < 0.9 else "النطاق ما ضاق")

    # 3) أيام امتصاص
    med_rng = float(np.median(rng))
    absorb = int(np.sum((vol >= 2 * avg50) & (rng <= med_rng * 1.2) & (cl >= prev_close * 0.99) & (clv >= 0)))
    add("absorb", "أيام امتصاص (فوليوم عالي بدون نزول)", 10, min(10, absorb * 4),
        f"{absorb} يوم فوليومه ضعف المعدل والسعر ما نزل", "" if absorb else "ما فيه امتصاص واضح")

    # 4) الإغلاق في أعلى الشمعة
    clv_pos = float(np.mean(np.where(hi > lo, (cl - lo) / (hi - lo), 0.5)))
    add("clv", "الإغلاق في أعلى الشمعة", 10, 10 if clv_pos >= 0.58 else 6 if clv_pos >= 0.52 else 0,
        f"متوسط موقع الإغلاق {clv_pos * 100:.0f}% من مدى الشمعة")

    # 5) فوليوم الصعود أكبر من الهبوط
    up_v = float(vol[cl >= prev_close].sum())
    dn_v = float(vol[cl < prev_close].sum()) or 1
    udr = up_v / dn_v
    add("updown", "فوليوم الصعود أكبر من الهبوط", 10, 10 if udr >= 1.3 else 6 if udr >= 1.1 else 0,
        f"فوليوم الصعود {udr:.2f}× فوليوم الهبوط")

    # 6) دوران الفلووت
    flt = info.get("floatShares")
    if flt and flt >= 50_000:
        turn = float(vol.sum()) / flt
        add("turnover", "دوران الفلووت داخل القاعدة", 10, 10 if 1 <= turn <= 6 else 6 if turn >= 0.5 else 0,
            f"تداول {turn:.1f}× الفلووت والسعر ما تحرك" + (" (كثير جداً — ممكن تصريف)" if turn > 6 else ""))
    else:
        turn = None
        add("turnover", "دوران الفلووت داخل القاعدة", 10, 3, "الفلووت غير معروف")

    # 7) هزة / سحب سيولة
    spring = None
    for i in range(len(lo)):
        if lo[i] < support * 0.97:
            nxt = cl[i:i + 3]
            if (nxt >= support).any():
                spring = {"low": float(lo[i]), "date": str(b.index[i].date())}
    add("spring", "هزة: كسر بسيط للدعم ورجوع فوقه", 10, 10 if spring else 0,
        f"نزل لـ {fp(spring['low'])} ({spring['date']}) ورجع فوق الدعم" if spring else "ما صار سحب سيولة",
        "" if spring else "أفضل دخول بعد هزة ورجوع")

    # 8) اختبار هادي
    tail5v = float(vol[-5:].mean())
    near = price <= support * 1.12
    dry = tail5v < base_avg * 0.6
    add("test", "اختبار هادي قرب الدعم بفوليوم ناشف", 10, (5 if dry else 0) + (5 if near else 0),
        f"فوليوم آخر 5 أيام {tail5v / base_avg * 100:.0f}% من معدل القاعدة · السعر عن الدعم {(price / support - 1) * 100:+.1f}%",
        "" if (dry and near) else ("ينتظر نزول هادي للدعم" if not near else "الفوليوم لسه مو ناشف"))

    # 9) هبوط قبل القاعدة
    pre_high = float(pre["Close"].max()) if len(pre) else base_top
    drop = (pre_high / base_top - 1) * 100
    add("decline", "هبوط/تعب قبل القاعدة", 5, 5 if drop >= 30 else 2 if drop >= 15 else 0,
        f"كان فوق القاعدة بـ {drop:.0f}% قبلها")

    bonus = []
    if short_readings and len(short_readings) >= 3:
        f0, f1 = short_readings[0].get("available"), short_readings[-1].get("available")
        if f0 and f1 is not None and f1 < f0 * 0.8:
            pts += 5
            bonus.append(f"الشورت المتاح يقل: {f0:,} ← {f1:,}")
    if flt and flt < 10e6:
        pts += 3
        bonus.append(f"فلووت صغير {flt / 1e6:.2f}M")
    pts = min(100.0, pts)

    status = ("ready" if pts >= A["ready_score"] and k >= A["ready_base"] else
              "near" if pts >= A["near_score"] else "radar")
    stop = base_low * (1 - A["stop_below_low_pct"] / 100)
    entry = [support, support * 1.08]
    t2 = pre_high if pre_high > base_top * 1.2 else base_top * 1.5
    country = (info.get("country") or "")
    chinese = country in ("China", "Hong Kong", "Macau", "Taiwan", "Singapore", "Malaysia")
    tail = df.tail(120)
    return {
        "price": _r(price), "date": str(df.index[-1].date()), "status": status, "score": round(pts, 1),
        "base_len": k, "support": _r(support), "base_top": _r(base_top), "base_low": _r(base_low),
        "range_pct": _r((b["Close"].max() / b["Close"].min() - 1) * 100, 1),
        "entry": [_r(x) for x in entry], "stop": _r(stop), "targets": [_r(base_top), _r(t2)],
        "spring": spring, "float_turnover": _r(turn, 2), "flow": _r(flow, 2), "updown": _r(udr, 2),
        "chinese": chinese, "country": country, "bonus": bonus, "checks": checks,
        "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(x) for x in tail["Open"]],
                  "h": [_r(x) for x in tail["High"]], "l": [_r(x) for x in tail["Low"]],
                  "c": [_r(x) for x in tail["Close"]], "v": [int(x) for x in tail["Volume"]],
                  "base_start": len(tail) - k},
    }
