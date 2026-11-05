"""
محرك قواعد نظرية عبدالله (رادار عبدالله).

الفكرة: موجة تدبيل فعلية ← تصحيح إلى دعم ← ثبات ← عودة طلب لموجة ثانية.
فرضية تداول نعمل عليها، وليست نظرية مثبتة النجاح.

الشروط:
 1) تدبيل سابق قريب ومثبت (قاع وتاريخه ← قمة وتاريخها، x2 أو أكثر). قفزة التقسيم العكسي لا تحسب.
 2) تصحيح إلى منطقة مهمة: قاعدة الانطلاقة، أو مقاومة مخترقة تتحول دعم، أو رف تماسك داخل الموجة.
 3) ثبات عند المنطقة: توقف القيعان الأدنى عدة جلسات (+ جفاف الفوليوم قرينة مساعدة).
 4) طلب جديد: استرداد مقاومة التماسك مع تحسن الحجم، والاحتفاظ بها (ملامسة المقاومة لا تكفي).
 5) إبطال واضح: مستوى يحدد مسبقاً ولا يتغير كلما هبط السهم.

الفريمات: اليومي للموجة والتصحيح، 4 ساعات لبنية الدعم والاسترداد، الساعة لعودة الطلب.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# أدوات عامة
# ----------------------------------------------------------------------------

def pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def fp(p):
    if p is None:
        return "-"
    return f"{p:.3f}" if p < 1 else f"{p:.2f}"


def _r(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def pivots(series: pd.Series, window: int, kind: str) -> list[int]:
    v = series.values
    out = []
    for i in range(window, len(v) - window):
        seg = v[i - window: i + window + 1]
        if (kind == "low" and v[i] == seg.min()) or (kind == "high" and v[i] == seg.max()):
            out.append(i)
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _d(df, i):
    return str(df.index[i])[:10]


# ----------------------------------------------------------------------------
# 1) موجة التدبيل
# ----------------------------------------------------------------------------

def split_days(df: pd.DataFrame) -> set[int]:
    """مواقع أيام التقسيم العكسي (عشان نستبعد القفزة الحسابية)."""
    if "Stock Splits" not in df.columns:
        return set()
    s = df["Stock Splits"].fillna(0).values
    return {i for i, v in enumerate(s) if 0 < v < 1}


def find_wave(df: pd.DataFrame, A: dict) -> dict | None:
    """
    أحدث موجة تدبيل: قمة (pivot) خلال آخر peak_within جلسة، وقاعها = أدنى سعر خلال
    wave_max_len جلسة قبل القمة. لازم القمة/القاع >= min_wave_mult.
    """
    n = len(df)
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    sdays = split_days(df)
    start = max(0, n - A["peak_within_sessions"])
    best = None
    for p in pivots(df["High"], 3, "high") + [int(np.argmax(hi[-4:])) + n - 4]:
        if p < start or p >= n:
            continue
        a = max(0, p - A["wave_max_len"])
        if p - a < 2:
            continue
        li = a + int(np.argmin(lo[a:p]))
        low, peak = float(lo[li]), float(hi[p])
        if low <= 0:
            continue
        mult = peak / low
        if mult < A["min_wave_mult"]:
            continue
        # استبعاد قفزة التقسيم: إذا الموجة فيها يوم تقسيم وقفزته هي اللي صنعت التدبيل
        split_inside = [i for i in sdays if li < i <= p]
        if split_inside:
            jump = max(cl[i] / cl[i - 1] for i in split_inside if i > 0)
            if peak / jump / low < A["min_wave_mult"]:
                continue
        # لازم السعر صحح بعد القمة (القمة ما تكون اليوم)، والقمة ما انكسرت بعدها
        if p >= n - 2:
            continue
        if float(hi[p + 1:].max()) > peak * 1.02 or float(cl[-1]) >= peak * 0.95:
            continue
        cand = {"low_i": li, "peak_i": p, "low": low, "peak": peak, "mult": mult,
                "low_date": _d(df, li), "peak_date": _d(df, p), "split_inside": bool(split_inside)}
        if best is None or p > best["peak_i"] or (p == best["peak_i"] and mult > best["mult"]):
            best = cand
    return best


# ----------------------------------------------------------------------------
# 2) منطقة الدعم بعد التصحيح
# ----------------------------------------------------------------------------

def support_zones(df: pd.DataFrame, w: dict) -> list[dict]:
    """مرشحات الدعم: قاعدة الانطلاقة، مقاومة قديمة اخترقتها الموجة، رفوف تماسك داخل الموجة."""
    li, p = w["low_i"], w["peak_i"]
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    zones = []
    # (أ) قاعدة الانطلاقة: من قاع الموجة إلى قمة التماسك قبل الانفجار
    j = li
    base_top = float(hi[li])
    while j + 1 < p and cl[j + 1] <= w["low"] * 1.25:
        j += 1
        base_top = max(base_top, float(hi[j]))
    base_top = min(base_top, w["low"] * 1.25)
    zones.append({"kind": "قاعدة الانطلاقة", "low": w["low"], "high": max(base_top, w["low"] * 1.04)})
    # (ب) مقاومات قديمة قبل الموجة كسرتها الموجة
    pre = df.iloc[max(0, li - 150):li]
    for i in pivots(pre["High"], 3, "high"):
        lvl = float(pre["High"].iloc[i])
        if w["low"] * 1.15 < lvl < w["peak"] * 0.85:
            zones.append({"kind": "مقاومة مخترقة → دعم", "low": lvl * 0.97, "high": lvl * 1.03})
    # (ج) رفوف تماسك داخل الموجة
    seg = df.iloc[li:p + 1]
    for i in pivots(seg["Low"], 2, "low"):
        lvl = float(seg["Low"].iloc[i])
        if w["low"] * 1.15 < lvl < w["peak"] * 0.85:
            zones.append({"kind": "رف تماسك داخل الموجة", "low": lvl * 0.98, "high": lvl * 1.04})
    zones.sort(key=lambda z: z["high"])
    # دمج المتداخل
    merged = []
    for z in zones:
        if merged and z["low"] <= merged[-1]["high"]:
            m = merged[-1]
            m["high"] = max(m["high"], z["high"])
            if z["kind"] not in m["kind"]:
                m["kind"] += " + " + z["kind"]
        else:
            merged.append(dict(z))
    return merged


def pick_zone(zones: list[dict], price: float, post_low: float) -> dict | None:
    """المنطقة الأقرب اللي السعر عندها أو فوقها (وما انكسرت بإغلاقات بعد القمة)."""
    cands = [z for z in zones if z["low"] <= price * 1.03 and post_low >= z["low"] * 0.93]
    if cands:
        return max(cands, key=lambda z: z["high"])
    return None


# ----------------------------------------------------------------------------
# التحليل الكامل
# ----------------------------------------------------------------------------

class C:
    def __init__(self, key, title, weight):
        self.key, self.title, self.weight = key, title, weight
        self.got, self.ok, self.detail, self.remaining, self.progress = 0.0, False, "", "", 0.0

    def d(self):
        return {"key": self.key, "title": self.title, "weight": self.weight, "got": round(self.got, 1),
                "ok": self.ok, "detail": self.detail, "remaining": self.remaining,
                "progress": round(self.progress, 2)}


def analyze(df: pd.DataFrame, info: dict | None, cfg: dict, state: dict | None = None) -> dict | None:
    """state = مستوى الإبطال المحفوظ من مسح سابق (عشان ما يتغير الدعم كلما هبط السهم)."""
    info = info or {}
    A = cfg["abdullah"]
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df[df["Close"] > 0]
    if len(df) < 60:
        return None
    df["Volume"] = df["Volume"].fillna(0)
    n = len(df)
    price = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])

    w = find_wave(df, A)
    if not w:
        return None
    li, p = w["low_i"], w["peak_i"]
    after = df.iloc[p + 1:]
    post_low = float(after["Low"].min()) if len(after) else price
    retrace = (w["peak"] - price) / (w["peak"] - w["low"]) * 100 if w["peak"] > w["low"] else None
    sessions_since_peak = n - 1 - p

    zones = support_zones(df, w)
    zone = pick_zone(zones, price, post_low)

    # --- الإبطال: يُحدد مرة وحدة ويبقى ثابت لنفس الموجة
    fixed = state if (state and state.get("peak_date") == w["peak_date"]) else None
    if fixed:
        inval = fixed["invalidation"]
        zone_fixed = {"kind": fixed.get("zone_kind", ""), "low": fixed["zone_low"], "high": fixed["zone_high"]}
        zone = zone_fixed
    else:
        inval = zone["low"] * (1 - A["invalidation_below_zone_pct"] / 100) if zone else None
    # ملغي إذا أغلق تحت الإبطال الحين، أو انكسر في مسح سابق لنفس الموجة
    invalidated = bool(inval and (price < inval or (fixed and fixed.get("broken"))))

    checks: list[C] = []

    # 1) التدبيل
    c = C("wave", "تدبيل سابق قريب ومثبت", 25)
    c.ok, c.progress = True, 1
    rec = sessions_since_peak <= A["recent_sessions"]
    c.got = 25 if rec else 18
    c.detail = (f"من {fp(w['low'])} ({w['low_date']}) إلى {fp(w['peak'])} ({w['peak_date']}) = x{w['mult']:.2f} "
                f"(+{(w['mult'] - 1) * 100:.0f}%) · قبل {sessions_since_peak} جلسة")
    if not rec:
        c.remaining = "الموجة صارت قديمة نسبياً"
    checks.append(c)

    # 2) التصحيح إلى منطقة مهمة
    c = C("zone", "تصحيح إلى منطقة دعم مهمة", 20)
    dist = pct(price, zone["high"]) if zone else None
    low10 = float(df["Low"].tail(10).min())
    touched = bool(zone and low10 <= zone["high"] * (1 + A["zone_near_pct"] / 100) and low10 >= zone["low"] * 0.93)
    if zone and price <= zone["high"] * (1 + A["zone_near_pct"] / 100):
        c.ok, c.got, c.progress = True, 20, 1
        c.detail = f"عند {zone['kind']} {fp(zone['low'])} – {fp(zone['high'])} · التصحيح {retrace:.0f}% من الموجة"
    elif touched and price > zone["high"] * (1 + A["max_bounce_pct"] / 100):
        c.got, c.progress = 8, 0.4
        c.detail = (f"لمس {zone['kind']} {fp(zone['low'])} – {fp(zone['high'])} وارتد بقوة "
                    f"(+{pct(price, low10):.0f}% من {fp(low10)}) — الحركة بدأت")
        c.remaining = "لا تطارد — انتظر إعادة اختبار المنطقة أو تماسك جديد"
    elif touched:
        c.ok, c.got, c.progress = True, 20, 1
        c.detail = (f"صحح إلى {zone['kind']} {fp(zone['low'])} – {fp(zone['high'])} (أدنى سعر {fp(low10)}) "
                    f"وارتد منها · التصحيح {retrace:.0f}%")
    elif zone:
        c.progress = max(0, 1 - (dist - A["zone_near_pct"]) / 40)
        c.got = 20 * c.progress * 0.5
        c.detail = f"السعر فوق {zone['kind']} ({fp(zone['low'])} – {fp(zone['high'])}) · التصحيح {retrace:.0f}%"
        c.remaining = f"يحتاج ينزل {abs(pct(zone['high'], price)):.1f}% ليوصل المنطقة {fp(zone['low'])} – {fp(zone['high'])}"
    else:
        c.detail = "كسر كل مناطق الدعم المحددة"
        c.remaining = "ما فيه منطقة دعم صالحة تحت السعر"
    checks.append(c)

    # 3) الثبات عند المنطقة
    c = C("hold", "ثبات عند المنطقة (توقف القيعان الأدنى)", 20)
    k = A["hold_sessions"]
    last = df.tail(k)
    prevk = df.iloc[-(2 * k):-k]
    no_lower = float(last["Low"].min()) >= float(prevk["Low"].min()) * 0.99 if len(prevk) else False
    in_zone_days = int(((df["Low"].tail(15) <= (zone["high"] * 1.03 if zone else 0)) &
                        (df["Close"].tail(15) >= (zone["low"] * 0.98 if zone else 1e9))).sum()) if zone else 0
    wave_vol = float(df["Volume"].iloc[li:p + 1].mean())
    corr_vol = float(after["Volume"].mean()) if len(after) else wave_vol
    vol_dry = corr_vol < wave_vol * A["vol_dry_ratio"]
    if no_lower and in_zone_days >= 3:
        c.ok, c.got, c.progress = True, 15 + (5 if vol_dry else 0), 1
        c.detail = f"ما سجل قاع أدنى آخر {k} جلسات · {in_zone_days} جلسات عند المنطقة"
    else:
        c.progress = 0.5 if no_lower else (0.3 if in_zone_days else 0)
        c.got = 8 if no_lower else (4 if in_zone_days else 0)
        c.detail = "لا يزال يسجل قيعان أدنى" if not no_lower else f"ثابت لكن {in_zone_days} جلسة فقط عند المنطقة"
        c.remaining = f"يحتاج يثبت {k} جلسات عند المنطقة بدون قيعان أدنى"
    c.detail += f" · فوليوم التصحيح {corr_vol / wave_vol * 100:.0f}% من فوليوم الموجة" + (" (جف ✓ قرينة مساعدة)" if vol_dry else "")
    checks.append(c)

    # 4) طلب جديد: استرداد مقاومة التماسك + حجم + احتفاظ
    c = C("demand", "ظهور طلب جديد (استرداد مقاومة + حجم + احتفاظ)", 25)
    cons = df.tail(A["consolidation_sessions"] + 3).iloc[:-3] if n > 20 else df.tail(10)
    res = float(cons["High"].max())
    avg_vol = float(df["Volume"].iloc[-31:-1].mean())
    closes = df["Close"].tail(3).values
    vols = df["Volume"].tail(3).values
    broke_i = next((i for i in range(3) if closes[i] > res), None)
    if broke_i is not None:
        held = all(x > res for x in closes[broke_i:])
        vol_ok = vols[broke_i] >= avg_vol * A["demand_vol_mult"]
        held_n = 3 - broke_i
        if held and vol_ok and held_n >= 2:
            c.ok, c.got, c.progress = True, 25, 1
            c.detail = f"استرد {fp(res)} بحجم {vols[broke_i] / avg_vol:.1f}x وثبت {held_n} جلسات فوقه"
        elif held:
            c.got, c.progress = 15, 0.6
            c.detail = f"استرد {fp(res)}" + (f" بحجم {vols[broke_i] / avg_vol:.1f}x" if vol_ok else " لكن الحجم ضعيف")
            c.remaining = "يحتاج احتفاظ جلستين فوق المستوى" + ("" if vol_ok else " مع تحسن الحجم")
        else:
            c.got, c.progress = 5, 0.2
            c.detail = f"لامس/تجاوز {fp(res)} ثم رجع تحته — الملامسة لا تكفي"
            c.remaining = f"يحتاج إغلاق واحتفاظ فوق {fp(res)}"
    else:
        c.progress = max(0, 1 - pct(res, price) / 25) * 0.4
        c.got = 0
        c.detail = f"مقاومة التماسك {fp(res)}"
        c.remaining = f"باقي {pct(res, price):.1f}% لاسترداد {fp(res)} مع حجم أعلى من المتوسط"
    checks.append(c)

    # 5) الإبطال
    c = C("inval", "مستوى إبطال واضح ومحترم", 10)
    if inval and not invalidated:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = f"الإبطال عند {fp(inval)} (السعر فوقه بـ {pct(price, inval):.1f}%)" + (" · محدد من مسح سابق" if fixed else "")
    elif inval:
        c.detail = f"أغلق تحت مستوى الإبطال {fp(inval)} — النموذج ضعيف/ملغي"
        c.remaining = "ينتظر موجة جديدة"
    else:
        c.detail = "ما أمكن تحديد مستوى إبطال"
    checks.append(c)

    score = sum(x.got for x in checks)
    ok = {x.key: x.ok for x in checks}
    if invalidated or not zone:
        status = "invalid"
    elif ok["zone"] and ok["hold"] and ok["demand"] and ok["inval"]:
        status = "ready"
    elif ok["zone"] and ok["inval"] and (ok["hold"] or checks[3].got >= 15):
        status = "near"
    else:
        status = "radar"

    # الأهداف: قمة الموجة، ونصف الطريق
    t_mid = (res + w["peak"]) / 2 if res < w["peak"] else None

    # السيولة
    dollar_vol = avg_vol * price

    tail = df.tail(120)
    chart = {"d": [str(i)[:10] for i in tail.index],
             "o": [_r(x) for x in tail["Open"]], "h": [_r(x) for x in tail["High"]],
             "l": [_r(x) for x in tail["Low"]], "c": [_r(x) for x in tail["Close"]],
             "v": [int(x) for x in tail["Volume"]],
             "wave_low_i": li - (n - len(tail)), "wave_peak_i": p - (n - len(tail))}

    r14 = rsi(df["Close"]).iloc[-1]
    return {
        "price": _r(price), "change_pct": _r(pct(price, prev), 2), "date": str(df.index[-1])[:10],
        "volume": int(df["Volume"].iloc[-1]), "avg_volume": int(avg_vol), "dollar_volume": int(dollar_vol),
        "rsi": _r(r14, 1),
        "wave": {"low": _r(w["low"]), "low_date": w["low_date"], "peak": _r(w["peak"]),
                 "peak_date": w["peak_date"], "mult": _r(w["mult"], 2), "sessions_since_peak": sessions_since_peak,
                 "retrace_pct": _r(retrace, 1), "split_inside": w["split_inside"]},
        "zone": {"kind": zone["kind"], "low": _r(zone["low"]), "high": _r(zone["high"])} if zone else None,
        "zones": [{"kind": z["kind"], "low": _r(z["low"]), "high": _r(z["high"])} for z in zones],
        "invalidation": _r(inval), "invalidated": invalidated, "fixed_from_state": bool(fixed),
        "resistance": _r(res), "targets": [_r(x) for x in (t_mid, w["peak"]) if x],
        "vol_dry": vol_dry, "corr_vol_ratio": _r(corr_vol / wave_vol if wave_vol else None, 2),
        "score": round(score, 1), "status": status, "checks": [x.d() for x in checks],
        "chart": chart,
    }


# ----------------------------------------------------------------------------
# فحص 4 ساعات و ساعة (للمرشحين)
# ----------------------------------------------------------------------------

def intraday_check(h1: pd.DataFrame, zone: dict | None, res: float | None) -> dict:
    out = {}
    if h1 is None or h1.empty:
        return out
    h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                "Close": "last", "Volume": "sum"}).dropna()
    if len(h4) >= 12:
        lows = h4["Low"].tail(12).values
        # بنية الدعم: هل آخر 6 شموع 4س بدون قاع أدنى من الـ 6 اللي قبلها؟
        out["h4_no_lower_low"] = bool(lows[-6:].min() >= lows[:6].min() * 0.99)
        if zone:
            out["h4_closes_in_zone"] = int(((h4["Close"].tail(12) >= zone["low"]) &
                                            (h4["Close"].tail(12) <= zone["high"] * 1.05)).sum())
        if res:
            out["h4_reclaim"] = bool(h4["Close"].iloc[-1] > res)
    if len(h1) >= 20 and res:
        last = h1.tail(7)  # تقريباً جلسة
        avgv = float(h1["Volume"].tail(70).mean())
        out["h1_above_res"] = bool(last["Close"].iloc[-1] > res)
        out["h1_vol_ratio"] = round(float(last["Volume"].mean()) / avgv, 2) if avgv else None
    return out
