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
        if float(hi[p + 1:].max()) > peak * 1.02 or float(cl[-1]) >= peak * 0.85:
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


def consolidation(df: pd.DataFrame, start: int, max_range_pct: float) -> dict | None:
    """
    أطول نافذة أخيرة (بعد القمة) السعر فيها متماسك: أعلى سعر ÷ أدنى سعر <= 1 + max_range.
    ترجع: طولها، الدعم (أدنى سعر)، السقف، وعدد مرات اختبار الدعم.
    """
    n = len(df)
    hi, lo = df["High"].values, df["Low"].values
    best = None
    for k in range(3, n - start):
        a = n - k
        h, l = float(hi[a:].max()), float(lo[a:].min())
        if l <= 0 or h / l > 1 + max_range_pct / 100:
            break
        best = (k, l, h)
    if not best:
        return None
    k, sup, top = best
    a = n - k
    # اختبارات الدعم: كل نزول للدعم (ضمن 4%) بعد ارتداد واضح عنه (القمة تبعد عن الدعم)
    band = sup * 1.04
    bounce = sup * (1 + max(0.04, min(0.08, (top / sup - 1) / 2)))
    touches, dates, armed = 0, [], True
    for i in range(a, n):
        if armed and lo[i] <= band:
            touches += 1
            dates.append(str(df.index[i])[:10])
            armed = False
        if not armed and hi[i] >= bounce and lo[i] > band:
            armed = True
    return {"len": k, "support": sup, "top": top, "range_pct": (top / sup - 1) * 100,
            "touches": touches, "touch_dates": dates, "start_i": a}


def analyze(df: pd.DataFrame, info: dict | None, cfg: dict, state: dict | None = None) -> dict | None:
    """
    state = الدعم ومستوى الإبطال المحفوظ من أول مرة ظهر فيها السهم (ما يتغير كلما هبط).
    يرجع None إذا ما فيه موجة تدبيل. الحالة "moved" = طلع/بدأ يطلع، "invalid" = كسر الدعم.
    """
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
    sessions_since_peak = n - 1 - p
    wave_len = p - li

    cons = consolidation(df, p + 1, A["consolidation_max_range_pct"])
    zones = support_zones(df, w)
    sup = cons["support"] if cons else float(after["Low"].min())
    retrace = (w["peak"] - sup) / (w["peak"] - w["low"]) * 100 if w["peak"] > w["low"] else 0
    zone_match = next((z for z in zones if z["low"] * 0.94 <= sup <= z["high"] * 1.06), None)

    # الإبطال: يتحدد أول مرة ويبقى ثابت لنفس الموجة
    fixed = state if (state and state.get("peak_date") == w["peak_date"]) else None
    if fixed:
        inval = fixed["invalidation"]
    else:
        inval = sup * (1 - A["invalidation_below_support_pct"] / 100)
    # كسر الدعم: السعر نزل تحت قاع التماسك السابق (قبل آخر 5 جلسات) بوضوح
    prior = after.iloc[:-5] if len(after) > 10 else None
    prior_sup = float(prior["Low"].tail(20).min()) if prior is not None and len(prior) else None
    broke_prior = bool(prior_sup and price < prior_sup * (1 - A["invalidation_below_support_pct"] / 100))
    broken = price < inval or broke_prior or bool(fixed and fixed.get("broken"))

    # طلع؟ (اخترق سقف التماسك أو ارتد بقوة من الدعم)
    prior_top = float(df["High"].iloc[-(cons["len"]):-2].max()) if cons and cons["len"] > 4 else None
    moved = price > sup * (1 + A["max_bounce_pct"] / 100) or \
        bool(prior_top and price > prior_top * (1 + A["breakout_pct"] / 100))

    checks: list[C] = []

    # 1) التدبيل خلال شهر-شهرين
    c = C("wave", "تدبيل سابق خلال شهر إلى شهرين", 20)
    c.ok, c.progress, c.got = True, 1, 20
    c.detail = (f"من {fp(w['low'])} ({w['low_date']}) إلى {fp(w['peak'])} ({w['peak_date']}) "
                f"= x{w['mult']:.2f} خلال {wave_len} جلسة · القمة قبل {sessions_since_peak} جلسة")
    checks.append(c)

    # 2) تصحيح فعلي
    c = C("correction", "تصحيح بعد الموجة", 10)
    if retrace >= A["min_retrace_pct"]:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = f"صحح {retrace:.0f}% من الموجة (من {fp(w['peak'])} إلى {fp(sup)})"
    else:
        c.progress = retrace / A["min_retrace_pct"]
        c.got = 10 * c.progress
        c.detail = f"صحح {retrace:.0f}% فقط"
        c.remaining = f"التصحيح خفيف — غالباً يحتاج نزول لمنطقة أقوى"
    checks.append(c)

    # 3) التماسك (عدد الجلسات)
    c = C("consol", "تماسك على الدعم (عدة جلسات)", 20)
    L = cons["len"] if cons else 0
    need = A["consolidation_min_sessions"]
    if L >= need:
        c.ok, c.got, c.progress = True, 20, 1
        c.detail = f"متماسك {L} جلسة بين {fp(sup)} و {fp(cons['top'])} (مدى {cons['range_pct']:.0f}%)"
    else:
        c.progress = L / need
        c.got = 20 * c.progress
        c.detail = f"متماسك {L} جلسة فقط" if L else "لا يزال ينزل — ما بدأ تماسك"
        c.remaining = f"باقي {need - L} جلسات تماسك تقريباً"
    checks.append(c)

    # 4) اختبار الدعم مرات عديدة
    c = C("touches", "اختبر الدعم مرات عديدة وصمد", 25)
    t = cons["touches"] if cons else 0
    need_t = A["min_touches"]
    if t >= need_t:
        c.ok, c.got, c.progress = True, 25, 1
        c.detail = f"اختبر الدعم {fp(sup)} {t} مرات ({'، '.join(cons['touch_dates'][-5:])})"
    else:
        c.progress = t / need_t
        c.got = 25 * c.progress * 0.8
        c.detail = f"اختبر الدعم {t} مرة" if t else "ما اختبر الدعم بعد"
        c.remaining = f"يحتاج اختبار الدعم {need_t - t} مرة إضافية ويصمد"
    checks.append(c)

    # 5) تماسك ضيق
    c = C("tight", "مدى التماسك ضيق", 10)
    rp = cons["range_pct"] if cons else None
    if rp is not None and rp <= A["tight_range_pct"]:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = f"المدى {rp:.0f}% — تماسك ضيق"
    elif rp is not None:
        c.progress = 0.5
        c.got = 5
        c.detail = f"المدى {rp:.0f}% — تماسك واسع نوعاً ما"
    else:
        c.detail = "-"
    checks.append(c)

    # 6) الدعم منطقة مهمة من الشارت
    c = C("zone", "الدعم عند منطقة مهمة (قاعدة / مقاومة سابقة / رف)", 10)
    if zone_match:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = f"يتوافق مع {zone_match['kind']} {fp(zone_match['low'])} – {fp(zone_match['high'])}"
    else:
        c.detail = "الدعم ما يطابق قاعدة أو مقاومة سابقة واضحة"
    checks.append(c)

    # 7) جفاف الفوليوم (قرينة مساعدة)
    c = C("vol", "الفوليوم جف أثناء التصحيح (قرينة مساعدة)", 5)
    wave_vol = float(df["Volume"].iloc[li:p + 1].mean()) or 1
    cons_vol = float(df["Volume"].tail(max(L, 5)).mean())
    vr = cons_vol / wave_vol
    if vr < A["vol_dry_ratio"]:
        c.ok, c.got, c.progress = True, 5, 1
    c.detail = f"فوليوم التماسك {vr * 100:.0f}% من فوليوم الموجة"
    checks.append(c)

    score = sum(x.got for x in checks)
    ok = {x.key: x.ok for x in checks}
    if broken:
        status = "invalid"
    elif moved:
        status = "moved"
    elif ok["consol"] and ok["touches"] and ok["correction"]:
        status = "ready"
    elif L >= 4 and t >= 2:
        status = "near"
    else:
        status = "radar"

    remaining_total = [x.remaining for x in checks if x.remaining]
    entry = [sup, sup * (1 + A["entry_above_support_pct"] / 100)]
    targets = [x for x in ((cons["top"] if cons else None), (sup + w["peak"]) / 2, w["peak"]) if x and x > price]

    avg_vol = float(df["Volume"].iloc[-31:-1].mean())
    tail = df.tail(120)
    off = n - len(tail)
    chart = {"d": [str(i)[:10] for i in tail.index],
             "o": [_r(x) for x in tail["Open"]], "h": [_r(x) for x in tail["High"]],
             "l": [_r(x) for x in tail["Low"]], "c": [_r(x) for x in tail["Close"]],
             "v": [int(x) for x in tail["Volume"]],
             "wave_low_i": li - off, "wave_peak_i": p - off,
             "cons_start_i": (cons["start_i"] - off) if cons else None}

    return {
        "price": _r(price), "change_pct": _r(pct(price, prev), 2), "date": str(df.index[-1])[:10],
        "volume": int(df["Volume"].iloc[-1]), "avg_volume": int(avg_vol), "dollar_volume": int(avg_vol * price),
        "rsi": _r(rsi(df["Close"]).iloc[-1], 1),
        "wave": {"low": _r(w["low"]), "low_date": w["low_date"], "peak": _r(w["peak"]),
                 "peak_date": w["peak_date"], "mult": _r(w["mult"], 2), "len": wave_len,
                 "sessions_since_peak": sessions_since_peak, "retrace_pct": _r(retrace, 1),
                 "split_inside": w["split_inside"]},
        "support": _r(sup), "support_zone": zone_match["kind"] if zone_match else None,
        "consolidation": {"len": L, "top": _r(cons["top"]) if cons else None,
                          "range_pct": _r(rp, 1), "touches": t,
                          "touch_dates": cons["touch_dates"] if cons else []},
        "dist_support_pct": _r(pct(price, sup), 1),
        "entry": [_r(x) for x in entry], "invalidation": _r(inval), "invalidated": broken,
        "fixed_from_state": bool(fixed), "targets": [_r(x) for x in targets],
        "zones": [{"kind": z["kind"], "low": _r(z["low"]), "high": _r(z["high"])} for z in zones],
        "vol_ratio": _r(vr, 2), "remaining": remaining_total,
        "score": round(score, 1), "status": status, "checks": [x.d() for x in checks],
        "chart": chart,
    }


# ----------------------------------------------------------------------------
# فحص 4 ساعات و ساعة (للمرشحين)
# ----------------------------------------------------------------------------

def intraday_check(h1: pd.DataFrame, support: float | None, top: float | None) -> dict:
    out = {}
    if h1 is None or h1.empty:
        return out
    h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                "Close": "last", "Volume": "sum"}).dropna()
    if len(h4) >= 12:
        lows = h4["Low"].tail(12).values
        # بنية الدعم: هل آخر 6 شموع 4س بدون قاع أدنى من الـ 6 اللي قبلها؟
        out["h4_no_lower_low"] = bool(lows[-6:].min() >= lows[:6].min() * 0.99)
        if support:
            # كم مرة لمست شموع 4 ساعات الدعم (ضمن 4%) آخر 60 شمعة
            lo = h4["Low"].tail(60).values
            tc, inside = 0, False
            for v in lo:
                if v <= support * 1.04:
                    if not inside:
                        tc += 1
                    inside = True
                elif v > support * 1.06:
                    inside = False
            out["h4_touches"] = tc
            out["h4_close_above_support"] = bool(h4["Close"].iloc[-1] >= support)
    if len(h1) >= 20:
        last = h1.tail(7)  # تقريباً جلسة
        avgv = float(h1["Volume"].tail(70).mean())
        out["h1_vol_ratio"] = round(float(last["Volume"].mean()) / avgv, 2) if avgv else None
        if top:
            out["h1_dist_top_pct"] = round((top / float(last["Close"].iloc[-1]) - 1) * 100, 1)
    return out
