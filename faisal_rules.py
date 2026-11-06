"""
محرك قواعد طريقة فيصل.

كل دالة هنا تأخذ بيانات يومية (Open/High/Low/Close/Volume) وتطلع قراءة رقمية،
ثم analyze() تجمعها وتحسب:
  - نسبة جاهزية سهم الارتكاز (0-100) + وش الناقص وكم باقي لكل شرط
  - نسبة جاهزية سهم الزخم (0-100)
  - السلوك (Former Runner، سحب سيولة، W، قروبات، تقسيم عكسي ...)
  - الخطة: الدعم، منطقة الطلب، الوقف، الأهداف الأربعة، المقاومات والشموع الساقطة فوق

لا يوجد هنا أي اتصال بالإنترنت؛ هذا الملف يمكن اختباره ببيانات تجريبية.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# مؤشرات فنية
# ----------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI بطريقة Wilder (نفس اللي في TradingView وويبل)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)
    return out


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def pivots(series: pd.Series, window: int, kind: str) -> list[int]:
    """مواقع القيعان/القمم المحلية (pivot) بنافذة window من كل جهة."""
    vals = series.values
    n = len(vals)
    out = []
    for i in range(window, n - window):
        seg = vals[i - window: i + window + 1]
        if kind == "low" and vals[i] == seg.min():
            out.append(i)
        elif kind == "high" and vals[i] == seg.max():
            out.append(i)
    return out


def _r(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def pct(a, b):
    """نسبة التغير من b إلى a."""
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def fmt_price(p):
    if p is None:
        return "-"
    return f"{p:.3f}" if p < 1 else f"{p:.2f}"


# ----------------------------------------------------------------------------
# مستويات: الدعم، المقاومات، الشموع الساقطة، الفجوات
# ----------------------------------------------------------------------------

def cluster_levels(prices: list[float], tol_pct: float = 4.0) -> list[dict]:
    """يجمع المستويات المتقاربة (ضمن tol%) في مستوى واحد ويعد عدد اللمسات."""
    clusters: list[dict] = []
    for p in sorted(prices):
        if clusters and p <= clusters[-1]["max"] * (1 + tol_pct / 100):
            c = clusters[-1]
            c["vals"].append(p)
            c["max"] = max(c["max"], p)
        else:
            clusters.append({"vals": [p], "max": p})
    out = []
    for c in clusters:
        out.append({"level": float(np.mean(c["vals"])), "low": min(c["vals"]),
                    "touches": len(c["vals"])})
    return out


def find_support(df: pd.DataFrame, lookback: int = 120) -> dict:
    """
    الدعم الحقيقي = منطقة لمسها السعر أكثر من مرة ورفض الهبوط تحتها.
    نأخذ قيعان آخر lookback شمعة، نجمعها، ونختار أقرب منطقة تحت السعر (أو عنده).
    إذا السعر كسر كل الدعوم، الدعم = أدنى قاع حديث (قاع جديد).
    """
    d = df.tail(lookback)
    price = float(d["Close"].iloc[-1])
    low_idx = pivots(d["Low"], 3, "low")
    lows = [float(d["Low"].iloc[i]) for i in low_idx]
    recent_min = float(d["Low"].tail(60).min())
    lows.append(recent_min)
    levels = cluster_levels(lows, tol_pct=4.0)
    below = [l for l in levels if l["low"] <= price * 1.02]
    if not below:
        return {"level": recent_min, "touches": 1, "kind": "قاع جديد"}
    # فضّل الأقرب للسعر، ومع التساوي فضّل الأكثر لمسات
    multi = [l for l in below if l["touches"] >= 2]
    best_multi = max(multi, key=lambda l: l["level"]) if multi else None
    nearest = max(below, key=lambda l: l["level"])
    chosen = best_multi if best_multi and pct(price, best_multi["level"]) <= 25 else nearest
    # الدعم هو حد المنطقة السفلي تقريباً (قاعها) وليس متوسطها
    level = chosen["low"] if chosen["touches"] >= 2 else chosen["level"]
    return {"level": float(level), "touches": int(chosen["touches"]),
            "kind": "دعم مكرر" if chosen["touches"] >= 2 else "قاع"}


def falling_candles(df: pd.DataFrame, lookback: int = 250, min_body_pct: float = 7.0) -> list[dict]:
    """
    الشمعة الساقطة: شمعة حمراء كبيرة هبط منها السهم ولم يرجع لها
    -> مقاومة وهدف للمضارب. (كل شمعة ساقطة حمراء تقابلها شمعة صاعدة خضراء)
    """
    d = df.tail(lookback).reset_index(drop=False)
    out = []
    highs = d["High"].values
    for i in range(len(d) - 1):
        o, c = float(d["Open"].iloc[i]), float(d["Close"].iloc[i])
        if c >= o or o <= 0:
            continue
        body = (o - c) / o * 100
        if body < min_body_pct:
            continue
        after_max = float(highs[i + 1:].max()) if i + 1 < len(d) else 0
        if after_max < o:  # لم يرجع لرأس الشمعة
            out.append({"top": o, "tail": c, "body_pct": body,
                        "date": str(d.iloc[i, 0])[:10], "covered": after_max >= c})
    return out


def unfilled_gaps_down(df: pd.DataFrame, lookback: int = 250, min_gap_pct: float = 3.0) -> list[dict]:
    d = df.tail(lookback).reset_index(drop=False)
    out = []
    for i in range(1, len(d)):
        prev_low = float(d["Low"].iloc[i - 1])
        hi = float(d["High"].iloc[i])
        if hi < prev_low and pct(prev_low, hi) >= min_gap_pct:
            after_max = float(d["High"].iloc[i:].max())
            if after_max < prev_low:
                out.append({"from": max(hi, after_max), "to": prev_low,
                            "date": str(d.iloc[i, 0])[:10]})
    return out


def resistances_above(df: pd.DataFrame, price: float, lookback: int = 250) -> list[dict]:
    d = df.tail(lookback)
    hi_idx = pivots(d["High"], 4, "high")
    levels = []
    for i in hi_idx:
        h = float(d["High"].iloc[i])
        if h > price * 1.03:
            levels.append(h)
    # أضف رؤوس الشموع الساقطة وبدايات الفجوات
    for fc in falling_candles(df, lookback):
        if fc["top"] > price * 1.03:
            levels.append(fc["top"])
    for g in unfilled_gaps_down(df, lookback):
        if g["to"] > price * 1.03:
            levels.append(g["to"])
    cl = cluster_levels(levels, tol_pct=3.0)
    return [{"level": c["level"], "touches": c["touches"]} for c in cl]


# ----------------------------------------------------------------------------
# السلوك
# ----------------------------------------------------------------------------

def former_runner(df: pd.DataFrame, window: int = 10) -> dict:
    """هل السهم سبق انفجر +100% خلال window جلسة؟ (Former Runner)"""
    lows = df["Low"].rolling(window, min_periods=1).min()
    ratio = df["High"] / lows
    if ratio.empty:
        return {"max_pct": 0, "date": None}
    idx = ratio.idxmax()
    return {"max_pct": float((ratio.max() - 1) * 100), "date": str(idx)[:10]}


def reverse_splits(df: pd.DataFrame) -> list[dict]:
    if "Stock Splits" not in df.columns:
        return []
    s = df["Stock Splits"]
    s = s[(s > 0) & (s < 1)]
    out = []
    last_date = df.index[-1]
    for dt, v in s.items():
        out.append({"date": str(dt)[:10], "ratio": f"1:{round(1 / v)}",
                    "days_ago": int((last_date - dt).days)})
    return out


def below_dollar_streak(df: pd.DataFrame) -> int:
    """عدد الجلسات المتتالية اللي أغلق فيها السهم تحت 1$ (امتثال ناسداك: 30 جلسة -> مهلة 180 يوم)."""
    c = df["Close"].values[::-1]
    n = 0
    for v in c:
        if v < 1:
            n += 1
        else:
            break
    return n


def pump_candles(df: pd.DataFrame, look: int = 30, wick_pct: float = 15.0) -> int:
    """شموع القروبات: ذيل علوي طويل (يصعد ويرجع) مع فوليوم منفجر."""
    d = df.tail(look + 30)
    avgv = d["Volume"].rolling(20, min_periods=5).mean().shift(1)
    d = d.assign(avgv=avgv).tail(look)
    n = 0
    for _, r in d.iterrows():
        top = max(r["Open"], r["Close"])
        if r["Close"] <= 0:
            continue
        upper = (r["High"] - top) / r["Close"] * 100
        if upper >= wick_pct and r["avgv"] and r["Volume"] >= 3 * r["avgv"]:
            n += 1
    return n


def liquidity_sweep(df: pd.DataFrame, support: float, days: int = 5) -> dict | None:
    """سحب سيولة: كسر بسيط للدعم ثم استرداد سريع فوقه (والسعر لسه قريب من الدعم)."""
    d = df.tail(days)
    if float(d["Close"].iloc[-1]) > support * 1.15:
        return None
    for i in range(len(d)):
        lo = float(d["Low"].iloc[i])
        if lo < support * 0.995 and lo > support * 0.85:
            # هل رجع فوق الدعم (إغلاق نفس الشمعة أو بعدها)؟
            closes_after = d["Close"].iloc[i:]
            if (closes_after > support).any() and float(d["Close"].iloc[-1]) >= support:
                return {"sweep_low": lo, "date": str(d.index[i])[:10]}
    return None


def broke_support_and_stayed(df: pd.DataFrame, support: float, days: int = 3) -> bool:
    d = df.tail(days)
    return bool((d["Close"] < support * 0.98).all())


def higher_lows(df: pd.DataFrame, look: int = 40) -> bool:
    d = df.tail(look)
    idx = pivots(d["Low"], 2, "low")
    if len(idx) < 3:
        return False
    l = [float(d["Low"].iloc[i]) for i in idx[-3:]]
    return l[0] < l[1] < l[2]


def no_new_low(df: pd.DataFrame, sessions: int = 5) -> bool:
    """أدنى شمعة خلال 5 جلسات: هل كُسرت؟ (فشل الهبوط = ما فيه قاع جديد)"""
    if len(df) < sessions + 10:
        return False
    last = df.tail(sessions)
    prev = df.iloc[-(sessions + 15):-sessions]
    ref_low = float(prev["Low"].min())
    first_low = float(last["Low"].iloc[0])
    # القاع داخل آخر 5 جلسات ما انكسر بعد تكوينه
    min_idx = int(np.argmin(last["Low"].values))
    after = last["Low"].values[min_idx + 1:]
    held = len(after) >= 2 and after.min() > last["Low"].values[min_idx]
    return bool(held or float(last["Low"].min()) >= min(ref_low, first_low) * 0.995)


def positive_candle(df: pd.DataFrame, support: float, days: int = 3) -> str | None:
    """شمعة إيجابية عند الدعم: Hammer / Engulfing / ذيل سفلي طويل."""
    d = df.tail(days + 1)
    for i in range(len(d) - 1, 0, -1):
        o, h, l, c = (float(d[k].iloc[i]) for k in ("Open", "High", "Low", "Close"))
        po, pc = float(d["Open"].iloc[i - 1]), float(d["Close"].iloc[i - 1])
        near = l <= support * 1.08
        body = abs(c - o)
        rng = h - l
        if rng <= 0:
            continue
        lower = min(o, c) - l
        upper = h - max(o, c)
        if near and lower >= 2 * max(body, rng * 0.05) and upper <= max(body, rng * 0.15) * 1.2:
            return "Hammer / ذيل سفلي"
        if c > o and pc < po and c >= po and o <= pc and near:
            return "Engulfing ابتلاعية"
        if near and lower / rng >= 0.55 and c >= o:
            return "رفض كسر (ذيل سفلي)"
    return None


def volume_read(df: pd.DataFrame) -> dict:
    """فوليوم الصعود مقابل فوليوم الهبوط، وهل فوليوم البيع يتناقص."""
    # يوم صعود أو هبوط = مقارنة الإغلاق بإغلاق أمس (مو بافتتاح نفس اليوم:
    # سهم يفتح بفجوة +57% ويغلق +37% شمعته حمراء لكنه يوم صعود)
    d = df.tail(21).copy()
    d["up"] = d["Close"] >= d["Close"].shift(1)
    d = d.tail(20)
    last10, prev10 = d.tail(10), d.head(10)
    up_v = last10.loc[last10["up"], "Volume"].mean()
    dn_v = last10.loc[~last10["up"], "Volume"].mean()
    dn_prev = prev10.loc[~prev10["up"], "Volume"].mean()
    up_v = 0 if pd.isna(up_v) else float(up_v)
    dn_v = 0 if pd.isna(dn_v) else float(dn_v)
    dn_prev = 0 if pd.isna(dn_prev) else float(dn_prev)
    return {"up_vs_down": (up_v / dn_v) if dn_v else (2.0 if up_v else 1.0),
            "selling_drying": bool(dn_prev and dn_v < dn_prev * 0.85)}


def w_pattern(df: pd.DataFrame, look: int = 60) -> dict | None:
    """نموذج W / Double Bottom: قاعين متقاربين وبينهم قمة = خط العنق."""
    d = df.tail(look)
    idx = pivots(d["Low"], 3, "low")
    if len(idx) < 2:
        return None
    best = None
    for a in range(len(idx) - 1):
        for b in range(a + 1, len(idx)):
            i1, i2 = idx[a], idx[b]
            if i2 - i1 < 5:
                continue
            l1, l2 = float(d["Low"].iloc[i1]), float(d["Low"].iloc[i2])
            if not (l1 * 0.97 <= l2 <= l1 * 1.06):
                continue
            neck = float(d["High"].iloc[i1:i2 + 1].max())
            if neck < max(l1, l2) * 1.12:
                continue
            if float(d["Low"].iloc[i2:].min()) < min(l1, l2) * 0.97:
                continue  # انكسر القاع الثاني -> النموذج بطل
            cand = {"low1": l1, "low2": l2, "neck": neck,
                    "date2": str(d.index[i2])[:10], "_i2": i2}
            if best is None or i2 > best["_i2"]:
                best = cand
    if best:
        best.pop("_i2", None)
        price = float(d["Close"].iloc[-1])
        best["broken"] = price > best["neck"]
        best["to_neck_pct"] = pct(best["neck"], price)
    return best


# ----------------------------------------------------------------------------
# النتيجة
# ----------------------------------------------------------------------------

@dataclass
class Check:
    key: str
    title: str
    weight: float
    got: float = 0.0
    ok: bool = False
    detail: str = ""
    remaining: str = ""   # كم باقي
    progress: float = 0.0  # 0..1 قربه من التحقق

    def to_dict(self):
        return {"key": self.key, "title": self.title, "weight": self.weight,
                "got": round(self.got, 1), "ok": self.ok, "detail": self.detail,
                "remaining": self.remaining, "progress": round(self.progress, 2)}


def _closeness(value, target_low, target_high, far):
    """0..1 كم القيمة قريبة من النطاق [target_low,target_high]؛ far = المسافة اللي عندها 0."""
    if value is None:
        return 0.0
    if target_low <= value <= target_high:
        return 1.0
    dist = (target_low - value) if value < target_low else (value - target_high)
    return max(0.0, 1 - dist / far)


def analyze(df: pd.DataFrame, info: dict | None, cfg: dict) -> dict | None:
    """يحلل سهم واحد. df يومي مرتب زمنياً."""
    info = info or {}
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df[df["Close"] > 0]
    if len(df) < 60:
        return None
    df["Volume"] = df["Volume"].fillna(0)

    U, IR, MO, BH = cfg["universe"], cfg["irtikaz"], cfg["momentum"], cfg["behavior"]
    close = df["Close"]
    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    today = df.iloc[-1]

    # --- مؤشرات
    r = rsi(close)
    rsi_now = _r(r.iloc[-1], 1)
    rsi_min10 = _r(r.tail(10).min(), 1)
    ma = {n: close.rolling(n).mean() for n in (20, 30, 50, 200)}
    ma_now = {n: _r(ma[n].iloc[-1], 4) for n in ma}
    m_line, m_sig, m_hist = macd(close)
    hist = m_hist.tail(4).values
    macd_improving = bool(len(hist) == 4 and hist[-1] > hist[-2] > hist[-3])
    macd_cross = bool(m_line.iloc[-1] > m_sig.iloc[-1] and m_line.iloc[-4] <= m_sig.iloc[-4])

    avg_vol = float(df["Volume"].iloc[-31:-1].mean()) if len(df) > 31 else float(df["Volume"].mean())
    today_vol = float(today["Volume"])
    rel_vol = today_vol / avg_vol if avg_vol else None
    change_pct = pct(price, prev_close)
    gap_pct = pct(float(today["Open"]), prev_close)
    hi52 = float(df["High"].tail(252).max())
    lo52 = float(df["Low"].tail(252).min())
    drawdown = pct(price, hi52)

    # --- بيانات الشركة
    mcap = info.get("marketCap")
    flt = info.get("floatShares")
    if flt is not None and flt < 50_000:  # رقم غير منطقي من المصدر
        flt = None
    shares_out = info.get("sharesOutstanding")
    short_shares = info.get("sharesShort")
    short_pct_float = info.get("shortPercentOfFloat")
    if short_pct_float is not None and short_pct_float <= 1:
        short_pct_float *= 100
    short_value = short_shares * price if short_shares else None
    short_avail = info.get("shortAvailable")  # من ملف IB (يضيفه scanner.py)
    short_to_mcap = (short_value / mcap * 100) if (short_value and mcap) else None

    # --- مستويات
    sup = find_support(df)
    support = sup["level"]
    dist_support = pct(price, support)
    entry_lo = support * 1.0
    entry_hi = support * (1 + IR["entry_zone_above_support_pct"] / 100)
    # الوقف (قرار المستخدم): أقل سعر وصله السهم عند الدعم − 10% — عشان مسح السيولة ما يضربه
    near_lows = [float(x) for x in df["Low"].tail(20) if float(x) >= support * 0.85]
    base_low = min([support] + near_lows)
    stop = base_low * (1 - IR.get("stop_below_low_pct", 10) / 100)
    res = resistances_above(df, price)
    fcs = [f for f in falling_candles(df) if f["top"] > price]
    gaps = [g for g in unfilled_gaps_down(df) if g["to"] > price]

    # الأهداف: أول مقاومة = هدف أول، والهدف الرئيسي = أبعد شمعة ساقطة/فجوة (حد أقصى 3x)
    tops = sorted({round(x["level"], 4) for x in res})
    t1 = tops[0] if tops else None
    major_candidates = [f["top"] for f in fcs] + [g["to"] for g in gaps] + tops
    major_candidates = [x for x in major_candidates if x <= max(price, entry_hi) * 3]
    main_target = max(major_candidates) if major_candidates else (t1 or None)
    ref_entry = min(price, entry_hi) if dist_support is not None and dist_support <= 25 else price
    four_targets = []
    if main_target and main_target > ref_entry:
        move = main_target - ref_entry
        four_targets = [ref_entry + move * k / 4 for k in (1, 2, 3, 4)]
    upside_pct = pct(t1, ref_entry) if t1 else (pct(main_target, ref_entry) if main_target else None)
    rr = None
    if t1 and ref_entry > stop:
        rr = (t1 - ref_entry) / (ref_entry - stop)

    # --- سلوك
    fr = former_runner(df)
    rsplits = reverse_splits(df)
    last_rs = rsplits[-1] if rsplits else None
    pumps = pump_candles(df, wick_pct=BH["pump_upper_wick_pct"])
    med_range = float(((df["High"] - df["Low"]) / df["Close"]).tail(10).median() * 100)
    sweep = liquidity_sweep(df, support)
    broke_stay = broke_support_and_stayed(df, support)
    hl = higher_lows(df)
    held = no_new_low(df)
    pc = positive_candle(df, support, days=IR.get("positive_candle_days", 10))
    vol = volume_read(df)
    w = w_pattern(df)
    low20 = float(df["Low"].tail(20).min())
    up_from_low = pct(price, low20)
    below_1 = below_dollar_streak(df)
    reclaimed_ma20 = bool(ma_now[20] and price > ma_now[20] and (close.iloc[-6:-1] < ma[20].iloc[-6:-1]).any())
    under_all_ma = all(ma_now[n] and price < ma_now[n] for n in (20, 30, 50))
    ma50_200 = None
    if ma_now[50] and ma_now[200]:
        cross_recent = bool((ma[50].tail(10) > ma[200].tail(10)).iloc[-1] and
                            (ma[50].tail(10) <= ma[200].tail(10)).any())
        ma50_200 = {"golden_recent": cross_recent,
                    "gap_pct": pct(ma_now[50], ma_now[200])}

    # ========================================================================
    # فلتر الارتكاز (حسب الجدول في الدليل)
    # ========================================================================
    checks: list[Check] = []

    # 1) قرب الدعم الحقيقي
    c = Check("support", "قريب من دعم حقيقي / منطقة طلب", 20)
    if dist_support is not None:
        if broke_stay:
            c.detail = f"كسر الدعم {fmt_price(support)} وبقي تحته — هذا ضعف وليس سحب سيولة"
            c.remaining = "انتظر قاع جديد واضح أو استرداد الدعم"
        elif dist_support <= IR["support_near_pct"]:
            c.ok, c.got, c.progress = True, 20, 1
            c.detail = f"السعر فوق الدعم {fmt_price(support)} بـ {dist_support:.1f}% ({sup['kind']}، لمسات: {sup['touches']})"
        else:
            c.progress = _closeness(dist_support, 0, IR["support_near_pct"], 40)
            c.got = 10 if dist_support <= IR["support_watch_pct"] else 20 * c.progress * 0.4
            need = pct(entry_hi, price)
            c.detail = f"السعر بعيد عن الدعم {fmt_price(support)} بـ {dist_support:.1f}%"
            c.remaining = f"يحتاج ينزل {abs(need):.1f}% ليدخل منطقة الطلب {fmt_price(entry_lo)} – {fmt_price(entry_hi)}"
    checks.append(c)

    # 2) RSI 23-27
    c = Check("rsi", "RSI مضغوط (23-27)", 15)
    lo_, hi_ = IR["rsi_ideal_low"], IR["rsi_ideal_high"]
    if rsi_now is not None:
        if lo_ <= rsi_now <= hi_:
            c.ok, c.got, c.progress = True, 15, 1
            c.detail = f"RSI = {rsi_now} داخل منطقة الضغط"
        elif rsi_now < lo_:
            c.ok, c.got, c.progress = True, 13, 0.9
            c.detail = f"RSI = {rsi_now} تشبع بيعي عميق جداً (انتبه لاستمرار الهبوط)"
        elif rsi_min10 is not None and rsi_min10 <= 30 and rsi_now <= IR["rsi_reject_above"]:
            c.ok, c.got, c.progress = True, 11, 0.8
            c.detail = f"انضغط قبل أيام (أدنى RSI {rsi_min10}) والحين {rsi_now} — الضغط تحقق"
        else:
            c.progress = _closeness(rsi_now, lo_, hi_, 25)
            c.got = 15 * c.progress * 0.5
            c.detail = f"RSI = {rsi_now}"
            if rsi_now > IR["rsi_reject_above"]:
                c.detail += " — فوق 40: جزء من الارتداد حصل، فيصل ما يفضله"
            c.remaining = f"باقي {rsi_now - hi_:.1f} نقطة ليوصل منطقة 23-27"
    checks.append(c)

    # 3) فشل الهبوط / ثبات / سحب سيولة
    c = Check("hold", "فشل الهبوط: ثبات أو سحب سيولة ثم استرداد", 15)
    if sweep:
        c.ok, c.got, c.progress = True, 15, 1
        c.detail = f"سحب سيولة إلى {fmt_price(sweep['sweep_low'])} ({sweep['date']}) ثم رجع فوق الدعم ✓"
    elif held and not broke_stay:
        c.ok, c.got, c.progress = True, 12, 0.85
        c.detail = "ما كسر أدنى قاع خلال آخر 5 جلسات — ثبات"
    else:
        c.progress = 0.2
        c.detail = "لا يزال يصنع قيعان جديدة"
        c.remaining = "ينتظر ثبات 3-5 جلسات بدون قاع جديد، أو سحب سيولة واسترداد"
    checks.append(c)

    # 4) شمعة إيجابية
    c = Check("candle", "شمعة إيجابية عند الدعم", 10)
    if pc:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = pc
    else:
        c.detail = "لا توجد Hammer / Engulfing / ذيل سفلي قرب الدعم"
        c.remaining = "انتظر شمعة رفض كسر عند الدعم"
    checks.append(c)

    # 5) MACD
    c = Check("macd", "MACD يتحسن", 10)
    if macd_cross:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = "تقاطع إيجابي حديث"
    elif macd_improving:
        c.ok, c.got, c.progress = True, 8, 0.8
        c.detail = "الهيستوجرام يتحسن 3 أيام متتالية"
    else:
        c.progress = 0.3 if hist[-1] > hist[-2] else 0
        c.got = 3 if hist[-1] > hist[-2] else 0
        c.detail = "الهيستوجرام ما زال يضعف" if hist[-1] <= hist[-2] else "بداية تحسن (يوم واحد)"
        c.remaining = "يحتاج تحسن الهيستوجرام أو تقاطع إيجابي"
    checks.append(c)

    # 6) الفوليوم
    c = Check("volume", "الفوليوم يؤكد (بيع يجف / شراء أقوى)", 10)
    if vol["up_vs_down"] >= 1.2 or vol["selling_drying"]:
        c.ok, c.got, c.progress = True, 10 if vol["up_vs_down"] >= 1.2 else 7, 1
        parts = []
        if vol["up_vs_down"] >= 1.2:
            parts.append(f"فوليوم أيام الصعود {vol['up_vs_down']:.1f}x أيام الهبوط")
        if vol["selling_drying"]:
            parts.append("فوليوم البيع يتناقص")
        c.detail = " · ".join(parts)
    else:
        c.progress = min(1, vol["up_vs_down"] / 1.2) * 0.6
        c.got = 3 * c.progress
        c.detail = f"فوليوم الصعود/الهبوط = {vol['up_vs_down']:.2f}"
        c.remaining = "انتظر فوليوم أخضر أقوى من الأحمر"
    checks.append(c)

    # 7) المتوسطات
    c = Check("ma", "المتوسطات 20/30/50: ضغط ثم استعادة", 10)
    if reclaimed_ma20:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = f"استعاد MA20 ({fmt_price(ma_now[20])}) — بداية تحسن"
    elif under_all_ma:
        c.got, c.progress = 5, 0.5
        c.detail = "تحت كل المتوسطات = ضغط (مناسب للارتكاز بس يحتاج تأكيد)"
        c.remaining = f"باقي {pct(ma_now[20], price):.1f}% لاستعادة MA20 عند {fmt_price(ma_now[20])}"
    elif ma_now[20] and price > ma_now[20]:
        c.ok, c.got, c.progress = True, 7, 0.8
        c.detail = f"فوق MA20 ({fmt_price(ma_now[20])})"
    else:
        c.got, c.progress = 4, 0.4
        c.detail = "بين المتوسطات"
        if ma_now[20]:
            c.remaining = f"باقي {pct(ma_now[20], price):.1f}% لاستعادة MA20"
    checks.append(c)

    # 8) مساحة صعود
    c = Check("upside", "مساحة صعود / أهداف فوق (شموع ساقطة، فجوات)", 10)
    if upside_pct is not None and upside_pct >= IR["min_upside_pct"]:
        c.ok, c.got, c.progress = True, 10, 1
        c.detail = f"أول هدف {fmt_price(t1)} (+{upside_pct:.0f}%)"
    elif upside_pct is not None:
        c.progress = max(0, upside_pct / IR["min_upside_pct"])
        c.got = 10 * c.progress * 0.6
        c.detail = f"أول مقاومة قريبة {fmt_price(t1)} (+{upside_pct:.0f}%) — الدخول تحت مقاومة مباشرة غلط"
        c.remaining = f"يحتاج اختراق {fmt_price(t1)} وثبات"
    else:
        c.got, c.progress, c.ok = 6, 0.6, False
        c.detail = "لا توجد مقاومات واضحة فوق (قمم تاريخية؟)"
    checks.append(c)

    base = sum(x.got for x in checks)

    # ---- تعديلات السلوك (مكافآت / خصومات)
    adj = []
    if fr["max_pct"] >= BH["former_runner_pct"]:
        adj.append(("Former Runner", +5, f"سبق انفجر +{fr['max_pct']:.0f}% ({fr['date']})"))
    if flt and flt < 5e6:
        adj.append(("فلووت أقل من 5M", +3, f"فلووت {flt/1e6:.2f}M — خفيف وسريع (وخطر)"))
    if w and not w["broken"]:
        adj.append(("نموذج W", +3, f"قاعين {fmt_price(w['low1'])}/{fmt_price(w['low2'])} وخط عنق {fmt_price(w['neck'])}"))
    if w and w["broken"]:
        adj.append(("كسر خط عنق W", +4, f"فوق خط العنق {fmt_price(w['neck'])} — يحتاج ثبات"))
    if hl:
        adj.append(("قيعان أعلى", +3, "تجميع: قيعان أعلى"))
    if pumps >= 2:
        adj.append(("سلوك قروبات", -10, f"{pumps} شموع ذيل علوي طويل مع فوليوم منفجر آخر 30 يوم"))
    if up_from_low is not None and up_from_low >= IR["chase_from_low_pct"]:
        adj.append(("لا تطارد", -15, f"صعد {up_from_low:.0f}% من قاع آخر 20 يوم — انتظر Retest"))
    if short_avail is not None:
        # طريقة فيصل: الشورت = المتاح للاقتراض (IB)، مو Short Interest الرسمي القديم
        if short_avail > BH.get("max_short_shares", 50000):
            adj.append(("شورت متاح مرتفع", -8, f"الشورت المتاح {short_avail:,.0f} سهم"))
    elif short_pct_float is not None and short_pct_float >= BH["high_short_pct_float"]:
        adj.append(("شورت مرتفع", -8, f"الشورت {short_pct_float:.1f}% من الفلووت"))
    elif short_to_mcap is not None and short_to_mcap >= BH["high_short_value_to_mcap_pct"]:
        adj.append(("شورت مرتفع نسبة للقيمة", -6, f"قيمة الشورت {short_to_mcap:.1f}% من القيمة السوقية"))
    if last_rs and last_rs["days_ago"] <= 20:
        adj.append(("تقسيم عكسي حديث", -3,
                    f"تقسيم {last_rs['ratio']} قبل {last_rs['days_ago']} يوم — القاعدة: أعطه 20-50 يوم"))
    if broke_stay:
        adj.append(("كسر دعم", -10, "كسر الدعم وبقي تحته"))

    score = max(0.0, min(100.0, base + sum(a[1] for a in adj)))

    # مطابق = لازم الدعم + فشل الهبوط + RSI مضغوط (فيصل ما يفضل RSI فوق 40)
    core_ok = all(x.ok for x in checks if x.key in ("support", "hold", "rsi"))
    near_ok = (dist_support is not None and dist_support <= IR["support_watch_pct"]
               and not broke_stay and (rsi_now or 100) <= IR["rsi_reject_above"])
    if score >= 75 and core_ok:
        status = "ready"
    elif score >= 60 and near_ok:
        status = "near"
    elif score >= 40:
        status = "radar"
    else:
        status = "no"

    # جملة الجاهزية (بنفس أسلوب جدول فيصل: متى يكون جاهز؟)
    readiness = []
    if dist_support is not None and dist_support > IR["support_near_pct"]:
        readiness.append(f"نزول إلى منطقة الطلب {fmt_price(entry_lo)} – {fmt_price(entry_hi)} ثم ثبات")
    if t1:
        readiness.append(("أو " if readiness else "") + f"اختراق {fmt_price(t1)} بثبات ثم Retest ناجح")
    if not readiness:
        readiness.append("الشروط شبه مكتملة — راقب الثبات فوق الدعم وضع طلبات Limit")

    # أنواع السلوك (tags)
    tags = []
    if status in ("ready", "near") or (dist_support is not None and dist_support <= IR["support_watch_pct"]):
        tags.append("ارتكاز")
    if fr["max_pct"] >= BH["former_runner_pct"]:
        tags.append("Former Runner")
    if fcs or gaps:
        tags.append("Gap Fill")
    if w:
        tags.append("W")
    if sweep:
        tags.append("سحب سيولة")
    if pumps >= 2:
        tags.append("قروبات ⚠")
    if last_rs and last_rs["days_ago"] <= 120:
        tags.append("تقسيم عكسي")
    if below_1 >= 30:
        tags.append("تحت 1$ (امتثال)")
    if drawdown is not None and drawdown <= -70:
        tags.append("استنفد النزول")

    # ========================================================================
    # فلتر الزخم
    # ========================================================================
    m_checks: list[Check] = []

    def mc(key, title, w_, ok, got, detail, remaining=""):
        m_checks.append(Check(key, title, w_, got if ok else got, ok, detail, remaining,
                              1.0 if ok else max(0.0, got / w_ if w_ else 0)))

    rv = rel_vol or 0
    mc("rvol", "Relative Volume فوق 2 (الأفضل 3)", 20, rv >= MO["min_rel_volume"],
       20 if rv >= MO["strong_rel_volume"] else (15 if rv >= MO["min_rel_volume"] else 20 * min(1, rv / MO["min_rel_volume"]) * 0.5),
       f"RVOL = {rv:.1f}x", "" if rv >= MO["min_rel_volume"] else f"يحتاج فوليوم {MO['min_rel_volume']}x المتوسط")
    mc("tvol", "فوليوم اليوم فوق 200K (الأفضل 500K)", 15, today_vol >= MO["min_today_volume"],
       15 if today_vol >= MO["strong_today_volume"] else (10 if today_vol >= MO["min_today_volume"] else 15 * today_vol / MO["min_today_volume"] * 0.5),
       f"{today_vol/1e3:,.0f}K", "" if today_vol >= MO["min_today_volume"] else f"باقي {(MO['min_today_volume']-today_vol)/1e3:,.0f}K سهم")
    ch = change_pct or 0
    mc("change", "السهم صاعد اليوم", 10, ch > 0, 10 if ch > 0 else 0, f"{ch:+.1f}%")
    gp = gap_pct or 0
    mc("gap", "فجوة افتتاح 5%+", 10, gp >= MO["min_gap_pct"], 10 if gp >= MO["min_gap_pct"] else max(0, gp) / MO["min_gap_pct"] * 5,
       f"Gap {gp:+.1f}%")
    if flt:
        mc("float", "فلووت أقل من 20M", 15, flt < MO["max_float"],
           15 if flt < MO["max_float"] else (8 if flt < 50e6 else 0), f"{flt/1e6:.2f}M")
    else:
        mc("float", "فلووت أقل من 20M", 15, False, 5, "غير معروف")
    mc("runner", "سلوك سابق (Former Runner)", 15, fr["max_pct"] >= BH["former_runner_pct"],
       15 if fr["max_pct"] >= BH["former_runner_pct"] else 15 * fr["max_pct"] / BH["former_runner_pct"] * 0.5,
       f"أكبر انفجار سابق +{fr['max_pct']:.0f}%")
    if short_avail is not None:
        short_ok = short_avail <= BH.get("max_short_shares", 50000)
        s_txt = f"المتاح {short_avail:,.0f} سهم"
    else:
        short_ok = (short_pct_float is None or short_pct_float < 10)
        s_txt = f"{short_pct_float:.1f}% من الفلووت" if short_pct_float is not None else "غير متوفر"
    mc("short", "شورت قليل", 5, short_ok, 5 if short_ok else 0, s_txt)
    mc("news", "خبر / محفز اليوم", 10, False, 0, "يتم فحصه في التحليل العميق")
    m_score = sum(x.got for x in m_checks)
    m_warn = []
    if ch >= MO["late_chase_pct"]:
        m_warn.append(f"صعد {ch:.0f}% اليوم — متأخر. قاعدة فيصل: إذا ما دخلت أول موجة 5 دقائق، اصرف النظر")
    if pumps >= 2:
        m_warn.append("سلوك قروبات وهم سابق — احذر التصريف")
    m_score = max(0, min(100, m_score - (15 if ch >= MO["late_chase_pct"] else 0)))
    if m_score >= 70 and rv >= MO["min_rel_volume"] and ch > 0:
        m_status = "ready"
    elif m_score >= 50 and rv >= 1.5:
        m_status = "near"
    else:
        m_status = "no"
    if m_status != "no":
        tags.append("زخم")

    # فلتر القائمة (Universe)
    gate = []
    if not (U["min_price"] <= price <= U["max_price"]):
        gate.append(f"السعر {fmt_price(price)} خارج {U['min_price']}–{U['max_price']}$")
    if mcap and mcap > U["max_market_cap"]:
        gate.append(f"القيمة السوقية {mcap/1e6:.0f}M فوق {U['max_market_cap']/1e6:.0f}M")
    if flt and flt > U["max_float"]:
        gate.append(f"الفلووت {flt/1e6:.1f}M فوق {U['max_float']/1e6:.0f}M")
    if avg_vol < U["min_avg_volume"]:
        gate.append(f"متوسط الفوليوم {avg_vol/1e3:.0f}K أقل من {U['min_avg_volume']/1e3:.0f}K")

    # بيانات للرسم (آخر 100 شمعة)
    tail = df.tail(100)
    chart = {
        "d": [str(i)[:10] for i in tail.index],
        "o": [_r(x, 4) for x in tail["Open"]], "h": [_r(x, 4) for x in tail["High"]],
        "l": [_r(x, 4) for x in tail["Low"]], "c": [_r(x, 4) for x in tail["Close"]],
        "v": [int(x) for x in tail["Volume"]],
        "ma20": [_r(x, 4) for x in ma[20].tail(100)],
        "ma50": [_r(x, 4) for x in ma[50].tail(100)],
    }

    return {
        "price": _r(price, 4), "change_pct": _r(change_pct, 2), "gap_pct": _r(gap_pct, 2),
        "date": str(df.index[-1])[:10],
        "volume": int(today_vol), "avg_volume": int(avg_vol), "rel_volume": _r(rel_vol, 2),
        "market_cap": mcap, "float": flt, "shares_out": shares_out,
        "short_shares": short_shares, "short_pct_float": _r(short_pct_float, 2),
        "short_to_mcap": _r(short_to_mcap, 2),
        "rsi": rsi_now, "rsi_min10": rsi_min10,
        "ma": ma_now, "ma50_200": ma50_200,
        "hi52": _r(hi52, 4), "lo52": _r(lo52, 4), "drawdown_pct": _r(drawdown, 1),
        "up_from_low20_pct": _r(up_from_low, 1),
        "median_range_pct": _r(med_range, 1),
        "quiet": med_range <= BH["quiet_daily_range_pct"],
        "below_1_streak": below_1,
        "former_runner": {"max_pct": _r(fr["max_pct"], 0), "date": fr["date"]},
        "reverse_splits": rsplits,
        "pump_candles": pumps,
        "w": {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in w.items()} if w else None,
        "sweep": sweep,
        "plan": {
            "support": _r(support, 4), "support_kind": sup["kind"], "support_touches": sup["touches"],
            "dist_support_pct": _r(dist_support, 1),
            "entry_zone": [_r(entry_lo, 4), _r(entry_hi, 4)],
            "orders": [_r(entry_lo + (entry_hi - entry_lo) * k / 2, 4) for k in (0, 1, 2)],
            "stop": _r(stop, 4),
            "t1": _r(t1, 4), "main_target": _r(main_target, 4),
            "four_targets": [_r(x, 4) for x in four_targets],
            "resistances": [_r(x["level"], 4) for x in res[:6]],
            "falling_candles": [{"top": _r(f["top"], 4), "tail": _r(f["tail"], 4), "date": f["date"]} for f in fcs[-5:]],
            "gaps": [{"from": _r(g["from"], 4), "to": _r(g["to"], 4), "date": g["date"]} for g in gaps[-5:]],
            "upside_pct": _r(upside_pct, 1), "rr": _r(rr, 2),
        },
        "irtikaz": {"score": round(score, 1), "base": round(base, 1), "status": status,
                    "checks": [x.to_dict() for x in checks],
                    "adjustments": [{"name": a[0], "pts": a[1], "detail": a[2]} for a in adj],
                    "readiness": readiness},
        "momentum": {"score": round(m_score, 1), "status": m_status,
                     "checks": [x.to_dict() for x in m_checks], "warnings": m_warn},
        "gate_fail": gate,
        "tags": tags,
        "chart": chart,
    }
