"""
🎯 أداة الثقة لفيصل المطوّر — احتمال نجاح التوصية من تاريخ آلاف الحالات المشابهة.

النجاح = نفس تعريف سجل التوصيات: السهم يوصل الهدف الأول قبل ما يسكر تحت الوقف خلال 20 جلسة.
النموذج: انحدار لوجستي (Logistic Regression) مدرّب على بيانات Alpaca (سنتين+) ومختبر على فترات ما شافها.
الملف `confidence_model.json` فيه الأوزان (يطلع من diag/backtest_confidence.py على فرع diag-flye).

نفس دالة `features()` تُستخدم في الاختبار التاريخي وفي الرادار — عشان الحساب يطلع متطابق.
كل العوامل «وقتية» (point-in-time): ما تستخدم أي معلومة بعد يوم الإشارة.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "confidence_model.json"
CHECKS = ("support", "rsi", "hold", "candle", "macd", "volume", "ma", "upside")
DILUTION_FORMS = ("S-1", "F-1", "S-3", "F-3", "424B1", "424B2", "424B4", "424B5")

# أسماء عربية للعوامل (لشرح سبب الثقة في الصفحة)
NAMES_AR = {
    "score": "نقاط فيصل", "adj": "مكافآت/خصومات السلوك", "rsi": "RSI", "rsi_min10": "أدنى RSI آخر 10 أيام",
    "pc_support": "قرب الدعم", "pc_rsi": "ضغط RSI", "pc_hold": "فشل الهبوط", "pc_candle": "الشمعة الإيجابية",
    "pc_macd": "تحسن MACD", "pc_volume": "تأكيد الفوليوم", "pc_ma": "المتوسطات", "pc_upside": "مساحة الصعود",
    "dist_support": "البعد عن الدعم", "touches": "اختبارات الدعم", "upside": "المساحة لأول مقاومة",
    "stop_pct": "بعد الوقف", "t1_pct": "بعد الهدف الأول", "drawdown": "النزول من قمة السنة",
    "up_from_low20": "الصعود من قاع 20 يوم", "range_med": "تذبذب الشموع", "below1": "جلسات تحت 1$",
    "runner": "سبق انفجر (Former Runner)", "pumps": "شموع قروبات", "has_w": "نموذج W", "sweep": "سحب سيولة",
    "rel_vol": "الفوليوم النسبي", "atr20": "التذبذب اليومي", "ret5": "أداء 5 أيام", "ret20": "أداء 20 يوم",
    "ret60": "أداء 60 يوم", "vol_ratio": "الفوليوم الأخير ÷ المعتاد", "dollar_vol": "السيولة بالدولار",
    "price_log": "مستوى السعر", "close_loc": "موقع الإغلاق في الشمعة", "rs365": "تقسيمات عكسية آخر سنة",
    "days_rs": "أيام من آخر تقسيم عكسي", "dil90": "ملفات تخفيف آخر 90 يوم", "days_earn": "أيام من آخر أرباح",
    "earn_soon": "الأرباح قريبة", "iwm_up": "سوق الأسهم الصغيرة فوق متوسط 50", "iwm_ret20": "أداء الأسهم الصغيرة 20 يوم",
    "breadth": "نسبة الأسهم الصاعدة (فوق متوسط 50)", "is_ready": "مطابق", "is_near": "قريب",
}


def _f(x, default=0.0, lo=None, hi=None):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            v = default
    except (TypeError, ValueError):
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def corp_from_daily(d: pd.DataFrame) -> list:
    """تواريخ التقسيم العكسي من عمود Stock Splits (نسبة أقل من 1 = عكسي)."""
    if "Stock Splits" not in d:
        return []
    s = d["Stock Splits"]
    return [ix for ix, v in s.items() if v and 0 < float(v) < 1]


def features(a: dict, d: pd.DataFrame, rs_dates=None, dil_dates=None, earn_dates=None, mkt: dict | None = None) -> dict:
    """العوامل لسهم في يوم معيّن. d = الشارت اليومي لين يوم الإشارة (بدون أي بيانات بعده)."""
    I = a["irtikaz"]
    p = a.get("plan") or {}
    chk = {c["key"]: c for c in I["checks"]}
    price = _f(a.get("price"), 1.0, 1e-4)
    day = pd.Timestamp(d.index[-1]).normalize()
    c = d["Close"].astype(float)
    h, l, v = d["High"].astype(float), d["Low"].astype(float), d["Volume"].astype(float)

    def ret(k):
        return float(c.iloc[-1] / c.iloc[-1 - k] - 1) * 100 if len(c) > k and c.iloc[-1 - k] > 0 else 0.0

    rng = (h - l).tail(5)
    close_loc = float(((c - l).tail(5) / rng.replace(0, np.nan)).mean()) if len(rng) else 0.5
    stop = _f(p.get("stop"), price * 0.85)
    t1 = _f(p.get("t1"), price * 1.2)
    rs_dates = [pd.Timestamp(x).normalize() for x in (rs_dates if rs_dates is not None else corp_from_daily(d))]
    rs_past = [x for x in rs_dates if x <= day]
    dil_past = [pd.Timestamp(x) for x in (dil_dates or []) if pd.Timestamp(x) <= day]
    earn_past = [pd.Timestamp(x) for x in (earn_dates or []) if pd.Timestamp(x) <= day]
    days_earn = (day - max(earn_past)).days if earn_past else 200
    mkt = mkt or {}
    f = {
        "score": _f(I.get("score")), "adj": _f(I.get("score")) - _f(I.get("base")),
        "rsi": _f(a.get("rsi"), 50), "rsi_min10": _f(a.get("rsi_min10"), 50),
        **{f"pc_{k}": _f((chk.get(k) or {}).get("got")) / max(1e-9, _f((chk.get(k) or {}).get("weight"), 1)) for k in CHECKS},
        "dist_support": _f(p.get("dist_support_pct"), 30, -50, 100), "touches": _f(p.get("support_touches"), 0, 0, 10),
        "upside": _f(p.get("upside_pct"), 30, 0, 200), "stop_pct": (price - stop) / price * 100,
        "t1_pct": _f((t1 / price - 1) * 100, 20, -50, 300), "drawdown": _f(a.get("drawdown_pct"), -50, -100, 0),
        "up_from_low20": _f(a.get("up_from_low20_pct"), 0, 0, 300), "range_med": _f(a.get("median_range_pct"), 5, 0, 50),
        "below1": _f(a.get("below_1_streak"), 0, 0, 250),
        "runner": math.log1p(_f((a.get("former_runner") or {}).get("max_pct"), 0, 0, 5000)),
        "pumps": _f(a.get("pump_candles"), 0, 0, 10), "has_w": 1.0 if a.get("w") else 0.0,
        "sweep": 1.0 if a.get("sweep") else 0.0, "rel_vol": _f(a.get("rel_volume"), 1, 0, 20),
        "atr20": float(((h - l) / c).tail(20).mean() * 100), "ret5": ret(5), "ret20": ret(20), "ret60": ret(60),
        "vol_ratio": float(v.tail(5).mean() / max(1.0, v.tail(50).mean())),
        "dollar_vol": math.log10(max(1.0, float((c * v).tail(20).mean()))), "price_log": math.log10(price),
        "close_loc": _f(close_loc, 0.5, 0, 1),
        "rs365": float(sum(1 for x in rs_past if (day - x).days <= 365)),
        "days_rs": float(min(400, (day - max(rs_past)).days)) if rs_past else 400.0,
        "dil90": float(sum(1 for x in dil_past if (day - x).days <= 90)),
        "days_earn": float(min(200, days_earn)), "earn_soon": 1.0 if 75 <= days_earn <= 100 else 0.0,
        "iwm_up": _f(mkt.get("iwm_up"), 0.5), "iwm_ret20": _f(mkt.get("iwm_ret20"), 0), "breadth": _f(mkt.get("breadth"), 0.4),
        "is_ready": 1.0 if I.get("status") == "ready" else 0.0, "is_near": 1.0 if I.get("status") == "near" else 0.0,
    }
    for k in ("ret5", "ret20", "ret60"):
        f[k] = max(-90.0, min(300.0, f[k]))
    f["stop_pct"] = max(0.0, min(80.0, f["stop_pct"]))
    f["atr20"] = max(0.0, min(60.0, f["atr20"]))
    f["vol_ratio"] = max(0.0, min(20.0, f["vol_ratio"]))
    return f


def market_state(iwm: pd.DataFrame | None, closes: dict[str, pd.Series] | None, day=None) -> dict:
    """وضع سوق الأسهم الصغيرة في يوم معيّن: IWM فوق متوسط 50، أداء 20 يوم، ونسبة أسهمنا فوق متوسط 50."""
    out = {}
    if iwm is not None and len(iwm) > 55:
        s = iwm["Close"].astype(float)
        if day is not None:
            s = s[s.index <= day]
        if len(s) > 55:
            out["iwm_up"] = 1.0 if s.iloc[-1] > s.tail(50).mean() else 0.0
            out["iwm_ret20"] = float(s.iloc[-1] / s.iloc[-21] - 1) * 100
    if closes:
        up = tot = 0
        for s in closes.values():
            if day is not None:
                s = s[s.index <= day]
            if len(s) > 55:
                tot += 1
                up += 1 if s.iloc[-1] > s.tail(50).mean() else 0
        if tot:
            out["breadth"] = up / tot
    return out


# ----------------------------------------------------------------------------
# التنبؤ
# ----------------------------------------------------------------------------
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        except Exception:
            _MODEL = {}
    return _MODEL


def predict(f: dict, model: dict | None = None) -> dict | None:
    """احتمال النجاح + المستوى + أهم الأسباب (مع وضد)."""
    m = model if model is not None else load_model()
    if not m or "coef" not in m:
        return None
    z = m["intercept"]
    contrib = []
    for k, w in m["coef"].items():
        x = (f.get(k, m["mean"][k]) - m["mean"][k]) / (m["std"][k] or 1)
        x = max(-5.0, min(5.0, x))
        z += w * x
        contrib.append((k, w * x))
    pr = 1 / (1 + math.exp(-z))
    # معايرة (من فترات الاختبار): متوسط الاحتمال الخام لكل عُشر ← النسبة الفعلية (منحنى صاعد)
    cal = m.get("calibration")
    if cal:
        pr = float(np.interp(pr, [c[0] for c in cal], [c[1] for c in cal]))
    tiers = m.get("tiers", {"high": 0.4, "mid": 0.28})
    label = "عالية" if pr >= tiers["high"] else "متوسطة" if pr >= tiers["mid"] else "منخفضة"
    contrib.sort(key=lambda x: x[1])
    neg = [NAMES_AR.get(k, k) for k, v in contrib[:3] if v < -0.08]
    pos = [NAMES_AR.get(k, k) for k, v in reversed(contrib[-3:]) if v > 0.08]
    return {"p": round(pr, 3), "label": label, "base": m.get("base_rate"), "pos": pos, "neg": neg,
            "n_train": m.get("n_train"), "version": m.get("version")}
