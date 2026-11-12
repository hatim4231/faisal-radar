"""🧨 عالي مخاطرة — الأسهم اللي تنفجر بشكل متكرر (طلب المستخدم 2026-11-11: «خل لها رادار اسمه عالي مخاطرة ... كل شي مفتوح»).

النسخة 2 (2026-11-12، طلب المستخدم: «طوره وخله ثقة والتوصية تكون مفيدة وقوية وتبني محفظتي»):
- **انفجار حقيقي** = أعلى سعر باليوم ≥ ضعف إغلاق اليوم اللي قبله **و** تداول ذاك اليوم ≥ نص مليون دولار
  (النسخة الأولى كانت تحسب صفقة شاذة بسهم ما فيه تداول «انفجار» — يتوقعها النموذج بسهولة وما تنشرى).
- الميزات بنافذة ثابتة (آخر 252 جلسة) عشان الرادار الحي (420 يوم) يطابق الدراسة (7 سنين) بالضبط.
- 🌡 ميزان حرارة سوق الطيارات: مؤشرات على كل السوق (`market_frame` لكل سهم ← تتجمع ← `regime_series`).
نفس هالملف تستخدمه الدراسة `diag/backtest_highrisk2.py` (فرع diag-flye) والرادار الحي — مصدر واحد.
ما يغير أي رادار ثاني."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "highrisk_model.json"
H = 20
WIN = 252                 # نافذة «سبق انفجر/تقسيم عكسي»
EX_MIN_DV = 500_000       # الانفجار الحقيقي: تداول ذاك اليوم ≥ 500 ألف$
ACT_MIN_DV20 = 5_000      # سهم «نشط»: وسيط تداوله اليومي ≥ 5 آلاف$ وسعره ≥ 0.05$ (أقل من كذا ما ينشرى أصلاً)
ACT_MIN_PX = 0.05
FEATS = ["l_price", "l_dv20", "n_ep252", "l_days_ep", "recent_ep", "last_mag", "base_ratio", "from_peak",
         "vol5_60", "vol1_20", "atr5_60", "atr20p", "rng20", "pos60", "dd252", "ret5", "ret20", "rsi", "closepos10",
         "upv10", "obv20", "hl_up", "gap20", "rs_recent", "n_rs", "l_days_rs", "ret1", "closepos1", "rng1"]
MK_COLS = ["n", "ex", "big", "pn", "pn_ab", "pn_r20", "pn_r5", "exn_n", "exn_c", "exn_h", "pg_n", "pg_r", "pg_h"]
_MODEL = None


def model() -> dict | None:
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        except Exception:
            _MODEL = {}
    return _MODEL or None


def rsi_np(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def _roll(a, w, fn, minp=None):
    r = pd.Series(a).rolling(w, min_periods=minp or w)
    return getattr(r, fn)().to_numpy(float)


def _sh(a, k):
    return np.r_[np.full(k, np.nan), a[:-k]] if len(a) > k else np.full(len(a), np.nan)


def _fsh(a, k):
    return np.r_[a[k:], np.full(k, np.nan)] if len(a) > k else np.full(len(a), np.nan)


def episodes(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """jump = أعلى ÷ إغلاق أمس · ex = انفجار حقيقي (≥ ×2 وتداول اليوم ≥ 500 ألف$) · ep = بداية حلقة (أول انفجار بعد 20 جلسة بدونه)."""
    h, l, c, v = (d[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    pc = np.r_[np.nan, c[:-1]]
    jump = h / pc
    dvd = v * (h + l + c) / 3
    ex = (np.nan_to_num(jump, nan=0.0) >= 2.0) & (dvd >= EX_MIN_DV)
    ep = np.zeros(len(c), bool)
    last = -999
    for t in np.flatnonzero(ex):
        if t - last > 20:
            ep[t] = True
        last = t
    return jump, ex, ep


def features(d: pd.DataFrame) -> pd.DataFrame:
    """كل الميزات لكل يوم. d: يومي معدّل للتقسيم + عمود Stock Splits (النسبة يوم التقسيم، أقل من 1 = عكسي)."""
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    idx = np.arange(n)
    pc = np.r_[np.nan, c[:-1]]
    jump, ex, ep = episodes(d)
    cs = np.cumsum(ep)
    n_ep252 = cs - np.r_[np.zeros(WIN), cs[:-WIN]] if n > WIN else cs.astype(float)
    lastep = np.maximum.accumulate(np.where(ep, idx, -1))
    has = (lastep >= 0) & (idx - lastep <= WIN)             # آخر انفجار داخل النافذة بس
    days_ep = np.where(has, idx - lastep, 9999)
    li = np.maximum(lastep, 0)
    last_mag = np.where(has, jump[li], 0.0)
    base_before = np.where(has & (li >= 1), c[np.maximum(li - 1, 0)], np.nan)
    peak = np.where(has, h[li], np.nan)
    spl = d["Stock Splits"].to_numpy(float) if "Stock Splits" in d else np.zeros(n)
    rs = (spl > 0) & (spl < 1)
    rcs = np.cumsum(rs)
    n_rs = rcs - np.r_[np.zeros(WIN), rcs[:-WIN]] if n > WIN else rcs.astype(float)
    lastrs = np.maximum.accumulate(np.where(rs, idx, -1))
    has_rs = (lastrs >= 0) & (idx - lastrs <= WIN)
    days_rs = np.where(has_rs, idx - lastrs, 9999)
    tr = np.where(np.isfinite(pc), np.maximum(h, pc) - np.minimum(l, pc), h - l)
    dv20 = _roll(c * v, 20, "median")
    vol5_60 = _roll(v, 5, "mean") / np.maximum(_roll(v, 60, "mean"), 1)
    vol1_20 = v / np.maximum(_roll(v, 20, "median"), 1)
    atr20 = _roll(tr, 20, "mean")
    atr5_60 = _roll(tr, 5, "mean") / np.maximum(_roll(tr, 60, "mean"), 1e-9)
    hi20, lo20 = _roll(h, 20, "max"), _roll(l, 20, "min")
    hi60, lo60 = _roll(h, 60, "max"), _roll(l, 60, "min")
    hi252 = _roll(h, 252, "max", 120)
    c5, c20 = _sh(c, 5), _sh(c, 20)
    rsi = rsi_np(c)
    rngd = np.where(h > l, (c - l) / np.maximum(h - l, 1e-12), 0.5)
    up, dn = c > pc, c < pc
    upv10 = _roll(np.where(up, v, 0.0), 10, "sum") / (_roll(np.where(dn, v, 0.0), 10, "sum") + 1)
    obv = np.cumsum(np.where(up, v, np.where(dn, -v, 0.0)))
    obv20 = (obv - _sh(obv, 20)) / np.maximum(_roll(v, 60, "mean") * 20, 1)
    m5 = _roll(l, 5, "min")
    hl_up = (m5 > _sh(m5, 5)).astype(float) + (_sh(m5, 5) > _sh(m5, 10)).astype(float) + (_sh(m5, 10) > _sh(m5, 15)).astype(float)
    gap20 = _roll((o >= 1.1 * pc).astype(float), 20, "sum")
    return pd.DataFrame({
        "close": c, "dv20": dv20, "atr20": atr20, "lo20": lo20, "hi20": hi20, "jump": jump, "ex": ex, "ep": ep,
        "days_ep_raw": days_ep, "days_rs_raw": days_rs, "n_ep_all": cs,
        "l_price": np.log(np.maximum(c, 0.01)), "l_dv20": np.log10(np.maximum(dv20, 1)),
        "n_ep252": n_ep252, "l_days_ep": np.log1p(np.minimum(days_ep, 1000)),
        "recent_ep": (days_ep <= 60).astype(float), "last_mag": np.minimum(last_mag, 20),
        "base_ratio": np.where(np.isfinite(base_before), np.minimum(c / np.maximum(base_before, 1e-9), 10), 1.0),
        "from_peak": np.where(np.isfinite(peak), c / peak - 1, 0.0),
        "vol5_60": np.minimum(vol5_60, 20), "vol1_20": np.minimum(vol1_20, 50), "atr5_60": np.minimum(atr5_60, 10),
        "atr20p": atr20 / c, "rng20": np.minimum(hi20 / np.maximum(lo20, 1e-9) - 1, 10),
        "pos60": np.clip((c - lo60) / np.maximum(hi60 - lo60, 1e-12), 0, 1), "dd252": c / hi252 - 1,
        "ret5": np.clip(c / c5 - 1, -1, 5), "ret20": np.clip(c / c20 - 1, -1, 10), "rsi": rsi,
        "closepos10": _roll(rngd, 10, "mean"), "upv10": np.minimum(upv10, 20), "obv20": np.clip(obv20, -5, 5),
        "hl_up": hl_up, "gap20": gap20, "rs_recent": (days_rs <= 60).astype(float), "n_rs": n_rs,
        "l_days_rs": np.log1p(np.minimum(days_rs, 1000)),
        "ret1": np.clip(c / pc - 1, -0.9, 3), "closepos1": rngd, "rng1": np.clip((h - l) / np.maximum(pc, 1e-9), 0, 3),
    }, index=d.index)


# ----------------------------------------------------------------------------
# 🌡 حرارة سوق الطيارات — مساهمة كل سهم (تتجمع على كل الأسهم) ثم المؤشرات
# ----------------------------------------------------------------------------

def market_frame(d: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """صف لكل يوم: مساهمة هالسهم بمؤشرات السوق. كل شي «معروف بإغلاق ذاك اليوم» (النتايج اللاحقة تنحط بيوم ما تنعرف)."""
    h, c = d["High"].to_numpy(float), d["Close"].to_numpy(float)
    n = len(c)
    idx = np.arange(n)
    pc = np.r_[np.nan, c[:-1]]
    act = (np.nan_to_num(_sh(F["dv20"].to_numpy(float), 1)) >= ACT_MIN_DV20) & (np.nan_to_num(pc) >= ACT_MIN_PX)
    ex = F["ex"].to_numpy(bool) & act
    r1 = np.nan_to_num(c / pc - 1)
    pn = act & (pc < 5) & (idx >= 20)
    ma20 = _roll(c, 20, "mean")
    out = np.zeros((n, len(MK_COLS)), np.float32)
    col = {k: i for i, k in enumerate(MK_COLS)}
    out[:, col["n"]] = act
    out[:, col["ex"]] = ex
    out[:, col["big"]] = act & (r1 >= 0.5)
    out[:, col["pn"]] = pn
    out[:, col["pn_ab"]] = pn & (c > np.nan_to_num(ma20, nan=np.inf))
    out[:, col["pn_r20"]] = np.where(pn, np.clip(np.nan_to_num(c / _sh(c, 20) - 1), -0.9, 3), 0)
    out[:, col["pn_r5"]] = np.where(pn, np.clip(np.nan_to_num(c / _sh(c, 5) - 1), -0.9, 3), 0)
    E = np.flatnonzero(ex[:-1]) if n > 1 else np.array([], int)       # انفجار يوم E ← نتيجته تنعرف E+1
    if len(E):
        out[E + 1, col["exn_n"]] = 1
        out[E + 1, col["exn_c"]] = np.clip(c[E + 1] / c[E] - 1, -0.9, 3)
        out[E + 1, col["exn_h"]] = np.clip(h[E + 1] / c[E] - 1, -0.9, 5)
    if n > 6:   # المجموعة المرجعية (سبق انفجر بسنة + منهار 80%+): عائدها بعد 5 جلسات ينعرف d+5
        pg = act & (F["n_ep252"].to_numpy(float) >= 1) & (np.nan_to_num(F["dd252"].to_numpy(float)) <= -0.8) & ~F["ex"].to_numpy(bool)
        D = np.flatnonzero(pg[:n - 5])
        if len(D):
            hmax5 = _fsh(_roll(h, 5, "max"), 5)
            out[D + 5, col["pg_n"]] = 1
            out[D + 5, col["pg_r"]] = np.clip(c[D + 5] / c[D] - 1, -0.9, 3)
            out[D + 5, col["pg_h"]] = (hmax5[D] / c[D] - 1 >= 0.3)
    return pd.DataFrame(out, index=d.index, columns=MK_COLS)


REG_COLS = ["heat5", "heat10", "heat20", "big10", "pen_ab", "pen_r20", "pen_r5", "follow20", "followh20",
            "pg_r20", "pg_h20", "pg_r5", "iwm_r20", "iwm_r5", "iwm_ab50"]


def regime_series(MK: pd.DataFrame, iwm: pd.Series | None = None) -> pd.DataFrame:
    """مؤشرات السوق لكل يوم من مجموع market_frame على كل الأسهم (MK مفهرس بالتاريخ تصاعدي). iwm = إغلاقات IWM (اختياري)."""
    def rs(col, w):
        return MK[col].rolling(w, min_periods=max(3, w // 2)).sum()
    R = pd.DataFrame(index=MK.index)
    for w in (5, 10, 20):
        R[f"heat{w}"] = 1000 * rs("ex", w) / rs("n", w)
    R["big10"] = 1000 * rs("big", 10) / rs("n", 10)
    R["pen_ab"] = (MK["pn_ab"] / MK["pn"]).rolling(3, min_periods=1).mean()
    R["pen_r20"] = MK["pn_r20"] / MK["pn"]
    R["pen_r5"] = MK["pn_r5"] / MK["pn"]
    en = rs("exn_n", 20)
    R["follow20"] = (rs("exn_c", 20) / en).where(en >= 5)
    R["followh20"] = (rs("exn_h", 20) / en).where(en >= 5)
    R["pg_r20"] = rs("pg_r", 20) / rs("pg_n", 20)
    R["pg_h20"] = rs("pg_h", 20) / rs("pg_n", 20)
    R["pg_r5"] = rs("pg_r", 5) / rs("pg_n", 5)
    if iwm is not None and len(iwm):
        x = iwm.reindex(MK.index).ffill()
        R["iwm_r20"] = x / x.shift(20) - 1
        R["iwm_r5"] = x / x.shift(5) - 1
        R["iwm_ab50"] = (x > x.rolling(50, min_periods=30).mean()).astype(float)
    for k in REG_COLS:
        if k not in R:
            R[k] = np.nan
    return R[REG_COLS].replace([np.inf, -np.inf], np.nan)


def score(row: pd.Series) -> dict | None:
    """الاحتمال الخام من النموذج + نسبة مجموعته الفعلية خارج العينة (🧨/🟠)."""
    M = model()
    if not M:
        return None
    x = np.array([float(row.get(f, np.nan)) for f in M["feats"]])
    mu, sd = np.array(M["mean"]), np.array(M["std"])
    x = np.where(np.isfinite(x), x, mu)
    z = np.clip((x - mu) / np.where(sd == 0, 1, sd), -5, 5)
    w = np.array(M["w"])
    p = float(1 / (1 + np.exp(-(w[0] + z @ w[1:]))))
    th, tiers = M.get("thresholds") or {}, M.get("tiers") or {}
    tier = "ready" if p >= th.get("ready", 9) else "near" if p >= th.get("near", 9) else None
    T = tiers.get(tier) or {}
    contrib = sorted(((f, float(w[i + 1] * z[i])) for i, f in enumerate(M["feats"])), key=lambda t: -abs(t[1]))
    return {"p_raw": p, "tier": tier, "e10": T.get("e10"), "e5": T.get("e5"), "base": M.get("base_e10"), "top": contrib[:6]}


def history(d: pd.DataFrame, F: pd.DataFrame) -> dict:
    """تاريخ الانفجارات الحقيقية: كل حلقة بتاريخها وقوتها ووش صار بعدها (5 و20 جلسة من إغلاق ما قبلها)."""
    c, h = d["Close"].to_numpy(float), d["High"].to_numpy(float)
    n = len(c)
    eps = []
    for E in np.flatnonzero(F["ep"].to_numpy()):
        if E < 1:
            continue
        pre = c[E - 1]
        eps.append({"date": str(d.index[E].date()), "pre": round(pre, 4), "high": round(float(h[E]), 4),
                    "mag": round(float(h[E] / pre), 2),
                    "pk5": round(float(h[E:E + 6].max() / pre - 1) * 100, 0),
                    "c5": round(float(c[E + 5] / pre - 1) * 100, 0) if E + 5 < n else None,
                    "c20": round(float(c[E + 20] / pre - 1) * 100, 0) if E + 20 < n else None})
    gaps = np.diff([pd.Timestamp(e["date"]) for e in eps]).astype("timedelta64[D]").astype(float) if len(eps) >= 2 else []
    spl = d["Stock Splits"] if "Stock Splits" in d else pd.Series(dtype=float)
    rsx = [{"date": str(ix.date()), "ratio": f"1:{round(1 / r)}"} for ix, r in spl.items() if 0 < r < 1]
    return {"episodes": eps, "n_year": int(F["n_ep252"].iat[-1]), "n_all": int(F["n_ep_all"].iat[-1]),
            "days_since": int(F["days_ep_raw"].iat[-1]) if F["days_ep_raw"].iat[-1] < 9999 else None,
            "cycle_days": round(float(np.median(gaps))) if len(gaps) else None, "reverse_splits": rsx}
