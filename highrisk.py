"""🧨 عالي مخاطرة — الأسهم اللي تنفجر بشكل متكرر (طلب المستخدم 2026-11-11: «خل لها رادار اسمه عالي مخاطرة ... كل شي مفتوح»).

نفس معادلات الدراسة `diag/backtest_highrisk.py` (فرع diag-flye) بالضبط — الرادار يحسب الميزات على آخر يوم ويعطي:
- احتمال انفجار (أعلى سعر ≥ ضعف إغلاق اليوم اللي قبله) خلال 10 جلسات — من النموذج ومعاير على فترات ما شافها (`highrisk_model.json`).
- تاريخ انفجارات السهم (الحلقات: أول انفجار بعد 20 جلسة بدونه) + التقسيمات العكسية.
- بصمات التجميع قبل الانطلاقة (من الدراسة) + خطة: بيع Limit وقت القفزة.
ما يغير أي رادار ثاني."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "highrisk_model.json"
H = 20
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


def episodes(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """jump = أعلى ÷ إغلاق أمس، ex = انفجار (≥ ×2)، ep = بداية حلقة (أول انفجار بعد 20 جلسة بدونه)."""
    h, c = d["High"].to_numpy(float), d["Close"].to_numpy(float)
    pc = np.r_[np.nan, c[:-1]]
    jump = h / pc
    ex = np.nan_to_num(jump, nan=0.0) >= 2.0
    ep = np.zeros(len(c), bool)
    last = -999
    for t in np.flatnonzero(ex):
        if t - last > 20:
            ep[t] = True
        last = t
    return jump, ex, ep


def features(d: pd.DataFrame) -> pd.DataFrame:
    """كل الميزات لكل يوم (نفس الدراسة). d: يومي معدّل للتقسيم + عمود Stock Splits (النسبة يوم التقسيم، أقل من 1 = عكسي)."""
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    idx = np.arange(n)
    pc = np.r_[np.nan, c[:-1]]
    jump, ex, ep = episodes(d)
    cs = np.cumsum(ep)
    n_ep252 = cs - np.r_[np.zeros(252), cs[:-252]] if n > 252 else cs.astype(float)
    lastep = np.maximum.accumulate(np.where(ep, idx, -1))
    has = lastep >= 0
    days_ep = np.where(has, idx - lastep, 9999)
    li = np.maximum(lastep, 0)
    last_mag = np.where(has, jump[li], 0.0)
    base_before = np.where(has & (li >= 1), c[np.maximum(li - 1, 0)], np.nan)
    peak = np.where(has, h[li], np.nan)
    spl = d["Stock Splits"].to_numpy(float) if "Stock Splits" in d else np.zeros(n)
    rs = (spl > 0) & (spl < 1)
    rcs = np.cumsum(rs)
    n_rs = rcs - np.r_[np.zeros(252), rcs[:-252]] if n > 252 else rcs.astype(float)   # تقسيمات عكسية بآخر سنة (نفس الدراسة)
    lastrs = np.maximum.accumulate(np.where(rs, idx, -1))
    days_rs = np.where(lastrs >= 0, idx - lastrs, 9999)
    tr = np.where(np.isfinite(pc), np.maximum(h, pc) - np.minimum(l, pc), h - l)
    dv20 = _roll(c * v, 20, "median")
    vol5_60 = _roll(v, 5, "mean") / np.maximum(_roll(v, 60, "mean"), 1)
    vol1_20 = v / np.maximum(_roll(v, 20, "median"), 1)
    atr20 = _roll(tr, 20, "mean")
    atr5_60 = _roll(tr, 5, "mean") / np.maximum(_roll(tr, 60, "mean"), 1e-9)
    hi20, lo20 = _roll(h, 20, "max"), _roll(l, 20, "min")
    hi60, lo60 = _roll(h, 60, "max"), _roll(l, 60, "min")
    hi252 = _roll(h, 252, "max", 120)
    sh = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]
    c5, c20 = sh(c, 5), sh(c, 20)
    rsi = rsi_np(c)
    rngd = np.where(h > l, (c - l) / np.maximum(h - l, 1e-12), 0.5)
    up, dn = c > pc, c < pc
    upv10 = _roll(np.where(up, v, 0.0), 10, "sum") / (_roll(np.where(dn, v, 0.0), 10, "sum") + 1)
    obv = np.cumsum(np.where(up, v, np.where(dn, -v, 0.0)))
    obv20 = (obv - sh(obv, 20)) / np.maximum(_roll(v, 60, "mean") * 20, 1)
    m5 = _roll(l, 5, "min")
    hl_up = (m5 > sh(m5, 5)).astype(float) + (sh(m5, 5) > sh(m5, 10)).astype(float) + (sh(m5, 10) > sh(m5, 15)).astype(float)
    gap20 = _roll((o >= 1.1 * pc).astype(float), 20, "sum")
    return pd.DataFrame({
        "close": c, "dv20": dv20, "atr20": atr20, "lo20": lo20, "hi20": hi20, "jump": jump, "ex": ex, "ep": ep,
        "days_ep_raw": days_ep, "days_rs_raw": days_rs,
        "l_price": np.log(np.maximum(c, 0.01)), "l_dv20": np.log10(np.maximum(dv20, 1)),
        "n_ep252": n_ep252, "n_ep_all": cs, "l_days_ep": np.log1p(np.minimum(days_ep, 1000)),
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
    }, index=d.index)


def score(row: pd.Series) -> dict | None:
    """الاحتمال الخام من النموذج + المعايرة من فترات الاختبار (العُشر اللي يطيح فيه → نسبته الفعلية)."""
    M = model()
    if not M:
        return None
    x = np.array([float(row.get(f, np.nan)) for f in M["feats"]])
    mu, sd = np.array(M["mean"]), np.array(M["std"])
    x = np.where(np.isfinite(x), x, mu)
    z = np.clip((x - mu) / np.where(sd == 0, 1, sd), -5, 5)
    w = np.array(M["w"])
    p = float(1 / (1 + np.exp(-(w[0] + z @ w[1:]))))
    cal = None
    for k in M.get("calib", []):
        if p >= k["p_lo"]:
            cal = k
    cal = cal or (M.get("calib") or [None])[0]
    contrib = sorted(((f, float(w[i + 1] * z[i])) for i, f in enumerate(M["feats"])), key=lambda t: -abs(t[1]))
    return {"p_raw": p, "decile": cal["dec"] if cal else None, "e10": cal["e10"] if cal else None,
            "e20": cal["e20"] if cal else None, "tr50": cal.get("tr50") if cal else None, "mx20": cal.get("mx20") if cal else None,
            "base": M.get("base_e10"), "top": contrib[:6]}


def history(d: pd.DataFrame, F: pd.DataFrame) -> dict:
    """تاريخ الانفجارات: كل حلقة بتاريخها وقوتها ووش صار بعدها (5 و20 جلسة من إغلاق ما قبلها)."""
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
