"""🎯 «الجاهزة» — يرتب إشارات فيصل المطوّر + الأصلي + عبدالله ويطلع أقصر قائمة تكسب (طلب المستخدم 2026-11-13:
«شوف لي طريقة من الرادارات تكسبني ولاتوصي على مليون سهم أعطني الأسهم الجاهزة»).

عقلان (من diag/backtest_ready2.py، فرع diag-flye — 3 سنين، كل أسهم ناسداك مع المشطوبة، بعد التكاليف):
  🎯 الثبات (win): احتمال وصول الهدف الأول خلال 10 جلسات — أعلى 10% يوصل 70% من المرات (أهداف قريبة، كسب بسيط).
  🚀 الطيران (b30): احتمال حركة كبيرة (+30% خلال 20 جلسة) — ترابطه مع أقصى ربح +0.31 (صاد BYAH وAVAT بأعلى الترتيب).
الصدق: العقلان بعد التكاليف تعادل تقريباً بالمتوسط؛ الكسب يجي من اختيار قليل + دخول Limit + البيع داخل القفزة.
ولا عقل يتوقع الـ +300% (PEPG كان ترتيبه #14). هذي قائمة مراقبة مرتبة، مو ضمان.

يحتاج highrisk.py (ميزات 🧨/🩸) + ready_model.json. ما يغير أي رادار — طبقة ترتيب فوق النتايج الموجودة."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "ready_model.json"
HEADS = {"steady": "win", "runner": "b30"}      # 🎯 / 🚀
_MODEL = None


def model() -> dict | None:
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        except Exception:
            _MODEL = {}
    return _MODEL or None


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def feature_row(sig: dict, daily: pd.DataFrame, mkt: dict) -> dict | None:
    """ميزات إشارة وحدة. sig: {radar, status, entry, stop, t1, score}. daily: شارت يومي (آخر عمود = يوم الإشارة)."""
    import highrisk as HR
    import faisal_rules as fr
    M = model()
    if not M or daily is None or len(daily) < 120:
        return None
    try:
        F = HR.features(daily)
        sc = HR.score(F.iloc[-1])
    except Exception:
        return None
    if not sc:
        return None
    last = F.iloc[-1]
    e, stop, t1 = float(sig["entry"]), float(sig.get("stop") or 0), float(sig.get("t1") or 0)
    if e <= 0 or stop <= 0 or t1 <= 0:
        return None
    radar = sig["radar"]
    f = {"x_" + q: float(last[q]) for q in M["feats_b"]}
    f.update({
        "m_breadth": float(mkt.get("breadth", 0.4)), "m_iwm_up": float(mkt.get("iwm_up", 0.5)), "m_iwm_ret20": float(mkt.get("iwm_ret20", 0.0)),
        "lp": _logit(sc["p_raw"]), "lpb": _logit(sc["pb"] if sc.get("pb") is not None else 0.1),
        "t1c": t1 / e - 1, "stc": 1 - stop / e, "rsi14": float(fr.rsi(daily["Close"]).iloc[-1]),
        "score": float(sig.get("score") or 0), "is_ready": float(sig.get("status") == "ready"),
        "is_dev": float(radar == "faisal_dev"), "is_orig": float(radar == "faisal_orig"),
    })
    f["_p"], f["_pb"] = sc["p_raw"], sc.get("pb")           # 🧨 / 🩸 للعرض
    return f


def score_row(f: dict) -> dict:
    """يرجع درجات العقلين + النسبة المئوية لكل عقل (من حدود النموذج)."""
    M = model()
    mu, sd = np.array(M["mean"]), np.array(M["std"])
    x = np.array([f.get(k, np.nan) for k in M["feats"]], float)
    z = np.clip((np.where(np.isfinite(x), x, mu) - mu) / np.where(sd == 0, 1, sd), -5, 5)
    out = {}
    for flavor, head in HEADS.items():
        w = np.array(M["w"][head])
        s = float(w[0] + z @ w[1:])
        q = M["q"][head]                                   # [50%, 80%, 90%, 95%, 98%]
        tier = 4 if s >= q[4] else 3 if s >= q[3] else 2 if s >= q[2] else 1 if s >= q[1] else 0
        out[flavor] = {"score": s, "tier": tier}           # tier: 0 عادي · 1 أعلى20% · 2 أعلى10% · 3 أعلى5% · 4 أعلى2%
    return out
