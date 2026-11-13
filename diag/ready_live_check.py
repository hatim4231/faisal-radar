"""
🎯 «الجاهزة» — فحص حي على سجل التوصيات الحقيقي (بدون نشر): لو كان الترتيب شغّال من أول يوم، مين كان بيختار؟ ووش صار لهم؟
+ قائمة اليوم: أعلى التوصيات المفتوحة الجديدة حسب النموذج.
يقرا track.json من فرع track-data (المستودع عام)، ويحسب لكل توصية (المطوّر/الأصلي/عبدالله) ميزاتها **قبل يوم التوصية**
(ميزات عالي مخاطرة + هندسة الخطة + السوق) ثم درجات نموذج ready_model.json (من diag/backtest_ready2.py).
DATA_RUN = اسم الدرجة اللي نرتب بها (افتراضي tr).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import requests

import alpaca_data as A
import faisal_rules as fr
import highrisk as HR
from backtest_ready import prep_frame, score_all, universe

KEY = (os.environ.get("DATA_RUN") or "tr").strip()
M = json.loads((ROOT / "ready_model.json").read_text(encoding="utf-8"))
SRC = {"faisal_dev": "dev", "faisal_orig": "orig", "abdullah": "abd"}
WIN = ("t1", "t2", "t1_then_stop", "t1_expired")
T = requests.get("https://raw.githubusercontent.com/hatim4231/faisal-radar/track-data/track.json", timeout=60).json()
recs = [t for t in T["trades"] if t["radar"] in SRC and t.get("stop") and t.get("t1") and t.get("entry")]
syms = sorted({t["symbol"] for t in recs})
start = datetime.now(timezone.utc) - timedelta(days=700)
frames = {}
for i in range(0, len(syms), 100):
    b = syms[i:i + 100]
    raw = A.bars(b, "1Day", start)
    try:
        spl = A.splits(b, start)
    except Exception:
        spl = {}
    for s, x in raw.items():
        frames[s] = prep_frame(s, x, spl.get(s, []))
iw = A.bars(["IWM"], "1Day", start)["IWM"]
iw.index = pd.to_datetime(iw.index).tz_localize(None).normalize()
iw = iw[~iw.index.duplicated(keep="last")]["Close"].astype(float)
# اتساع السوق: عينة 700 سهم ناسداك عشوائية ثابتة (تقريب لاتساع الدراسة)
uni = sorted(s for s, st in universe().items() if st == "active")
rng = np.random.default_rng(7)
samp = [uni[i] for i in rng.choice(len(uni), size=min(700, len(uni)), replace=False)]
cl = {}
for i in range(0, len(samp), 100):
    try:
        for s, x in A.bars(samp[i:i + 100], "1Day", start).items():
            y = x["Close"].copy()
            y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
            cl[s] = y[~y.index.duplicated(keep="last")]
    except Exception as e:
        print("breadth err", repr(e)[:80])
closes = pd.DataFrame(cl)
above = (closes > closes.rolling(50).mean()).astype(float).where(closes.rolling(50).count() >= 50)
breadth = (above.sum(axis=1) / above.count(axis=1).replace(0, np.nan)).astype(float)
iwm_up, iwm_r20 = (iw > iw.rolling(50).mean()).astype(float), (iw / iw.shift(20) - 1) * 100
print(f"توصيات: {len(recs)} · أسهم {len(syms)} (لها بيانات {len(frames)}) · اتساع السوق اليوم {breadth.iloc[-1]:.2f} · IWM فوق متوسطه؟ {iwm_up.iloc[-1]} · IWM 20 يوم {iwm_r20.iloc[-1]:+.1f}%")
FB, FEATS = M["feats_b"], M["feats"]
mu, sd = np.array(M["mean"]), np.array(M["std"])
rows = []
for t in recs:
    d = frames.get(t["symbol"])
    if d is None:
        continue
    D = pd.Timestamp(t["date"])
    h = d[d.index < D]                       # قبل يوم التوصية (بدون أي نظرة للمستقبل)
    if len(h) < 200:
        continue
    F = HR.features(h)
    p, pb = score_all(F.tail(1))
    last = F.iloc[-1]
    day = h.index[-1]
    f = {"x_" + q: float(last[q]) for q in FB}
    f.update({"m_breadth": float(breadth.get(day, np.nan)), "m_iwm_up": float(iwm_up.get(day, np.nan)), "m_iwm_ret20": float(iwm_r20.get(day, np.nan)),
              "lp": float(np.log(np.clip(p[0], 1e-6, 1 - 1e-6) / (1 - np.clip(p[0], 1e-6, 1 - 1e-6)))),
              "lpb": float(np.log(np.clip(pb[0], 1e-6, 1 - 1e-6) / (1 - np.clip(pb[0], 1e-6, 1 - 1e-6)))),
              "t1c": t["t1"] / t["entry"] - 1, "stc": 1 - t["stop"] / t["entry"], "rsi14": float(fr.rsi(h["Close"]).iloc[-1]),
              "score": float(t.get("score") or 0), "is_ready": float(t.get("status") == "ready"),
              "is_dev": float(SRC[t["radar"]] == "dev"), "is_orig": float(SRC[t["radar"]] == "orig")})
    x = np.array([f.get(k, np.nan) for k in FEATS], float)
    z = np.clip((np.where(np.isfinite(x), x, mu) - mu) / np.where(sd == 0, 1, sd), -5, 5)
    sc = {k: float(w[0] + z @ np.array(w[1:])) for k, w in M["w"].items()}
    aft = d[d.index >= D]
    rows.append({"date": t["date"], "s": t["symbol"], "src": SRC[t["radar"]], "label": t.get("label"), "entry": t["entry"], "stop": t["stop"], "t1": t["t1"],
                 "p": float(p[0]), "pb": float(pb[0]), **{"s_" + k: v for k, v in sc.items()},
                 "mg": t.get("max_gain_pct"), "dd": t.get("max_dd_pct"), "ret": t.get("ret_pct"), "res": t["result"], "n": t.get("sessions"),
                 "last": float(d["Close"].iloc[-1]), "t1c": f["t1c"], "stc": f["stc"],
                 "hi": float(aft["High"].max() / t["entry"] - 1) * 100 if len(aft) else None})
R = pd.DataFrame(rows)
R = R.sort_values("s_" + KEY, ascending=False).drop_duplicates(["date", "s"])
q = M["q"][KEY]
print(f"\nالترتيب بدرجة «{KEY}» · حدود الدراسة (وسيط/80%/90%/95%/98%): {[round(x, 3) for x in q]}")
print("\n==== لو «الجاهزة» كانت شغّالة: أعلى 3 كل يوم (والنتيجة لين اليوم) ====")
tot = {"n": 0, "win": 0, "stop": 0, "mg": [], "all_mg": []}
for day, g in R.groupby("date"):
    top = g.head(3)
    rest = g.iloc[3:]
    print(f"\n{day} — {len(g)} توصية · وسيط أقصى ربح للكل {np.nanmedian(pd.to_numeric(g.mg, errors='coerce')):.1f}%")
    for _, r in top.iterrows():
        print(f"   🎯 {r.s:5s} [{r.src:4s}] {r.label} · دخول {r.entry} وقف {r.stop} هدف {r.t1} (+{r.t1c * 100:.0f}%/−{r.stc * 100:.0f}%) · 🩸 {r.pb:.2f} 🧨 {r.p:.3f}"
              f" · الدرجة {r['s_' + KEY]:+.3f} · أقصى ربح {r.mg}% · أقصى نزول {r.dd}% · الآن {r.ret}% · {r.res}")
        tot["n"] += 1
        tot["win"] += r.res in WIN
        tot["stop"] += r.res == "stop"
        if r.mg is not None:
            tot["mg"].append(r.mg)
    tot["all_mg"] += [x for x in rest.mg if x is not None]
    best = g.sort_values("mg", ascending=False).head(3)
    print("   (أعلى 3 فعلياً ذاك اليوم: " + " · ".join(f"{r.s} {r.mg}% ترتيبه {int((g['s_' + KEY] > r['s_' + KEY]).sum()) + 1}" for _, r in best.iterrows()) + ")")
if tot["n"]:
    print(f"\nالمجموع: أعلى 3 باليوم = {tot['n']} توصية · وصلت الهدف 1: {tot['win']} · وقف: {tot['stop']} · متوسط أقصى ربح {np.mean(tot['mg']):.1f}% (الباقي {np.mean(tot['all_mg']) if tot['all_mg'] else float('nan'):.1f}%)"
          f" · وسيط {np.median(tot['mg']):.1f}% (الباقي {np.median(tot['all_mg']) if tot['all_mg'] else float('nan'):.1f}%)")
# ترابط الدرجة مع أقصى ربح (كل التوصيات)
ok = R[pd.to_numeric(R.mg, errors="coerce").notna()]
for k in M["w"]:
    print(f"ترابط درجة «{k}» مع أقصى ربح (سبيرمان): {ok['s_' + k].corr(pd.to_numeric(ok.mg), method='spearman'):+.3f} (n={len(ok)})")
print("\n==== 🎯 قائمة اليوم: أحدث التوصيات (آخر جلستين) مرتبة ====")
last_days = sorted(R.date.unique())[-2:]
for _, r in R[R.date.isin(last_days)].head(12).iterrows():
    print(f"   {r.date} {r.s:5s} [{r.src:4s}] {r.label} · دخول {r.entry} · الآن {r['last']:.4g} · وقف {r.stop} · هدف {r.t1} (+{r.t1c * 100:.0f}%/−{r.stc * 100:.0f}%) · 🩸 {r.pb:.2f} 🧨 {r.p:.3f}"
          f" · الدرجة {r['s_' + KEY]:+.3f} ({'أعلى 5%' if r['s_' + KEY] >= q[3] else 'أعلى 10%' if r['s_' + KEY] >= q[2] else 'أعلى 20%' if r['s_' + KEY] >= q[1] else 'عادي'}) · {r.res}")
