"""
فيصل المطوّر — طرق الخروج حسب مستوى الثقة (تشخيص فقط).
نفس بيانات أداة الثقة (Alpaca ~3 سنين، كل جلستين) + لكل إشارة نحسب نتيجة كذا طريقة خروج:
  A  الحالية (السجل): الهدف الأول، الوقف بإغلاق اليوم تحت وقف الخطة، 20 جلسة.
  B  وقف −10% من الدخول (بالإغلاق) أو وقف الخطة إذا أقرب.
  C  وقف −15% من الدخول (بالإغلاق) أو وقف الخطة إذا أقرب.
  D  نص الكمية عند الهدف الأول، والباقي: الوقف يرتفع لسعر الدخول والهدف الثاني (20 جلسة).
  E  الحالية بس أقصى 10 جلسات.
  F  وقف الخطة باللمس (أمر وقف حقيقي: يتنفذ عند الوقف أو الافتتاح إذا فتح تحته). نفس اليوم = وقف.
  G  D + وقف −15% من الدخول.
  H  بدون وقف ولا هدف: الاحتفاظ 20 جلسة (المرجع الخام).
مستويات الثقة «خارج العينة» (Walk-forward 3 مراحل) مثل التقرير.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import faisal_rules as fr
import new_low_rule
import compliance_rule
import confidence as C
from backtest_confidence import sec_dates, fit_logit, prep, predict_raw, pct

STEP, H, DAYS = 2, 20, 1100
VARIANTS = "ABCDEFGH"


def sim(fut, entry, stop, t1, t2, *, max_days=20, half_be=False, touch=False):
    """يرجع العائد النهائي للصفقة."""
    fut = fut.iloc[:max_days]
    if half_be:
        part = None
        cur_stop = stop
        for _, r in fut.iterrows():
            if part is None:
                if t1 and r["High"] >= t1:
                    part = 0.5 * (t1 / entry - 1)
                    cur_stop = entry
                    if t2 and r["High"] >= t2:
                        return part + 0.5 * (t2 / entry - 1)
                    continue
                if cur_stop and r["Close"] < cur_stop:
                    return r["Close"] / entry - 1
            else:
                if t2 and r["High"] >= t2:
                    return part + 0.5 * (t2 / entry - 1)
                if r["Close"] < cur_stop:
                    return part + 0.5 * (r["Close"] / entry - 1)
        last = float(fut["Close"].iloc[-1]) / entry - 1
        return (part + 0.5 * last) if part is not None else last
    for _, r in fut.iterrows():
        if touch and stop and r["Low"] <= stop:
            fill = min(stop, r["Open"]) if r["Open"] < stop else stop
            return fill / entry - 1
        if t1 and r["High"] >= t1:
            return t1 / entry - 1
        if not touch and stop and r["Close"] < stop:
            return r["Close"] / entry - 1
    return float(fut["Close"].iloc[-1]) / entry - 1


def run_one(args):
    t, d, info, cfg, dil, earn, mkt = args
    out = []
    n = len(d)
    U = cfg["universe"]
    for end in range(220, n - H, STEP):
        hist = d.iloc[:end]
        px = float(hist["Close"].iloc[-1])
        if not (U["min_price"] <= px <= U["max_price"]) or float(hist["Volume"].tail(30).mean()) < U["min_avg_volume"] * 0.7:
            continue
        try:
            a = fr.analyze(hist, info, cfg)
            if not a:
                continue
            new_low_rule.apply(a, hist)
            compliance_rule.apply_faisal(a, hist)
        except Exception:
            continue
        if a["gate_fail"] or a["irtikaz"]["score"] < 40:
            continue
        day = hist.index[-1]
        try:
            f = C.features(a, hist, None, dil, earn, mkt.get(day))
        except Exception:
            continue
        p = a["plan"]
        stop, t1 = p.get("stop"), p.get("t1")
        ft = [x for x in (p.get("four_targets") or []) if x and t1 and x > t1 * 1.02]
        t2 = ft[0] if ft else (t1 * 1.15 if t1 else None)
        fut = d.iloc[end:end + H]
        hit = 0
        for _, r in fut.iterrows():
            if t1 and r["High"] >= t1:
                hit = 1
                break
            if stop and r["Close"] < stop:
                break
        y = {"y_hit": hit,
             "A": sim(fut, px, stop, t1, t2),
             "B": sim(fut, px, max(stop or 0, px * 0.90), t1, t2),
             "C": sim(fut, px, max(stop or 0, px * 0.85), t1, t2),
             "D": sim(fut, px, stop, t1, t2, half_be=True),
             "E": sim(fut, px, stop, t1, t2, max_days=10),
             "F": sim(fut, px, stop, t1, t2, touch=True),
             "G": sim(fut, px, max(stop or 0, px * 0.85), t1, t2, half_be=True),
             "H": float(fut["Close"].iloc[-1]) / px - 1}
        out.append({"t": t, "date": str(day.date()), "status": a["irtikaz"]["status"], **y, **f})
    return out


def collect():
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    daily = {}
    for i in range(0, len(tickers), 100):
        try:
            daily.update(A.daily(tickers[i:i + 100], days=DAYS))
        except Exception as e:
            print("daily err", repr(e)[:120], flush=True)
    daily = {t: d for t, d in daily.items() if len(d) > 260}
    iwm = A.daily(["IWM"], days=DAYS).get("IWM")
    closes = pd.DataFrame({t: d["Close"] for t, d in daily.items()})
    above = (closes > closes.rolling(50).mean()).astype(float).where(closes.rolling(50).count() >= 50)
    breadth = (above.sum(axis=1) / above.count(axis=1).replace(0, np.nan)).astype(float)
    iw = iwm["Close"].astype(float)
    iwm_up = (iw > iw.rolling(50).mean()).astype(float)
    iwm_r20 = (iw / iw.shift(20) - 1) * 100
    mkt = {}
    for day in closes.index:
        br, up, r20 = breadth.get(day, np.nan), iwm_up.get(day, np.nan), iwm_r20.get(day, np.nan)
        mkt[day] = {"breadth": 0.4 if pd.isna(br) else float(br), "iwm_up": 0.5 if pd.isna(up) else float(up),
                    "iwm_ret20": 0.0 if pd.isna(r20) else float(r20)}
    infos = scanner.get_info(list(daily), cfg)
    sec = {}
    for t in daily:
        try:
            sec[t] = sec_dates(t)
        except Exception:
            sec[t] = ([], [])
    jobs = [(t, d, infos.get(t) or {}, cfg, sec[t][0], sec[t][1], mkt) for t, d in daily.items()]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=3)):
            rows += r
            if i % 100 == 0:
                print("analyze", i, len(rows), round(time.time() - t0), flush=True)
    return pd.DataFrame(rows)


NAMES = {"A": "الحالية (الهدف 1 + وقف الخطة بالإغلاق)", "B": "وقف −10% بالإغلاق", "C": "وقف −15% بالإغلاق",
         "D": "نص عند الهدف 1 + الباقي وقف الدخول وهدف 2", "E": "الحالية بأقصى 10 جلسات",
         "F": "وقف الخطة باللمس (أمر وقف)", "G": "D + وقف −15%", "H": "احتفاظ 20 جلسة بدون وقف/هدف"}


def table(g, title):
    print(f"\n## {title} (n={len(g)})")
    for v in VARIANTS:
        x = g[v] * 100
        print(f"   {v} {NAMES[v]:<42} متوسط {x.mean():+6.2f}% · وسيط {x.median():+6.1f}% · رابحة {(x > 0).mean() * 100:3.0f}% · "
              f"أسوأ 10% {np.percentile(x, 10):+6.1f}%")


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/exits_data.csv.gz", index=False, compression="gzip")
    model = json.loads((ROOT / "confidence_model.json").read_text(encoding="utf-8")) if (ROOT / "confidence_model.json").exists() else None
    feats = list(model["coef"]) if model else [c for c in df.columns if c not in ("t", "date", "status", "y_hit", *VARIANTS)]
    dates = np.sort(df.date.unique())
    cuts = [dates[int(len(dates) * q)] for q in (0.5, 0.67, 0.83)] + ["9999"]
    oos = []
    for k in range(3):
        tr = df[df.date < cuts[k]]
        te = df[(df.date >= cuts[k]) & (df.date < cuts[k + 1])]
        Z, mu, sd = prep(tr, feats)
        w = fit_logit(Z, tr.y_hit.values.astype(float))
        oos.append(te.assign(p=predict_raw(te, feats, w, mu, sd)))
    O = pd.concat(oos)
    S = O[O.status.isin(["ready", "near"])].copy()
    q = S.p.rank(pct=True)
    S["tier"] = np.where(q >= 0.75, "عالية", np.where(q >= 0.25, "متوسطة", "منخفضة"))
    print(f"الفترة {df.date.min()} → {df.date.max()} · صفوف {len(df)} · ✅+🟡 خارج العينة {len(S)}")
    table(S, "كل ✅+🟡 (خارج العينة)")
    for t in ("عالية", "متوسطة", "منخفضة"):
        table(S[S.tier == t], f"ثقة {t}")
    for st, nm in (("near", "🟡 قريب"), ("ready", "✅ مطابق")):
        table(S[(S.status == st) & (S.tier == "عالية")], f"{nm} + ثقة عالية")
    # ثبات: كل مرحلة لحالها لأفضل الطرق
    print("\n## الثبات (متوسط العائد لكل مرحلة اختبار) — ثقة عالية")
    S["fold"] = np.searchsorted(np.array(cuts[:3]), S.date.values, side="right")
    for v in VARIANTS:
        vals = [S[(S.fold == k) & (S.tier == "عالية")][v].mean() * 100 for k in (1, 2, 3)]
        print(f"   {v} {NAMES[v]:<42} " + " · ".join(f"م{k}: {x:+.2f}%" for k, x in zip((1, 2, 3), vals)))


if __name__ == "__main__":
    main()
