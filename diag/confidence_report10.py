"""
أداة الثقة — نسخة 10 جلسات (قرار المستخدم: مدة توصية فيصل المطوّر 10 جلسات بدل 20).
تقرا بيانات اختبار طرق الخروج المحفوظة (DATA_RUN = رقم تشغيل backtest_exits.py) وتعيد التدريب على:
  النجاح = الهدف الأول قبل الإغلاق تحت وقف الخطة خلال 10 جلسات.
  hit10 = (وصل الهدف خلال 20) و (عائد طريقة E «أقصى 10 جلسات» = عائد الطريقة الحالية A) — نفس الحساب بالضبط لما يوصل الهدف قبل الجلسة 11.
  عائد الصفقة = E (الهدف 1، أو الوقف بالإغلاق، أو إغلاق الجلسة العاشرة).
Walk-forward 3 مراحل + معايرة + مستويات (أعلى 25% / الوسط / أقل 25% من ✅+🟡) + إحصائيات لكل مستوى. يطبع JSON.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
from backtest_confidence import fit_logit, prep, predict_raw, auc, pct
from confidence_report import pav


def load():
    rid = os.environ.get("DATA_RUN")
    subprocess.run(["gh", "run", "download", rid, "-n", "diag-out", "-D", "diag_in"], check=True)
    return pd.read_csv("diag_in/exits_data.csv.gz")


def stats(g):
    if not len(g):
        return {"n": 0}
    w, m = g[g.hit == 1], g[g.hit == 0]
    return {"n": int(len(g)), "hit": float(g.hit.mean()), "lose": float((g.E < 0).mean()),
            "mean": float(g.E.mean()), "median": float(g.E.median()),
            "avg_win": float(w.E.mean()) if len(w) else None, "avg_miss": float(m.E.mean()) if len(m) else None}


def line(name, s):
    if not s.get("n"):
        return f"  {name}: لا يوجد"
    return (f"  {name:<8} n={s['n']:<5} نجاح {pct(s['hit'])} · خسرت {pct(s['lose'])} · متوسط {s['mean'] * 100:+.2f}% · "
            f"وسيط {s['median'] * 100:+.1f}% · لما تنجح {(s['avg_win'] or 0) * 100:+.1f}% · لما ما تنجح {(s['avg_miss'] or 0) * 100:+.1f}%")


def main():
    df = load()
    df["hit"] = ((df.y_hit == 1) & (df.E == df.A)).astype(int)
    model20 = json.loads((ROOT / "confidence_model.json").read_text(encoding="utf-8"))
    feats = list(model20["coef"])
    print(f"صفوف {len(df)} · نجاح خلال 20: {pct(df.y_hit.mean())} · خلال 10: {pct(df.hit.mean())}")
    dates = np.sort(df.date.unique())
    cuts = [dates[int(len(dates) * q)] for q in (0.5, 0.67, 0.83)] + ["9999"]
    oos = []
    for k in range(3):
        tr = df[df.date < cuts[k]]
        te = df[(df.date >= cuts[k]) & (df.date < cuts[k + 1])]
        Z, mu, sd = prep(tr, feats)
        w = fit_logit(Z, tr.hit.values.astype(float))
        oos.append(te.assign(p=predict_raw(te, feats, w, mu, sd), fold=k + 1))
    O = pd.concat(oos)
    O["dec"] = pd.qcut(O.p, 10, labels=False, duplicates="drop")
    cal = pav([(g.p.mean(), g.hit.mean(), len(g)) for _, g in O.groupby("dec")])
    O["pc"] = np.interp(O.p, [c[0] for c in cal], [c[1] for c in cal])
    S = O[O.status.isin(["ready", "near"])].copy()
    hi, lo = float(S.pc.quantile(0.75)), float(S.pc.quantile(0.25))
    S["tier"] = np.where(S.pc >= hi, "عالية", np.where(S.pc >= lo, "متوسطة", "منخفضة"))
    print(f"AUC خارج العينة: الكل {auc(O.hit, O.p):.3f} · ✅+🟡 {auc(S.hit, S.p):.3f} · نقاط فيصل {auc(O.hit, O.score):.3f}")
    print(f"حدود المستويات: عالية ≥ {hi:.3f} · متوسطة ≥ {lo:.3f}")
    print("\n==== ✅+🟡 حسب الثقة (10 جلسات، خارج العينة) ====")
    ts = {}
    for t in ("عالية", "متوسطة", "منخفضة"):
        ts[t] = stats(S[S.tier == t])
        print(line(t, ts[t]))
    allst = stats(S)
    print(line("الكل", allst))
    print("\n==== الثبات لكل مرحلة ====")
    for k in (1, 2, 3):
        F = S[S.fold == k]
        print(f" المرحلة {k}:")
        for t in ("عالية", "متوسطة", "منخفضة"):
            print("   " + line(t, stats(F[F.tier == t])))
    print("\n-- المعايرة (أعشار): متوقع ← فعلي --")
    for _, g in O.groupby("dec"):
        print(f"  {pct(g.p.mean())} ← {pct(g.hit.mean())} (n={len(g)})")
    Z, mu, sd = prep(df, feats)
    w = fit_logit(Z, df.hit.values.astype(float))
    model = {"version": datetime.now().strftime("%Y-%m-%d") + "-10s", "horizon": 10,
             "target": "T1 before close<stop within 10 sessions", "intercept": float(w[0]),
             "coef": {f: float(x) for f, x in zip(feats, w[1:])}, "mean": {f: float(x) for f, x in zip(feats, mu)},
             "std": {f: float(x) for f, x in zip(feats, sd)}, "calibration": cal, "base_rate": allst["hit"],
             "n_train": int(len(df)), "tiers": {"high": hi, "mid": lo}, "tier_stats": ts, "all_stats": allst,
             "oos_auc": auc(O.hit, O.p), "oos_auc_sig": auc(S.hit, S.p), "oos_auc_score": auc(O.hit, O.score),
             "period": [str(df.date.min()), str(df.date.max())]}
    print("\nMODEL_JSON_BEGIN")
    print(json.dumps(model, ensure_ascii=False))
    print("MODEL_JSON_END")


if __name__ == "__main__":
    main()
