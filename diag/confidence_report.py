"""
تقرير أداة الثقة (المرحلة 2) — على نفس البيانات المحفوظة من تشغيل سابق (DATA_RUN = رقم التشغيل).
- نفس Walk-forward (3 مراحل)، والمعايرة من فترات الاختبار.
- مستويات الثقة على ✅+🟡: عالية = أعلى 25%، متوسطة = الوسط 50%، منخفضة = أقل 25%.
- إحصائيات كل مستوى (نجاح/وقف/متوسط العائد/الوسيط) لكل مرحلة لحالها (ثبات) ولكلهم مع بعض.
- يطبع النموذج النهائي JSON (مع إحصائيات المستويات) بين علامتين.
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


def load():
    rid = os.environ.get("DATA_RUN")
    subprocess.run(["gh", "run", "download", rid, "-n", "diag-out", "-D", "diag_in"], check=True)
    return pd.read_csv("diag_in/confidence_data.csv.gz")


def pav(points):
    blocks = [[x * n, y * n, n] for x, y, n in points]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] / blocks[i][2] > blocks[i + 1][1] / blocks[i + 1][2]:
            blocks[i] = [a + b for a, b in zip(blocks[i], blocks[i + 1])]
            del blocks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    return [[b[0] / b[2], b[1] / b[2]] for b in blocks]


def tier_stats(g):
    if not len(g):
        return {"n": 0}
    win = g[g.res == "hit"]
    lose = g[g.res == "stop"]
    return {"n": int(len(g)), "hit": float(g.hit.mean()), "stop": float((g.res == "stop").mean()),
            "open": float((g.res == "open").mean()), "mean": float(g.xret.mean()), "median": float(g.xret.median()),
            "avg_win": float(win.xret.mean()) if len(win) else None, "avg_loss": float(lose.xret.mean()) if len(lose) else None,
            "ret20": float(g.ret20.mean())}


def line(name, s):
    if not s.get("n"):
        return f"  {name}: لا يوجد"
    return (f"  {name:<10} n={s['n']:<5} نجاح {pct(s['hit'])} · وقف {pct(s['stop'])} · مفتوحة {pct(s['open'])} · "
            f"متوسط العائد {s['mean'] * 100:+.2f}% · وسيط {s['median'] * 100:+.1f}% · "
            f"متوسط الربح لما ينجح {(s['avg_win'] or 0) * 100:+.1f}% · متوسط الخسارة بالوقف {(s['avg_loss'] or 0) * 100:+.1f}%")


def main():
    df = load()
    feats = [c for c in df.columns if c not in ("t", "date", "status", "hit", "res", "xret", "sess", "u10", "ret20")]
    pop = df[df.score >= 40].copy()
    dates = np.sort(pop.date.unique())
    cuts = [dates[int(len(dates) * q)] for q in (0.5, 0.67, 0.83)] + ["9999"]
    oos = []
    for k in range(3):
        tr = pop[pop.date < cuts[k]]
        te = pop[(pop.date >= cuts[k]) & (pop.date < cuts[k + 1])]
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
    print(f"حدود المستويات (احتمال معاير): عالية ≥ {hi:.3f} · متوسطة ≥ {lo:.3f}")
    print(f"AUC خارج العينة: الكل {auc(O.hit, O.p):.3f} · ✅+🟡 {auc(S.hit, S.p):.3f} · نقاط فيصل {auc(O.hit, O.score):.3f}")
    print("\n==== ✅+🟡 حسب مستوى الثقة — كل فترات الاختبار ====")
    stats = {}
    for t in ("عالية", "متوسطة", "منخفضة"):
        stats[t] = tier_stats(S[S.tier == t])
        print(line(t, stats[t]))
    allst = tier_stats(S)
    print(line("الكل", allst))
    print("\n==== الثبات: نفس المقارنة في كل فترة اختبار لحالها ====")
    for k in (1, 2, 3):
        F = S[S.fold == k]
        print(f" المرحلة {k} ({F.date.min()} → {F.date.max()}):")
        for t in ("عالية", "متوسطة", "منخفضة"):
            print("   " + line(t, tier_stats(F[F.tier == t])))
    print("\n==== ✅ مطابق و 🟡 قريب لحالهم ====")
    for st, nm in (("ready", "✅"), ("near", "🟡")):
        for t in ("عالية", "متوسطة", "منخفضة"):
            print("  " + nm + line(t, tier_stats(S[(S.status == st) & (S.tier == t)])))
    print("\n==== تحت/فوق 1$ ====")
    for nm, m in (("تحت 1$", S.price_log < 0), ("فوق 1$", S.price_log >= 0)):
        for t in ("عالية", "منخفضة"):
            print("  " + nm + line(t, tier_stats(S[m & (S.tier == t)])))
    # النموذج النهائي على كل البيانات
    Z, mu, sd = prep(pop, feats)
    w = fit_logit(Z, pop.hit.values.astype(float))
    model = {"version": datetime.now().strftime("%Y-%m-%d"), "target": "T1 before close<stop within 20 sessions",
             "intercept": float(w[0]), "coef": {f: float(x) for f, x in zip(feats, w[1:])},
             "mean": {f: float(x) for f, x in zip(feats, mu)}, "std": {f: float(x) for f, x in zip(feats, sd)},
             "calibration": cal, "base_rate": allst["hit"], "n_train": int(len(pop)),
             "tiers": {"high": hi, "mid": lo}, "tier_stats": stats, "all_stats": allst,
             "oos_auc": auc(O.hit, O.p), "oos_auc_sig": auc(S.hit, S.p), "oos_auc_score": auc(O.hit, O.score),
             "period": [str(df.date.min()), str(df.date.max())]}
    print("\nMODEL_JSON_BEGIN")
    print(json.dumps(model, ensure_ascii=False))
    print("MODEL_JSON_END")


if __name__ == "__main__":
    main()
