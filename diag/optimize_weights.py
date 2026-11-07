"""
تجربة أوزان جديدة لفيصل المطوّر وعبدالله (تشخيص فقط — ما يغير الرادار).

الطريقة (عشان ما نخدع نفسنا):
- نجمع نفس بيانات backtest_features (سنة، كل 3 جلسات، Alpaca).
- نقسمها: النص الأول من السنة = «تدريب»، النص الثاني = «اختبار» (التحليل ما يشوفه وهو يختار الأوزان).
- لكل مجموعة أوزان: نحسب النقاط الجديدة، وناخذ أعلى نسبة أسهم بنفس عدد المرشحين الحالي (✅+🟡)،
  ونقارنهم بعشوائي بنفس التذبذب: الزيادة في نسبة +20% قبل −10%، والزيادة في العائد بعد 10 جلسات.
- نجرب: الأوزان الحالية · أوزان مقترحة من التحليل · أفضل أوزان بالبحث على «التدريب» ← ونشوف وش صار لها في «الاختبار».
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import backtest_features as BF

STEPS = [0, 5, 10, 15, 20, 25, 30]
PROPOSED = {
    "FAISAL": {"support": 20, "rsi": 25, "hold": 10, "candle": 5, "macd": 15, "volume": 5, "ma": 5, "upside": 15},
    "ABDULLAH": None,  # يتحدد من الأوزان الحالية + تعديل التصحيح (تحت)
}


def baselines(rnd):
    edges = rnd["atr"].quantile([0, .2, .4, .6, .8, 1]).to_numpy().copy()
    edges[0], edges[-1] = -1, 99
    q = pd.cut(rnd["atr"], edges, labels=False)
    return edges, rnd.groupby(q)["u10"].mean(), rnd.groupby(q)["ret10"].median(), rnd.groupby(q)["dd10"].mean()


def evaluate(pop, rnd, keys, w, share, adj=True):
    s = sum(pop["pc_" + k].fillna(0) * w.get(k, 0) for k in keys)
    if adj and "adj" in pop:
        s = s + pop["adj"].fillna(0)
    n = max(1, int(round(len(pop) * share)))
    sel = pop.loc[s.sort_values(ascending=False).index[:n]]
    edges, bu, br, bd = baselines(rnd)
    q = pd.cut(sel["atr"], edges, labels=False)
    eu, er, ed = q.map(bu).mean(), q.map(br).mean(), q.map(bd).mean()
    return {"n": n, "u10": sel.u10.mean(), "exp_u": eu, "ex_u": sel.u10.mean() - eu,
            "ret": sel.ret10.median(), "ex_r": sel.ret10.median() - er,
            "dd": sel.dd10.mean(), "ex_dd": sel.dd10.mean() - ed, "up30": sel.up30.mean()}


def line(name, r):
    return (f"  {name:<28} n={r['n']:<5} +20%قبل−10% {r['u10'] * 100:5.1f}% (عشوائي مماثل {r['exp_u'] * 100:4.1f}% ← {r['ex_u'] * 100:+.1f}) · "
            f"عائد10 وسيط {r['ret'] * 100:+.1f}% ({r['ex_r'] * 100:+.1f}) · نزل−10% {r['dd'] * 100:.0f}% ({r['ex_dd'] * 100:+.1f}) · طار30% {r['up30'] * 100:.0f}%")


def obj(r):
    return r["ex_u"] + r["ex_r"] - 0.5 * r["ex_dd"]


def run(df, radar, rnd_all):
    pop = df[(df.radar == radar) & (~df.status.isin(["moved", "invalid"]))].dropna(axis=1, how="all").copy()
    keys = sorted(c[3:] for c in pop.columns if c.startswith("pc_"))
    cur = {k: float(pop["w_" + k].dropna().mode().iloc[0]) for k in keys}
    share = pop.status.isin(["ready", "near"]).mean()
    dates = pop["date"].sort_values()
    mid = dates.iloc[len(dates) // 2]
    tr, te = pop[pop.date < mid], pop[pop.date >= mid]
    rtr, rte = rnd_all[rnd_all.date < mid], rnd_all[rnd_all.date >= mid]
    use_adj = radar == "FAISAL"
    print(f"\n\n######## {radar} ########")
    print(f"الشروط والأوزان الحالية: {cur}")
    print(f"نسبة المرشحين الحالية (✅+🟡): {share * 100:.1f}% · تدريب: لين {mid} (n={len(tr)}) · اختبار: من {mid} (n={len(te)})")
    prop = PROPOSED.get(radar)
    if prop is None:
        prop = dict(cur)
        if "correction" in prop:
            prop["correction"] = cur["correction"] + 10
        for k in ("tight", "consol", "touches"):
            if k in prop:
                prop[k] = max(0, cur[k] - 5)
    prop = {k: prop.get(k, 0) for k in keys}
    # بحث عشوائي على التدريب فقط
    rng = np.random.default_rng(11)
    cands = []
    for _ in range(2500):
        w = {k: float(rng.choice(STEPS)) for k in keys}
        if sum(w.values()) == 0:
            continue
        tot = sum(w.values())
        w = {k: v * 100 / tot for k, v in w.items()}
        cands.append((obj(evaluate(tr, rtr, keys, w, share, use_adj)), w))
    cands.sort(key=lambda x: -x[0])
    print("\n-- التدريب (النص الأول) --")
    print(line("الأوزان الحالية", evaluate(tr, rtr, keys, cur, share, use_adj)))
    print(line("المقترحة", evaluate(tr, rtr, keys, prop, share, use_adj)))
    print(line("أفضل بحث (تدريب)", evaluate(tr, rtr, keys, cands[0][1], share, use_adj)))
    print("\n-- الاختبار (النص الثاني — ما شافه التحليل) --")
    rc = evaluate(te, rte, keys, cur, share, use_adj)
    rp = evaluate(te, rte, keys, prop, share, use_adj)
    print(line("الأوزان الحالية", rc))
    print(line("المقترحة", rp))
    for i, (o, w) in enumerate(cands[:5]):
        print(line(f"بحث #{i + 1}", evaluate(te, rte, keys, w, share, use_adj)))
    top = pd.DataFrame([w for _, w in cands[:25]]).mean().round(0)
    print("\nمتوسط أوزان أفضل 25 بالبحث (على 100):", dict(top))
    rt = evaluate(te, rte, keys, top.to_dict(), share, use_adj)
    print(line("متوسط أفضل 25 (اختبار)", rt))
    print(f"\nالمقترحة: {prop}")
    print(f"الخلاصة {radar}: الحالية {obj(rc) * 100:+.1f} · المقترحة {obj(rp) * 100:+.1f} · متوسط البحث {obj(rt) * 100:+.1f} "
          f"(المقياس = زيادة النجاح + زيادة العائد − نص زيادة النزول، كلها مقابل العشوائي المماثل، على فترة الاختبار)")


def main():
    df = BF.collect()
    rnd = df[df.radar == "RANDOM"]
    print("الفترة:", rnd.date.min(), "→", rnd.date.max(), "· أيام-أسهم:", len(rnd))
    for r in ("FAISAL", "ABDULLAH"):
        try:
            run(df, r, rnd)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(r, "خطأ", repr(e))


if __name__ == "__main__":
    main()
