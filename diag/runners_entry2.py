"""
🚀 الطيارات — متابعة دراسة الدخول (طلب أبوفرح: «شغل 1»): هل ضعف 2026 من الإشارة ولا من السوق؟ وهل أفضلية تحت 1$ تصمد؟

الدراسة الأولى (runners_entry.py) كانت ناقصة بثلاث نقاط، وهذي تعالجها:
  1) العشوائي كان مجمّع على 3 سنين ← الحين **مطابق بالسنة** (A = مجمّع قديم · B = نفس السنة · C = نفس السنة + نفس فئة السعر تحت/فوق 1$).
  2) ما فيه تكاليف ← الحين بعد التكاليف (تحت 1$: 4% · تحت 5$: 2% · غير كذا: 1% للصفقة المنفذة، للإشارة وللعشوائي بسعره هو).
  3) أفضلية تحت 1$ طلعت من تقسيمات كثيرة (احتمال صدفة) ← الحين نكررها على إشارات «🟡 قريب» لوحدها (V4 بدون الجاهز: مستقلة)،
     ونفصلها بالسنة والسيولة.
نفس الإشارات ونفس آلية الأوامر والمحاكاة من runners_entry.py بالضبط.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from backtest_runners2 import first_only  # noqa: E402
import runners_entry as RE  # noqa: E402

COST = lambda px: np.where(px < 1, 0.04, np.where(px < 5, 0.02, 0.01))
MIN_POOL = int(__import__("os").environ.get("MIN_POOL", 150))


def main():
    SG, RD = RE.run()
    print(f"\nإشارات V4+: {len(SG)} · عشوائي {len(RD)}", flush=True)
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "P"} for x in SG])
    df["i"] = np.arange(len(df))
    df["year"] = df.date.str[:4]
    df["sub1"] = df.px < 1
    rd = pd.DataFrame([{k: v for k, v in x.items() if k != "P"} for x in RD])
    rd["year"] = rd.date.str[:4]
    rd["sub1"] = rd.px < 1
    qa = np.nanquantile(rd.atr20, [0.2, 0.4, 0.6, 0.8])
    rd["q"] = np.searchsorted(qa, rd.atr20.fillna(0).values)
    df["q"] = np.searchsorted(qa, df.atr20.fillna(0).values)
    rng = np.random.default_rng(11)
    cache = {}

    def pool(key):
        """key = (year|None, sub1|None, q) → مجموعة عشوائي (حد أقصى 2500) + أسعارها."""
        if key in cache:
            return cache[key]
        m = (rd.q == key[2]).values.copy()
        if key[0] is not None:
            m &= (rd.year == key[0]).values
        if key[1] is not None:
            m &= (rd.sub1 == key[1]).values
        ids = np.where(m)[0]
        if len(ids) < MIN_POOL:
            cache[key] = None
            return None
        if len(ids) > 2500:
            ids = rng.choice(ids, 2500, replace=False)
        cache[key] = (RE.stack([RD[k] for k in ids]), rd.px.values[ids])
        return cache[key]

    def get(mode, year, sub1, q):
        keys = {"A": [(None, None, q)],
                "B": [(year, None, q), (None, None, q)],
                "C": [(year, sub1, q), (None, sub1, q), (None, None, q)]}[mode]
        for k in keys:
            p = pool(k)
            if p is not None:
                return p
        return None

    Sall = RE.stack(SG)
    LV = RE.levels(df)
    cost_s = COST(df.px.values)
    R = {}
    t0 = time.time()
    for k, (kind, lv, s) in LV.items():
        has, hit, stp, ret = RE.simulate(Sall, lv, s, kind)
        out = {"has": has, "hit": hit, "ret": ret, "net": ret - cost_s * has}
        for mode in "ABC":
            eg, en = np.full(len(df), np.nan), np.full(len(df), np.nan)
            for i in range(len(df)):
                p = get(mode, df.year.iat[i], bool(df.sub1.iat[i]), int(df.q.iat[i]))
                if p is None:
                    continue
                P, px_r = p
                h2, _, _, r2 = RE.simulate(P, lv[i], s[i], kind)
                eg[i], en[i] = r2.mean(), (r2 - COST(px_r) * h2).mean()
            out["g" + mode], out["n" + mode] = eg, en
        R[k] = out
        print(f"  {k} · {round(time.time() - t0)}ث", flush=True)

    def stat(a, b, g):
        ok = np.isfinite(a) & np.isfinite(b)
        d = pd.Series(a[ok] - b[ok]).groupby(g[ok]).mean()
        return (a[ok] - b[ok]).mean() * 100, 1.96 * d.std() / max(1, len(d)) ** 0.5 * 100

    def report(mask, title, entries=("E1", "E2", "E5", "E6")):
        g = first_only(df[mask])
        if len(g) < 25:
            print(f"\n── {title}: n={len(g)} (قليل)"); return
        ix, st = g.i.values, g.t.values
        print(f"\n══ {title} — n={len(g)} · {g.t.nunique()} سهم ══")
        print(f"{'':<4}{'انملأ':>6}{'قبل التكاليف':>13}{'فرق مجمّع':>13}{'فرق نفس السنة':>16}{'فرق سنة+سعر':>15}{'│ بعد التكاليف':>15}{'فرق سنة+سعر':>16}")
        for k in entries:
            r = R[k]
            fa, fb = stat(r["ret"][ix], r["gA"][ix], st), stat(r["ret"][ix], r["gB"][ix], st)
            fc, fn = stat(r["ret"][ix], r["gC"][ix], st), stat(r["net"][ix], r["nC"][ix], st)
            print(f"{k:<4}{r['has'][ix].mean()*100:5.0f}%{r['ret'][ix].mean()*100:+12.1f}%"
                  f"{fa[0]:+8.1f}±{fa[1]:<4.1f}{fb[0]:+10.1f}±{fb[1]:<4.1f}{fc[0]:+9.1f}±{fc[1]:<4.1f}"
                  f"{r['net'][ix].mean()*100:+10.1f}%{fn[0]:+10.1f}±{fn[1]:<4.1f}")

    ready = df.V5 | df.V6
    report(ready, "🚀 جاهز — كل الإشارات", ("E0", "E1", "E2", "E3", "E4", "E5", "E6"))
    for y in sorted(df.year.unique()):
        report(ready & (df.year == y), f"🚀 جاهز · سنة {y}")
    report(ready & df.sub1, "🚀 جاهز · تحت 1$")
    report(ready & ~df.sub1, "🚀 جاهز · فوق 1$")
    print("\n" + "#" * 70 + "\n# تكرار مستقل: إشارات 🟡 قريب فقط (V4 بدون الجاهز — ما تتداخل مع اللي فوق)\n" + "#" * 70)
    near = ~ready
    report(near, "🟡 قريب — كل الإشارات")
    report(near & df.sub1, "🟡 قريب · تحت 1$")
    report(near & ~df.sub1, "🟡 قريب · فوق 1$")
    for y in sorted(df.year.unique()):
        report(near & df.sub1 & (df.year == y), f"🟡 قريب · تحت 1$ · سنة {y}")
    print("\n" + "#" * 70 + "\n# تحت 1$ (جاهز + قريب) حسب السيولة اليومية\n" + "#" * 70)
    sub = df.sub1
    report(sub & (df.dv20 < 1e5), "تحت 1$ · سيولة < 100 ألف$")
    report(sub & (df.dv20 >= 1e5) & (df.dv20 < 4e5), "تحت 1$ · سيولة 100-400 ألف$")
    report(sub & (df.dv20 >= 4e5), "تحت 1$ · سيولة ≥ 400 ألف$")
    # حساسية التكاليف تحت 1$
    g = first_only(df[sub])
    if len(g) > 40:
        ix = g.i.values
        r = R["E1"]
        for c in (0.04, 0.08, 0.12):
            net = r["ret"][ix] - c * r["has"][ix]
            print(f"\nحساسية: تحت 1$ (جاهز+قريب) E1 بتكلفة {c*100:.0f}% → متوسط {net.mean()*100:+.1f}% · وسيط {np.median(net)*100:+.1f}%", end="")
    print("\n\n⚠ تشخيص فقط — ليس توصية. الفرق = عائد الإشارة − عشوائي بنفس آلية الأمر (±95%، مجمّع بالسهم).")


if __name__ == "__main__":
    main()
