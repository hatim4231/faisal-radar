"""
🚀 الطيارات — كم مدة الاحتفاظ المناسبة؟ (طلب أبوفرح: «60 جلسة كثير»)

نفس إشارات الطيارات (V4+، 7,742 إشارة، 3 سنين مع المشطوبة) ونفس العشوائي المطابق (نفس السنة + فئة السعر تحت/فوق 1$ + شريحة التذبذب)،
بس نغيّر المدة: 10 / 20 / 30 / 60 جلسة، ونقيس شيئين:
  أ) احتمال الطيران: أعلى سعر خلال المدة ≥ +30% / +50% / +100% من إغلاق الإشارة — الإشارة مقابل العشوائي (هذي الأفضلية الحقيقية المثبتة قبل).
  ب) خطة التداول: دخول فوري (افتتاح بكرة) + هدف +50% أو +100% + نفس الوقف (إغلاق تحت القاع الكبير −10%) + خروج بنهاية المدة،
     بعد التكاليف (4%/2%/1%)، الإشارة مقابل عشوائي بنفس الآلية والسنة والسعر.
الفرق = الإشارة − العشوائي (±95%، مجمّع بالسهم).
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
HS = (10, 20, 30, 60)
XS = (0.30, 0.50, 1.00)
PLANS = ((1.5, "هدف +50%"), (2.0, "هدف +100%"))
MIN_POOL = 150


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
    rng = np.random.default_rng(13)
    cache = {}

    def pool(key):
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
        P = RE.stack([RD[k] for k in ids])
        cache[key] = {"P": P, "px": rd.px.values[ids],
                      "mfe": {h: P["ph"][:, 1:h + 1].max(1) - 1 for h in HS}}
        return cache[key]

    def get(year, sub1, q):
        for k in ((year, sub1, q), (None, sub1, q), (None, None, q)):
            p = pool(k)
            if p is not None:
                return p
        return None

    Sall = RE.stack(SG)
    LV = RE.levels(df)
    _, lv, s = LV["E1"]
    cost_s = COST(df.px.values)
    n = len(df)
    pools = [get(df.year.iat[i], bool(df.sub1.iat[i]), int(df.q.iat[i])) for i in range(n)]
    t0 = time.time()

    # أ) احتمال الطيران
    prob = {}
    for h in HS:
        smfe = Sall["ph"][:, 1:h + 1].max(1) - 1
        for x in XS:
            sig = (smfe >= x).astype(float)
            rnd = np.array([np.nan if p is None else (p["mfe"][h] >= x).mean() for p in pools])
            prob[(h, x)] = (sig, rnd)
    print(f"  احتمال الطيران محسوب · {round(time.time() - t0)}ث", flush=True)

    # ب) خطة التداول (دخول فوري)
    plan = {}
    for h in HS:
        for tgt, _ in PLANS:
            has, hit, stp, ret = RE.simulate(Sall, lv, s, "open", H=h, tgt=tgt)
            net = ret - cost_s * has
            er = np.full(n, np.nan)
            en = np.full(n, np.nan)
            for i in range(n):
                p = pools[i]
                if p is None:
                    continue
                h2, _, _, r2 = RE.simulate(p["P"], lv[i], s[i], "open", H=h, tgt=tgt)
                er[i], en[i] = r2.mean(), (r2 - COST(p["px"]) * h2).mean()
            plan[(h, tgt)] = {"hit": hit, "stp": stp, "ret": ret, "net": net, "er": er, "en": en}
        print(f"  خطة {h} جلسة · {round(time.time() - t0)}ث", flush=True)

    def stat(a, b, g):
        ok = np.isfinite(a) & np.isfinite(b)
        d = pd.Series(a[ok] - b[ok]).groupby(g[ok]).mean()
        return (a[ok] - b[ok]).mean() * 100, 1.96 * d.std() / max(1, len(d)) ** 0.5 * 100

    def report(mask, title):
        g = first_only(df[mask])
        if len(g) < 30:
            print(f"\n── {title}: n={len(g)} (قليل)"); return
        ix, st = g.i.values, g.t.values
        print(f"\n══ {title} — n={len(g)} · {g.t.nunique()} سهم ══")
        print("أ) احتمال الطيران (أعلى سعر خلال المدة من إغلاق الإشارة): الإشارة | عشوائي مطابق | الفرق")
        for h in HS:
            cells = []
            for x in XS:
                sg, rn = prob[(h, x)]
                d, ci = stat(sg[ix], rn[ix], st)
                cells.append(f"+{int(x*100)}%: {sg[ix].mean()*100:4.1f}|{np.nanmean(rn[ix])*100:4.1f}|{d:+5.1f}±{ci:.1f}")
            print(f"   {h:>2} جلسة   " + "   ".join(cells))
        print("ب) خطة التداول (دخول بافتتاح بكرة، وقف القاع−10%، خروج بنهاية المدة، بعد التكاليف):")
        print(f"   {'المدة':<8}{'الخطة':<11}{'وصل الهدف':>10}{'وقف':>6}{'عائد صافي':>11}{'عشوائي':>9}{'الفرق':>14}")
        for h in HS:
            for tgt, nm in PLANS:
                r = plan[(h, tgt)]
                d, ci = stat(r["net"][ix], r["en"][ix], st)
                print(f"   {h:>2} جلسة {nm:<11}{r['hit'][ix].mean()*100:9.1f}%{r['stp'][ix].mean()*100:5.0f}%"
                      f"{r['net'][ix].mean()*100:+10.1f}%{np.nanmean(r['en'][ix])*100:+8.1f}%{d:+8.1f}±{ci:.1f}")

    ready = df.V5 | df.V6
    report(ready, "🚀 جاهز")
    report(~ready, "🟡 قريب (مستقل)")
    report(np.ones(n, bool), "V4 كلها")
    for y in sorted(df.year.unique()):
        report(ready & (df.year == y), f"🚀 جاهز · سنة {y}")
    print("\n⚠ تشخيص فقط — ليس توصية. الأرقام المطلقة مبالغ فيها شوي (بيانات معدّلة للتقسيم)، المعتمد الفرق عن العشوائي.")


if __name__ == "__main__":
    main()
