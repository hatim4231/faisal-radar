"""
📅 إعادة اختبار «قائمة الأسبوع» **خارج العينة بالكامل** (طلب أبوفرح 2026-11-28: «عيدها وتأكد من نتائجك»).
الخلل بالاختبار الأول (movers_weekly.py): استخدم highrisk_model.json النهائي المدرّب على 2021-01 → 2026-10، فالأسابيع المختبرة كانت من بيانات تعلّمها.
هنا: Walk-forward ربع بربع — لكل ربع (2024-Q4 → 2026-Q4) نموذج **جديد** يتدرّب على بيانات **قبله بس** (فجوة 20 يوم عشان ما يتداخل الهدف)،
بنفس الميزات (highrisk.features، 29 ميزة) ونفس الهدف (انفجار ×2 بيوم بتداول ≥ 500K$ خلال 10 جلسات) ونفس طريقة التدريب (انحدار لوجستي L2).
ثم كل جمعة بالربع: أعلى 5/10/20 بالاحتمال (سعر حقيقي 0.2-20$، سيولة ≥ 30K$، تاريخ ≥ 260 جلسة) ← الأسبوع الجاي.
للمقارنة: نفس الأسابيع بالنموذج النهائي (المنحاز) عشان نشوف حجم الانحياز. هامش 95% مجمّع بالسهم. تشخيص فقط.
"""
import os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import alpaca_data as A
import highrisk as HR
from backtest_runners2 import universe
from backtest_highrisk2 import fit_logit, auc

DAYS = 1500
M = HR.model(); FE = M["feats"]
MU0 = np.array(M["mean"])[:len(FE)]; SD0 = np.array(M["std"])[:len(FE)]; W0 = np.array(M["w"])
COST = lambda p: np.where(p < 1, 0.04, np.where(p < 5, 0.02, 0.01))


def collect():
    syms = sorted(universe())
    TR, WK, t0 = [], [], time.time()
    for i in range(0, len(syms), 100):
        try:
            raw = A.daily(syms[i:i + 100], days=DAYS)
        except Exception as e:
            print("daily err", repr(e)[:90], flush=True); continue
        for s, d in raw.items():
            if d is None or len(d) < 300:
                continue
            d = d[~d.index.duplicated(keep="last")]
            try:
                F = HR.features(d)
            except Exception:
                continue
            X = F[FE].to_numpy(np.float32)
            o, h, l, c = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
            ex = F.ex.values; praw = F.praw.values; dv = F.dv20.values
            idx = pd.to_datetime(d.index); idx = idx.tz_localize(None) if idx.tz is not None else idx
            n = len(c)
            exc = np.r_[0, np.cumsum(ex.astype(int))]
            y10 = np.array([exc[min(t + 11, n)] - exc[t + 1] > 0 if t + 10 < n else np.nan for t in range(n)], dtype=float)
            elig = (np.arange(n) >= 260) & (praw >= 0.2) & (praw <= 20) & (dv >= 30_000)
            for t in np.flatnonzero(elig & np.isfinite(y10))[::3]:
                TR.append((idx[t].value, X[t], y10[t], s))
            fri = np.flatnonzero(elig & (idx.dayofweek == 4) & (np.arange(n) + 5 < n))
            for t in fri:
                a, b = t + 1, t + 6
                mo = o[a]
                WK.append({"t": s, "date": idx[t], "x": X[t], "px": praw[t], "exW": bool(ex[a:b].any()),
                           "hit50": bool(h[a:b].max() >= mo * 1.5), "hit100": bool(h[a:b].max() >= mo * 2),
                           "clW_o": c[b - 1] / mo - 1, "loW_o": l[a:b].min() / mo - 1})
        if (i // 100) % 10 == 0:
            print(f"  {i + 100}/{len(syms)} · تدريب {len(TR):,} · جمعات {len(WK):,} · {round(time.time() - t0)}ث", flush=True)
    return TR, WK


def ci(x, g):
    x = pd.Series(x, dtype=float).reset_index(drop=True); g = pd.Series(np.asarray(g))
    ok = x.notna()
    if ok.sum() < 10:
        return "   -   "
    m = x[ok].groupby(g[ok]).mean()
    return f"{x[ok].mean()*100:6.1f}±{1.96*m.std()/max(1,len(m))**.5*100:.1f}"


def main():
    TR, WK = collect()
    tdate = np.array([r[0] for r in TR], dtype="int64"); TX = np.vstack([r[1] for r in TR]).astype(float); Ty = np.array([r[2] for r in TR])
    Wk = pd.DataFrame([{k: v for k, v in r.items() if k != "x"} for r in WK]); WX = np.vstack([r["x"] for r in WK]).astype(float)
    print(f"\nصفوف تدريب: {len(Ty):,} (انفجار خلال 10 جلسات {Ty.mean()*100:.2f}%) · سهم-جمعة {len(Wk):,} · {Wk.date.min().date()} → {Wk.date.max().date()}", flush=True)
    # احتمال النموذج النهائي (المنحاز) للمقارنة
    Z0 = np.clip((np.where(np.isfinite(WX), WX, MU0) - MU0) / np.where(SD0 == 0, 1, SD0), -5, 5)
    Wk["p_final"] = 1 / (1 + np.exp(-(W0[0] + Z0 @ W0[1:])))
    Wk["p_oos"] = np.nan
    qs = pd.date_range("2024-10-01", Wk.date.max() + pd.Timedelta(days=1), freq="QS")
    for qa in qs:
        qb = qa + pd.offsets.QuarterBegin(startingMonth=1)
        mtest = (Wk.date >= qa) & (Wk.date < qb)
        if not mtest.any():
            continue
        cut = (qa - pd.Timedelta(days=20)).value
        mt = tdate < cut
        X = TX[mt]; y = Ty[mt]
        mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1
        Z = np.clip((np.where(np.isfinite(X), X, mu) - mu) / sd, -5, 5)
        w = fit_logit(Z, y)
        Zt = np.clip((np.where(np.isfinite(WX[mtest.values]), WX[mtest.values], mu) - mu) / sd, -5, 5)
        Wk.loc[mtest, "p_oos"] = 1 / (1 + np.exp(-(w[0] + Zt @ w[1:])))
        # AUC خارج العينة على أيام التدريب للربع نفسه (مو الجمعات بس)
        mq = (tdate >= qa.value) & (tdate < qb.value)
        Zq = np.clip((np.where(np.isfinite(TX[mq]), TX[mq], mu) - mu) / sd, -5, 5)
        a = auc(Ty[mq], Zq @ w[1:]) if mq.any() else float("nan")
        print(f"  ربع {qa.date()}: تدريب {mt.sum():,} صف (لين {pd.Timestamp(cut).date()}) · AUC خارج العينة {a:.3f} · جمعات {mtest.sum():,}", flush=True)
    Wk = Wk[Wk.p_oos.notna()].copy()
    cost = COST(Wk.px.values)
    Wk["r_hold"] = Wk.clW_o - cost
    Wk["r_lim50"] = np.where(Wk.hit50, 0.5, Wk.clW_o) - cost
    Wk["r_lim100"] = np.where(Wk.hit100, 1.0, Wk.clW_o) - cost
    Wk["ds"] = Wk.date.dt.strftime("%Y-%m-%d")
    mid = Wk.ds.sort_values().iloc[len(Wk) // 2]
    print(f"\nالأسابيع المختبرة: {Wk.ds.nunique()} جمعة · {Wk.ds.min()} → {Wk.ds.max()} · المنتصف {mid}")
    hdr = f"{'المجموعة':<34} {'n':>6} {'انفجر ×2':>11} {'لمس +50%':>11} {'لمس +100%':>11} {'الاثنين→الجمعة':>15} {'Limit +50%':>12} {'Limit +100%':>12} {'أسوأ نزول':>9}"
    for col, title in (("p_oos", "✅ خارج العينة (نموذج كل ربع ما شاف بياناته)"), ("p_final", "⚠ النموذج النهائي (منحاز — للمقارنة بس)")):
        Wk["rank"] = Wk.groupby("ds")[col].rank(ascending=False, method="first")
        print("\n" + "#" * 72 + f"\n# {title}\n" + "#" * 72)
        for nm, part in (("كل الفترة", Wk), ("النص الأول", Wk[Wk.ds < mid]), ("النص الثاني", Wk[Wk.ds >= mid])):
            print(f"\n── {nm} ──\n{hdr}")
            for lab, m in (("كل الأسهم", part.index == part.index), ("أعلى 20", part["rank"] <= 20), ("أعلى 10", part["rank"] <= 10), ("أعلى 5", part["rank"] <= 5)):
                g = part[m]; G = g.t.values
                print(f"{lab:<34} {len(g):>6} {ci(g.exW,G):>11} {ci(g.hit50,G):>11} {ci(g.hit100,G):>11} {ci(g.r_hold,G):>15} {ci(g.r_lim50,G):>12} {ci(g.r_lim100,G):>12} {g.loW_o.median()*100:8.0f}%")
        t10 = Wk[Wk["rank"] <= 10]
        by = t10.groupby(t10.ds.str[:7]).agg(ex=("exW", "mean"), r=("r_lim100", "mean"), n=("t", "size"))
        print("أعلى 10 بالشهر (انفجر% · Limit+100): " + " · ".join(f"{k} {v.ex*100:.0f}%/{v.r*100:+.0f}%" for k, v in by.iterrows()))
    print("\n⚠ تشخيص — التكاليف 4%/2%/1%، بيانات Alpaca معدّلة للتقسيم. الانفجار = أعلى ≥ ×2 إغلاق أمس وتداول ≥ 500K$.")


if __name__ == "__main__":
    main()
