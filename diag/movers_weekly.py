"""
🚀 تشريح الأسهم اللي تطير ×2 بيوم + «قائمة أسبوعية متوقعة» (طلب أبوفرح 2026-11-28: «شوف كيف كل يوم بعض الأسهم البيني والصينية والسنغافورية
واليابانية ترتفع فوق 100% ... راقب السلوك واعرف النمط وكيف الدخول الصحيح ... قائمة أسبوعية متوقعة»).
الانفجار = أعلى سعر ≥ ضعف إغلاق أمس وتداول اليوم ≥ 500 ألف$ (highrisk.episodes). الاحتمال = نموذج الانفجار المعاير (highrisk_model.json).
A) التشريح: قبل الانفجار بيوم — السعر الحقيقي، عدد الانفجارات السابقة، الأيام من آخر انفجار، التقسيم العكسي، الانهيار من القمة، الفوليوم، رتبة النموذج · والدولة (ياهو).
B) القائمة الأسبوعية: كل جمعة نرتب كل الأسهم (سعر حقيقي 0.2-20$، سيولة ≥ 30 ألف$) بالاحتمال ← أعلى 10/20/50 ← الأسبوع الجاي:
   انفجر؟ · أعلى سعر · خطط الدخول/الخروج بعد التكاليف (4%/2%/1%) مقابل كل الأسهم بنفس الجمعة. نصين بالتاريخ.
C) التوقيت داخل اليوم (شموع ساعة، آخر 9 شهور): متى بدأ (أفتر أمس/بري/الجلسة)، وقت القمة، كم ذاب لين الإغلاق، وخطط دخول لحظية.
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

DAYS = 800
COST = lambda p: np.where(p < 1, 0.04, np.where(p < 5, 0.02, 0.01))
M = HR.model()
FE = M["feats"]; MU = np.array(M["mean"])[:len(FE)]; SD = np.array(M["std"])[:len(FE)]; W = np.array(M["w"])


def prob(F):
    X = F[FE].to_numpy(float)
    X = np.where(np.isfinite(X), X, MU)
    Z = np.clip((X - MU) / np.where(SD == 0, 1, SD), -5, 5)
    return 1 / (1 + np.exp(-(W[0] + Z @ W[1:])))


def collect():
    syms = sorted(universe())
    weekly, events, t0 = [], [], time.time()
    for i in range(0, len(syms), 100):
        try:
            raw = A.daily(syms[i:i + 100], days=DAYS)
        except Exception as e:
            print("daily err", repr(e)[:90], flush=True); continue
        for s, d in raw.items():
            if d is None or len(d) < 260:
                continue
            d = d[~d.index.duplicated(keep="last")]
            try:
                F = HR.features(d)
            except Exception:
                continue
            p = prob(F)
            o, h, l, c = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
            praw, dv, ex = F.praw.values, F.dv20.values, F.ex.values
            idx = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index, "tz", None) else pd.to_datetime(d.index)
            n = len(c)
            for t in np.flatnonzero(ex):
                if t < 260:
                    continue
                y = t - 1
                hi_t = h[t]
                events.append({"t": s, "date": str(idx[t].date()), "px": praw[y], "dv20": dv[y], "p_prev": p[y],
                               "n_ep252": F.n_ep252.values[y], "days_ep": F.days_ep_raw.values[y], "days_rs": F.days_rs_raw.values[y],
                               "dd252": F.dd252.values[y], "vol5_60": F.vol5_60.values[y], "rsi": F.rsi.values[y], "ret5": F.ret5.values[y],
                               "jump": h[t] / c[y], "open_gap": o[t] / c[y] - 1, "close_ret": c[t] / c[y] - 1, "close_vs_high": c[t] / hi_t,
                               "next_open": (o[t + 1] / c[t] - 1) if t + 1 < n else np.nan,
                               "next5_close": (c[min(t + 5, n - 1)] / c[t] - 1) if t + 5 < n else np.nan,
                               "dow": idx[t].dayofweek})
            fri = np.flatnonzero((idx.dayofweek == 4) & (np.arange(n) >= 260) & (np.arange(n) + 5 < n))
            for t in fri:
                if not (0.2 <= praw[t] <= 20) or not (dv[t] >= 30_000):
                    continue
                a, b = t + 1, t + 6
                mo = o[a]
                hiW = h[a:b].max(); clW = c[b - 1]
                weekly.append({"t": s, "date": str(idx[t].date()), "p": p[t], "px": praw[t], "c": c[t], "mo": mo,
                               "exW": bool(ex[a:b].any()), "hiW_c": hiW / c[t] - 1, "hiW_o": hiW / mo - 1, "clW_o": clW / mo - 1,
                               "loW_o": l[a:b].min() / mo - 1,
                               # أول يوم بالأسبوع لمس +50%/+100% من افتتاح الاثنين (للبيع Limit)
                               "hit50": bool((h[a:b] >= mo * 1.5).any()), "hit100": bool((h[a:b] >= mo * 2).any()),
                               # دخول زناد: أول يوم يلمس افتتاحه +20% ← شراء عند +20% وبيع بإغلاق نفس اليوم
                               "trig_ret": next(((c[k] / (o[k] * 1.2) - 1) for k in range(a, b) if h[k] >= o[k] * 1.2), np.nan)})
        if (i // 100) % 10 == 0:
            print(f"  {i + 100}/{len(syms)} · جمعات {len(weekly):,} · انفجارات {len(events):,} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(weekly), pd.DataFrame(events)


def ci(x, g):
    x = pd.Series(x).astype(float); ok = x.notna()
    if ok.sum() < 10:
        return "   -   "
    m = x[ok].groupby(np.asarray(g)[ok.values]).mean()
    return f"{x[ok].mean()*100:6.1f}±{1.96*m.std()/max(1,len(m))**.5*100:.1f}"


def part_a(E, countries):
    print("\n" + "#" * 70 + "\n# A) تشريح الانفجارات ×2 (أعلى ≥ ضعف إغلاق أمس، تداول ≥ 500 ألف$)\n" + "#" * 70)
    print(f"انفجارات: {len(E):,} · أسهم {E.t.nunique():,} · {E.date.min()} → {E.date.max()} · بالشهر ~{len(E)/ (DAYS/30.4):.0f}")
    E = E.copy(); E["ym"] = E.date.str[:7]
    print("بالشهر: " + " · ".join(f"{k} {v}" for k, v in E.groupby("ym").size().tail(12).items()))
    q = lambda m: f"{m.mean()*100:.0f}%"
    print(f"قبلها بيوم: سعر تحت 1$ {q(E.px<1)} · 1-5$ {q((E.px>=1)&(E.px<5))} · فوق 5$ {q(E.px>=5)}")
    print(f"  طيّار سابق (انفجار خلال سنة) {q(E.n_ep252>=1)} · مرتين+ {q(E.n_ep252>=2)} · آخر انفجار خلال 60 جلسة {q(E.days_ep<=60)} · خلال 20 {q(E.days_ep<=20)}")
    print(f"  تقسيم عكسي خلال سنة {q(E.days_rs<=252)} · خلال 60 جلسة {q(E.days_rs<=60)} · منهار ≥70% من قمة السنة {q(E.dd252<=-0.7)} · ≥90% {q(E.dd252<=-0.9)}")
    print(f"  فوليوم 5 أيام ÷ 60 ≥ 2 (سخن قبلها) {q(E.vol5_60>=2)} · RSI<35 {q(E.rsi<35)} · طلع ≥20% بآخر 5 أيام {q(E.ret5>=0.2)}")
    print(f"  رتبة النموذج يوم قبل: احتمال ≥ عتبة 🧨 ({M['thresholds']['ready']:.3f}) {q(E.p_prev>=M['thresholds']['ready'])} · ≥ 🟠 {q(E.p_prev>=M['thresholds']['near'])}")
    print("\nسلوك يوم الانفجار (وسيط): " + f"فجوة الافتتاح {E.open_gap.median()*100:+.0f}% · القفزة القصوى ×{E.jump.median():.2f} · الإغلاق {E.close_ret.median()*100:+.0f}% من أمس · "
          f"الإغلاق = {E.close_vs_high.median()*100:.0f}% من القمة · افتتاح بكرة {E.next_open.median()*100:+.0f}% · بعد 5 جلسات {E.next5_close.median()*100:+.0f}%")
    print(f"  فتح بفجوة ≥ +50% (يعني الطيران صار بالبري/الأفتر) {q(E.open_gap>=0.5)} · فجوة ≥ +100% {q(E.open_gap>=1.0)} · أغلق فوق ×2 {q(E.close_ret>=1.0)} · أغلق تحت +30% {q(E.close_ret<0.3)}")
    print("  باليوم بالأسبوع: " + " · ".join(f"{['اثنين','ثلاثاء','أربعاء','خميس','جمعة'][k]} {v}" for k, v in E.groupby('dow').size().items() if k < 5))
    if countries:
        E["country"] = E.t.map(countries).fillna("؟")
        g = E.groupby("country").agg(n=("t", "size"), stocks=("t", "nunique")).sort_values("n", ascending=False)
        tot_u = pd.Series(countries).value_counts()
        print("\nالدولة (مقر الشركة من ياهو): عدد الانفجارات · أسهم · نسبة من الانفجارات مقابل نسبة الدولة من المجال")
        for ct, r in g.head(14).iterrows():
            print(f"  {ct:<22} {r.n:>5} · {r.stocks:>4} سهم · {r.n/len(E)*100:5.1f}% من الانفجارات مقابل {tot_u.get(ct,0)/max(1,tot_u.sum())*100:5.1f}% من المجال")


def part_b(Wk):
    print("\n" + "#" * 70 + "\n# B) القائمة الأسبوعية (كل جمعة نرتب بالاحتمال ← أداء الأسبوع الجاي)\n" + "#" * 70)
    Wk = Wk.copy(); Wk["rank"] = Wk.groupby("date").p.rank(ascending=False, method="first")
    cost = COST(Wk.px.values)
    Wk["r_hold"] = Wk.clW_o - cost
    Wk["r_lim50"] = np.where(Wk.hit50, 0.5, Wk.clW_o) - cost
    Wk["r_lim100"] = np.where(Wk.hit100, 1.0, Wk.clW_o) - cost
    Wk["r_trig"] = Wk.trig_ret - cost
    mid = Wk.date.sort_values().iloc[len(Wk) // 2]
    print(f"جمعات: {Wk.date.nunique()} · سهم-أسبوع {len(Wk):,} · المنتصف {mid}")
    hdr = f"{'المجموعة':<26} {'n':>6} {'انفجر ×2':>10} {'لمس +50%':>10} {'لمس +100%':>10} {'شراء الاثنين→الجمعة':>20} {'Limit +50%':>12} {'Limit +100%':>12} {'زناد +20%':>11} {'أسوأ نزول':>10}"
    for nm, part in (("كل الفترة", Wk), ("النص الأول", Wk[Wk.date < mid]), ("النص الثاني", Wk[Wk.date >= mid])):
        print(f"\n── {nm} ──\n{hdr}")
        for lab, m in (("كل الأسهم", part.index == part.index), ("أعلى 50", part["rank"] <= 50), ("أعلى 20", part["rank"] <= 20),
                       ("أعلى 10", part["rank"] <= 10), ("أعلى 5", part["rank"] <= 5)):
            g = part[m]; G = g.t.values
            print(f"{lab:<26} {len(g):>6} {ci(g.exW,G):>10} {ci(g.hit50,G):>10} {ci(g.hit100,G):>10} {ci(g.r_hold,G):>20} {ci(g.r_lim50,G):>12} {ci(g.r_lim100,G):>12} {ci(g.r_trig,G):>11} {g.loW_o.median()*100:9.0f}%")
    t10 = Wk[Wk["rank"] <= 10]
    print("\nأعلى 10 حسب السعر: " + " · ".join(f"{nm}: انفجر {g.exW.mean()*100:.0f}% · Limit+100 {g.r_lim100.mean()*100:+.1f}% (n={len(g)})"
                                       for nm, g in (("تحت 1$", t10[t10.px < 1]), ("1-5$", t10[(t10.px >= 1) & (t10.px < 5)]), ("5$+", t10[t10.px >= 5]))))
    by = t10.groupby(t10.date.str[:7]).agg(n=("t", "size"), ex=("exW", "mean"), r=("r_lim100", "mean"))
    print("أعلى 10 بالشهر (انفجر% · Limit+100 متوسط): " + " · ".join(f"{k} {v.ex*100:.0f}%/{v.r*100:+.0f}%" for k, v in by.tail(14).iterrows()))
    return Wk


def part_c(E):
    print("\n" + "#" * 70 + "\n# C) التوقيت داخل اليوم (شموع ساعة) — آخر 9 شهور\n" + "#" * 70)
    E = E[E.date >= str((datetime.now() - timedelta(days=270)).date())]
    rows, t0 = [], time.time()
    for s, g in E.groupby("t"):
        try:
            hb = A.bars([s], "1Hour", datetime.fromisoformat(g.date.min()).replace(tzinfo=timezone.utc) - timedelta(days=3),
                        datetime.fromisoformat(g.date.max()).replace(tzinfo=timezone.utc) + timedelta(days=2)).get(s)
        except Exception:
            continue
        if hb is None or hb.empty:
            continue
        hb = hb.copy(); hb.index = hb.index.tz_convert("America/New_York")
        m = hb.index.hour * 60 + hb.index.minute
        sess = np.where(m < 570, "pm", np.where(m < 960, "rth", "ah"))
        days = sorted(set(hb.index.date))
        for dstr in g.date:
            D = pd.Timestamp(dstr).date()
            if D not in days or days.index(D) == 0:
                continue
            P = days[days.index(D) - 1]
            prev_rth = hb[(hb.index.date == P) & (sess == "rth")]
            if prev_rth.empty:
                continue
            pc = float(prev_rth.Close.iloc[-1])
            ah = hb[(hb.index.date == P) & (sess == "ah")]; dd = hb[hb.index.date == D]
            sd = sess[hb.index.date == D]
            pm, rth = dd[sd == "pm"], dd[sd == "rth"]
            if rth.empty:
                continue
            ahh = float(ah.High.max()) / pc - 1 if len(ah) else 0
            pmh = float(pm.High.max()) / pc - 1 if len(pm) else 0
            full = pd.concat([ah, dd])
            t_hi = full.High.idxmax(); hi = float(full.High.max())
            start = "أفتر أمس" if ahh >= 0.3 else ("البري" if pmh >= 0.3 else "الجلسة")
            # خطة لحظية: شراء أول ما يلمس +30% من إغلاق أمس (بأي جلسة) ثم: بيع +30% من الدخول Limit وإلا إغلاق اليوم
            trig = full[full.High >= pc * 1.3]
            r30 = np.nan; r_close = np.nan
            if len(trig):
                e = pc * 1.3 * 1.02
                after = full.loc[trig.index[0]:]
                r30 = 0.30 if (after.High >= e * 1.3).any() else float(rth.Close.iloc[-1]) / e - 1
                r_close = float(rth.Close.iloc[-1]) / e - 1
            rows.append({"t": s, "start": start, "ah_pct": ahh, "pm_pct": pmh, "hi_hour": t_hi.hour if t_hi.date() == D else -1,
                         "close_hi": float(rth.Close.iloc[-1]) / hi, "r30": r30, "r_close": r_close, "px": pc})
    C = pd.DataFrame(rows)
    if C.empty:
        print("ما فيه بيانات ساعة"); return
    print(f"انفجارات بشموع ساعة: {len(C)}")
    for st, g in C.groupby("start"):
        cost = COST(g.px.values)
        print(f"  بدأ في {st:<9}: {len(g)/len(C)*100:4.0f}% · الإغلاق = {g.close_hi.median()*100:.0f}% من القمة (وسيط) · "
              f"زناد +30% ← Limit +30% وإلا الإغلاق: {np.nanmean(g.r30 - cost)*100:+.1f}% · إمساك للإغلاق {np.nanmean(g.r_close - cost)*100:+.1f}%")
    hh = C[C.hi_hour >= 0].hi_hour.value_counts().sort_index()
    print("  ساعة القمة (نيويورك): " + " · ".join(f"{h}:00 {v}" for h, v in hh.items()))
    print(f"  القمة كانت بالأفتر أمس (قبل اليوم): {(C.hi_hour < 0).mean()*100:.0f}%")


def main():
    os.makedirs("diag_out", exist_ok=True)
    Wk, E = collect()
    Wk.to_csv("diag_out/movers_weekly.csv.gz", index=False, compression="gzip"); E.to_csv("diag_out/movers_events.csv.gz", index=False, compression="gzip")
    countries = {}
    try:
        import scanner
        cfg = scanner.load_cfg()
        syms = sorted(set(E.t) | set(Wk[Wk.groupby("date").p.rank(ascending=False) <= 50].t))
        info = scanner.get_info(syms, cfg)
        countries = {s: (v or {}).get("country") or "؟" for s, v in info.items()}
        # عيّنة من باقي المجال للنسبة
        rest = sorted(set(Wk.t) - set(syms))[:1500]
        info2 = scanner.get_info(rest, cfg)
        countries.update({s: (v or {}).get("country") or "؟" for s, v in info2.items()})
        print(f"الدول: {len(countries)} سهم", flush=True)
    except Exception as e:
        print("الدول تعطلت:", repr(e)[:150], flush=True)
    part_a(E, countries)
    part_b(Wk)
    part_c(E)
    print("\n⚠ تشخيص — بيانات Alpaca معدّلة للتقسيم (السعر الحقيقي من raw_factor)، التكاليف 4%/2%/1%، السبريد الحقيقي بالبيني ممكن أسوأ.")


if __name__ == "__main__":
    main()
