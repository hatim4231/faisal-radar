"""
🧨 عالي مخاطرة 3 — بعد ما الدراسة 2 بيّنت إن الشراء قبل الانفجار خسران كل سنة وبكل حرارة سوق (تشخيص فقط):
أ) السعر الحقيقي وقتها (البيانات معدّلة للتقسيم) ← تكاليف وشرائح صحيحة.
ب) **قفزات البري والأفتر**: «تذكرة + أمر بيع Limit شغّال بالتداول الممتد» — الاختبار اليومي ما كان يشوفها (مثل RUBI الفجر).
   لأعلى 3 و10 كل يوم: شموع الساعة (فيها الممتد) لـ 20 جلسة بعد الإشارة ← نفس الخطط بقمم الممتد (لازم تتعدى الهدف بـ 2%).
ج) نموذج مباشر للربح (انحدار على عائد الخطة نفسها) — لو فيه أي جيب رابح بهالميزات بيلقاه. سنة بسنة.
د) 🩸 نموذج «النزيف»: احتمال السهم يلمس −30% خلال 20 جلسة (للفاحص: أي سهم ← خطره) — سنة بسنة + معايرة.
المخرجات: جداول + JSON بين HR3_BEGIN/END (أسطر H3|).
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
import highrisk as HR
import backtest_highrisk2 as B2
from backtest_highrisk2 import job, fit_logit, auc, zs, predict, pr, net_cols, PL, START, FEATS, LIQ

FAST = os.environ.get("DATA_RUN", "") == "fast"      # تشغيل سريع: بدون قسم التداول الممتد (شموع الساعة ~15 دقيقة)
PB_EDGES = [0, .03, .06, .10, .15, .20, .25, .30, .40, .50, 1.01]
P_EDGES = [0, .001, .002, .005, .01, .02, .04, .0632, .10, .15, .25, 1.01]
FB = FEATS + ["m60", "m120", "m250"]            # ميزات نموذج النزيف والنموذج المباشر
GRID = [(T, S, Hh) for Hh in (5, 10, 20) for T in (0.3, 0.5, 1.0, 2.0) for S in (None, 0.3)]


def universe():
    return B2.universe()


def cost_of(px):
    return np.where(px < 1, 0.04, np.where(px < 5, 0.02, 0.01))


def ext_highs(picks, calD, calpos):
    """قمم البري (قبل 9 الصبح) والأفتر (من 4 العصر) لكل (سهم، يوم) بنوافذ الاختيارات — من شموع الساعة.
    الشمعة الخفيفة (تداول أقل من 10 آلاف$ أو أقل من 10 صفقات) ما تنحسب (طبعة وحدة ما تعبّي أمر)."""
    need = {}
    for t, D in zip(picks.t.astype(str), picks.D.to_numpy()):
        i = calpos.get(int(D))
        if i is None:
            continue
        for m in set(int(x) for x in calD[i + 1:i + 22] // 100):
            need.setdefault(m, set()).add(t)
    XH, nreq, t0 = {}, 0, time.time()
    lim = datetime.now(timezone.utc) - timedelta(minutes=30)
    months = sorted(need)
    for mi, m in enumerate(months):
        y, mo = divmod(m, 100)
        st = datetime(y, mo, 1, tzinfo=timezone.utc)
        en = min(datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc), lim)
        if en <= st:
            continue
        syms = sorted(need[m])
        for j in range(0, len(syms), 30):
            bars = None
            for attempt in range(3):
                try:
                    bars = A.bars(syms[j:j + 30], "1Hour", st, en)
                    break
                except Exception as e:
                    print("  ساعة err", m, repr(e)[:90], flush=True)
                    time.sleep(10 * (attempt + 1))
            nreq += 1
            for s_, b in (bars or {}).items():
                ix = b.index
                Dd = (ix.year * 10000 + ix.month * 100 + ix.day).to_numpy()
                hr = ix.hour.to_numpy()
                hi = b["High"].to_numpy(float)
                px = b["VWAP"].to_numpy(float) if "VWAP" in b else b["Close"].to_numpy(float)
                ok = (b["Volume"].to_numpy(float) * px >= 10_000)
                if "Trades" in b:
                    ok &= b["Trades"].to_numpy(float) >= 10
                g = pd.DataFrame({"D": Dd, "pre": np.where((hr <= 8) & ok, hi, np.nan), "post": np.where((hr >= 16) & ok, hi, np.nan),
                                  "mid": np.where((hr >= 10) & (hr <= 15), hi, np.nan)}).groupby("D").max()
                for D_, a_, h_, m_ in zip(g.index.to_numpy(), g["pre"].to_numpy(), g["post"].to_numpy(), g["mid"].to_numpy()):
                    XH[(s_, int(D_))] = (float(a_), float(h_), float(m_))
        if mi % 12 == 0:
            print(f"  شموع الساعة {m} · طلبات {nreq} · أيام-أسهم {len(XH):,} · {round(time.time() - t0)}ث", flush=True)
    print(f"  شموع الساعة: طلبات {nreq} · أيام-أسهم {len(XH):,} · {round(time.time() - t0)}ث", flush=True)
    return XH


def run_plan(ent, T, S, Hh, reg, cl, pre, post):
    """دخول افتتاح بكرة · أمر بيع Limit عند +T · وقف بإغلاق تحت −S · أقصى Hh جلسة. الترتيب داخل اليوم: البري ← الجلسة ← الإغلاق ← الأفتر."""
    tgt = ent * (1 + T)
    for k in range(Hh):
        if (k > 0 and pre[k] >= tgt) or reg[k] >= tgt:
            return T
        if S is not None and cl[k] <= ent * (1 - S):
            return cl[k] / ent - 1
        if k < Hh - 1 and post[k] >= tgt:
            return T
    return cl[Hh - 1] / ent - 1


def ext_eval(picks, PATH, XH):
    """لكل اختيار: نتيجة كل خطة بالقمم العادية «r»، وبقمم التداول الممتد بهامش 2% «x»، وبهامش 10% «z» (أشد)."""
    rows, ratios = [], []
    NEG = np.full(20, -np.inf)
    for t, D, px in zip(picks.t.astype(str), picks.D.to_numpy(), picks.praw.to_numpy(float)):
        P = PATH.get(t)
        if P is None:
            continue
        Dint, O, Hh_, L, C = P
        i = int(np.searchsorted(Dint, D))
        if i >= len(Dint) or Dint[i] != D or i + 20 >= len(Dint):
            continue
        ent = float(O[i + 1])
        if not np.isfinite(ent) or ent <= 0:
            continue
        reg = Hh_[i + 1:i + 21].astype(float)
        cl = C[i + 1:i + 21].astype(float)
        pre, post, has = NEG.copy(), NEG.copy(), 0
        for k in range(20):
            a = XH.get((t, int(Dint[i + 1 + k])))
            if a:
                has += 1
                if np.isfinite(a[0]):
                    pre[k] = a[0]
                if np.isfinite(a[1]):
                    post[k] = a[1]
                if np.isfinite(a[2]) and reg[k] > 0:
                    ratios.append(a[2] / reg[k])
        V = {"r": (NEG, NEG), "x": (pre / 1.02, post / 1.02), "z": (pre / 1.10, post / 1.10)}
        rec = {"t": t, "D": int(D), "Y": int(D) // 10000, "cost": float(cost_of(px)), "px": px, "has": has,
               "rmax20": float(reg.max() / ent - 1),
               "xmax20": float(max(reg.max(), pre[1:].max() / 1.02, post[:-1].max() / 1.02) / ent - 1)}
        for T, S, Hh in GRID:
            for nm, (pr_, po_) in V.items():
                rec[f"{nm}:{T}:{S}:{Hh}"] = run_plan(ent, T, S, Hh, reg, cl, pr_, po_)
        last = cl[19] / ent - 1
        for nm, (pr_, po_) in V.items():          # سلّم: ثلث عند +50 / +100 / +200 بدون وقف 20 جلسة
            mx = max(reg.max(), pr_[1:].max(), po_[:-1].max()) / ent - 1
            rec[f"{nm}:lad"] = float(np.mean([L_ if mx >= L_ else last for L_ in (0.5, 1.0, 2.0)]))
        rows.append(rec)
    return pd.DataFrame(rows), np.array(ratios)


def fit_ridge(X, y, lam=200.0):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    R = lam * np.eye(Xb.shape[1])
    R[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)


def main():
    t0 = time.time()
    syms = universe()
    iw = A.bars(["IWM"], "1Day", START)["IWM"]
    iw.index = pd.to_datetime(iw.index).tz_localize(None).normalize()
    iw = iw[~iw.index.duplicated(keep="last")]
    calD = iw.index.strftime("%Y%m%d").astype(int).to_numpy(np.int64)
    calpos = {int(d): i for i, d in enumerate(calD)}
    parts, PATH = [], {}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            batch = syms[i:i + 100]
            raw = None
            for attempt in range(3):
                try:
                    raw = A.bars(batch, "1Day", START)
                    break
                except Exception as e:
                    print("bars err", i, repr(e)[:100], flush=True)
                    time.sleep(15 * (attempt + 1))
            if raw is None:
                continue
            try:
                spl = A.splits(batch, START)
            except Exception:
                spl = {}
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")], spl.get(s, [])))
            for (s, _, _), res in zip(jobs, ex.map(job, jobs, chunksize=4)):
                if res is None:
                    continue
                if len(res[0]):
                    parts.append(res[0])
                PATH[s] = res[4]
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · صفوف {sum(len(p) for p in parts):,} · {round(time.time() - t0)}ث", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["t"] = df["t"].astype("category")
    for f_ in FB:
        x = df[f_].replace([np.inf, -np.inf], np.nan)
        df[f_] = x.fillna(x.median()).astype(np.float32)
    df["Y"] = (df.D // 10000).astype(np.int16)
    df["B"] = (df.dd20 <= -0.3).astype(np.float32)
    print(f"\nالصفوف: {len(df):,} سهم-يوم · أسهم {df.t.nunique():,} · من {df.D.min()} لين {df.D.max()} · {round(time.time() - t0)}ث", flush=True)
    print(f"الأساس: انفجار حقيقي خلال 10 جلسات {pr(df.E10.mean(), 2)} · يلمس −30% خلال 20 جلسة {pr(df.B.mean())} · وسيط السعر الحقيقي {df.praw.median():.2f}$"
          f" · تحت 1$ {pr((df.praw < 1).mean())} · 1-5$ {pr(((df.praw >= 1) & (df.praw < 5)).mean())}")

    rng = np.random.default_rng(1)
    XF = df[FB].to_numpy(np.float32)
    nE = len(FEATS)
    yE, yB = df.E10.to_numpy(float), df.B.to_numpy(float)
    for c_ in ("p", "q99", "q95", "pb"):
        df[c_] = np.float32(np.nan)
    TG = ("o5", "olot", "h5", "d1")
    cost_all = cost_of(df.praw.to_numpy(float))
    for tg in TG:
        df["r_" + tg] = np.float32(np.nan)
    print("\n==== 1) النماذج سنة بسنة (كل سنة بنموذج ما شافها) — 🧨 الانفجار · 🩸 النزيف (يلمس −30% خلال 20) ====")
    for Y in range(2021, 2027):
        trm = (df.D < (Y - 1) * 10000 + 1201).to_numpy()
        tem = (df.Y == Y).to_numpy()
        if trm.sum() < 50_000 or tem.sum() < 1000:
            continue
        tri = np.flatnonzero(trm)
        if len(tri) > 1_500_000:
            tri = rng.choice(tri, 1_500_000, replace=False)
        Xtr = XF[tri].astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        Ztr = zs(Xtr, mu, sd)
        del Xtr
        w = fit_logit(Ztr[:, :nE], yE[tri], lam=5.0)
        ptr = 1 / (1 + np.exp(-(w[0] + Ztr[:400_000, :nE] @ w[1:])))
        q99, q95 = float(np.quantile(ptr, 0.99)), float(np.quantile(ptr, 0.95))
        wb = fit_logit(Ztr, yB[tri], lam=5.0)
        Zte = zs(XF[tem].astype(float), mu, sd)
        pte = 1 / (1 + np.exp(-(w[0] + Zte[:, :nE] @ w[1:])))
        pbe = 1 / (1 + np.exp(-(wb[0] + Zte @ wb[1:])))
        df.loc[tem, "p"], df.loc[tem, "pb"] = pte.astype(np.float32), pbe.astype(np.float32)
        df.loc[tem, "q99"], df.loc[tem, "q95"] = np.float32(q99), np.float32(q95)
        for tg in TG:   # النموذج المباشر: انحدار على عائد الخطة (مقصوص)
            y = np.clip(df[tg].to_numpy(float)[tri] - cost_all[tri], -0.6, 1.0)
            ok = np.isfinite(y)
            beta = fit_ridge(Ztr[ok], y[ok])
            df.loc[tem, "r_" + tg] = (beta[0] + Zte @ beta[1:]).astype(np.float32)
        print(f"  {Y}: تدريب {trm.sum():,} · اختبار {tem.sum():,} · 🧨 AUC {auc(yE[tem], pte):.3f} (الأساس {pr(yE[tem].mean(), 2)}) · "
              f"🩸 AUC {auc(yB[tem], pbe):.3f} (الأساس {pr(yB[tem].mean())})", flush=True)
        del Ztr, Zte
    # النماذج النهائية على كل البيانات (للرادار الحي)
    tri = np.arange(len(df))
    if len(tri) > 2_000_000:
        tri = rng.choice(tri, 2_000_000, replace=False)
    Xall = XF[tri].astype(float)
    mu_f, sd_f = Xall.mean(0), Xall.std(0)
    sd_f[sd_f == 0] = 1
    Zall = zs(Xall, mu_f, sd_f)
    del Xall
    w_f = fit_logit(Zall[:, :nE], yE[tri], lam=5.0)
    wb_f = fit_logit(Zall, yB[tri], lam=5.0)
    pall = 1 / (1 + np.exp(-(w_f[0] + Zall[:600_000, :nE] @ w_f[1:])))
    thr_f = {"ready": float(np.quantile(pall, 0.99)), "near": float(np.quantile(pall, 0.95))}
    pball = 1 / (1 + np.exp(-(wb_f[0] + Zall[:600_000] @ wb_f[1:])))
    pb_q = [float(x) for x in np.quantile(pball, np.linspace(0, 1, 101))]
    del Zall, XF
    print("أوزان 🧨: " + " · ".join(f"{f_} {c_:+.2f}" for f_, c_ in sorted(zip(FEATS, w_f[1:]), key=lambda x: -abs(x[1]))[:12]))
    print("أوزان 🩸: " + " · ".join(f"{f_} {c_:+.2f}" for f_, c_ in sorted(zip(FB, wb_f[1:]), key=lambda x: -abs(x[1]))[:14]))

    T = df[df.p.notna()].copy()
    del df
    T["vq"] = pd.qcut(T["atr20p"].rank(method="first"), 5, labels=False)
    T["pq"] = pd.cut(T["praw"], [0, 1, 5, 1e9], labels=False)
    T["liq"] = T.dv20 >= LIQ

    # ---- 2) 🩸 النزيف: معايرة خارج العينة
    print("\n==== 2) 🩸 خطر النزيف — معايرة خارج العينة (2021 ← 2026): العُشر · المتوقع · الفعلي يلمس −30% · وسيط عائد 20 جلسة · يطلع +30% خلال 20 ====")
    T["bdec"] = pd.qcut(T.pb.rank(method="first"), 10, labels=False)
    bcal = []
    for k, g in T.groupby("bdec"):
        bcal.append({"dec": int(k), "p_lo": float(g.pb.min()), "p_hi": float(g.pb.max()), "rate": float(g.B.mean()),
                     "r20_med": float(g.h20.median()), "up30": float((g.mx20 >= 0.3).mean()), "n": int(len(g))})
        print(f"  {k}: {pr(g.pb.mean())} · {pr(g.B.mean())} · {pr(g.h20.median())} · {pr((g.mx20 >= 0.3).mean())} · n={len(g):,}")
    top1 = T[T.pb >= T.pb.quantile(0.99)]
    top5 = T[T.pb >= T.pb.quantile(0.95)]
    print(f"  أعلى 1% خطر: يلمس −30% {pr(top1.B.mean())} · وسيط 20 جلسة {pr(top1.h20.median())} | أعلى 5%: {pr(top5.B.mean())} · {pr(top5.h20.median())}")
    print("  بكل سنة — أعلى 5% خطر: " + " · ".join(f"{int(y)}: {pr(g.B.mean())} (الأساس {pr(T[T.Y == y].B.mean())})" for y, g in top5.groupby("Y")))
    lq = T[T.liq & (T.praw >= 0.7) & (T.praw <= 10)]
    lq5 = lq[lq.pb >= lq.pb.quantile(0.9)]
    lq1 = lq[lq.pb <= lq.pb.quantile(0.5)]
    print(f"  داخل مجال راداراتنا (0.7-10$ وسيولة): الأساس يلمس −30% {pr(lq.B.mean())} · أعلى 10% خطر {pr(lq5.B.mean())} (وسيط 20 جلسة {pr(lq5.h20.median())})"
          f" · النص الأقل خطر {pr(lq1.B.mean())} (وسيط {pr(lq1.h20.median())})")
    print("  🧨 × 🩸: " + " · ".join(f"{nm} n={len(g):,} يلمس −30% {pr(g.B.mean())} انفجر {pr(g.E10.mean())}" for nm, g in (
        ("🧨 وخطر نزيف عالي (أعلى 10%)", T[(T.p >= T.q99) & (T.bdec == 9)]), ("🧨 وخطر أقل", T[(T.p >= T.q99) & (T.bdec < 9)]))))

    def band_table(g, col, edges):
        out = []
        ix = np.digitize(g[col].to_numpy(float), edges) - 1
        for k in range(len(edges) - 1):
            x = g[ix == k]
            rec = {"lo": edges[k], "hi": edges[k + 1], "n": int(len(x))}
            if len(x) >= 150:
                rec.update({"p": float(x[col].mean()), "rate": float(x.B.mean()), "dn20": float((x.dd20 <= -0.2).mean()),
                            "r20_med": float(x.h20.median()), "r20_mean": float(x.h20.mean()), "up30": float((x.mx20 >= 0.3).mean()),
                            "e5": float(x.E5.mean()), "e10": float(x.E10.mean())})
            out.append(rec)
        return out

    def show_bands(title, tb, key="rate"):
        print(f"  {title}")
        for r in tb:
            if "rate" not in r:
                print(f"    {r['lo']:.3f}-{r['hi']:.3f}: n={r['n']:,} (قليل)")
                continue
            print(f"    {r['lo']:.3f}-{r['hi']:.3f}: n={r['n']:,} · متوقع {pr(r['p'], 2)} · يلمس −30% {pr(r['rate'])} · يلمس −20% {pr(r['dn20'])} · وسيط 20 جلسة {pr(r['r20_med'])}"
                  f" (متوسط {pr(r['r20_mean'])}) · يطلع +30% {pr(r['up30'])} · انفجر خلال 5 {pr(r['e5'], 2)} · خلال 10 {pr(r['e10'], 2)}")

    rr = T[T.liq & (T.praw >= 0.7) & (T.praw <= 10)]
    rdy = T[T.p >= T.q99]
    print("\n==== 2ب) معايرة بالشرائح المطلقة (خارج العينة) ====")
    cal = {"pb_all": band_table(T, "pb", PB_EDGES), "pb_radar": band_table(rr, "pb", PB_EDGES), "pb_ready": band_table(rdy, "pb", PB_EDGES),
           "p_all": band_table(T, "p", P_EDGES), "p_liq": band_table(T[T.liq], "p", P_EDGES)}
    show_bands("🩸 كل الأسهم (حسب خطر النزيف):", cal["pb_all"])
    show_bands("🩸 مجال راداراتنا 0.7-10$ وسيولة ≥ 30 ألف$:", cal["pb_radar"])
    show_bands("🩸 داخل 🧨 (أعلى 1% انفجار):", cal["pb_ready"])
    show_bands("🧨 كل الأسهم (حسب احتمال الانفجار):", cal["p_all"])
    show_bands("🧨 بسيولة ≥ 30 ألف$:", cal["p_liq"])
    cal["radar_q"] = {str(q): float(rr.pb.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
    cal["all_q"] = {str(q): float(T.pb.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
    cal["radar_base"] = {"rate": float(rr.B.mean()), "dn20": float((rr.dd20 <= -0.2).mean()), "r20_med": float(rr.h20.median()), "n": int(len(rr))}
    print(f"  حدود 🩸 بمجال راداراتنا: {cal['radar_q']} · كل الأسهم: {cal['all_q']} · أساس المجال: {cal['radar_base']}")
    for nm, g in (("كل الأسهم", T), ("مجال راداراتنا", rr)):
        for th_ in (0.25, 0.40):
            x = g[g.pb >= th_]
            print(f"  ثبات 🩸 ≥ {th_} — {nm}: " + " · ".join(f"{int(y)}: {pr(v.B.mean())} (n={len(v):,}، أساس {pr(g[g.Y == y].B.mean())})" for y, v in x.groupby("Y")))
        x = g[g.pb < 0.06]
        print(f"  ثبات 🩸 < 0.06 — {nm}: " + " · ".join(f"{int(y)}: {pr(v.B.mean())} (n={len(v):,})" for y, v in x.groupby("Y")))

    # ---- 3) المجموعات والخطط بعد التكاليف (بالسعر الحقيقي)
    T["rkL"] = T[T.liq].groupby("D").p.rank(ascending=False, method="first")
    G = {"top3": T[T.rkL <= 3], "top10": T[T.rkL <= 10], "q99": T[T.liq & (T.p >= T.q99)]}
    GA = {"top3": "أعلى 3 كل يوم (سيولة ≥ 30 ألف$)", "top10": "أعلى 10 كل يوم", "q99": "🧨 فوق حد 99%"}
    NT = net_cols(T[T.liq])
    NT[["Y", "vq", "pq"]] = T.loc[T.liq, ["Y", "vq", "pq"]].to_numpy()
    base_net = NT.groupby(["Y", "vq", "pq"])[list(PL)].mean()
    base_e = T[T.liq].groupby(["Y", "vq", "pq"])[["E5", "E10"]].mean()
    del NT

    def matched(g, tbl, col):
        k = g.groupby(["Y", "vq", "pq"]).size()
        k = k[k.index.isin(tbl.index)]
        v = tbl.loc[k.index, col]
        ok = v.notna()
        return float((v[ok] * k[ok]).sum() / max(k[ok].sum(), 1))

    print("\n==== 3) 🧨 المجموعات (خارج العينة) والخطط بعد التكاليف بالسعر الحقيقي ====")
    tiers = {}
    for k, g in G.items():
        print(f"  {GA[k]}: n={len(g):,} · انفجر خلال 5 {pr(g.E5.mean())} ({pr(matched(g, base_e, 'E5'))}) · خلال 10 {pr(g.E10.mean())} ({pr(matched(g, base_e, 'E10'))})"
              f" · يلمس −30% خلال 20: {pr(g.B.mean())} · وسيط 20 جلسة {pr(g.h20.median())} · وسيط السعر الحقيقي {g.praw.median():.2f}$ · تحت 1$ {pr((g.praw < 1).mean(), 0)}")
        X = net_cols(g)
        X[["Y", "vq", "pq"]] = g[["Y", "vq", "pq"]].to_numpy()
        for p in ("o5", "o20", "olot", "lad", "on", "d1L", "h5", "s15b", "s15d5"):
            x = X[p].dropna()
            by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in X.groupby("Y")[p].mean().items() if np.isfinite(v))
            print(f"     {PL[p]}: {pr(x.mean())} · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean(), 0)} · المماثل {pr(matched(X.dropna(subset=[p]), base_net, p))} · [{by}]")
        tiers[k] = {"n": int(len(g)), "e5": float(g.E5.mean()), "e10": float(g.E10.mean()), "exp_e10": matched(g, base_e, "E10"),
                    "bleed": float(g.B.mean()), "r20_med": float(g.h20.median()), "o5": float(X.o5.mean()), "olot": float(X.olot.mean()),
                    "o20": float(X.o20.mean()), "by_year_o5": {int(y): float(v) for y, v in X.groupby("Y").o5.mean().items()}}
    # الجديد بالقائمة (ما كان بأعلى 10 في آخر ~10 جلسات) مقابل القديم · وشرائح خطر النزيف داخل أعلى 10
    T["in10"] = (T.rkL <= 10).astype(np.float32)
    T["inq"] = (T.liq & (T.p >= T.q99)).astype(np.float32)
    T = T.sort_values(["t", "D"])
    for c_ in ("in10", "inq"):
        T["prev_" + c_] = T.groupby("t", observed=True)[c_].transform(lambda z: z.shift(1).rolling(5, min_periods=1).max()).fillna(0)
    T = T.sort_index()

    def show(nm, g, plans=("o5", "o20", "olot", "h5")):
        if len(g) < 30:
            print(f"  {nm}: n={len(g)} (قليل)")
            return
        X = net_cols(g)
        X["Y"] = g["Y"].to_numpy()
        txt = " · ".join(f"{p_} {pr(X[p_].mean())}" for p_ in plans)
        by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in X.groupby("Y")[plans[0]].mean().items() if np.isfinite(v))
        print(f"  {nm}: n={len(g):,} · انفجر خلال 10 {pr(g.E10.mean())} · يلمس −30% {pr(g.B.mean())} · {txt} · [{plans[0]} بالسنة {by}]")

    print("\n==== 3ب) تقسيمات داخل المجموعات (بعد التكاليف) ====")
    show("أعلى 10 — جديد بالقائمة", T[(T.in10 > 0) & (T.prev_in10 == 0)])
    show("أعلى 10 — قديم بالقائمة", T[(T.in10 > 0) & (T.prev_in10 > 0)])
    show("🧨 99% — جديد", T[(T.inq > 0) & (T.prev_inq == 0)])
    show("🧨 99% — قديم", T[(T.inq > 0) & (T.prev_inq > 0)])
    g10 = T[T.in10 > 0]
    if len(g10) > 300:
        q1, q2 = g10.pb.quantile([1 / 3, 2 / 3])
        show("أعلى 10 — خطر نزيف أقل (ثلث)", g10[g10.pb <= q1])
        show("أعلى 10 — خطر نزيف وسط", g10[(g10.pb > q1) & (g10.pb <= q2)])
        show("أعلى 10 — خطر نزيف أعلى (ثلث)", g10[g10.pb > q2])
    show("🧨 99% وفوليوم آخر جلسة ≥ 2× المعتاد", T[(T.inq > 0) & (T.vol1_20 >= 2)])
    show("🧨 99% وفوليوم آخر جلسة < 2×", T[(T.inq > 0) & (T.vol1_20 < 2)])
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 5), (5, 1e9)):
        show(f"🧨 99% — السعر الحقيقي {lo}-{hi if hi < 1e8 else '∞'}$", T[(T.inq > 0) & (T.praw >= lo) & (T.praw < hi)])
    for lo, hi in ((30_000, 300_000), (300_000, 3_000_000), (3_000_000, 1e18)):
        show(f"🧨 99% — وسيط التداول {lo:,.0f}-{hi if hi < 1e17 else '∞'}$", T[(T.inq > 0) & (T.dv20 >= lo) & (T.dv20 < hi)])
    for k_, g in (("ready", T[T.p >= T.q99]), ("near", T[(T.p >= T.q95) & (T.p < T.q99)])):
        tiers[k_] = {"n": int(len(g)), "e5": float(g.E5.mean()), "e10": float(g.E10.mean()), "bleed": float(g.B.mean()), "r20_med": float(g.h20.median())}

    # ---- 4) النموذج المباشر للربح: أعلى 3 و10 كل يوم حسب العائد المتوقع (سيولة ≥ 30 ألف$، سعر ≥ 0.2$)
    print("\n==== 4) النموذج المباشر (يتوقع عائد الخطة نفسها) — أعلى 3 / 10 كل يوم بعد التكاليف، بكل سنة ====")
    elig = T.liq & (T.praw >= 0.2)
    direct = {}
    for tg in ("o5", "olot", "h5", "d1"):
        for K in (3, 10):
            rk = T[elig].groupby("D")["r_" + tg].rank(ascending=False, method="first")
            g = T.loc[rk[rk <= K].index]
            X = net_cols(g)
            X[["Y", "vq", "pq"]] = g[["Y", "vq", "pq"]].to_numpy()
            x = X[tg].dropna()
            by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in X.groupby("Y")[tg].mean().items() if np.isfinite(v))
            print(f"  هدف {tg} · أعلى {K}: {pr(x.mean())} · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean(), 0)} · المماثل {pr(matched(X.dropna(subset=[tg]), base_net, tg))} · [{by}]"
                  f" · وسيط السعر {g.praw.median():.2f}$ · وسيط السيولة {g.dv20.median():,.0f}$")
            direct[f"{tg}:{K}"] = {"mean": float(x.mean()), "by": {int(y): float(v) for y, v in X.groupby("Y")[tg].mean().items()}}
        rk = T[elig & (T.p >= T.q95)].groupby("D")["r_" + tg].rank(ascending=False, method="first")
        g = T.loc[rk[rk <= 3].index]
        X = net_cols(g)
        x = X[tg].dropna()
        X["Y"] = g["Y"].to_numpy()
        by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in X.groupby("Y")[tg].mean().items() if np.isfinite(v))
        print(f"  هدف {tg} · أعلى 3 من مرشحي الانفجار (أعلى 5%): {pr(x.mean())} · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean(), 0)} · انفجر خلال 10 {pr(g.E10.mean())} · [{by}]")
        # بسيولة أعلى (≥ 300 ألف$) — أقل تأثر بفرق السعر
        rk = T[elig & (T.dv20 >= 300_000)].groupby("D")["r_" + tg].rank(ascending=False, method="first")
        g = T.loc[rk[rk <= 3].index]
        X = net_cols(g)
        x = X[tg].dropna()
        X["Y"] = g["Y"].to_numpy()
        by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in X.groupby("Y")[tg].mean().items() if np.isfinite(v))
        print(f"  هدف {tg} · أعلى 3 بسيولة ≥ 300 ألف$: {pr(x.mean())} · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean(), 0)} · [{by}]")

    ext_json = {}
    if not FAST:
        # ---- 5) قفزات البري والأفتر: تذكرة + أمر بيع Limit شغّال بالتداول الممتد
        print("\n==== 5) التداول الممتد — نفس الخطة (دخول افتتاح بكرة) بقمم الجلسة بس «عادي» مقابل قمم البري والأفتر «ممتد» (بعد التكاليف) ====", flush=True)
        wk = T[(T.inq > 0) & (T.vol1_20 >= 2)]
        wk = wk[wk.groupby("D").p.rank(ascending=False, method="first") <= 3]
        picks = pd.concat([G["top10"][["t", "D", "praw", "rkL"]], wk[["t", "D", "praw", "rkL"]]])
        picks = picks[~picks.index.duplicated()].copy()
        picks["wake"] = picks.index.isin(wk.index)
        XH = ext_highs(picks, calD, calpos)
        EV, ratios = ext_eval(picks, PATH, XH)
        key = pd.MultiIndex.from_arrays([EV.t, EV.D])
        pk = picks.set_index([picks.t.astype(str), picks.D.astype(int)])
        EV["rk"] = pk.rkL.reindex(key).to_numpy()
        EV["wake"] = pk.wake.reindex(key).to_numpy()
        if len(ratios):
            print(f"  فحص التطابق: أعلى سعر شموع الساعة (10ص-4م) ÷ أعلى اليومي — وسيط {np.median(ratios):.3f} · 1% {np.quantile(ratios, 0.01):.3f} · 99% {np.quantile(ratios, 0.99):.3f}"
                  f" · فوق 1.02: {pr((ratios > 1.02).mean())} · اختيارات بدون أي شموع ساعة: {pr((EV.has == 0).mean())}")
        GA["wake3"] = "أعلى 3 من 🧨 صحى فوليومها (≥ 2×)"
        for gk, g in (("top3", EV[EV.rk <= 3]), ("top10", EV[EV.rk <= 10]), ("wake3", EV[EV.wake == True])):   # noqa: E712
            if len(g) < 50:
                print(f"-- {GA[gk]}: n={len(g)} (قليل)")
                continue
            print(f"\n-- {GA[gk]} (n={len(g):,}) · أعلى سعر خلال 20 جلسة ≥ +50%: عادي {pr((g.rmax20 >= 0.5).mean())} ← ممتد {pr((g.xmax20 >= 0.5).mean())}"
                  f" · ≥ +100%: {pr((g.rmax20 >= 1).mean())} ← {pr((g.xmax20 >= 1).mean())} · ≥ +200%: {pr((g.rmax20 >= 2).mean())} ← {pr((g.xmax20 >= 2).mean())}")
            res = []
            for T_, S, Hh in GRID:
                k_ = f"{T_}:{S}:{Hh}"
                res.append((float((g["x:" + k_] - g.cost).mean()), T_, S, Hh))
            res.sort(key=lambda z_: -z_[0])
            ext_json[gk] = {}
            for xm, T_, S, Hh in res:
                k_ = f"{T_}:{S}:{Hh}"
                r, x, z = g["r:" + k_] - g.cost, g["x:" + k_] - g.cost, g["z:" + k_] - g.cost
                by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in x.groupby(g.Y).mean().items())
                hitx = float((g["x:" + k_] >= T_ - 1e-9).mean())
                hitr = float((g["r:" + k_] >= T_ - 1e-9).mean())
                print(f"   هدف +{int(T_ * 100)}% · وقف {('−' + str(int(S * 100)) + '%') if S else 'بدون'} · {Hh} جلسة: عادي {pr(r.mean())} ← ممتد {pr(x.mean())} (أشد {pr(z.mean())})"
                      f" · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean(), 0)} · وصل الهدف {pr(hitr, 0)} ← {pr(hitx, 0)} · [{by}]")
                ext_json[gk][k_] = {"reg": float(r.mean()), "ext": float(x.mean()), "strict": float(z.mean()), "med": float(x.median()), "win": float((x > 0).mean()),
                                    "hit_r": hitr, "hit_x": hitx, "n": int(len(g)), "by": {int(y): float(v) for y, v in x.groupby(g.Y).mean().items()}}
            r, x, z = g["r:lad"] - g.cost, g["x:lad"] - g.cost, g["z:lad"] - g.cost
            by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in x.groupby(g.Y).mean().items())
            print(f"   سلّم ثلث عند +50/+100/+200% بدون وقف 20 جلسة: عادي {pr(r.mean())} ← ممتد {pr(x.mean())} (أشد {pr(z.mean())}) · وسيط {pr(x.median())} · [{by}]")
            ext_json[gk]["lad"] = {"reg": float(r.mean()), "ext": float(x.mean()), "strict": float(z.mean()), "by": {int(y): float(v) for y, v in x.groupby(g.Y).mean().items()}}
            xm, T_, S, Hh = res[0]                      # أفضل خطة ممتدة: حسب شريحة السعر الحقيقي
            k_ = f"{T_}:{S}:{Hh}"
            x, r = g["x:" + k_] - g.cost, g["r:" + k_] - g.cost
            for nm, m in (("تحت 1$", g.px < 1), ("1-5$", (g.px >= 1) & (g.px < 5)), ("فوق 5$", g.px >= 5)):
                if m.sum() >= 30:
                    print(f"   أفضل خطة (+{int(T_ * 100)}%/{S}/{Hh}) — {nm}: n={int(m.sum())} · ممتد {pr(x[m].mean())} · عادي {pr(r[m].mean())}")

    model = {"version": "hr3-" + datetime.now(timezone.utc).strftime("%Y%m%d"), "feats": FEATS, "w": [float(x) for x in w_f],
             "feats_b": FB, "wb": [float(x) for x in wb_f], "mean": [float(x) for x in mu_f], "std": [float(x) for x in sd_f],
             "thresholds": thr_f, "tiers": tiers, "bleed_cal": bcal, "pb_q": pb_q, "liq": LIQ,
             "base_e10": float(T.E10.mean()), "base_e5": float(T.E5.mean()), "base_bleed": float(T.B.mean()),
             "n_rows": int(len(T)), "period": [int(T.D.min()), int(T.D.max())], "direct": direct, "ext": ext_json, "cal": cal,
             "top3_year": {int(y): {"e10": float(g.E10.mean()), "e5": float(g.E5.mean()), "bleed": float(g.B.mean()), "n": int(len(g))}
                           for y, g in G["top3"].groupby("Y")}}
    js = json.dumps(model, ensure_ascii=False)
    print("HR3_BEGIN")
    for k in range(0, len(js), 900):
        print("H3|" + js[k:k + 900])
    print("HR3_END")
    print(f"\nخلص في {round(time.time() - t0)} ثانية", flush=True)


if __name__ == "__main__":
    main()
