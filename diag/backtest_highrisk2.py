"""
🧨 عالي مخاطرة 2 — «طوره وخله ثقة والتوصية تكون مفيدة وقوية وتبني محفظتي» (طلب المستخدم 2026-11-12) — تشخيص فقط.

الجديد عن الدراسة الأولى:
1) انفجار **حقيقي** = أعلى سعر ≥ ضعف إغلاق أمس **وتداول اليوم ≥ 500 ألف$** (الأولى كانت تحسب صفقة شاذة بسهم ميت «انفجار»).
2) تاريخ أطول (من 2018-09، الصفوف من ~2019-10) — عشان نشوف أكثر من دورة سوق.
3) 🌡 حرارة السوق: مؤشرات على كل السوق (معروفة بإغلاق اليوم) — هل تفصل الفترات الرابحة عن الخاسرة؟
   الاختيار على فترة تطوير (2021 ← 2024) والحكم على فترة ما انلمست (2025 ← 2026).
4) 17 طريقة دخول/خروج **بعد التكاليف** (فرق السعر والانزلاق) + سيولة تسمح بالشراء فعلاً (وسيط التداول ≥ 30 ألف$).
5) «اليوم الثاني»: بعد يوم الانطلاقة (إغلاق قوي) — هل يكمل؟
6) محاكاة محفظة بسيطة للقاعدة المختارة.
الميزات ومؤشرات السوق من highrisk.py (نفس ملف الرادار الحي).
المخرجات: جداول + JSON بين HR2_BEGIN/END (أسطر H2| مقطعة).
"""
import bisect
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
import highrisk as HR

START = datetime(2018, 9, 1, tzinfo=timezone.utc)
H = 20
FEATS = HR.FEATS
LIQ = 30_000                      # سيولة تسمح بتذكرة صغيرة: وسيط التداول اليومي ≥ 30 ألف$
DEV_END, TEST_START = 20241231, 20250101
PL = {"o5": "افتتاح بكرة · +30% · وقف −15% · 5", "o20": "افتتاح بكرة · +50% · وقف −30% · 20", "olot": "افتتاح بكرة · +100% · بدون وقف · 20",
      "c20": "إغلاق · +50% · وقف −30% · 20 جلسة", "c10": "إغلاق · +50% · وقف −20% · 10", "c5": "إغلاق · +30% · وقف −15% · 5",
      "c3": "إغلاق · +20% · وقف −10% · 3", "lot": "إغلاق · +100% · بدون وقف · 20 (يانصيب)",
      "lad": "إغلاق · سلّم ثلث عند +50/+100/+200% · 20 جلسة", "on": "إغلاق ← افتتاح بكرة", "d1": "إغلاق ← إغلاق بكرة",
      "d1L": "إغلاق ← بيع +20% بكرة وإلا إغلاقها", "h5": "احتفاظ 5 جلسات", "h20": "احتفاظ 20 جلسة",
      "s15": "شراء Stop +15% ← +50% / −20% / 5", "s15b": "شراء Stop +15% ← +30% / −15% / 3", "s25": "شراء Stop +25% ← +50% / −20% / 5",
      "s15d": "شراء Stop +15% ← بيع إغلاق نفس اليوم", "s15d5": "شراء Stop +15% ← +50% وإلا إغلاق نفس اليوم",
      "s30d": "شراء Stop +30% ← بيع إغلاق نفس اليوم"}
STOP_PL = {"s15", "s15b", "s25", "s15d", "s15d5", "s30d"}
HOLD = {"o5": 5, "o20": 20, "olot": 20, "c20": 20, "c10": 10, "c5": 5, "c3": 3, "lot": 20, "lad": 20, "on": 1, "d1": 1, "d1L": 1, "h5": 5, "h20": 20,
        "s15": 5, "s15b": 3, "s25": 5, "s15d": 1, "s15d5": 1, "s30d": 1}


def universe():
    from backtest_highrisk import universe as u
    return u()


def job(args):
    sym, d, spl = args
    if d is None or len(d) < 300:
        return None
    d = d.copy()
    d["Stock Splits"] = 0.0
    for dt, ratio in spl or []:
        dt = pd.Timestamp(dt)
        dt = (dt.tz_localize(None) if dt.tzinfo else dt).normalize()
        if dt in d.index:
            d.loc[dt, "Stock Splits"] = ratio
    F = HR.features(d)
    mk = HR.market_frame(d, F)
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    idx = np.arange(n)
    fsh, roll = HR._fsh, HR._roll
    ex = F["ex"].to_numpy(bool)
    jump = F["jump"].to_numpy(float)
    ex_raw = np.nan_to_num(jump) >= 2
    exf = ex.astype(float)
    pc = np.r_[np.nan, c[:-1]]
    dvd = v * (h + l + c) / 3
    o1, h1, l1, c1 = fsh(o, 1), fsh(h, 1), fsh(l, 1), fsh(c, 1)
    E5, E10, E20 = (fsh(roll(exf, k, "max"), k) for k in (5, 10, 20))
    mx5, mx10, mx20 = (fsh(roll(h, k, "max"), k) / c - 1 for k in (5, 10, 20))
    dd20 = fsh(roll(l, 20, "min"), 20) / c - 1
    r20 = fsh(c, 20) / c - 1

    def close_entry(T, S, Hh):
        out = np.full(n, np.nan)
        done = np.zeros(n, bool)
        for k in range(1, Hh + 1):
            hk, ck = fsh(h, k), fsh(c, k)
            hit = ~done & (hk >= c * (1 + T))
            out[hit] = T
            done |= hit
            if S is not None:
                st = ~done & (ck <= c * (1 - S))
                out[st] = ck[st] / c[st] - 1
                done |= st
            if k == Hh:
                rest = ~done & np.isfinite(ck)
                out[rest] = ck[rest] / c[rest] - 1
        return out

    def stop_entry(trig, T, S, Hh):
        lvl = c * (1 + trig)
        trg = np.isfinite(h1) & (h1 >= lvl)
        ent = np.where(np.isfinite(o1) & (o1 > lvl), o1, lvl)
        out = np.full(n, np.nan)
        done = ~trg
        hit = ~done & (h1 >= ent * (1 + T))
        out[hit] = T
        done |= hit
        st = ~done & (c1 <= ent * (1 - S))
        out[st] = c1[st] / ent[st] - 1
        done |= st
        for k in range(2, Hh + 1):
            hk, ck = fsh(h, k), fsh(c, k)
            hit = ~done & (hk >= ent * (1 + T))
            out[hit] = T
            done |= hit
            st = ~done & (ck <= ent * (1 - S))
            out[st] = ck[st] / ent[st] - 1
            done |= st
            if k == Hh:
                rest = ~done & np.isfinite(ck)
                out[rest] = ck[rest] / ent[rest] - 1
        return out, trg

    def open_entry(T, S, Hh):
        """دخول عند افتتاح بكرة (الإشارة بعد الإغلاق) — الهدف والوقف من سعر الافتتاح، واليوم الأول محسوب."""
        ent = o1
        out = np.full(n, np.nan)
        done = ~np.isfinite(ent)
        for k in range(1, Hh + 1):
            hk, ck = fsh(h, k), fsh(c, k)
            hit = ~done & (hk >= ent * (1 + T))
            out[hit] = T
            done |= hit
            if S is not None:
                st = ~done & (ck <= ent * (1 - S))
                out[st] = ck[st] / ent[st] - 1
                done |= st
            if k == Hh:
                rest = ~done & np.isfinite(ck)
                out[rest] = ck[rest] / ent[rest] - 1
        return out

    P = {"o5": open_entry(0.3, 0.15, 5), "o20": open_entry(0.5, 0.30, 20), "olot": open_entry(1.0, None, 20),
         "c20": close_entry(0.5, 0.30, 20), "c10": close_entry(0.5, 0.20, 10), "c5": close_entry(0.3, 0.15, 5),
         "c3": close_entry(0.2, 0.10, 3), "lot": close_entry(1.0, None, 20)}
    P["lad"] = (np.where(mx20 >= 0.5, 0.5, r20) + np.where(mx20 >= 1.0, 1.0, r20) + np.where(mx20 >= 2.0, 2.0, r20)) / 3
    P["on"] = o1 / c - 1
    P["d1"] = c1 / c - 1
    P["d1L"] = np.where(o1 >= c * 1.2, o1 / c - 1, np.where(h1 >= c * 1.2, 0.2, c1 / c - 1))
    P["h5"] = fsh(c, 5) / c - 1
    P["h20"] = r20
    P["s15"], trg15 = stop_entry(0.15, 0.5, 0.20, 5)
    P["s15b"], _ = stop_entry(0.15, 0.3, 0.15, 3)
    P["s25"], _ = stop_entry(0.25, 0.5, 0.20, 5)
    lvl = c * 1.15
    ent = np.where(np.isfinite(o1) & (o1 > lvl), o1, lvl)
    tg = np.isfinite(h1) & (h1 >= lvl)
    P["s15d"] = np.where(tg, c1 / ent - 1, np.nan)
    P["s15d5"] = np.where(tg, np.where(h1 >= ent * 1.5, 0.5, c1 / ent - 1), np.nan)
    lvl = c * 1.30
    ent = np.where(np.isfinite(o1) & (o1 > lvl), o1, lvl)
    tg = np.isfinite(h1) & (h1 >= lvl)
    P["s30d"] = np.where(tg, c1 / ent - 1, np.nan)

    Dint = d.index.strftime("%Y%m%d").astype(int).to_numpy(np.int32)
    dv20 = np.nan_to_num(F["dv20"].to_numpy(float))
    praw = F["praw"].to_numpy(float)
    sh = HR._sh
    cols = {"t": sym, "D": Dint, "close": c, "praw": praw, "dv20": dv20,
            "m60": np.clip(np.nan_to_num(c / sh(c, 60) - 1), -1, 5), "m120": np.clip(np.nan_to_num(c / sh(c, 120) - 1), -1, 8),
            "m250": np.clip(np.nan_to_num(c / sh(c, 250) - 1), -1, 10)}
    for f_ in FEATS:
        cols[f_] = F[f_].to_numpy(float)
    cols.update({"E5": E5, "E10": E10, "E20": E20, "mx5": mx5, "mx10": mx10, "mx20": mx20, "dd20": dd20, "trg15": trg15.astype(float)})
    cols.update(P)
    keep = (idx >= 260) & (idx <= n - H - 1) & (idx % 2 == 0) & (praw >= HR.ACT_MIN_PX) & (dv20 >= HR.ACT_MIN_DV20) & ~ex_raw
    rows = pd.DataFrame(cols)[keep]
    for col in rows.columns:
        if rows[col].dtype == np.float64:
            rows[col] = rows[col].astype(np.float32)
    # «اليوم الثاني»: أيام الانطلاقة (انفجار حقيقي، أو +30% بإغلاق وتداول ≥ مليون$)
    r1 = c / pc - 1
    evm = (idx >= 60) & (idx <= n - 7) & (ex | ((r1 >= 0.3) & (dvd >= 1e6))) & (np.nan_to_num(pc) >= HR.ACT_MIN_PX)
    E = np.flatnonzero(evm)
    ev = pd.DataFrame({"t": sym, "D": Dint[E], "pc": (pc * (praw / np.maximum(c, 1e-12)))[E], "close": c[E], "praw": praw[E], "r1": r1[E], "jump": jump[E], "isex": ex[E].astype(float),
                       "cp": F["closepos1"].to_numpy(float)[E], "vol": F["vol1_20"].to_numpy(float)[E], "dvd": dvd[E],
                       "n_prior": F["n_ep252"].to_numpy(float)[np.maximum(E - 1, 0)], "dd_prior": F["dd252"].to_numpy(float)[np.maximum(E - 1, 0)],
                       "gap": (o / pc - 1)[E], "on1": (o1 / c - 1)[E], "hi1": (h1 / c - 1)[E], "lo1": (l1 / c - 1)[E], "c1": (c1 / c - 1)[E],
                       "hi5": mx5[E], "c5": (fsh(c, 5) / c - 1)[E]})
    yr = Dint // 10000
    info = {"first": int(Dint[0]), "last": int(Dint[-1]),
            "raw": {int(y): int(k) for y, k in zip(*np.unique(yr[ex_raw], return_counts=True))},
            "real": {int(y): int(k) for y, k in zip(*np.unique(yr[ex], return_counts=True))},
            "raw_dv": [float(x) for x in dvd[ex_raw]]}
    path = (Dint, o.astype(np.float32), h.astype(np.float32), l.astype(np.float32), c.astype(np.float32))
    return rows, (Dint, mk.to_numpy(np.float32)), ev, info, path


def fit_logit(X, y, lam=5.0, iters=40):
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(k + 1)
    R = lam * np.eye(k + 1)
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        W = p * (1 - p)
        g = Xb.T @ (y - p) - R @ w
        Hm = (Xb * W[:, None]).T @ Xb + R
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-7:
            break
    return w


def auc(y, s):
    y = np.asarray(y)
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def zs(X, mu, sd):
    return np.clip((X - mu) / sd, -5, 5)


def predict(X, w, mu, sd, chunk=400_000):
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        Z = zs(X[i:i + chunk].astype(float), mu, sd)
        out[i:i + chunk] = 1 / (1 + np.exp(-(w[0] + Z @ w[1:])))
    return out


def exp_pct(vals, minp=250):
    """النسبة المئوية مقارنة بالماضي بس (بدون نظرة للمستقبل)."""
    out = np.full(len(vals), np.nan)
    hist = []
    for i, x in enumerate(vals):
        if np.isfinite(x):
            if len(hist) >= minp:
                out[i] = bisect.bisect_right(hist, x) / len(hist)
            bisect.insort(hist, x)
    return out


def pr(x, d=1):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{d}f}%"


def net_cols(G):
    """عوائد الخطط بعد التكاليف: ذهاب وعودة 4% تحت 1$ · 2% تحت 5$ · 1% فوقها (+1% لأوامر Stop)."""
    px = G["praw"].to_numpy(float)
    cost = np.where(px < 1, 0.04, np.where(px < 5, 0.02, 0.01))
    X = pd.DataFrame({p: G[p].to_numpy(float) - cost - (0.01 if p in STOP_PL else 0) for p in PL})
    X["D"] = G["D"].to_numpy()
    return X


def tstat(x, hold=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    n_eff = max(len(x) / max(hold, 1), 3)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(n_eff) + 1e-12))


def sim(rets, f=0.03):
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return 1.0, 0.0
    eq = np.cumprod(1 + f * r)
    return float(eq[-1]), float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    syms = universe()
    iw = None
    for attempt in range(3):
        try:
            iw = A.bars(["IWM"], "1Day", START)["IWM"]
            break
        except Exception as e:
            print("IWM err", repr(e)[:100], flush=True)
            time.sleep(10)
    iw.index = pd.to_datetime(iw.index).tz_localize(None).normalize()
    iw = iw[~iw.index.duplicated(keep="last")]
    cal = iw.index
    calD = cal.strftime("%Y%m%d").astype(int).to_numpy(np.int64)
    ACC = np.zeros((len(cal), len(HR.MK_COLS)))
    parts, evs, infos = [], [], []
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
            for res in ex.map(job, jobs, chunksize=4):
                if res is None:
                    continue
                rows, (Dm, vals), ev, info = res[:4]
                pos = np.searchsorted(calD, Dm)
                ok = (pos < len(calD)) & (calD[np.minimum(pos, len(calD) - 1)] == Dm)
                ACC[pos[ok]] += vals[ok]
                if len(rows):
                    parts.append(rows)
                if len(ev):
                    evs.append(ev)
                infos.append(info)
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · صفوف {sum(len(p) for p in parts):,} · {round(time.time() - t0)}ث", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["t"] = df["t"].astype("category")
    for f_ in FEATS:
        x = df[f_].replace([np.inf, -np.inf], np.nan)
        df[f_] = x.fillna(x.median()).astype(np.float32)
    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    df["Y"] = (df.D // 10000).astype(np.int16)
    print(f"\nالصفوف: {len(df):,} سهم-يوم · أسهم {df.t.nunique():,} · من {df.D.min()} لين {df.D.max()} · أيام انطلاقة {len(ev):,} · {round(time.time() - t0)}ث", flush=True)

    # ---- 0) جودة البيانات: الانفجار الخام مقابل الحقيقي + البقاء
    raw_y, real_y = {}, {}
    for inf in infos:
        for y, k in inf["raw"].items():
            raw_y[y] = raw_y.get(y, 0) + k
        for y, k in inf["real"].items():
            real_y[y] = real_y.get(y, 0) + k
    print("\n==== 0) الانفجار «الخام» (×2 بأي تداول) مقابل «الحقيقي» (×2 وتداول ≥ 500 ألف$) — أيام بكل سنة ====")
    for y in sorted(raw_y):
        print(f"  {y}: خام {raw_y[y]:,} · حقيقي {real_y.get(y, 0):,} ({real_y.get(y, 0) / max(raw_y[y], 1) * 100:.0f}%)")
    dvs = np.array([x for inf in infos for x in inf["raw_dv"]])
    if len(dvs):
        print("  تداول يوم الانفجار الخام ($): " + " · ".join(f"{int(q * 100)}% {np.quantile(dvs, q):,.0f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
    last_y = pd.Series([inf["last"] // 10000 for inf in infos]).value_counts().sort_index()
    print("  آخر سنة تداول لكل سهم (البقاء): " + " · ".join(f"{y}: {k}" for y, k in last_y.items()))
    print("  أسهم نشطة بكل سنة (لها صفوف): " + " · ".join(f"{y}: {k}" for y, k in df.groupby("Y").t.nunique().items()))
    print(f"  نسبة الانفجار الحقيقي خلال 10 جلسات (الأساس): {pr(df.E10.mean(), 2)} · خلال 5: {pr(df.E5.mean(), 2)}")

    # ---- 1) مؤشرات السوق
    MK = pd.DataFrame(ACC, index=cal, columns=HR.MK_COLS)
    R = HR.regime_series(MK, iw["Close"])
    P_ = pd.DataFrame({k: exp_pct(R[k].to_numpy(float)) for k in HR.REG_COLS}, index=cal)
    R.index = calD
    P_.index = calD
    print("\nمؤشرات السوق (وسيط كل سنة): " + " | ".join(HR.REG_COLS))
    yrs = pd.Series(calD // 10000, index=calD)
    for y, g in R.groupby(yrs):
        print(f"  {y}: " + " | ".join(f"{g[k].median():.3f}" for k in HR.REG_COLS))

    # ---- 2) النموذج سنة بسنة (تدريب على كل اللي قبل، حجر شهر)
    df["p"] = np.float32(np.nan)
    df["q99"] = np.float32(np.nan)
    df["q95"] = np.float32(np.nan)
    rng = np.random.default_rng(1)
    XF = df[FEATS].to_numpy(np.float32)
    yE = df.E10.to_numpy(float)
    print("\n==== 2) النموذج (انفجار حقيقي خلال 10 جلسات) — اختبار كل سنة بنموذج ما شافها ====")
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
        w = fit_logit(zs(Xtr, mu, sd), yE[tri], lam=5.0)
        ptr = predict(Xtr[:400_000], w, mu, sd)
        q99, q95 = float(np.quantile(ptr, 0.99)), float(np.quantile(ptr, 0.95))
        del Xtr
        pte = predict(XF[tem], w, mu, sd)
        df.loc[tem, "p"] = pte
        df.loc[tem, "q99"] = q99
        df.loc[tem, "q95"] = q95
        yte = yE[tem]
        top = pte >= q99
        print(f"  {Y}: تدريب {trm.sum():,} · اختبار {tem.sum():,} · AUC {auc(yte, pte):.3f} · الأساس {pr(yte.mean(), 2)} · "
              f"🧨 (فوق حد التدريب 99%) n={int(top.sum()):,} انفجر {pr(yte[top].mean())}", flush=True)
    # النموذج النهائي على كل البيانات (للرادار الحي)
    tri = np.arange(len(df))
    if len(tri) > 2_000_000:
        tri = rng.choice(tri, 2_000_000, replace=False)
    Xall = XF[tri].astype(float)
    mu_f, sd_f = Xall.mean(0), Xall.std(0)
    sd_f[sd_f == 0] = 1
    w_f = fit_logit(zs(Xall, mu_f, sd_f), yE[tri], lam=5.0)
    pall = predict(Xall[:600_000], w_f, mu_f, sd_f)
    thr_f = {"ready": float(np.quantile(pall, 0.99)), "near": float(np.quantile(pall, 0.95))}
    del Xall, XF, yE
    print("أوزان النموذج النهائي: " + " · ".join(f"{f_} {c_:+.2f}" for f_, c_ in sorted(zip(FEATS, w_f[1:]), key=lambda x: -abs(x[1]))[:14]))
    base_all = {"e10": float(df.E10.mean()), "e5": float(df.E5.mean()), "n": int(len(df))}

    T = df[df.p.notna()].copy()
    del df
    T["vq"] = pd.qcut(T["atr20p"].rank(method="first"), 5, labels=False)
    T["pq"] = pd.cut(T["praw"], [0, 1, 5, 1e9], labels=False)
    T["liq"] = T.dv20 >= LIQ
    T["rkL"] = T[T.liq].groupby("D").p.rank(ascending=False, method="first")
    Bm = T.liq & (T.p >= T.q99) & (T.vol1_20 >= 2)
    T["rkB"] = T[Bm].groupby("D").p.rank(ascending=False, method="first")
    G = {"top3": T[T.rkL <= 3], "top10": T[T.rkL <= 10], "q99": T[T.liq & (T.p >= T.q99)], "q95": T[T.liq & (T.p >= T.q95) & (T.p < T.q99)],
         "top3B": T[T.rkB <= 3], "pg": T[T.liq & (T.n_ep252 >= 1) & (T.dd252 <= -0.8)],
         "illiq99": T[~T.liq & (T.p >= T.q99)]}
    GA = {"top3": "أعلى 3 كل يوم (سيولة ≥ 30 ألف$)", "top10": "أعلى 10 كل يوم", "q99": "🧨 فوق حد 99%", "q95": "🟠 بين 95% و99%",
          "top3B": "أعلى 3 من 🧨 + فوليوم آخر جلسة ≥ 2×", "pg": "مرجعية: سبق انفجر + منهار 80%+", "illiq99": "🧨 بسيولة أقل من 30 ألف$ (للمقارنة)"}
    base_cell = T[T.liq].groupby(["Y", "vq", "pq"])
    NT = net_cols(T[T.liq])
    NT[["Y", "vq", "pq"]] = T.loc[T.liq, ["Y", "vq", "pq"]].to_numpy()
    base_net = NT.groupby(["Y", "vq", "pq"])[list(PL)].mean()
    base_e = base_cell[["E5", "E10"]].mean()

    def matched(g, tbl, col):
        k = g.groupby(["Y", "vq", "pq"]).size()
        k = k[k.index.isin(tbl.index)]
        v = tbl.loc[k.index, col]
        ok = v.notna()
        return float((v[ok] * k[ok]).sum() / max(k[ok].sum(), 1))

    print("\n==== 3) المجموعات (خارج العينة 2021 ← 2026) — انفجار حقيقي خلال 5 / 10 جلسات مقابل نفس السنة والتذبذب والسعر ====")
    for k, g in G.items():
        if len(g) < 50:
            print(f"  {GA[k]}: عدد قليل ({len(g)})")
            continue
        print(f"  {GA[k]}: n={len(g):,} · خلال 5 {pr(g.E5.mean())} ({pr(matched(g, base_e, 'E5'))}) · خلال 10 {pr(g.E10.mean())} ({pr(matched(g, base_e, 'E10'))})"
              f" · +50% خلال 10 {pr((g.mx10 >= 0.5).mean())} · −30% خلال 20 {pr((g.dd20 <= -0.3).mean())} · وسيط السعر الحقيقي {g.praw.median():.2f}$ · وسيط السيولة {g.dv20.median():,.0f}$")
    print("\nبكل سنة — أعلى 3 كل يوم: انفجر خلال 10 (المماثل) · وأعلى 10:")
    for Y, g in G["top3"].groupby("Y"):
        g10 = G["top10"][G["top10"].Y == Y]
        print(f"  {Y}: أعلى 3 n={len(g):,} {pr(g.E10.mean())} ({pr(matched(g, base_e, 'E10'))}) · أعلى 10 n={len(g10):,} {pr(g10.E10.mean())} ({pr(matched(g10, base_e, 'E10'))})")

    print("\n==== 4) طرق الدخول والخروج **بعد التكاليف** — متوسط الصفقة · وسيط · رابحة · (المماثل بنفس السنة والتذبذب والسعر) ====")
    NG = {}
    for k in ("top3", "top10", "q99", "top3B", "pg"):
        g = G[k]
        if len(g) < 50:
            continue
        X = net_cols(g)
        X[["Y", "vq", "pq"]] = g[["Y", "vq", "pq"]].to_numpy()
        X["E5"], X["E10"] = g["E5"].to_numpy(float), g["E10"].to_numpy(float)
        NG[k] = X
        print(f"\n-- {GA[k]} (n={len(g):,})")
        for p in PL:
            x = X[p].dropna()
            if len(x) < 30:
                continue
            by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in X.groupby("Y")[p].mean().items() if np.isfinite(v))
            print(f"   {PL[p]}: {pr(x.mean())} · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean(), 0)} · n={len(x):,} · المماثل {pr(matched(X.dropna(subset=[p]), base_net, p))} · بالسنة [{by}]")

    # ---- 5) حرارة السوق: هل تفصل الأيام الرابحة؟ (تطوير 2021-2024 · اختبار 2025-2026)
    print("\n==== 5) 🌡 حرارة السوق — متوسط اليوم بعد التكاليف: حار (أعلى ثلث تاريخياً) مقابل بارد (أقل ثلث) ====")
    # المقاييس كلها على مستوى أيام التقويم (معروفة بإغلاق اليوم): نسبة كل مؤشر مقارنة بماضيه + مركب بسيط بدون تدريب
    GZ = {k: P_[k].to_numpy(float) for k in HR.REG_COLS}
    naive_raw = np.nanmean(np.vstack([GZ[k] for k in ("heat10", "follow20", "pg_r20", "pen_ab")]), axis=0)
    GZ["naive4"] = exp_pct(naive_raw, minp=200)
    Zi = np.column_stack([R[k].to_numpy(float) for k in HR.REG_COLS])
    devc, tstc = calD <= DEV_END, calD >= TEST_START
    mu_r, sd_r = np.nanmean(Zi[devc], axis=0), np.nanstd(Zi[devc], axis=0)
    sd_r[~np.isfinite(sd_r) | (sd_r == 0)] = 1
    mu_r[~np.isfinite(mu_r)] = 0
    Zr = np.nan_to_num(np.clip((Zi - mu_r) / sd_r, -4, 4))
    out_reg = {"naive_q": [float(x) for x in np.nanquantile(naive_raw, np.linspace(0, 1, 101))], "ridge": {}, "stats": {}, "last": {}}
    calpos = pd.Series(np.arange(len(calD)), index=calD)
    best_rows = []
    BUCK = (("cold", lambda q: q <= 1 / 3), ("mid", lambda q: (q > 1 / 3) & (q < 2 / 3)), ("hot", lambda q: q >= 2 / 3), ("vhot", lambda q: q >= 0.85))
    for gk in ("top10", "top3", "q99"):
        if gk not in NG:
            continue
        Xg = NG[gk]
        dm = Xg.groupby("D")[list(PL)].mean()
        idxD = dm.index.to_numpy()
        pos = calpos.reindex(idxD).to_numpy()
        dev, tst = idxD <= DEV_END, idxD >= TEST_START
        gz = dict(GZ)
        # مقياس انحدار (معاملاته من فترة التطوير بس): يتوقع متوسط اليوم لخطة مرجعية من مؤشرات السوق
        for ref in ("o5", "c5", "s15b", "d1L"):
            y = dm[ref].to_numpy(float)
            ok = dev & np.isfinite(y)
            if ok.sum() < 200:
                continue
            Xr = np.column_stack([np.ones(ok.sum()), Zr[pos[ok]]])
            lam = 50.0 * np.eye(Xr.shape[1])
            lam[0, 0] = 0
            beta = np.linalg.solve(Xr.T @ Xr + lam, Xr.T @ y[ok])
            pred = beta[0] + Zr @ beta[1:]
            gz[f"ridge_{ref}"] = exp_pct(pred, minp=200)
            out_reg["ridge"][f"{gk}:{ref}"] = {"beta": [round(float(b), 6) for b in beta], "mu": [float(x) for x in mu_r], "sd": [float(x) for x in sd_r],
                                               "q": [round(float(x), 6) for x in np.nanquantile(pred, np.linspace(0, 1, 101))]}
        rows_ = []
        for ind, full in gz.items():
            pct = full[pos]
            for p in PL:
                x = dm[p].to_numpy(float)
                st = {}
                for nm, sel in (("dev", dev), ("tst", tst)):
                    hot, cold, mid = x[sel & (pct >= 2 / 3)], x[sel & (pct <= 1 / 3)], x[sel & (pct > 1 / 3) & (pct < 2 / 3)]
                    st[nm] = (np.nanmean(hot) if np.isfinite(hot).sum() >= 15 else np.nan, tstat(hot, HOLD[p]), int(np.isfinite(hot).sum()),
                              np.nanmean(cold) if np.isfinite(cold).sum() >= 15 else np.nan, int(np.isfinite(cold).sum()),
                              np.nanmean(mid) if np.isfinite(mid).sum() >= 15 else np.nan)
                rows_.append((gk, ind, p, st))
        rows_ = [r for r in rows_ if r[3]["dev"][2] >= 100 and np.isfinite(r[3]["dev"][1])]
        rows_.sort(key=lambda r: -r[3]["dev"][1])
        print(f"\n-- {GA[gk]}: أفضل 16 (مؤشر × خطة) حسب فترة التطوير — ثم نتيجتها بفترة الاختبار اللي ما انلمست")
        print("   المؤشر · الخطة | تطوير: حار (t، أيام) · وسط · بارد | اختبار: حار (t، أيام) · وسط · بارد")
        for gk_, ind, p, st in rows_[:16]:
            d_, t_ = st["dev"], st["tst"]
            print(f"   {ind} · {p} | {pr(d_[0])} (t={d_[1]:.1f}، {d_[2]}) · {pr(d_[5])} · {pr(d_[3])} | {pr(t_[0])} (t={t_[1]:.1f}، {t_[2]}) · {pr(t_[5])} · {pr(t_[3])}")
        best_rows += rows_[:40]
        print("   -- كل مؤشر على 6 خطط مرجعية: تطوير حار/بارد ← اختبار حار/بارد")
        for ind, full in gz.items():
            pct = full[pos]
            line = []
            for p in ("o5", "c5", "s15b", "d1L", "olot", "o20"):
                x = dm[p].to_numpy(float)
                a = [np.nanmean(x[s & c_]) if np.isfinite(x[s & c_]).sum() >= 15 else np.nan
                     for s in (dev, tst) for c_ in (pct >= 2 / 3, pct <= 1 / 3)]
                line.append(f"{p}: {pr(a[0])}/{pr(a[1])} ← {pr(a[2])}/{pr(a[3])}")
            print(f"   {ind:>10}: " + " | ".join(line))
        # إحصائيات مضغوطة للرادار: (مقياس × حرارة × فترة) ← عدد، انفجار خلال 5/10، ولكل خطة [متوسط، رابحة، عدد]
        SELG = [k for k in ("naive4", "heat10", "follow20", "pg_r20", "pg_h20", "pen_ab", "ridge_o5", "ridge_c5") if k in gz]
        Dg = Xg.D.to_numpy()
        posg = calpos.reindex(Dg).to_numpy()
        sj = {}
        for ind in SELG:
            q = gz[ind][posg]
            sj[ind] = {}
            for bn, bf in BUCK:
                bm = bf(q)
                sj[ind][bn] = {}
                for pn_, pm in (("dev", Dg <= DEV_END), ("tst", Dg >= TEST_START), ("all", np.ones(len(Dg), bool))):
                    g_ = Xg[bm & pm]
                    rec = {"n": int(len(g_)), "days": int(g_.D.nunique()), "e5": round(float(g_.E5.mean()), 4) if len(g_) else None,
                           "e10": round(float(g_.E10.mean()), 4) if len(g_) else None}
                    for p in PL:
                        x = g_[p].dropna()
                        rec[p] = [round(float(x.mean()), 4), round(float((x > 0).mean()), 3), int(len(x))] if len(x) >= 10 else None
                    sj[ind][bn][pn_] = rec
        out_reg["stats"][gk] = sj
        out_reg["last"] = {k: (round(float(v[-1]), 4) if np.isfinite(v[-1]) else None) for k, v in gz.items()}
        # ---- 6) محاكاة محفظة: أفضل 3 تركيبات بالتطوير (3% من المحفظة لكل صفقة)
        for _, ind, p, st in rows_[:3]:
            hotm = gz[ind][posg] >= 2 / 3
            for nm, sel in (("تطوير 2021-2024", Dg <= DEV_END), ("اختبار 2025-2026", Dg >= TEST_START)):
                tr_hot = Xg.loc[sel & hotm].sort_values("D")[p].dropna()
                tr_all = Xg.loc[sel].sort_values("D")[p].dropna()
                e1, d1 = sim(tr_hot)
                e2, d2 = sim(tr_all)
                print(f"   محفظة ({nm}) — {ind} حار × {p}: صفقات {len(tr_hot):,} · متوسط {pr(tr_hot.mean())} · رابحة {pr((tr_hot > 0).mean(), 0)} · "
                      f"المحفظة ×{e1:.2f} (أقصى نزول {pr(d1, 0)}) | بدون فلتر الحرارة: صفقات {len(tr_all):,} · متوسط {pr(tr_all.mean())} · ×{e2:.2f} ({pr(d2, 0)})")
    if best_rows:
        pos_ = sum(1 for r in best_rows if np.isfinite(r[3]["tst"][0]) and r[3]["tst"][0] > 0)
        val = sum(1 for r in best_rows if np.isfinite(r[3]["tst"][0]))
        print(f"\nمن أفضل {len(best_rows)} تركيبة بالتطوير: {pos_} من {val} بقت موجبة بفترة الاختبار.")
    # جدول شهري (للفهم): حرارة السوق مقابل نتيجة أعلى 10 ذاك الشهر
    if "top10" in NG:
        Xg = NG["top10"]
        mon = Xg.D // 100
        calmon = pd.Series(calD // 100, index=calD)
        print("\nشهرياً — أعلى 10: n · انفجر خلال 10 · o5 · c5 · olot · s15b | مؤشرات آخر الشهر: heat10 · follow20 · pg_r20 · pen_ab · naive4 (نسبة تاريخية)")
        nv = pd.Series(GZ["naive4"], index=calD)
        for m, g_ in Xg.groupby(mon):
            dlast = calmon[calmon == m].index.max()
            print(f"  {m}: {len(g_):>4} · {pr(g_.E10.mean())} · {pr(g_.o5.mean())} · {pr(g_.c5.mean())} · {pr(g_.olot.mean())} · {pr(g_.s15b.mean())} | "
                  f"{R.heat10.get(dlast, np.nan):.2f} · {R.follow20.get(dlast, np.nan):+.3f} · {R.pg_r20.get(dlast, np.nan):+.3f} · {R.pen_ab.get(dlast, np.nan):.2f} · {nv.get(dlast, np.nan):.2f}")


    # ---- 7) «اليوم الثاني» بعد الانطلاقة
    if len(ev):
        ev = ev.replace([np.inf, -np.inf], np.nan).dropna(subset=["on1", "c1", "hi1"])
        ev["Y"] = ev.D // 10000
        cost = np.where(ev.praw < 1, 0.04, np.where(ev.praw < 5, 0.02, 0.01))
        ev["n_on"] = ev.on1 - cost
        ev["n_d1"] = ev.c1 - cost
        ev["n_L20"] = np.where(ev.on1 >= 0.2, ev.on1, np.where(ev.hi1 >= 0.2, 0.2, ev.c1)) - cost
        ev["n_L50"] = np.where(ev.on1 >= 0.5, ev.on1, np.where(ev.hi1 >= 0.5, 0.5, ev.c1)) - cost
        ev["n_h5"] = ev.c5 - cost
        ev["n_oc"] = (1 + ev.c1) / (1 + ev.on1) - 1 - cost                      # شراء افتتاح بكرة ← بيع إغلاقها
        ev["n_oL20"] = np.where((1 + ev.hi1) / (1 + ev.on1) - 1 >= 0.2, 0.2, (1 + ev.c1) / (1 + ev.on1) - 1) - cost   # افتتاح بكرة ← +20% وإلا الإغلاق
        for k in ("heat10", "follow20", "pg_r20"):
            ev[k] = P_[k].reindex(ev.D).to_numpy(float)
        print(f"\n==== 7) «اليوم الثاني» — الشراء بإغلاق يوم الانطلاقة (بعد التكاليف) · {len(ev):,} يوم انطلاقة (انفجار حقيقي أو +30% بتداول ≥ مليون$) ====")

        def line(nm, g):
            if len(g) < 30:
                return f"  {nm}: عدد قليل ({len(g)})"
            return (f"  {nm}: n={len(g):,} · افتتاح بكرة {pr(g.n_on.mean())} (وسيط {pr(g.n_on.median())}، رابحة {pr((g.n_on > 0).mean(), 0)}) · إغلاق بكرة {pr(g.n_d1.mean())}"
                    f" · بيع +20% وإلا الإغلاق {pr(g.n_L20.mean())} · +50% وإلا الإغلاق {pr(g.n_L50.mean())} · 5 جلسات {pr(g.n_h5.mean())} · أعلى بكرة ≥ +20%: {pr((g.hi1 >= 0.2).mean(), 0)}"
                    f" | دخول افتتاح بكرة: ← إغلاقها {pr(g.n_oc.mean())} · ← +20% وإلا الإغلاق {pr(g.n_oL20.mean())}")
        print(line("الكل", ev))
        for nm, m in (("إغلاق قوي (أعلى 20% من مدى اليوم)", ev.cp >= 0.8), ("إغلاق وسط (50-80%)", (ev.cp >= 0.5) & (ev.cp < 0.8)), ("إغلاق ضعيف (< 50%)", ev.cp < 0.5),
                      ("انفجار حقيقي ×2", ev.isex > 0), ("انفجار ×2 + إغلاق قوي", (ev.isex > 0) & (ev.cp >= 0.8)),
                      ("+30% إلى +60% بإغلاق قوي", (ev.r1 >= 0.3) & (ev.r1 < 0.6) & (ev.cp >= 0.8)), ("+60%+ بإغلاق قوي", (ev.r1 >= 0.6) & (ev.cp >= 0.8)),
                      ("سبق انفجر بسنة + إغلاق قوي", (ev.n_prior >= 1) & (ev.cp >= 0.8)), ("أول مرة + إغلاق قوي", (ev.n_prior < 1) & (ev.cp >= 0.8)),
                      ("تحت 1$ + إغلاق قوي", (ev.pc < 1) & (ev.cp >= 0.8)), ("1-5$ + إغلاق قوي", (ev.pc >= 1) & (ev.pc < 5) & (ev.cp >= 0.8)),
                      ("فوق 5$ + إغلاق قوي", (ev.pc >= 5) & (ev.cp >= 0.8)), ("تداول ≥ 10 مليون$ + إغلاق قوي", (ev.dvd >= 1e7) & (ev.cp >= 0.8))):
            print(line(nm, ev[m]))
        sc = ev[ev.cp >= 0.8]
        print("  إغلاق قوي بكل سنة (بيع +20% وإلا الإغلاق): " + " · ".join(f"{int(y)}: {pr(g.n_L20.mean())} (n={len(g)})" for y, g in sc.groupby("Y")))
        for k in ("heat10", "follow20", "pg_r20"):
            for nm, sel in (("تطوير", sc.D <= DEV_END), ("اختبار", sc.D >= TEST_START)):
                g = sc[sel]
                hot, cold = g[g[k] >= 2 / 3], g[g[k] <= 1 / 3]
                print(f"  إغلاق قوي × {k} ({nm}): حار n={len(hot)} +20%-وإلا-الإغلاق {pr(hot.n_L20.mean())} · افتتاح {pr(hot.n_on.mean())} | بارد n={len(cold)} {pr(cold.n_L20.mean())} · {pr(cold.n_on.mean())}")

    # ---- JSON للرادار
    tiers = {}
    for k_, g in (("ready", G["q99"]), ("near", G["q95"])):
        if len(g):
            tiers[k_] = {"e5": float(g.E5.mean()), "e10": float(g.E10.mean()), "exp_e10": matched(g, base_e, "E10"),
                         "mx10_50": float((g.mx10 >= 0.5).mean()), "dd20_30": float((g.dd20 <= -0.3).mean()), "n": int(len(g))}
    model = {"version": "hr2-" + datetime.now(timezone.utc).strftime("%Y%m%d"), "feats": FEATS, "w": [float(x) for x in w_f],
             "mean": [float(x) for x in mu_f], "std": [float(x) for x in sd_f], "thresholds": thr_f, "tiers": tiers,
             "base_e10": base_all["e10"], "base_e5": base_all["e5"], "n_rows": base_all["n"], "liq": LIQ,
             "period": [int(T.D.min()), int(T.D.max())],
             "reg_q": {k: [float(x) for x in np.nanquantile(R[k].to_numpy(float), np.linspace(0, 1, 101))] for k in HR.REG_COLS},
             "reg_last": {k: (float(R[k].iloc[-1]) if np.isfinite(R[k].iloc[-1]) else None) for k in HR.REG_COLS},
             "reg": out_reg}
    js = json.dumps(model, ensure_ascii=False)
    print("HR2_BEGIN")
    for k in range(0, len(js), 900):
        print("H2|" + js[k:k + 900])
    print("HR2_END")
    print(f"\nخلص في {round(time.time() - t0)} ثانية", flush=True)


if __name__ == "__main__":
    main()
