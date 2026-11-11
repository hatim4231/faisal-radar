"""
🧨 «عالي مخاطرة» — الأسهم اللي تنفجر بشكل متكرر (طلب المستخدم 2026-11-11) — تشخيص فقط، ما يغير شي.

الانفجار = أعلى سعر باليوم ≥ ضعف إغلاق اليوم اللي قبله (مثل OMH/QMMM/RUBI). «حلقة» = أول انفجار بعد 20 جلسة بدونه.
المجال: كل أسهم Alpaca (NASDAQ/NYSE/AMEX) مع المشطوبة، يومي ~3 سنين، **بدون فلاتر** (بس سهم له تداول: سيولة 20 يوم ≥ 5 آلاف$).
س1  التكرار: احتمال انفجار خلال 10/20 جلسة حسب عدد الانفجارات بآخر سنة — ومقابل نفس شريحة التذبذب (عشان ما ننخدع).
س2  بصمة المضارب قبل الانطلاقة: الميزات قبل الانفجار بـ 1/5/10 جلسات مقابل الأيام الهادية لنفس السهم.
س3  نموذج «احتمال انفجار خلال 10 جلسات» walk-forward (مرحلتين خارج العينة): AUC + الرفع فوق الأساس.
س4  خطة بسيطة (دخول إغلاق، بيع Limit عند +50% أو +100%، وقف إغلاق −30%، أقصى 20 جلسة) للإشارات العالية
    مقابل دخول عشوائي بنفس شريحة التذبذب والسعر.
المخرجات: جداول + نموذج JSON بين HR_MODEL_BEGIN/END (أسطر HM| مقطعة) + diag_out/highrisk_*.csv.gz
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
import re
import requests
from backtest_runners2 import rsi_np, ETF_RX

# مو أسهم عادية (Warrants بأي حرف مثل OPENZ/AMPGZ، حقوق، وحدات، ممتازة، سندات) — أول تشغيل دخلت وقلبت أعلى القائمة
NOT_COMMON_RX = re.compile(r"\b(Warrants?|Wts?|Rights?|Units?|Preferred|Pref|Depositary Shares? Representing|Notes?|Debentures?|"
                           r"Subordinated|Perpetual|Contingent Value)\b", re.I)


def universe():
    """كل أسهم Alpaca (نشط + موقوف/مشطوب) — أسهم عادية بس (بدون ETF/Warrants/حقوق/وحدات/ممتازة/سندات)."""
    import alpaca_data as A
    syms, drop = {}, 0
    for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            for st in ("active", "inactive"):
                r = requests.get(f"{base}/v2/assets", headers=A._headers(),
                                 params={"status": st, "asset_class": "us_equity"}, timeout=90)
                r.raise_for_status()
                for a in r.json():
                    s, ex, nm = a.get("symbol", ""), a.get("exchange", ""), a.get("name") or ""
                    if ex not in ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS") or not re.fullmatch(r"[A-Z]{1,5}", s):
                        continue
                    if (len(s) == 5 and s[-1] in "WRUZ") or ETF_RX.search(nm) or NOT_COMMON_RX.search(nm):
                        drop += 1
                        continue
                    syms[s] = st
            break
        except Exception as e:
            print("assets:", base, repr(e)[:120], flush=True)
            syms = {}
    act = sum(v == "active" for v in syms.values())
    print(f"المجال: {len(syms)} سهم عادي (نشط {act} · موقوف/مشطوب {len(syms) - act}) · مستبعد مو سهم عادي {drop}", flush=True)
    return sorted(syms)
from backtest_confidence import fit_logit, prep, auc

DAYS = 1180
H = 20  # أقصى مدة
FEATS = ["l_price", "l_dv20", "n_ep252", "l_days_ep", "recent_ep", "last_mag", "base_ratio", "from_peak",
         "vol5_60", "vol1_20", "atr5_60", "atr20p", "rng20", "pos60", "dd252", "ret5", "ret20", "rsi", "closepos10",
         "upv10", "obv20", "hl_up", "gap20", "rs_recent", "n_rs", "l_days_rs"]


def roll(a, w, fn, minp=None):
    s = pd.Series(a)
    r = s.rolling(w, min_periods=minp or w)
    return getattr(r, fn)().to_numpy(float)


def feat(job):
    sym, d, spl = job
    if d is None or len(d) < 300:
        return None, []
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    idx = np.arange(n)
    pc = np.r_[np.nan, c[:-1]]
    jump = h / pc
    ex = np.nan_to_num(jump, nan=0.0) >= 2.0
    ep = np.zeros(n, bool)
    last = -999
    for t in np.flatnonzero(ex):
        if t - last > 20:
            ep[t] = True
        last = t
    cs = np.cumsum(ep)
    lagd = lambda a, k: (a - np.r_[np.zeros(k), a[:-k]]) if len(a) > k else a.astype(float)
    n_ep252 = lagd(cs, 252)
    lastep = np.maximum.accumulate(np.where(ep, idx, -1))
    has = lastep >= 0
    days_ep = np.where(has, idx - lastep, 9999)
    li = np.maximum(lastep, 0)
    last_mag = np.where(has, jump[li], 0.0)
    base_before = np.where(has & (li >= 1), c[np.maximum(li - 1, 0)], np.nan)
    peak = np.where(has, h[li], np.nan)
    # التقسيمات العكسية
    rs = np.zeros(n, bool)
    for dt, ratio in spl or []:
        if ratio < 1:
            p = d.index.searchsorted(pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt))
            if 0 <= p < n:
                rs[p] = True
    rcs = np.cumsum(rs)
    n_rs = lagd(rcs, 252)
    lastrs = np.maximum.accumulate(np.where(rs, idx, -1))
    days_rs = np.where(lastrs >= 0, idx - lastrs, 9999)
    # ميزات السعر والفوليوم
    tr = np.where(np.isfinite(pc), np.maximum(h, pc) - np.minimum(l, pc), h - l)
    dv20 = roll(c * v, 20, "median")
    vol5_60 = roll(v, 5, "mean") / np.maximum(roll(v, 60, "mean"), 1)
    vol1_20 = v / np.maximum(roll(v, 20, "median"), 1)
    atr20 = roll(tr, 20, "mean")
    atr5_60 = roll(tr, 5, "mean") / np.maximum(roll(tr, 60, "mean"), 1e-9)
    hi20, lo20 = roll(h, 20, "max"), roll(l, 20, "min")
    hi60, lo60 = roll(h, 60, "max"), roll(l, 60, "min")
    hi252 = roll(h, 252, "max", 120)
    c5 = np.r_[np.full(5, np.nan), c[:-5]]
    c20 = np.r_[np.full(20, np.nan), c[:-20]]
    rsi = rsi_np(c)
    rngd = np.where(h > l, (c - l) / np.maximum(h - l, 1e-12), 0.5)
    up, dn = c > pc, c < pc
    upv10 = roll(np.where(up, v, 0.0), 10, "sum") / (roll(np.where(dn, v, 0.0), 10, "sum") + 1)
    obv = np.cumsum(np.where(up, v, np.where(dn, -v, 0.0)))
    obv20 = (obv - np.r_[np.full(20, np.nan), obv[:-20]]) / np.maximum(roll(v, 60, "mean") * 20, 1)
    m5 = roll(l, 5, "min")
    sh = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]
    hl_up = (m5 > sh(m5, 5)).astype(float) + (sh(m5, 5) > sh(m5, 10)).astype(float) + (sh(m5, 10) > sh(m5, 15)).astype(float)
    gap20 = roll((o >= 1.1 * pc).astype(float), 20, "sum")
    # النتايج (من إغلاق اليوم)
    fsh = lambda a, k: np.r_[a[k:], np.full(k, np.nan)]
    exf = ex.astype(float)
    E5 = fsh(roll(exf, 5, "max"), 5)
    E10 = fsh(roll(exf, 10, "max"), 10)
    E20 = fsh(roll(exf, 20, "max"), 20)
    mx10 = fsh(roll(h, 10, "max"), 10) / c - 1
    mx20 = fsh(roll(h, 20, "max"), 20) / c - 1
    dd20 = fsh(roll(l, 20, "min"), 20) / c - 1
    r20 = fsh(c, 20) / c - 1
    # خطة: Limit بيع عند +50% / +100%، وقف إغلاق −30%، أقصى 20 جلسة (الهدف يُفحص أول)
    res = {}
    for T in (0.5, 1.0):
        out = np.full(n, np.nan)
        done = np.zeros(n, bool)
        for k in range(1, H + 1):
            hk, ck = fsh(h, k), fsh(c, k)
            hit = ~done & (hk >= c * (1 + T))
            out[hit] = T
            done |= hit
            stop = ~done & (ck <= c * 0.7)
            out[stop] = ck[stop] / c[stop] - 1
            done |= stop
            if k == H:
                rest = ~done & np.isfinite(ck)
                out[rest] = ck[rest] / c[rest] - 1
        res[T] = out
    # طرق ثانية: (أ) دخول الإغلاق بمدد/وقف أضيق  (ب) أمر شراء فوق السعر (يدخل بس إذا السهم بدأ يتحرك فعلاً) — الهدف من سعر الدخول
    def close_entry(T, S, Hh):
        out = np.full(n, np.nan); done = np.zeros(n, bool)
        for k in range(1, Hh + 1):
            hk, ck = fsh(h, k), fsh(c, k)
            hit = ~done & (hk >= c * (1 + T)); out[hit] = T; done |= hit
            if S is not None:
                st = ~done & (ck <= c * (1 - S)); out[st] = ck[st] / c[st] - 1; done |= st
            if k == Hh:
                rest = ~done & np.isfinite(ck); out[rest] = ck[rest] / c[rest] - 1
        return out

    def stop_entry(trig, T, S, Hh):
        """شراء بأمر Stop عند إغلاق اليوم × (1+trig) باليوم التالي (لو فتح فوقه = سعر الافتتاح)، ثم بيع Limit عند +T من الدخول،
        ووقف إغلاق −S من الدخول، أقصى Hh جلسة. يوم الدخول نفسه: إذا الأعلى وصل الهدف نحسبه هدف (الأعلى جا بعد الاختراق غالباً)."""
        o1, h1, c1 = fsh(o, 1), fsh(h, 1), fsh(c, 1)
        lvl = c * (1 + trig)
        trg = np.isfinite(h1) & (h1 >= lvl)
        ent = np.where(np.isfinite(o1) & (o1 > lvl), o1, lvl)
        out = np.full(n, np.nan); done = ~trg
        hit = ~done & (h1 >= ent * (1 + T)); out[hit] = T; done |= hit
        st = ~done & (c1 <= ent * (1 - S)); out[st] = c1[st] / ent[st] - 1; done |= st
        for k in range(2, Hh + 1):
            hk, ck = fsh(h, k), fsh(c, k)
            hit = ~done & (hk >= ent * (1 + T)); out[hit] = T; done |= hit
            st = ~done & (ck <= ent * (1 - S)); out[st] = ck[st] / ent[st] - 1; done |= st
            if k == Hh:
                rest = ~done & np.isfinite(ck); out[rest] = ck[rest] / ent[rest] - 1
        return out, trg.astype(float)

    v_c10 = close_entry(0.5, 0.20, 10)
    v_c5 = close_entry(0.3, 0.15, 5)
    v_lot = close_entry(1.0, None, 20)
    v_s15, trg15 = stop_entry(0.15, 0.5, 0.20, 5)
    v_s25, trg25 = stop_entry(0.25, 0.5, 0.20, 5)
    v_s15b, _ = stop_entry(0.15, 0.3, 0.15, 3)
    F = pd.DataFrame({
        "t": sym, "D": d.index.strftime("%Y%m%d").astype(int).to_numpy(np.int32), "close": c, "dv20": dv20,
        "l_price": np.log(np.maximum(c, 0.01)), "l_dv20": np.log10(np.maximum(dv20, 1)),
        "n_ep252": n_ep252, "n_ep_all": cs, "l_days_ep": np.log1p(np.minimum(days_ep, 1000)),
        "recent_ep": (days_ep <= 60).astype(float), "last_mag": np.minimum(last_mag, 20),
        "base_ratio": np.where(np.isfinite(base_before), np.minimum(c / np.maximum(base_before, 1e-9), 10), 1.0),
        "from_peak": np.where(np.isfinite(peak), c / peak - 1, 0.0),
        "vol5_60": np.minimum(vol5_60, 20), "vol1_20": np.minimum(vol1_20, 50), "atr5_60": np.minimum(atr5_60, 10),
        "atr20p": atr20 / c, "rng20": np.minimum(hi20 / np.maximum(lo20, 1e-9) - 1, 10),
        "pos60": np.clip((c - lo60) / np.maximum(hi60 - lo60, 1e-12), 0, 1), "dd252": c / hi252 - 1,
        "ret5": np.clip(c / c5 - 1, -1, 5), "ret20": np.clip(c / c20 - 1, -1, 10), "rsi": rsi,
        "closepos10": roll(rngd, 10, "mean"), "upv10": np.minimum(upv10, 20), "obv20": np.clip(obv20, -5, 5),
        "hl_up": hl_up, "gap20": gap20, "rs_recent": (days_rs <= 60).astype(float), "n_rs": n_rs,
        "l_days_rs": np.log1p(np.minimum(days_rs, 1000)),
        "E5": E5, "E10": E10, "E20": E20, "mx10": mx10, "mx20": mx20, "dd20": dd20, "r20": r20,
        "tr50": res[0.5], "tr100": res[1.0], "ex_today": exf,
        "v_c10": v_c10, "v_c5": v_c5, "v_lot": v_lot, "v_s15": v_s15, "v_s25": v_s25, "v_s15b": v_s15b,
        "trg15": trg15, "trg25": trg25,
    })
    keep = (idx >= 260) & (idx <= n - H - 1) & (idx % 2 == 0) & (c >= 0.05) & (np.nan_to_num(dv20) >= 5_000) & ~ex
    rows = F[keep].copy()
    for col in rows.columns:
        if rows[col].dtype == np.float64:
            rows[col] = rows[col].astype(np.float32)
    # الحلقات: الميزات قبل الانفجار بـ 1/5/10 جلسات مقابل الأيام الهادية لنفس السهم
    quiet = np.ones(n, bool)
    for t in np.flatnonzero(ex):
        quiet[max(0, t - 25):t + 26] = False
    quiet &= (idx >= 260)
    eps = []
    cols = ["vol5_60", "vol1_20", "atr5_60", "rng20", "pos60", "dd252", "ret5", "ret20", "rsi", "closepos10", "upv10",
            "obv20", "hl_up", "gap20", "base_ratio", "from_peak", "l_days_rs", "n_ep252"]
    qmed = F.loc[quiet, cols].median() if quiet.sum() >= 30 else None
    for E in np.flatnonzero(ep):
        if E < 260 or qmed is None:
            continue
        rec = {"t": sym, "E": str(d.index[E].date()), "pre": c[E - 1], "jump": jump[E], "dv_pre": dv20[E - 1],
               "n_prior": int(n_ep252[E - 1]), "days_rs": int(days_rs[E - 1]),
               "c_E": c[E] / c[E - 1] - 1, "c5": (c[E + 5] / c[E - 1] - 1) if E + 5 < n else np.nan,
               "c20": (c[E + 20] / c[E - 1] - 1) if E + 20 < n else np.nan,
               "pk5": (h[E:E + 6].max() / c[E - 1] - 1)}
        for lag in (1, 5, 10):
            if E - lag >= 0:
                for col in cols:
                    rec[f"{col}@{lag}"] = float(F[col].iat[E - lag])
        for col in cols:
            rec[f"{col}@q"] = float(qmed[col])
        eps.append(rec)
    return rows, eps


def matched_expect(df_sig, df_all, col, strata=("vq", "pq")):
    """المتوقع لو دخول عشوائي بنفس الشرائح (التذبذب × السعر)."""
    base = df_all.groupby(list(strata))[col].mean()
    w = df_sig.groupby(list(strata)).size()
    w = w[w.index.isin(base.index)]
    return float((base.loc[w.index] * w).sum() / max(w.sum(), 1))


def pr(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.1f}%"


def main():
    t0 = time.time()
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    parts, eps = [], []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            batch = syms[i:i + 100]
            try:
                raw = A.bars(batch, "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:100], flush=True)
                continue
            try:
                spl = A.splits(batch, start)
            except Exception:
                spl = {}
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")], spl.get(s, [])))
            for rows, e in ex.map(feat, jobs, chunksize=4):
                if rows is not None and len(rows):
                    parts.append(rows)
                eps += e
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · صفوف {sum(len(p) for p in parts):,} · حلقات {len(eps)} · {round(time.time() - t0)}ث", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    for f_ in FEATS:  # تنظيف: لا نهايات ولا فراغات
        x = df[f_].replace([np.inf, -np.inf], np.nan)
        df[f_] = x.fillna(x.median()).astype(np.float32)
    ev = pd.DataFrame(eps)
    os.makedirs("diag_out", exist_ok=True)
    ev.to_csv("diag_out/highrisk_episodes.csv.gz", index=False, compression="gzip")
    print(f"\nالصفوف: {len(df):,} سهم-يوم · أسهم {df.t.nunique():,} · من {df.D.min()} لين {df.D.max()} · حلقات انفجار {len(ev):,}", flush=True)
    # شرائح للمقارنة العادلة
    df["vq"] = pd.qcut(df["atr20p"].rank(method="first"), 5, labels=False)
    df["pq"] = pd.cut(df["close"], [0, 1, 3, 10, 1e9], labels=False)
    df["epb"] = pd.cut(df["n_ep252"], [-1, 0, 1, 2, 3, 5, 1e9], labels=["0", "1", "2", "3", "4-5", "6+"])

    print("\n==== س1: التكرار — احتمال انفجار (×2 بيوم) خلال 10 / 20 جلسة حسب انفجارات آخر سنة ====")
    print(f"الأساس: 10 جلسات {pr(df.E10.mean())} · 20 جلسة {pr(df.E20.mean())}")
    for b, g in df.groupby("epb", observed=True):
        print(f"  انفجارات آخر سنة {b:>4}: n={len(g):>9,} · خلال 10 {pr(g.E10.mean())} · خلال 20 {pr(g.E20.mean())} "
              f"· أعلى ربح 20 (وسيط) {pr(g.mx20.median())} · متوقع بنفس التذبذب والسعر {pr(matched_expect(g, df, 'E10'))}")
    print("\nنفس الشي داخل كل شريحة تذبذب (خلال 10 جلسات) — الصفوف: شريحة التذبذب 0 (هادي) → 4 (متذبذب):")
    tab = df.pivot_table(index="vq", columns="epb", values="E10", aggfunc="mean", observed=True)
    print((tab * 100).round(1).to_string())
    ser = df[df.n_ep252 >= 1]
    ser = ser.assign(dsb=pd.cut(np.expm1(ser.l_days_ep), [0, 10, 20, 40, 60, 120, 250, 1e9]))
    print("\nداخل الأسهم اللي انفجرت بآخر سنة: حسب الجلسات من آخر انفجار:")
    for b, g in ser.groupby("dsb", observed=True):
        print(f"  {str(b):>14}: n={len(g):>8,} · خلال 10 {pr(g.E10.mean())} · متوقع بنفس التذبذب {pr(matched_expect(g, df, 'E10'))}")

    print("\n==== س2: بصمة المضارب — الميزات قبل الانفجار مقابل الأيام الهادية لنفس السهم (وسيط) ====")
    cols = ["vol5_60", "vol1_20", "atr5_60", "rng20", "pos60", "dd252", "ret5", "ret20", "rsi", "closepos10", "upv10",
            "obv20", "hl_up", "gap20", "base_ratio", "from_peak"]
    for grp, g in (("كل الحلقات", ev), ("المتكرر (انفجر قبل خلال سنة)", ev[ev.n_prior >= 1]), ("أول مرة", ev[ev.n_prior == 0])):
        print(f"\n-- {grp} (n={len(g)}) · وسيط: قبل 10 جلسات | قبل 5 | قبل يوم | أيامه الهادية")
        for col in cols:
            vals = [g[f"{col}@{lag}"].median() for lag in (10, 5, 1)] + [g[f"{col}@q"].median()]
            print(f"   {col:>11}: " + " | ".join(f"{x:.3f}" for x in vals))
    print("\nبعد الانفجار (من إغلاق ما قبله): نفس اليوم وسيط " + pr(ev.c_E.median()) + " · بعد 5 جلسات " + pr(ev.c5.median())
          + " · بعد 20 " + pr(ev.c20.median()) + " · أعلى 5 جلسات وسيط " + pr(ev.pk5.median()))
    print("التقسيم العكسي قبل الانفجار: خلال 30 جلسة " + pr((ev.days_rs <= 30).mean()) + " · خلال 60 " + pr((ev.days_rs <= 60).mean())
          + f" (الأيام العادية: خلال 60 {pr((np.expm1(df.l_days_rs) <= 60).mean())})")
    print("سعر ما قبل الانفجار: " + " · ".join(f"{a}-{b}$ {pr(((ev.pre >= a) & (ev.pre < b)).mean())}" for a, b in ((0, 1), (1, 3), (3, 10), (10, 1e9))))

    print("\n==== س3: النموذج (احتمال انفجار خلال 10 جلسات) — walk-forward ====")
    folds = [(20250501, 20251101), (20251101, 20261101)]
    test_all = []
    for a, b in folds:
        cut = int((pd.Timestamp(str(a)) - pd.Timedelta(days=30)).strftime("%Y%m%d"))
        tr = df[df.D < cut]
        te = df[(df.D >= a) & (df.D < b)].copy()
        Z, mu, sd = prep(tr, FEATS)
        w = fit_logit(Z, tr.E10.astype(float).values, lam=5.0)
        Zt, _, _ = prep(te, FEATS, mu, sd)
        te["p"] = 1 / (1 + np.exp(-(w[0] + Zt @ w[1:])))
        test_all.append(te)
        s_ser = te[te.n_ep252 >= 1]
        print(f"  اختبار {a}→{b}: n={len(te):,} · AUC {auc(te.E10.values, te.p.values):.3f} · داخل المتكرر "
              f"{auc(s_ser.E10.values, s_ser.p.values):.3f} · الأساس {pr(te.E10.mean())}")
        for q in (0.01, 0.05, 0.10):
            top = te[te.p >= te.p.quantile(1 - q)]
            print(f"     أعلى {int(q * 100)}%: انفجار خلال 10 {pr(top.E10.mean())} (متوقع بنفس التذبذب والسعر {pr(matched_expect(top, te, 'E10'))})"
                  f" · خلال 20 {pr(top.E20.mean())} · خطة +50% متوسط {pr(top.tr50.mean())} وسيط {pr(top.tr50.median())}"
                  f" · خطة +100% متوسط {pr(top.tr100.mean())}")
        daily = te.sort_values("p", ascending=False).groupby("D").head(10)
        print(f"     أعلى 10 كل يوم: انفجار خلال 10 {pr(daily.E10.mean())} · متوقع بنفس الشرائح {pr(matched_expect(daily, te, 'E10'))}"
              f" · خطة +50% متوسط {pr(daily.tr50.mean())} (عشوائي بنفس الشرائح {pr(matched_expect(daily, te, 'tr50'))})"
              f" · +100% متوسط {pr(daily.tr100.mean())} (عشوائي {pr(matched_expect(daily, te, 'tr100'))})")
        heur = te[(te.n_ep252 >= 2)]
        print(f"     مقارنة: «انفجر مرتين+ بسنة» لحاله: n={len(heur):,} · خلال 10 {pr(heur.E10.mean())} · خطة +50% {pr(heur.tr50.mean())}")
    print("\n==== قواعد بسيطة (كل الفترة) — مقابل عشوائي بنفس التذبذب والسعر ====")
    dse = np.expm1(df.l_days_ep)
    rules = {
        "انفجر مرتين+ بسنة": df.n_ep252 >= 2,
        "انفجر 3+ بسنة": df.n_ep252 >= 3,
        "مرتين+ وآخر انفجار قبل 20-120 جلسة": (df.n_ep252 >= 2) & dse.between(20, 120),
        "مرتين+ ورجع لقاعدته (≤1.3× ما قبل آخر انفجار)": (df.n_ep252 >= 2) & (df.base_ratio <= 1.3),
        "مرتين+ وفوليوم ناشف (5/60 < 0.6)": (df.n_ep252 >= 2) & (df.vol5_60 < 0.6),
        "مرتين+ وفوليوم اليوم ≥ 3× (بدون انفجار)": (df.n_ep252 >= 2) & (df.vol1_20 >= 3),
        "انفجر مرة+ وتقسيم عكسي خلال 60 جلسة": (df.n_ep252 >= 1) & (df.rs_recent > 0),
        "تقسيم عكسي خلال 60 جلسة (أي سهم)": df.rs_recent > 0,
        "مرتين+ وقيعان متصاعدة (3) وتذبذب منكمش": (df.n_ep252 >= 2) & (df.hl_up >= 3) & (df.atr5_60 < 0.8),
        "سعر تحت 1$ ومرتين+": (df.n_ep252 >= 2) & (df.close < 1),
    }
    for name, m in rules.items():
        g = df[m]
        if len(g) < 200:
            print(f"  {name}: عدد قليل ({len(g)})")
            continue
        print(f"  {name}: n={len(g):,} · خلال 10 {pr(g.E10.mean())} (عشوائي {pr(matched_expect(g, df, 'E10'))})"
              f" · خطة +50% متوسط {pr(g.tr50.mean())} (عشوائي {pr(matched_expect(g, df, 'tr50'))})"
              f" · +100% {pr(g.tr100.mean())} (عشوائي {pr(matched_expect(g, df, 'tr100'))})")
    T = pd.concat(test_all)
    VAR = [("tr50", "دخول إغلاق · +50% · وقف −30% · 20 جلسة"), ("v_c10", "دخول إغلاق · +50% · وقف −20% · 10 جلسات"),
           ("v_c5", "دخول إغلاق · +30% · وقف −15% · 5 جلسات"), ("v_lot", "دخول إغلاق · +100% · بدون وقف · 20 جلسة (يانصيب)"),
           ("v_s15", "شراء Stop +15% (لو بدأ يتحرك) · +50% · وقف −20% · 5 جلسات"),
           ("v_s25", "شراء Stop +25% · +50% · وقف −20% · 5 جلسات"),
           ("v_s15b", "شراء Stop +15% · +30% · وقف −15% · 3 جلسات")]
    print("\n==== س4: طرق الدخول والخروج (خارج العينة) — متوسط الصفقة | وسيط | مقابل عشوائي بنفس التذبذب والسعر ====")
    for nm_, g in (("أعلى 1%", T[T.p >= T.p.quantile(0.99)]), ("أعلى 5%", T[T.p >= T.p.quantile(0.95)]),
                   ("أعلى 10 كل يوم", T.sort_values("p", ascending=False).groupby("D").head(10)),
                   ("مرتين+ وفوليوم اليوم ≥ 3×", T[(T.n_ep252 >= 2) & (T.vol1_20 >= 3)])):
        print(f"\n-- {nm_} (n={len(g):,})")
        for col, lab in VAR:
            x = g[col].dropna()
            trg = ""
            if col in ("v_s15", "v_s15b"):
                trg = f" · تفعّل الشراء {pr(g.trg15.mean())}"
            elif col == "v_s25":
                trg = f" · تفعّل الشراء {pr(g.trg25.mean())}"
            base = T.dropna(subset=[col])
            print(f"   {lab}: متوسط {pr(x.mean())} · وسيط {pr(x.median())} · رابحة {pr((x > 0).mean())} (n={len(x):,})"
                  f" · العشوائي {pr(matched_expect(g.dropna(subset=[col]), base, col))}{trg}")
    T["dec"] = pd.qcut(T.p.rank(method="first"), 10, labels=False)
    print("\nمعايرة (خارج العينة، المرحلتين): العُشر · المتوقع · الفعلي خلال 10 · خلال 20 · خطة +50% متوسط · وسيط أعلى ربح 20")
    calib = []
    for k, g in T.groupby("dec"):
        calib.append({"dec": int(k), "p_lo": float(g.p.min()), "p_hi": float(g.p.max()), "e10": float(g.E10.mean()),
                      "e20": float(g.E20.mean()), "tr50": float(g.tr50.mean()), "mx20": float(g.mx20.median()), "n": int(len(g))})
        print(f"  {k}: {g.p.mean() * 100:.2f}% · {pr(g.E10.mean())} · {pr(g.E20.mean())} · {pr(g.tr50.mean())} · {pr(g.mx20.median())}")
    top1 = T[T.p >= T.p.quantile(0.99)]
    print(f"\nأعلى 1% (خارج العينة): +50% خلال 20 {pr((top1.mx20 >= 0.5).mean())} · +100% {pr((top1.mx20 >= 1).mean())} "
          f"· نزل −30% خلال 20 {pr((top1.dd20 <= -0.3).mean())} · وسيط العائد بعد 20 {pr(top1.r20.median())}")

    print("\n==== أوزان النموذج النهائي (كل البيانات) ====")
    Z, mu, sd = prep(df, FEATS)
    w = fit_logit(Z, df.E10.astype(float).values, lam=5.0)
    for f_, c_ in sorted(zip(FEATS, w[1:]), key=lambda x: -abs(x[1])):
        print(f"  {f_:>11}: {c_:+.3f}")
    # حدود الحالات من توزيع الاحتمال خارج العينة: 🧨 أعلى 1% · 🟠 أعلى 5% — ونتايجها الفعلية (للعرض بالصفحة)
    q99, q95 = float(T.p.quantile(0.99)), float(T.p.quantile(0.95))
    tiers = {}
    for k_, g in (("ready", T[T.p >= q99]), ("near", T[(T.p >= q95) & (T.p < q99)])):
        tiers[k_] = {"e10": float(g.E10.mean()), "e20": float(g.E20.mean()), "exp_e10": matched_expect(g, T, "E10"),
                     "mx20_50": float((g.mx20 >= 0.5).mean()), "mx20_100": float((g.mx20 >= 1).mean()),
                     "dd20_30": float((g.dd20 <= -0.3).mean()), "r20_med": float(g.r20.median()), "n": int(len(g)),
                     **{c_: float(g[c_].mean()) for c_ in ("tr50", "v_c10", "v_c5", "v_lot", "v_s15", "v_s25", "v_s15b")},
                     **{f"{c_}_rand": matched_expect(g.dropna(subset=[c_]), T.dropna(subset=[c_]), c_)
                        for c_ in ("tr50", "v_c10", "v_c5", "v_lot", "v_s15", "v_s25", "v_s15b")}}
    print("\nحدود الحالات:", {"ready_p": round(q99, 4), "near_p": round(q95, 4)}, json.dumps(tiers, ensure_ascii=False)[:1500])
    model = {"version": "hr1-" + datetime.now(timezone.utc).strftime("%Y%m%d"), "feats": FEATS, "w": [float(x) for x in w],
             "thresholds": {"ready": q99, "near": q95}, "tiers": tiers,
             "mean": [float(x) for x in mu], "std": [float(x) for x in sd], "calib": calib,
             "base_e10": float(df.E10.mean()), "base_e20": float(df.E20.mean()), "n_rows": int(len(df)), "n_eps": int(len(ev)),
             "period": [int(df.D.min()), int(df.D.max())]}
    js = json.dumps(model, ensure_ascii=False)
    print("HR_MODEL_BEGIN")
    for k in range(0, len(js), 900):
        print("HM|" + js[k:k + 900])
    print("HR_MODEL_END")
    print(f"\nخلص في {round(time.time() - t0)} ثانية", flush=True)


if __name__ == "__main__":
    main()
