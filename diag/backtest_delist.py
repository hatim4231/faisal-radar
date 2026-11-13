"""
⛔ توقّع الشطب — طلب المستخدم 2026-11-13: «أهم شي أنه السهم ماينشطب».
نبني نموذج: احتمال السهم ينشطب (يموت) خلال ~90 جلسة. ناسداك العادية **مع المشطوبة** (Alpaca active+inactive).

«شطب سيئ» = السهم صار inactive **و** آخر سعر حقيقي < 1.5$ **و** انهار (آخر سعر ÷ أعلى 120 جلسة < 0.5) — موت فعلي (مو استحواذ).
التسمية: يوم i موجب لو السهم شطب سيئ و(يوم الشطب − i) بين 1 و90 جلسة. أيام آخر 90 جلسة للأسهم inactive «الحسنة» (استحواذ، سعر عالي) تُستبعد.
الميزات: السعر الحقيقي + تحت 1$ (عداد ناسداك) + التقسيم العكسي + الانهيار + السيولة + الزخم + التذبذب (من highrisk.features).
walk-forward سنة بسنة · AUC · معايرة بالعُشر. المخرجات: جدول + JSON بين DL_BEGIN/END (أسطر DL|).
"""
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
import requests

import alpaca_data as A
import highrisk as HR
from backtest_ready import prep_frame, fit_logit, auc, pr, pp
from backtest_highrisk import ETF_RX, NOT_COMMON_RX

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
STEP = 3
HORIZON = 90
FEATS = ["l_price", "u1_streak", "u1_60", "l_dv20", "dd252", "from_peak", "ret20", "ret60", "rsi", "atr20p",
         "n_rs", "l_days_rs", "rs_recent", "pos60", "vol5_60", "rng20", "l_days_ep", "n_ep252", "below_now"]
_sh, _roll = HR._sh, HR._roll


def universe():
    """ناسداك العادية: نشط + موقوف/مشطوب (الحالة لكل سهم)."""
    syms = {}
    for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            for st in ("active", "inactive"):
                r = requests.get(f"{base}/v2/assets", headers=A._headers(), params={"status": st, "asset_class": "us_equity"}, timeout=90)
                r.raise_for_status()
                for a in r.json():
                    s, ex, nm = a.get("symbol", ""), a.get("exchange", ""), a.get("name") or ""
                    import re
                    if ex != "NASDAQ" or not re.fullmatch(r"[A-Z]{1,5}", s):
                        continue
                    if (len(s) == 5 and s[-1] in "WRUZ") or ETF_RX.search(nm) or NOT_COMMON_RX.search(nm):
                        continue
                    syms.setdefault(s, st)
            break
        except Exception as e:
            print("assets:", base, repr(e)[:120], flush=True)
            syms = {}
    print(f"المجال: {len(syms)} سهم ناسداك عادي (نشط {sum(v == 'active' for v in syms.values())} · موقوف/مشطوب {sum(v == 'inactive' for v in syms.values())})", flush=True)
    return syms


def job(args):
    sym, d, inactive = args
    c = d["Close"].to_numpy(float)
    v = d["Volume"].to_numpy(float)
    n = len(c)
    if n < 300:
        return None
    F = HR.features(d)
    praw = F["praw"].to_numpy(float)
    Dint = d.index.strftime("%Y%m%d").astype(int).to_numpy(np.int32)
    under1 = (praw < 1).astype(float)
    # عداد الأيام المتتالية تحت 1$ (قاعدة ناسداك: 30 جلسة = عجز، 180 يوم مهلة)
    streak = np.zeros(n)
    for i in range(1, n):
        streak[i] = streak[i - 1] + 1 if under1[i] else 0
    u1_60 = _roll(under1, 60, "mean", 20)
    # شطب سيئ؟ (موت فعلي، مو استحواذ)
    peak120 = float(np.nanmax(praw[-120:])) if n >= 120 else float(np.nanmax(praw))
    last_px = praw[-1]
    crashed = peak120 > 0 and last_px / peak120 < 0.5
    bad = bool(inactive and last_px < 1.5 and crashed)
    good_inactive = bool(inactive and not bad)
    cols = {f: F[f].to_numpy(float) for f in ("l_price", "l_dv20", "dd252", "from_peak", "ret20", "rsi", "atr20p",
                                              "n_rs", "l_days_rs", "rs_recent", "pos60", "vol5_60", "rng20", "l_days_ep", "n_ep252")}
    cols["ret60"] = np.clip(np.nan_to_num(c / _sh(c, 60) - 1), -1, 5)
    cols["u1_streak"] = np.minimum(streak, 250)
    cols["u1_60"] = u1_60
    cols["below_now"] = under1
    idx = np.arange(n)
    keep = (idx >= 260) & (idx <= n - 2) & (idx % STEP == 0)
    if good_inactive:                      # استبعد آخر 90 جلسة للأسهم الموقوفة «الحسنة» (ملتبسة)
        keep &= idx < n - HORIZON
    y = np.zeros(n)
    if bad:
        de = n - 1
        y[(idx >= de - HORIZON) & (idx < de)] = 1.0
        keep &= (idx < de)                 # ما بعد الشطب ما ينحسب
    rows = pd.DataFrame({"t": sym, "D": Dint, "y": y, "praw": praw, **cols})[keep]
    rows = rows.astype({c_: np.float32 for c_ in rows.columns if rows[c_].dtype == np.float64})
    info = {"bad": bad, "good_inactive": good_inactive, "last_px": float(last_px), "last_D": int(Dint[-1]), "n": int(keep.sum())}
    return rows, info


def main():
    t0 = time.time()
    syms = universe()
    names = sorted(syms)
    parts, bad_syms, good_syms = [], [], []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(names), 100):
            batch = names[i:i + 100]
            raw = None
            for attempt in range(3):
                try:
                    raw = A.bars(batch, "1Day", START)
                    break
                except Exception as e:
                    print("bars err", i, repr(e)[:90], flush=True)
                    time.sleep(12 * (attempt + 1))
            if raw is None:
                continue
            try:
                spl = A.splits(batch, START)
            except Exception:
                spl = {}
            jobs = [(s, prep_frame(s, b, spl.get(s, [])), syms[s] == "inactive") for s, b in raw.items() if len(b) >= 320]
            for (s, _, _), res in zip(jobs, ex.map(job, [(s, d, ia) for s, d, ia in jobs], chunksize=4)):
                if res is None:
                    continue
                if len(res[0]):
                    parts.append(res[0])
                if res[1]["bad"]:
                    bad_syms.append(s)
                elif res[1]["good_inactive"]:
                    good_syms.append(s)
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(names)} · صفوف {sum(len(p) for p in parts):,} · شطب سيئ {len(bad_syms)} · {round(time.time() - t0)}ث", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    for f_ in FEATS:
        x = df[f_].replace([np.inf, -np.inf], np.nan)
        df[f_] = x.fillna(x.median()).astype(np.float32)
    df["Y"] = (df.D // 10000).astype(int)
    print(f"\nالصفوف: {len(df):,} سهم-يوم · أسهم {df.t.nunique():,} · شطب سيئ {len(bad_syms)} سهم · استحواذ/موقوف حسن {len(good_syms)}"
          f" · من {df.D.min()} لين {df.D.max()} · {round(time.time() - t0)}ث")
    print(f"الأساس: ينشطب (سيئ) خلال 90 جلسة {pp(df.y.mean(), 2)} · تحت 1$ الآن {pp((df.below_now > 0).mean())} · وسيط السعر {df.praw.median():.2f}$")

    rng = np.random.default_rng(1)
    X = df[FEATS].to_numpy(np.float32)
    yv = df.y.to_numpy(float)
    df["p"] = np.float32(np.nan)
    print("\n==== النموذج سنة بسنة (كل سنة بنموذج ما شافها) ====")
    for Y in range(2021, 2027):
        trm = (df.D < (Y - 1) * 10000 + 1001).to_numpy()      # التدريب لازم ينضج قبل بداية الاختبار (90 جلسة ~ 4.5 شهر)
        tem = (df.Y == Y).to_numpy()
        if trm.sum() < 50_000 or tem.sum() < 1000:
            continue
        tri = np.flatnonzero(trm)
        if len(tri) > 1_500_000:
            tri = rng.choice(tri, 1_500_000, replace=False)
        Xtr = X[tri].astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        w = fit_logit(np.clip((Xtr - mu) / sd, -5, 5), yv[tri], lam=5.0)
        Zte = np.clip((X[tem].astype(float) - mu) / sd, -5, 5)
        pte = 1 / (1 + np.exp(-(w[0] + Zte @ w[1:])))
        df.loc[tem, "p"] = pte.astype(np.float32)
        print(f"  {Y}: تدريب {trm.sum():,} · اختبار {tem.sum():,} · ينشطب فعلاً {pp(yv[tem].mean(), 2)} · AUC {auc(yv[tem], pte):.3f}", flush=True)

    T = df[df.p.notna()].copy()
    print("\n==== المعايرة خارج العينة (العُشر · المتوقع · الفعلي ينشطب خلال 90) ====")
    T["dec"] = pd.qcut(T.p.rank(method="first"), 10, labels=False)
    cal = []
    for k, g in T.groupby("dec"):
        cal.append({"dec": int(k), "p_lo": float(g.p.min()), "p_hi": float(g.p.max()), "rate": float(g.y.mean()), "n": int(len(g))})
        print(f"  {k}: متوقع {pp(g.p.mean(), 2)} · فعلي {pp(g.y.mean(), 2)} · n={len(g):,} · وسيط السعر {g.praw.median():.2f}$ · تحت 1$ {pp((g.below_now > 0).mean())}")
    top1 = T[T.p >= T.p.quantile(0.99)]
    top5 = T[T.p >= T.p.quantile(0.95)]
    print(f"  أعلى 1% خطر: ينشطب {pp(top1.y.mean())} (n={len(top1):,}) · أعلى 5%: {pp(top5.y.mean())} (n={len(top5):,})")

    # هل القواعد البسيطة (تحت 1$ + عداد ناسداك + تقسيم عكسي) تكفي؟
    print("\n==== قواعد بسيطة شفافة (للتحذير الأحمر) ====")
    for nm, m in (("تحت 1$ الآن", T.below_now > 0), ("تحت 1$ ≥ 30 جلسة متتالية (عجز ناسداك)", T.u1_streak >= 30),
                  ("تقسيم عكسي خلال 180 جلسة", T.rs_recent > 0), ("تحت 1$ + تقسيم عكسي خلال سنة", (T.below_now > 0) & (T.n_rs >= 1)),
                  ("منهار 90%+ من قمة السنة", T.dd252 <= -0.9), ("تحت 1$ + منهار 85%+", (T.below_now > 0) & (T.dd252 <= -0.85)),
                  ("فوق 1$ وما انهار (آمن نسبياً)", (T.below_now == 0) & (T.dd252 > -0.7))):
        g = T[m]
        print(f"  {nm}: n={len(g):,} ({pp(len(g) / len(T))} من الكل) · ينشطب خلال 90: {pp(g.y.mean(), 2)} (الأساس {pp(T.y.mean(), 2)})")

    # النموذج النهائي على كل البيانات
    tri = np.arange(len(df))
    if len(tri) > 2_000_000:
        tri = rng.choice(tri, 2_000_000, replace=False)
    Xa = X[tri].astype(float)
    mu, sd = Xa.mean(0), Xa.std(0)
    sd[sd == 0] = 1
    wf = fit_logit(np.clip((Xa - mu) / sd, -5, 5), yv[tri], lam=5.0)
    pall = 1 / (1 + np.exp(-(wf[0] + np.clip((X[:600_000].astype(float) - mu) / sd, -5, 5) @ wf[1:])))
    thr = {"high": float(np.quantile(pall, 0.95)), "med": float(np.quantile(pall, 0.85))}
    print("\nأوزان: " + " · ".join(f"{f_} {w_:+.2f}" for f_, w_ in sorted(zip(FEATS, wf[1:]), key=lambda z: -abs(z[1]))[:14]))
    model = {"version": "delist1-" + datetime.now(timezone.utc).strftime("%Y%m%d"), "feats": FEATS,
             "mean": [float(x) for x in mu], "std": [float(x) for x in sd], "w": [float(x) for x in wf],
             "cal": cal, "thresholds": thr, "base": float(df.y.mean()), "n": int(len(df)), "n_bad": len(bad_syms),
             "period": [int(df.D.min()), int(df.D.max())], "horizon": HORIZON}
    js = json.dumps(model, ensure_ascii=False)
    print("DL_BEGIN")
    for k in range(0, len(js), 900):
        print("DL|" + js[k:k + 900])
    print("DL_END")
    print(f"\nخلص في {round(time.time() - t0)} ثانية", flush=True)


if __name__ == "__main__":
    main()
