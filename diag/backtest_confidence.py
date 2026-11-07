"""
🎯 أداة الثقة لفيصل المطوّر — بناء واختبار نموذج «احتمال نجاح التوصية» (تشخيص فقط).

- البيانات: Alpaca يومي ~3 سنين لكل أسهمنا (0.5-5$) + IWM + ملفات SEC (تخفيف/أرباح) بتواريخها.
- كل سهم كل جلستين: نشغل فيصل المطوّر (+ طبقة القاع الجديد والامتثال) ونحسب العوامل (confidence.features) والنتيجة:
  نجاح = لمس الهدف الأول قبل الإغلاق تحت الوقف خلال 20 جلسة (نفس السجل). + العائد عند الخروج.
- الاختبار: 3 مراحل «تدريب على الماضي ← اختبار على اللي بعده» (Walk-forward). كل الأرقام النهائية من فترات الاختبار بس.
- يطبع النموذج النهائي JSON بين علامتين عشان نستخدمه في الرادار.
تنبيه: الأسهم الحالية فقط (المشطوبة ما تظهر) ← النتايج متفائلة شوي، والمقارنات (نفس العينة) هي الأهم.
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
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import sec_tools
import faisal_rules as fr
import new_low_rule
import compliance_rule
import confidence as C

STEP, HORIZON, DAYS = 2, 20, 1100
FEATS = None  # تتحدد من أول صف


# ----------------------------------------------------------------------------
# جمع البيانات
# ----------------------------------------------------------------------------
def sec_dates(t):
    cik = sec_tools._cik(t)
    if not cik:
        return [], []
    js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
    dil, earn = [], []
    for form, date, items in zip(js["form"], js["filingDate"], js.get("items", [""] * len(js["form"]))):
        fu = form.upper()
        if fu.startswith(C.DILUTION_FORMS) and not fu.startswith("S-1/A") and "424B3" not in fu:
            dil.append(date)
        if fu == "8-K" and "3.02" in (items or ""):
            dil.append(date)
        if fu == "8-K" and "2.02" in (items or ""):
            earn.append(date)
    return sorted(dil), sorted(earn)


def outcome(fut, entry, stop, t1):
    """نفس قواعد السجل لفيصل المطوّر: الهدف يُفحص أول، والوقف بإغلاق اليوم."""
    for i, (_, r) in enumerate(fut.iterrows()):
        if t1 and r["High"] >= t1:
            return 1, "hit", t1 / entry - 1, i + 1
        if stop and r["Close"] < stop:
            return 0, "stop", r["Close"] / entry - 1, i + 1
    return 0, "open", float(fut["Close"].iloc[-1]) / entry - 1, len(fut)


def run_one(args):
    t, d, info, cfg, dil, earn, mkt = args
    out = []
    n = len(d)
    U = cfg["universe"]
    for end in range(220, n - HORIZON, STEP):
        hist = d.iloc[:end]
        px = float(hist["Close"].iloc[-1])
        if not (U["min_price"] <= px <= U["max_price"]) or float(hist["Volume"].tail(30).mean()) < U["min_avg_volume"] * 0.7:
            continue
        try:
            a = fr.analyze(hist, info, cfg)
            if not a:
                continue
            new_low_rule.apply(a, hist)
            compliance_rule.apply_faisal(a, hist)
        except Exception:
            continue
        if a["gate_fail"]:
            continue
        day = hist.index[-1]
        try:
            f = C.features(a, hist, None, dil, earn, mkt.get(day))
        except Exception:
            continue
        p = a["plan"]
        fut = d.iloc[end:end + HORIZON]
        hit, res, xret, sess = outcome(fut, px, p.get("stop"), p.get("t1"))
        u10 = 0
        for _, r in fut.iloc[:10].iterrows():
            if r["Low"] < px * 0.9:
                break
            if r["High"] >= px * 1.2:
                u10 = 1
                break
        out.append({"t": t, "date": str(day.date()), "status": a["irtikaz"]["status"], "hit": hit, "res": res,
                    "xret": xret, "sess": sess, "u10": u10,
                    "ret20": float(fut["Close"].iloc[-1]) / px - 1, **f})
    return out


def collect():
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    print("tickers", len(tickers), flush=True)
    daily = {}
    for i in range(0, len(tickers), 100):
        try:
            daily.update(A.daily(tickers[i:i + 100], days=DAYS))
        except Exception as e:
            print("daily err", repr(e)[:120], flush=True)
    daily = {t: d for t, d in daily.items() if len(d) > 260}
    iwm = A.daily(["IWM"], days=DAYS).get("IWM")
    print("أسهم فيها بيانات كافية:", len(daily), "· IWM:", None if iwm is None else len(iwm), flush=True)
    # وضع السوق لكل يوم
    closes = pd.DataFrame({t: d["Close"] for t, d in daily.items()})
    above = (closes > closes.rolling(50).mean()).astype(float).where(closes.rolling(50).count() >= 50)
    breadth = (above.sum(axis=1) / above.count(axis=1).replace(0, np.nan)).astype(float)
    iw = iwm["Close"].astype(float)
    iwm_up = (iw > iw.rolling(50).mean()).astype(float)
    iwm_r20 = (iw / iw.shift(20) - 1) * 100
    mkt = {}
    for day in closes.index:
        br = breadth.get(day, np.nan)
        up = iwm_up.get(day, np.nan)
        r20 = iwm_r20.get(day, np.nan)
        mkt[day] = {"breadth": 0.4 if pd.isna(br) else float(br), "iwm_up": 0.5 if pd.isna(up) else float(up),
                    "iwm_ret20": 0.0 if pd.isna(r20) else float(r20)}
    infos = scanner.get_info(list(daily), cfg)
    t0 = time.time()
    sec = {}
    for i, t in enumerate(daily):
        try:
            sec[t] = sec_dates(t)
        except Exception:
            sec[t] = ([], [])
        if i % 200 == 0:
            print("SEC", i, round(time.time() - t0), flush=True)
    jobs = [(t, d, infos.get(t) or {}, cfg, sec[t][0], sec[t][1], mkt) for t, d in daily.items()]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=3)):
            rows += r
            if i % 100 == 0:
                print("analyze", i, len(rows), round(time.time() - t0), flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# النموذج
# ----------------------------------------------------------------------------
def fit_logit(X, y, lam=2.0, iters=40):
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(k + 1)
    R = lam * np.eye(k + 1)
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        W = p * (1 - p)
        g = Xb.T @ (y - p) - R @ w
        H = (Xb * W[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-7:
            break
    return w


def prep(df, feats, mean=None, std=None):
    X = df[feats].astype(float).values
    if mean is None:
        mean, std = X.mean(0), X.std(0)
        std[std == 0] = 1
    Z = np.clip((X - mean) / std, -5, 5)
    return Z, mean, std


def predict_raw(df, feats, w, mean, std):
    Z, _, _ = prep(df, feats, mean, std)
    return 1 / (1 + np.exp(-(w[0] + Z @ w[1:])))


def auc(y, s):
    y = np.asarray(y)
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/confidence_data.csv.gz", index=False, compression="gzip")
    feats = [c for c in df.columns if c not in ("t", "date", "status", "hit", "res", "xret", "sess", "u10", "ret20")]
    print(f"\nالصفوف: {len(df)} · الأسهم: {df.t.nunique()} · الفترة: {df.date.min()} → {df.date.max()}")
    print("العوامل:", len(feats))
    sig = df.status.isin(["ready", "near"])
    print("\n==== نسب الأساس (النجاح = الهدف الأول قبل الإغلاق تحت الوقف خلال 20 جلسة) ====")
    for name, sub in (("كل الأسهم-الأيام", df), ("نقاط 40+", df[df.score >= 40]), ("✅+🟡 (السجل)", df[sig]),
                      ("✅ مطابق", df[df.status == "ready"]), ("🟡 قريب", df[df.status == "near"]), ("رادار", df[df.status == "radar"])):
        if len(sub):
            print(f"  {name:<18} n={len(sub):<7} نجاح {pct(sub.hit.mean())} · وقف {pct((sub.res == 'stop').mean())} · "
                  f"مفتوحة {pct((sub.res == 'open').mean())} · متوسط العائد عند الخروج {sub.xret.mean() * 100:+.2f}% · "
                  f"وسيط {sub.xret.median() * 100:+.1f}% · +20%قبل−10% {pct(sub.u10.mean())}")

    # --- Walk-forward
    pop = df[df.score >= 40].copy()
    dates = np.sort(pop.date.unique())
    cuts = [dates[int(len(dates) * q)] for q in (0.5, 0.67, 0.83)] + ["9999"]
    oos = []
    coefs = []
    print("\n==== Walk-forward (تدريب على الماضي ← اختبار على اللي بعده) ====")
    for k in range(3):
        tr = pop[pop.date < cuts[k]]
        te = pop[(pop.date >= cuts[k]) & (pop.date < cuts[k + 1])]
        Ztr, mu, sd = prep(tr, feats)
        w = fit_logit(Ztr, tr.hit.values.astype(float))
        p = predict_raw(te, feats, w, mu, sd)
        te = te.assign(p=p)
        oos.append(te)
        coefs.append(dict(zip(feats, w[1:])))
        s = te[te.status.isin(["ready", "near"])]
        print(f"  مرحلة {k + 1}: تدريب n={len(tr)} (لين {cuts[k]}) · اختبار n={len(te)} · "
              f"AUC النموذج {auc(te.hit, te.p):.3f} · AUC نقاط فيصل {auc(te.hit, te.score):.3f} · "
              f"✅+🟡 AUC {auc(s.hit, s.p):.3f}")
    O = pd.concat(oos)
    S = O[O.status.isin(["ready", "near"])]
    print(f"\n==== النتيجة خارج العينة (كل فترات الاختبار مجموعة: n={len(O)}, ✅+🟡 n={len(S)}) ====")
    print(f"  AUC النموذج {auc(O.hit, O.p):.3f} مقابل نقاط فيصل لحالها {auc(O.hit, O.score):.3f} (0.5 = عشوائي)")
    print("\n-- المعايرة: الاحتمال المتوقع مقابل الفعلي (أعشار) --")
    O["dec"] = pd.qcut(O.p, 10, labels=False, duplicates="drop")
    cal = []
    for dcl, g in O.groupby("dec"):
        print(f"  عُشر {int(dcl) + 1}: متوقع {pct(g.p.mean())} · فعلي {pct(g.hit.mean())} · متوسط العائد {g.xret.mean() * 100:+.2f}% · n={len(g)}")
        cal.append([float(g.p.mean()), float(g.hit.mean()), len(g)])
    # منحنى صاعد (Pool Adjacent Violators) عشان الثقة الأعلى ما تعطي نسبة أقل
    blocks = [[c[0] * c[2], c[1] * c[2], c[2]] for c in cal]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] / blocks[i][2] > blocks[i + 1][1] / blocks[i + 1][2]:
            blocks[i] = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    cal = [[b_[0] / b_[2], b_[1] / b_[2]] for b_ in blocks]
    print("  منحنى المعايرة:", [[round(x, 3), round(y, 3)] for x, y in cal])
    print("\n-- ✅+🟡 حسب الثقة (خارج العينة) --")
    for lab, lo, hi in (("أعلى 25%", 0.75, 1.01), ("الوسط 25-75%", 0.25, 0.75), ("أقل 25%", 0, 0.25)):
        q = S.p.rank(pct=True)
        g = S[(q >= lo) & (q < hi)]
        print(f"  {lab:<14} n={len(g):<6} نجاح {pct(g.hit.mean())} · وقف {pct((g.res == 'stop').mean())} · "
              f"متوسط العائد {g.xret.mean() * 100:+.2f}% · وسيط {g.xret.median() * 100:+.1f}% · احتمال متوقع {pct(g.p.mean())}")
    print(f"  كل ✅+🟡: نجاح {pct(S.hit.mean())} · متوسط العائد {S.xret.mean() * 100:+.2f}%")
    print("\n-- كل المرشحين (نقاط 40+) حسب الثقة --")
    q = O.p.rank(pct=True)
    for lab, lo, hi in (("أعلى 10%", 0.9, 1.01), ("أعلى 10-25%", 0.75, 0.9), ("الباقي", 0, 0.75)):
        g = O[(q >= lo) & (q < hi)]
        print(f"  {lab:<12} n={len(g):<6} نجاح {pct(g.hit.mean())} · متوسط العائد {g.xret.mean() * 100:+.2f}% · "
              f"منها ✅+🟡 {pct(g.status.isin(['ready', 'near']).mean())}")
    # ثبات الإشارات
    print("\n-- العوامل: اتجاه الأثر في المراحل الثلاث (+ يرفع النجاح، − ينزله) --")
    cf = pd.DataFrame(coefs)
    for f_ in cf.columns:
        v = cf[f_].values
        st = "✅ ثابت" if (np.sign(v) == np.sign(v[0])).all() else "⚠ متقلب"
        print(f"  {C.NAMES_AR.get(f_, f_):<36} {' '.join(f'{x:+.3f}' for x in v)}  {st}")
    # فلاتر واضحة
    print("\n-- فلاتر واضحة على ✅+🟡 (كل الفترة) --")
    T = df[sig]

    def cmp(name, m):
        a_, b_ = T[m], T[~m]
        if len(a_) >= 30 and len(b_) >= 30:
            print(f"  {name:<34} نعم: نجاح {pct(a_.hit.mean())} عائد {a_.xret.mean() * 100:+.2f}% (n={len(a_)}) · "
                  f"لا: نجاح {pct(b_.hit.mean())} عائد {b_.xret.mean() * 100:+.2f}% (n={len(b_)})")
    cmp("ملف تخفيف خلال 90 يوم", T.dil90 >= 1)
    cmp("تقسيم عكسي خلال سنة", T.rs365 >= 1)
    cmp("تقسيمين عكسيين+ خلال سنة", T.rs365 >= 2)
    cmp("تقسيم عكسي خلال 45 يوم", T.days_rs <= 45)
    cmp("الأرباح قريبة (75-100 يوم من آخرها)", T.earn_soon == 1)
    cmp("IWM فوق متوسط 50", T.iwm_up == 1)
    cmp("نسبة الأسهم الصاعدة > 50%", T.breadth > 0.5)
    cmp("نسبة الأسهم الصاعدة < 30%", T.breadth < 0.3)
    cmp("السعر تحت 1$", T.price_log < 0)
    cmp("RSI تحت 30", T.rsi < 30)
    cmp("الوقف أبعد من 25%", T.stop_pct > 25)
    cmp("الهدف الأول أقل من 10%", T.t1_pct < 10)
    cmp("سبق انفجر (Former Runner)", T.runner > np.log1p(100))
    cmp("صعد 20%+ من قاع 20 يوم", T.up_from_low20 > 20)
    cmp("الفوليوم الأخير أقل من المعتاد", T.vol_ratio < 0.7)
    cmp("إغلاق قريب من أعلى الشمعة (5 أيام)", T.close_loc > 0.6)
    # النموذج النهائي على كل البيانات
    Z, mu, sd = prep(pop, feats)
    w = fit_logit(Z, pop.hit.values.astype(float))
    tiers_q = S.p.quantile([0.4, 0.75]).values
    # نحول حدود المستويات من الاحتمال الخام إلى المعاير (نفس الأعشار)
    model = {"version": datetime.now().strftime("%Y-%m-%d"), "target": "T1 before close<stop within 20 sessions",
             "intercept": float(w[0]), "coef": {f_: float(x) for f_, x in zip(feats, w[1:])},
             "mean": {f_: float(x) for f_, x in zip(feats, mu)}, "std": {f_: float(x) for f_, x in zip(feats, sd)},
             "calibration": cal, "base_rate": float(df[sig].hit.mean()), "n_train": int(len(pop)),
             "tiers_raw": [float(x) for x in tiers_q],
             "oos_auc": auc(O.hit, O.p), "oos_auc_score": auc(O.hit, O.score)}
    # المستويات على الاحتمال المعاير: أعلى 25% من ✅+🟡 = عالية، 40-75% = متوسطة
    sp = pd.Series(np.interp(S.p.values, [c[0] for c in cal], [c[1] for c in cal]))
    model["tiers"] = {"high": float(sp.quantile(0.75)), "mid": float(sp.quantile(0.40))}
    print("\nMODEL_JSON_BEGIN")
    print(json.dumps(model, ensure_ascii=False))
    print("MODEL_JSON_END")


if __name__ == "__main__":
    main()
