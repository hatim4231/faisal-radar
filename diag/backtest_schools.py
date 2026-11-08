"""
🧭 رادار المدارس الثلاث — اختبار تاريخي (تشخيص فقط).

- Alpaca يومي ~3 سنين لكل أسهمنا (0.7-10$ حسب config) + IWM + ملفات SEC.
- كل سهم كل جلستين: نشغل schools.detect على البيانات لين ذاك اليوم بس، ونسجل كل فرصة + الخطة المجمعة.
- النتيجة بنفس قواعد السجل: الهدف الأول يُفحص أول (لمس)، الوقف بإغلاق اليوم تحته، خلال 10 و 20 جلسة.
- المرجع العادل: صفقات عشوائية (أسهم وأيام عشوائية) **بنفس بعد الهدف والوقف** ← نموذج «الهندسة فقط»
  يعطي النسبة المتوقعة للفرصة لو ما فيها أي ميزة. الفرق = قوة المدرسة الحقيقية.
- Walk-forward (3 مراحل) لنموذج ثقة للرادار + JSON بين علامتين.
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import schools as SC
from backtest_confidence import sec_dates, fit_logit, prep, predict_raw, auc, pct
from confidence_report import pav

STEP, DAYS = 2, 1100
RAND_P = 0.15


def outcome(fut, entry, stop, t1, H):
    f = fut.iloc[:H]
    for k, (_, r) in enumerate(f.iterrows()):
        if r["High"] >= t1:
            return 1, t1 / entry - 1, "hit", k + 1
        if r["Close"] < stop:
            return 0, r["Close"] / entry - 1, "stop", k + 1
    return 0, float(f["Close"].iloc[-1]) / entry - 1, "open", len(f)


def context(hist, dil, earn, mkt):
    c, h, l, v = (hist[k].to_numpy(float) for k in ("Close", "High", "Low", "Volume"))
    day = hist.index[-1]
    rs = [ix for ix, x in hist["Stock Splits"].items() if x and 0 < float(x) < 1] if "Stock Splits" in hist else []
    dp = [pd.Timestamp(x) for x in dil if pd.Timestamp(x) <= day]
    ep = [pd.Timestamp(x) for x in earn if pd.Timestamp(x) <= day]
    de = (day - max(ep)).days if ep else 200
    ma50 = c[-50:].mean()
    m = mkt.get(day) or {}
    return {"price_log": float(np.log10(c[-1])), "dollar_vol": float(np.log10(max(1, (c[-20:] * v[-20:]).mean()))),
            "atr20": float(((h - l) / c)[-20:].mean() * 100), "ret20": float(c[-1] / c[-21] - 1) * 100,
            "ret60": float(c[-1] / c[-61] - 1) * 100 if len(c) > 61 else 0.0,
            "drawdown": float(c[-1] / h[-252:].max() - 1) * 100, "vol_ratio": float(v[-5:].mean() / max(1, v[-50:].mean())),
            "above_ma50": float(c[-1] > ma50), "ma50_up": float(ma50 > c[-60:-10].mean()),
            "rs365": float(sum(1 for x in rs if (day - x).days <= 365)),
            "days_rs": float(min(400, (day - max(rs)).days)) if rs else 400.0,
            "dil90": float(sum(1 for x in dp if (day - x).days <= 90)), "days_earn": float(min(200, de)),
            "earn_soon": float(75 <= de <= 100), "iwm_up": float(m.get("iwm_up", 0.5)),
            "iwm_ret20": float(m.get("iwm_ret20", 0)), "breadth": float(m.get("breadth", 0.4))}


def run_one(args):
    t, d, cfg, dil, earn, mkt, seed = args
    rnd = random.Random(seed)
    out = []
    U = cfg["universe"]
    n = len(d)
    for end in range(220, n - 20, STEP):
        hist = d.iloc[:end]
        px = float(hist["Close"].iloc[-1])
        if not (U["min_price"] <= px <= U["max_price"]) or float(hist["Volume"].tail(30).mean()) < U["min_avg_volume"] * 0.7:
            continue
        fut = d.iloc[end:end + 20]
        ctx = None
        if rnd.random() < RAND_P and len(fut) == 20:
            ctx = context(hist, dil, earn, mkt)
            out.append({"t": t, "date": str(hist.index[-1].date()), "kind": "random", "setup": "RANDOM",
                        "ph": (fut["High"].to_numpy(float) / px).round(5).tolist(),
                        "pc": (fut["Close"].to_numpy(float) / px).round(5).tolist(), **ctx})
        try:
            r = SC.detect(hist)
        except Exception:
            r = None
        if not r:
            continue
        ctx = ctx or context(hist, dil, earn, mkt)
        base = {"t": t, "date": str(hist.index[-1].date()), "confluence": r["confluence"],
                "schools": "+".join(r["schools"]), "n_setups": len(r["setups"]), **ctx}
        for s in r["setups"] + [dict(setup="COMBO", school="combo", quality=max(x["quality"] for x in r["setups"]),
                                     **{k: r["plan"][k] for k in ("entry", "stop", "t1", "t2")}, parts={}, type="+".join(r["types"]))]:
            st_pct = (s["entry"] - s["stop"]) / s["entry"] * 100
            t1_pct = (s["t1"] / s["entry"] - 1) * 100
            if st_pct <= 0 or t1_pct <= 0:
                continue
            h10, x10, r10, _ = outcome(fut, px, s["stop"], s["t1"], 10)
            h20, x20, r20, _ = outcome(fut, px, s["stop"], s["t1"], 20)
            fh, fl_, fc = fut["High"].to_numpy(float), fut["Low"].to_numpy(float), fut["Close"].to_numpy(float)
            row = {**base, "kind": "setup", "setup": s["setup"], "school": s["school"], "quality": s["quality"],
                   "mfe10": fh[:10].max() / px - 1, "mfe20": fh.max() / px - 1, "mae10": fl_[:10].min() / px - 1,
                   "c10": fc[min(9, len(fc) - 1)] / px - 1, "c20": fc[-1] / px - 1,
                   "type": s.get("type"), "stop_pct": st_pct, "t1_pct": t1_pct,
                   "t2_pct": ((s["t2"] / s["entry"] - 1) * 100) if s.get("t2") else None,
                   "hit10": h10, "x10": x10, "res10": r10, "hit20": h20, "x20": x20, "res20": r20}
            for k, v in (s.get("parts") or {}).items():
                row[f"p_{s['setup']}_{k}"] = v
            for code in SC.SCHOOLS:
                row[f"has_{code}"] = float(any(x["setup"] == code for x in r["setups"]))
            out.append(row)
    return out


def collect():
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    daily = {}
    for i in range(0, len(tickers), 100):
        try:
            daily.update(A.daily(tickers[i:i + 100], days=DAYS))
        except Exception as e:
            print("daily err", repr(e)[:120], flush=True)
    daily = {t: d for t, d in daily.items() if len(d) > 260}
    iwm = A.daily(["IWM"], days=DAYS).get("IWM")
    closes = pd.DataFrame({t: d["Close"] for t, d in daily.items()})
    above = (closes > closes.rolling(50).mean()).astype(float).where(closes.rolling(50).count() >= 50)
    breadth = (above.sum(axis=1) / above.count(axis=1).replace(0, np.nan)).astype(float)
    iw = iwm["Close"].astype(float)
    iwm_up = (iw > iw.rolling(50).mean()).astype(float)
    iwm_r20 = (iw / iw.shift(20) - 1) * 100
    mkt = {}
    for day in closes.index:
        br, up, r20 = breadth.get(day, np.nan), iwm_up.get(day, np.nan), iwm_r20.get(day, np.nan)
        mkt[day] = {"breadth": 0.4 if pd.isna(br) else float(br), "iwm_up": 0.5 if pd.isna(up) else float(up),
                    "iwm_ret20": 0.0 if pd.isna(r20) else float(r20)}
    sec = {}
    for t in daily:
        try:
            sec[t] = sec_dates(t)
        except Exception:
            sec[t] = ([], [])
    print("أسهم:", len(daily), flush=True)
    jobs = [(t, d, cfg, sec[t][0], sec[t][1], mkt, i) for i, (t, d) in enumerate(daily.items())]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=3)):
            rows += r
            if i % 100 == 0:
                print("analyze", i, len(rows), round(time.time() - t0), flush=True)
    return pd.DataFrame(rows)


GEO = ["lt1", "lst", "ratio", "atr20", "price_log"]


U_GRID = 1 + np.arange(0.005, 1.0001, 0.005)      # الهدف من +0.5% إلى +100%
S_GRID = 1 - np.arange(0.005, 0.6001, 0.005)      # الوقف من −0.5% إلى −60%


def geometry_tables(PH, PC, H):
    """جداول دقيقة (0.5%) لنسبة النجاح ومتوسط العائد لصفقات عشوائية، لكل هدف ووقف — نفس قواعد السجل بالضبط."""
    ph, pc = PH[:, :H], PC[:, :H]
    n = len(ph)
    M = np.maximum.accumulate(ph, axis=1)                       # أعلى سعر لين اليوم j
    mprev = np.minimum.accumulate(np.hstack([np.full((n, 1), np.inf), pc[:, :-1]]), axis=1)  # أقل إغلاق قبل اليوم j
    HIT = np.zeros((len(U_GRID), len(S_GRID)))
    RET = np.zeros_like(HIT)
    for js, sv in enumerate(S_GRID):
        ok = mprev >= sv                                         # ما ضرب الوقف قبل اليوم j
        Us = np.max(np.where(ok, M, 0), axis=1)                  # أعلى هدف ممكن قبل الوقف
        stop_mat = pc < sv
        k = np.where(stop_mat.any(1), stop_mat.argmax(1), H)
        nohit_ret = np.where(k < H, pc[np.arange(n), np.minimum(k, H - 1)], pc[:, -1]) - 1
        hit = Us[None, :] >= U_GRID[:, None]
        HIT[:, js] = hit.mean(1)
        RET[:, js] = np.where(hit, (U_GRID - 1)[:, None], nohit_ret[None, :]).mean(1)
    return HIT, RET


def _interp2(T, u, s):
    iu = np.clip((u - U_GRID[0]) / 0.005, 0, len(U_GRID) - 1.0001)
    is_ = np.clip((S_GRID[0] - s) / 0.005, 0, len(S_GRID) - 1.0001)
    a, b = iu.astype(int), is_.astype(int)
    fa, fb = iu - a, is_ - b
    return (T[a, b] * (1 - fa) * (1 - fb) + T[a + 1, b] * fa * (1 - fb) + T[a, b + 1] * (1 - fa) * fb + T[a + 1, b + 1] * fa * fb)


def matched_baseline(Sx, R, H):
    """لكل فرصة: نفس الهدف والوقف بالضبط (جدول 0.5%) على صفقات عشوائية بنفس شريحة التذبذب ← المتوقع بدون أي ميزة."""
    PH, PC = np.array(R.ph.tolist()), np.array(R.pc.tolist())
    qs = np.quantile(R.atr20, [0.2, 0.4, 0.6, 0.8])
    rq = np.searchsorted(qs, R.atr20.values)
    sq = np.searchsorted(qs, Sx.atr20.values)
    u = 1 + Sx.t1_pct.values / 100
    s_ = 1 - Sx.stop_pct.values / 100
    exp, xr = np.full(len(Sx), np.nan), np.full(len(Sx), np.nan)
    for q in range(5):
        m = rq == q
        HIT, RET = geometry_tables(PH[m] if m.sum() >= 200 else PH, PC[m] if m.sum() >= 200 else PC, H)
        w = sq == q
        exp[w], xr[w] = _interp2(HIT, u[w], s_[w]), _interp2(RET, u[w], s_[w])
    return exp, xr


def line(name, g, H="10"):
    g = g[g["first"]] if "first" in g else g
    if len(g) < 30:
        return f"  {name:<34} n={len(g):<6} (قليل)"
    hit, exp = g[f"hit{H}"].mean(), g[f"exp{H}"].mean()
    per = (g[f"hit{H}"] - g[f"exp{H}"]).groupby(g.t).mean()
    ci = 1.96 * per.std() / max(1, len(per)) ** 0.5 if len(per) > 2 else float("nan")
    x, xr = g[f"x{H}"].mean(), g[f"xrand{H}"].mean()
    return (f"  {name:<34} n={len(g):<6} نجاح {pct(hit)} (لو عشوائي بنفس الهدف/الوقف {pct(exp)} ← {(hit - exp) * 100:+.1f} ±{ci * 100:.1f}) · "
            f"وقف {pct((g[f'res{H}'] == 'stop').mean())} · متوسط العائد {x * 100:+.2f}% (العشوائي المماثل {xr * 100:+.2f}%) · "
            f"وسيط {g[f'x{H}'].median() * 100:+.1f}% · الهدف {g.t1_pct.median():.0f}% / الوقف {g.stop_pct.median():.0f}%")


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/schools_data.csv.gz", index=False, compression="gzip")
    R = df[df.kind == "random"].copy()
    Sx = df[df.kind == "setup"].copy()
    print(f"\nالفترة {df.date.min()} → {df.date.max()} · فرص {len(Sx)} · عشوائي {len(R)} · أسهم {df.t.nunique()}")
    # أول ظهور لكل فرصة (التنبيه الحقيقي): نفس السهم ونفس النوع ما يتكرر قبل 14 يوم
    Sx = Sx.sort_values(["t", "setup", "date"])
    dt = pd.to_datetime(Sx.date)
    first = np.zeros(len(Sx), bool)
    last = {}
    for i, (t_, st_, d_) in enumerate(zip(Sx.t.values, Sx.setup.values, dt.values)):
        k = (t_, st_)
        if k not in last or (d_ - last[k]) > np.timedelta64(14, "D"):
            first[i] = True
            last[k] = d_
        elif (d_ - last[k]) <= np.timedelta64(14, "D"):
            last[k] = d_
    Sx["first"] = first
    print(f"أول ظهور (تنبيهات حقيقية): {first.sum()} من {len(Sx)} صف")
    # المرجع المطابق لكل فرصة (نفس الهدف/الوقف + نفس شريحة التذبذب)
    for H in ("10", "20"):
        Sx[f"exp{H}"], Sx[f"xrand{H}"] = matched_baseline(Sx, R, int(H))
    for H in ("10", "20"):
        print(f"\n==== النتيجة خلال {H} جلسات ====")
        print("-- كل فرصة لحالها --")
        for code in list(SC.SCHOOLS) + ["COMBO"]:
            print(line(f"{code} {SC.SETUP_AR.get(code, 'الخطة المجمعة')[:26]}", Sx[Sx.setup == code], H))
        print("-- التوافق (الخطة المجمعة) --")
        C = Sx[Sx.setup == "COMBO"]
        for k in (1, 2, 3):
            print(line(f"توافق {k} مدرسة", C[C.confluence == k], H))
        print(line("توافق 2+ مدرسة", C[C.confluence >= 2], H))
        print("-- الجودة (الخطة المجمعة) --")
        for lo, hi in ((0, 50), (50, 65), (65, 80), (80, 101)):
            print(line(f"جودة {lo}-{hi}", C[(C.quality >= lo) & (C.quality < hi)], H))
        print("-- السياق (الخطة المجمعة) --")
        print(line("سوق الأسهم الصغيرة صاعد (IWM)", C[C.iwm_up == 1], H))
        print(line("سوق الأسهم الصغيرة نازل", C[C.iwm_up == 0], H))
        print(line("فوق متوسط 50", C[C.above_ma50 == 1], H))
        print(line("تحت متوسط 50", C[C.above_ma50 == 0], H))
        print(line("بدون ملفات تخفيف 90 يوم", C[C.dil90 == 0], H))
        print(line("فيه ملفات تخفيف", C[C.dil90 >= 1], H))
    # ==== اقتناص الطيارات: هل الأنماط تمسك الأسهم اللي تطير؟ (بدون هدف/وقف) ====
    PH = np.array(R.ph.tolist()); PC = np.array(R.pc.tolist())
    R["mfe10"], R["mfe20"] = PH[:, :10].max(1) - 1, PH.max(1) - 1
    R["c10"], R["c20"] = PC[:, 9] - 1, PC[:, -1] - 1
    qa = np.quantile(R.atr20, [0.2, 0.4, 0.6, 0.8]); qp = np.quantile(R.price_log, [1 / 3, 2 / 3])
    R["cell"] = np.searchsorted(qa, R.atr20) * 3 + np.searchsorted(qp, R.price_log)
    Sx["cell"] = np.searchsorted(qa, Sx.atr20) * 3 + np.searchsorted(qp, Sx.price_log)
    cellm = R.groupby("cell").agg(r30=("mfe20", lambda x: (x >= 0.3).mean()), r50=("mfe20", lambda x: (x >= 0.5).mean()),
                                  r15_10=("mfe10", lambda x: (x >= 0.15).mean()), c20=("c20", "mean"), c10=("c10", "mean"))

    def runners(name, g):
        g = g[g["first"]] if "first" in g else g
        if len(g) < 40:
            return f"  {name:<34} n={len(g)} (قليل)"
        e = cellm.reindex(g.cell)
        a30, b30 = (g.mfe20 >= 0.3).mean(), e.r30.mean()
        a50, b50 = (g.mfe20 >= 0.5).mean(), e.r50.mean()
        a15, b15 = (g.mfe10 >= 0.15).mean(), e.r15_10.mean()
        per = ((g.mfe20 >= 0.3).astype(float).values - e.r30.values)
        per = pd.Series(per, index=g.index).groupby(g.t).mean()
        ci = 1.96 * per.std() / max(1, len(per)) ** 0.5
        return (f"  {name:<34} n={len(g):<5} طار +30% خلال 20: {pct(a30)} (عشوائي مماثل {pct(b30)} ← {(a30 - b30) * 100:+.1f} ±{ci * 100:.1f}) · "
                f"+50%: {pct(a50)} ({pct(b50)}) · +15% خلال 10: {pct(a15)} ({pct(b15)}) · عائد 20 جلسة {g.c20.mean() * 100:+.1f}% ({e.c20.mean() * 100:+.1f}%)")
    print("\n==== اقتناص الطيارات (أقصى صعود بدون هدف/وقف) مقابل أسهم عشوائية بنفس التذبذب والسعر ====")
    for code in list(SC.SCHOOLS) + ["COMBO"]:
        print(runners(f"{code} {SC.SETUP_AR.get(code, 'الخطة المجمعة')[:26]}", Sx[Sx.setup == code]))
    C_ = Sx[Sx.setup == "COMBO"]
    print(runners("توافق 2+ مدرسة", C_[C_.confluence >= 2]))
    print(runners("جودة 80+", C_[C_.quality >= 80]))
    print(runners("سوق صاعد + فوق متوسط 50", C_[(C_.iwm_up == 1) & (C_.above_ma50 == 1)]))
    # الثبات
    mid = Sx.date.sort_values().iloc[len(Sx) // 2]
    print(f"\n==== الثبات: النص الأول (قبل {mid}) مقابل الثاني — 10 جلسات ====")
    for code in list(SC.SCHOOLS) + ["COMBO"]:
        a_, b_ = Sx[(Sx.setup == code) & (Sx.date < mid)], Sx[(Sx.setup == code) & (Sx.date >= mid)]
        if len(a_) > 30 and len(b_) > 30:
            print(f"  {code:<5} الفرق: {(a_.hit10.mean() - a_.exp10.mean()) * 100:+.1f} ثم {(b_.hit10.mean() - b_.exp10.mean()) * 100:+.1f} · "
                  f"العائد: {(a_.x10.mean() - a_.xrand10.mean()) * 100:+.2f} ثم {(b_.x10.mean() - b_.xrand10.mean()) * 100:+.2f}")
    # نموذج الثقة (Walk-forward) على الخطة المجمعة
    C = Sx[Sx.setup == "COMBO"].copy()
    feats = ["quality", "confluence", "n_setups", "stop_pct", "t1_pct", "price_log", "dollar_vol", "atr20", "ret20", "ret60",
             "drawdown", "vol_ratio", "above_ma50", "ma50_up", "rs365", "days_rs", "dil90", "days_earn", "earn_soon",
             "iwm_up", "iwm_ret20", "breadth"] + [f"has_{c}" for c in SC.SCHOOLS]
    C[feats] = C[feats].fillna(0)
    dates = np.sort(C.date.unique())
    cuts = [dates[int(len(dates) * q)] for q in (0.5, 0.67, 0.83)] + ["9999"]
    oos = []
    for k in range(3):
        tr, te_ = C[C.date < cuts[k]], C[(C.date >= cuts[k]) & (C.date < cuts[k + 1])]
        Z, mu, sd = prep(tr, feats)
        w = fit_logit(Z, tr.hit10.values.astype(float))
        oos.append(te_.assign(p=predict_raw(te_, feats, w, mu, sd), fold=k + 1))
    O = pd.concat(oos)
    print(f"\n==== نموذج الثقة (خارج العينة) — AUC {auc(O.hit10, O.p):.3f} (الهندسة لحالها {auc(O.hit10, O.exp10):.3f}) ====")
    O["dec"] = pd.qcut(O.p, 10, labels=False, duplicates="drop")
    cal = pav([(g.p.mean(), g.hit10.mean(), len(g)) for _, g in O.groupby("dec")])
    O["pc"] = np.interp(O.p, [c_[0] for c_ in cal], [c_[1] for c_ in cal])
    hi, lo = float(O.pc.quantile(0.8)), float(O.pc.quantile(0.4))
    O["tier"] = np.where(O.pc >= hi, "عالية", np.where(O.pc >= lo, "متوسطة", "منخفضة"))
    ts = {}
    for t_ in ("عالية", "متوسطة", "منخفضة"):
        g = O[O.tier == t_]
        print(line(f"ثقة {t_}", g))
        win, miss = g[g.hit10 == 1], g[g.hit10 == 0]
        ts[t_] = {"n": int(len(g)), "hit": float(g.hit10.mean()), "lose": float((g.x10 < 0).mean()), "mean": float(g.x10.mean()),
                  "median": float(g.x10.median()), "avg_win": float(win.x10.mean()) if len(win) else None,
                  "avg_miss": float(miss.x10.mean()) if len(miss) else None, "expected_random": float(g.exp10.mean())}
    for k in (1, 2, 3):
        F = O[O.fold == k]
        print(f"  المرحلة {k}: " + " · ".join(f"{t_} {pct(F[F.tier == t_].hit10.mean())} (عشوائي {pct(F[F.tier == t_].exp10.mean())}) "
                                              f"عائد {F[F.tier == t_].x10.mean() * 100:+.1f}%" for t_ in ("عالية", "متوسطة", "منخفضة")))
    Z, mu, sd = prep(C, feats)
    w = fit_logit(Z, C.hit10.values.astype(float))
    model = {"version": datetime.now().strftime("%Y-%m-%d"), "horizon": 10, "target": "T1 before close<stop within 10 sessions",
             "intercept": float(w[0]), "coef": {f: float(x) for f, x in zip(feats, w[1:])},
             "mean": {f: float(x) for f, x in zip(feats, mu)}, "std": {f: float(x) for f, x in zip(feats, sd)},
             "calibration": cal, "tiers": {"high": hi, "mid": lo}, "tier_stats": ts, "base_rate": float(C.hit10.mean()),
             "n_train": int(len(C)), "oos_auc": auc(O.hit10, O.p), "oos_auc_geometry": auc(O.hit10, O.exp10),
             "period": [str(df.date.min()), str(df.date.max())]}
    print("\nMODEL_JSON_BEGIN")
    print(json.dumps(model, ensure_ascii=False))
    print("MODEL_JSON_END")


if __name__ == "__main__":
    main()
