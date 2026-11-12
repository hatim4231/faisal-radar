"""
🎯 «الجاهزة» 2 — نفس فكرة backtest_ready.py لكن من **كل رادارات تيليجرام**: فيصل المطوّر + فيصل الأصلي + عبدالله (تشخيص فقط).
السبب: بسجل أول أسبوع وصل المستخدم 159 توصية، وأكبر الرابحين (PEPG +157%، BYAH +77%، TVRD +68%) كانوا من «الأصلي» و«عبدالله»
وضاعوا بالزحمة. السؤال: هل فيه ترتيب يطلّع من كل التوصيات **أعلى 1-3 باليوم** تكسب بعد التكاليف وتصيد اللي تطير؟

- ناسداك مع المشطوبة · السعر الحقيقي ذاك اليوم · دخول افتتاح اليوم التالي · بعد التكاليف (4%/2%/1%).
- الميزات مشتركة لكل الرادارات (ما تعتمد على رادار معيّن): ميزات «عالي مخاطرة» (حمض الطيارات 🧨 + النزيف 🩸) + هندسة الخطة + السوق.
- نماذج ترتيب خارج العينة (كل نص سنة بنموذج تدرّب على اللي قبلها) ← أعلى 1 و3 باليوم من كل الرادارات مع بعض.
المخرجات: جداول + JSON بين RD2_BEGIN/END (أسطر R2|).
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
sys.path.insert(0, str(ROOT / "abdullah"))
sys.path.insert(0, str(ROOT / "faisal_original"))
import numpy as np
import pandas as pd

import alpaca_data as A
import scanner
import faisal_rules as fr
import faisal_rules_original as fo
import abdullah_rules as ar
import new_low_rule
import compliance_rule
import highrisk as HR
import backtest_ready as R1
from backtest_ready import (START, STEP, H, PLANS, RSI_PRE, universe, score_all, sim_plan, sim_half, sim_trail, prep_frame,
                            fit_logit, fit_ridge, auc, pr, pp, cost_of, tday, _sh, _fsh, _roll)

MIN_TRAIN = 1500
ORIG_STEP = 4                      # الأصلي إشاراته كثيرة وتحليله ثقيل ← كل 4 جلسات
SRC_AR = {"dev": "📡 فيصل المطوّر", "orig": "📜 فيصل الأصلي", "abd": "🌊 عبدالله"}


def masks2(d, U, AB):
    c, h, l, v = (d[k].to_numpy(float) for k in ("Close", "High", "Low", "Volume"))
    n = len(c)
    idx = np.arange(n)
    kf = HR.raw_factor(d)
    praw = c * kf
    v30 = pd.Series(v / kf).rolling(30, min_periods=20).mean().to_numpy()
    rsi = fr.rsi(d["Close"]).to_numpy(float)
    base = (idx >= 260) & (idx <= n - H - 2) & (idx % STEP == 0) & (praw >= U["min_price"]) & (praw <= U["max_price"]) & \
           (np.nan_to_num(v30) >= U["min_avg_volume"] * 0.7)
    dev = base & (np.nan_to_num(rsi, nan=99) <= RSI_PRE)
    orig = base & (idx % ORIG_STEP == 0)
    # عبدالله: لازم موجة تدبيل (قمة ÷ قاع ≥ 2) قمتها خلال آخر 120 جلسة والسعر تحتها بـ 15%+ ← فلتر سريع قبل التحليل
    hi120 = _roll(h, AB["peak_within_sessions"], "max", 20)
    lo = _roll(l, AB["peak_within_sessions"] + AB["wave_max_len"], "min", 20)
    abd = base & (np.nan_to_num(hi120 / np.maximum(lo, 1e-9)) >= AB["min_wave_mult"]) & (c <= hi120 * 0.86)
    return kf, praw, rsi, base, dev, orig, abd


def job(args):
    sym, d, info, cfg, cfg_abd, mkt = args
    U, AB = cfg["universe"], cfg_abd["abdullah"]
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    kf, praw, rsi, base, m_dev, m_orig, m_abd = masks2(d, U, AB)
    if not base.any():
        return None
    F = HR.features(d)
    p_raw, pb = score_all(F)
    FB = HR.model()["feats_b"]
    Fv = F[FB].to_numpy(float)
    Dint = d.index.strftime("%Y%m%d").astype(int).to_numpy(np.int32)
    o1 = _fsh(o, 1)
    B = pd.DataFrame({"praw": praw, "rsi": rsi, "H10": _fsh(c, 10) / o1 - 1, "H20": _fsh(c, 20) / o1 - 1,
                      "mx20": _fsh(_roll(h, 20, "max"), 20) / o1 - 1, "mx40": _fsh(_roll(h, 40, "max"), 40) / o1 - 1,
                      "mn20": _fsh(_roll(l, 20, "min"), 20) / o1 - 1})[base].astype(np.float32)
    B["D"] = Dint[base]
    rows, calls = [], [0, 0, 0]
    for i in np.flatnonzero(m_dev | m_orig | m_abd):
        end, k = i + 1, kf[i]
        hist = d.iloc[max(0, end - 500):end]
        if k != 1:
            hist = hist.copy()
            hist[["Open", "High", "Low", "Close"]] *= k
            hist["Volume"] /= k
        inf = info
        if info:
            inf = dict(info)
            so, fl = info.get("sharesOutstanding"), info.get("floatShares")
            if so:
                inf["marketCap"] = so * c[i]
            if fl:
                inf["floatShares"] = fl / k
        e = o[i + 1] * k
        Hh, Cc = h[i + 1:i + 1 + H] * k, c[i + 1:i + 1 + H] * k
        if not (np.isfinite(e) and e > 0) or len(Cc) < H:
            continue
        px = c[i] * k
        day = hist.index[-1]
        sig = []                                         # (المصدر، الحالة، النقاط، الوقف، الهدف 1)
        if m_dev[i]:
            calls[0] += 1
            try:
                a = fr.analyze(hist, inf, cfg)
                if a:
                    new_low_rule.apply(a, hist)
                    compliance_rule.apply_faisal(a, hist)
                    if not a["gate_fail"] and a["irtikaz"]["status"] in ("ready", "near"):
                        sig.append(("dev", a["irtikaz"]["status"], a["irtikaz"]["score"], a["plan"].get("stop"), a["plan"].get("t1")))
            except Exception:
                pass
        if m_orig[i]:
            calls[1] += 1
            try:
                a = fo.analyze(hist, inf, cfg)
                if a:
                    new_low_rule.apply(a, hist)
                    compliance_rule.apply_faisal(a, hist)
                    if not a["gate_fail"] and a["irtikaz"]["status"] in ("ready", "near"):
                        sig.append(("orig", a["irtikaz"]["status"], a["irtikaz"]["score"], a["plan"].get("stop"), a["plan"].get("t1")))
            except Exception:
                pass
        if m_abd[i]:
            calls[2] += 1
            try:
                a = ar.analyze(hist, inf, cfg_abd, None)
                mc, fl = (inf or {}).get("marketCap"), (inf or {}).get("floatShares")
                big = (mc and mc > U["max_market_cap"]) or (fl and fl >= 50_000 and fl > U["max_float"])
                if a and not big and a.get("status") in ("ready", "near"):
                    tg = a.get("targets") or []
                    sig.append(("abd", a["status"], a.get("score"), a.get("invalidation"), tg[0] if tg else None))
            except Exception:
                pass
        for src, st, score, stop, t1 in sig:
            if not stop or not t1 or stop >= px or t1 <= px:
                continue
            skip = 1 if e >= t1 else 2 if e <= stop else 0
            a10 = sim_plan(e, Hh, Cc, t1, stop, 10)
            rows.append({"t": sym, "D": int(Dint[i]), "src": src, "status": st, "score": float(score or 0), "praw": float(praw[i]), "skip": skip,
                         "t1c": float(t1 / px - 1), "stc": float(1 - stop / px), "p": float(p_raw[i]), "pb": float(pb[i]),
                         "rsi14": float(rsi[i]) if np.isfinite(rsi[i]) else 50.0,
                         **{"x_" + q: float(Fv[i, j]) for j, q in enumerate(FB)},
                         **{"m_" + q: float(val) for q, val in (mkt.get(day) or {}).items()},
                         "A10": a10, "A20": sim_plan(e, Hh, Cc, t1, stop, 20), "L30": sim_plan(e, Hh, Cc, e * 1.3, stop, 20),
                         "L50": sim_plan(e, Hh, Cc, e * 1.5, stop, 20), "L100": sim_plan(e, Hh, Cc, e * 2.0, stop, 40),
                         "HALF": sim_half(e, Hh, Cc, t1, stop, e * 1.5, 20), "TR": sim_trail(e, Hh, Cc, t1, stop, 40),
                         "H10": Cc[9] / e - 1, "H20": Cc[19] / e - 1, "hit10": float(a10 >= t1 / e - 1 - 1e-9),
                         "mx20": float(Hh[:20].max() / e - 1), "mx40": float(Hh.max() / e - 1)})
    return B, rows, calls


def line(nm, g, plans=("A10", "A20", "L30", "L50", "HALF", "TR")):
    if len(g) < 20:
        print(f"  {nm}: n={len(g)} (قليل)")
        return
    by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in g.groupby("Y").nA10.mean().items())
    by2 = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in g.groupby("Y").nTR.mean().items())
    print(f"  {nm}: n={len(g):,} ({g.D.nunique()} يوم) · هدف1 خلال 10: {pp(g.hit10.mean())} · "
          + " · ".join(f"{p_} {pr(g['n' + p_].mean())}" for p_ in plans)
          + f" · وسيط A10 {pr(g.nA10.median())} · رابحة {pp((g.nA10 > 0).mean())} · t {tday(g, 'nA10'):.1f}/{tday(g, 'nTR'):.1f}"
          + f" · +30%: {pp((g.mx20 >= .3).mean())} · +50%: {pp((g.mx20 >= .5).mean())} · دبّل/40: {pp((g.mx40 >= 1).mean(), 1)} · [A10 {by}] [TR {by2}]")


def main():
    t0 = time.time()
    cfg = scanner.load_cfg()
    cfg_abd = json.loads((ROOT / "abdullah" / "config.json").read_text(encoding="utf-8"))
    U, AB = cfg["universe"], cfg_abd["abdullah"]
    syms = universe()
    iw = A.bars(["IWM"], "1Day", START)["IWM"]
    iw.index = pd.to_datetime(iw.index).tz_localize(None).normalize()
    iw = iw[~iw.index.duplicated(keep="last")]["Close"].astype(float)
    frames, names = {}, sorted(syms)
    for i in range(0, len(names), 100):
        batch = names[i:i + 100]
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
        for s, b in raw.items():
            if len(b) >= 320:
                d = prep_frame(s, b, spl.get(s, []))
                if len(d) >= 320:
                    frames[s] = d
        if (i // 100) % 10 == 0:
            print(f"  جلب {i + 100}/{len(names)} · أسهم {len(frames)} · {round(time.time() - t0)}ث", flush=True)
    keep, tot = {}, np.zeros(4, int)
    for s, d in frames.items():
        kf, praw, rsi, base, m1, m2, m3 = masks2(d, U, AB)
        if base.any():
            keep[s] = int((m1 | m2 | m3).sum())
            tot += [int(base.sum()), int(m1.sum()), int(m2.sum()), int(m3.sum())]
    print(f"أسهم لها أيام بالمجال: {len(keep)} · أيام بالمجال {tot[0]:,} · مرشحين: المطوّر {tot[1]:,} · الأصلي {tot[2]:,} · عبدالله {tot[3]:,} · {round(time.time() - t0)}ث", flush=True)
    closes = pd.DataFrame({s: frames[s]["Close"] for s in keep})
    above = (closes > closes.rolling(50).mean()).astype(float).where(closes.rolling(50).count() >= 50)
    breadth = (above.sum(axis=1) / above.count(axis=1).replace(0, np.nan)).astype(float)
    del closes, above
    iwm_up, iwm_r20 = (iw > iw.rolling(50).mean()).astype(float), (iw / iw.shift(20) - 1) * 100
    mkt = {}
    for day in breadth.index:
        br, up, r20 = breadth.get(day, np.nan), iwm_up.get(day, np.nan), iwm_r20.get(day, np.nan)
        mkt[day] = {"breadth": 0.4 if pd.isna(br) else float(br), "iwm_up": 0.5 if pd.isna(up) else float(up),
                    "iwm_ret20": 0.0 if pd.isna(r20) else float(r20)}
    act = [s for s in keep if syms.get(s) == "active" and keep[s] > 0]
    try:
        infos = scanner.get_info(act, cfg)
    except Exception as e:
        print("ياهو تعطل:", repr(e)[:120], flush=True)
        infos = {}
    print(f"بيانات ياهو: {sum(1 for v in infos.values() if v and v.get('sharesOutstanding'))} من {len(act)} · {round(time.time() - t0)}ث", flush=True)
    jobs = [(s, frames[s], infos.get(s) or {}, cfg, cfg_abd, mkt) for s in keep]
    Bs, rows, calls = [], [], np.zeros(3, int)
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, res in enumerate(ex.map(job, jobs, chunksize=4)):
            if res is None:
                continue
            Bs.append(res[0])
            rows += res[1]
            calls += res[2]
            if i % 300 == 0:
                print(f"  تحليل {i}/{len(jobs)} · إشارات {len(rows):,} · نداءات {calls.tolist()} · {round(time.time() - t0)}ث", flush=True)
    del frames
    B = pd.concat(Bs, ignore_index=True)
    S = pd.DataFrame(rows)
    del Bs, rows
    for g in (B, S):
        g["Y"] = (g.D // 10000).astype(int)
        g["cost"] = cost_of(g.praw.to_numpy(float))
    print(f"\nالأساس: {len(B):,} سهم-يوم · الإشارات ✅/🟡: {len(S):,} — " + " · ".join(f"{SRC_AR[k]} {int((S.src == k).sum()):,} (✅ {int(((S.src == k) & (S.status == 'ready')).sum()):,})" for k in SRC_AR)
          + f" · من {S.D.min()} لين {S.D.max()} · {round(time.time() - t0)}ث")
    print(f"فتح فوق الهدف: {pp((S.skip == 1).mean(), 1)} · تحت الوقف: {pp((S.skip == 2).mean(), 1)} · وسيط الهدف 1: {pr(S.t1c.median())} · وسيط الوقف: −{S.stc.median() * 100:.1f}%")
    S = S[S.skip == 0].copy()
    for p_ in PLANS:
        S["n" + p_] = S[p_] - S.cost
    cal = {int(x): k for k, x in enumerate(iw.index.strftime("%Y%m%d").astype(int))}
    S["ci"] = S.D.map(cal).fillna(-1).astype(int)
    S = S.sort_values(["src", "t", "D"]).reset_index(drop=True)
    prev = S.groupby(["src", "t"]).ci.shift(1)
    S["first"] = (prev.isna() | (S.ci - prev > 10)).to_numpy()

    print("\n==== 1) خط الأساس: أي سهم بالمجال (دخول افتتاح بكرة، بعد التكاليف) ====")
    for nm, g in (("أي سهم بالمجال", B), ("RSI ≤ 40", B[B.rsi <= 40])):
        by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in (g.H10 - g.cost).groupby(g.Y).mean().items())
        print(f"  {nm}: n={len(g):,} · احتفاظ 10: {pr((g.H10 - g.cost).mean())} (وسيط {pr((g.H10 - g.cost).median())}) · 20: {pr((g.H20 - g.cost).mean())}"
              f" · +30% خلال 20: {pp((g.mx20 >= .3).mean())} · +50%: {pp((g.mx20 >= .5).mean())} · دبّل خلال 40: {pp((g.mx40 >= 1).mean(), 1)} · لمس −30%: {pp((g.mn20 <= -.3).mean())} · [{by}]")

    print("\n==== 2) كل رادار لحاله (بعد التكاليف) ====")
    Fi = S[S["first"]]
    for k in SRC_AR:
        line(f"{SRC_AR[k]} — كل الإشارات", S[S.src == k])
        line(f"{SRC_AR[k]} — أول ظهور", Fi[Fi.src == k])
        line(f"{SRC_AR[k]} — أول ظهور ✅", Fi[(Fi.src == k) & (Fi.status == "ready")])
        line(f"{SRC_AR[k]} — أول ظهور 🟡", Fi[(Fi.src == k) & (Fi.status == "near")])
        g = Fi[Fi.src == k]
        if len(g):
            print(f"     توصيات جديدة باليوم (وسيط): {g.groupby('D').size().median() * (ORIG_STEP if k == 'orig' else STEP) / 1:.0f} تقريباً (بعد تعويض العينة)")
    print("  كل الخطط — أول ظهور، كل الرادارات: " + " · ".join(f"{p_} {pr(Fi['n' + p_].mean())} (قبل التكاليف {pr(Fi[p_].mean())})" for p_ in PLANS))

    print("\n==== 3) وش يفرق؟ (أول ظهور، كل الرادارات) ====")
    for nm, m in (("🩸 أقل من 0.10", Fi.pb < .10), ("🩸 0.10-0.20", (Fi.pb >= .10) & (Fi.pb < .20)), ("🩸 0.20-0.30", (Fi.pb >= .20) & (Fi.pb < .30)), ("🩸 0.30-0.45", (Fi.pb >= .30) & (Fi.pb < .45)), ("🩸 0.45+", Fi.pb >= .45),
                  ("🧨 p < 0.005", Fi.p < .005), ("🧨 0.005-0.02", (Fi.p >= .005) & (Fi.p < .02)), ("🧨 0.02-0.0635", (Fi.p >= .02) & (Fi.p < .0635)), ("🧨 0.0635+", Fi.p >= .0635),
                  ("انفجر بسنة", Fi.x_n_ep252 >= 1), ("ما انفجر بسنة", Fi.x_n_ep252 < 1),
                  ("الهدف 1 ≥ 30%", Fi.t1c >= .30), ("الهدف 1 15-30%", (Fi.t1c >= .15) & (Fi.t1c < .30)), ("الهدف 1 < 15%", Fi.t1c < .15),
                  ("السوق فوق متوسطه", Fi.m_iwm_up >= .5), ("السوق تحت متوسطه", Fi.m_iwm_up < .5), ("اتساع ≥ 45%", Fi.m_breadth >= .45), ("اتساع < 30%", Fi.m_breadth < .30),
                  ("تحت 1$", Fi.praw < 1), ("1-3$", (Fi.praw >= 1) & (Fi.praw < 3)), ("3-10$", Fi.praw >= 3),
                  ("فوليوم آخر جلسة ≥ 2×", Fi.x_vol1_20 >= 2), ("فوليوم 5 أيام ناشف < 0.7", Fi.x_vol5_60 < .7)):
        line(nm, Fi[m])

    # ---- 4) نماذج الترتيب خارج العينة (ميزات مشتركة لكل الرادارات)
    S["lp"] = np.log(np.clip(S.p, 1e-6, 1 - 1e-6) / (1 - np.clip(S.p, 1e-6, 1 - 1e-6)))
    S["lpb"] = np.log(np.clip(S.pb, 1e-6, 1 - 1e-6) / (1 - np.clip(S.pb, 1e-6, 1 - 1e-6)))
    for k in SRC_AR:
        S["is_" + k] = (S.src == k).astype(float)
    S["is_ready"] = (S.status == "ready").astype(float)
    FEATS = [c_ for c_ in S.columns if c_.startswith("x_") or c_.startswith("m_")] + ["lp", "lpb", "t1c", "stc", "rsi14", "score", "is_ready", "is_dev", "is_orig"]
    for f_ in FEATS:
        x = S[f_].replace([np.inf, -np.inf], np.nan).astype(float)
        S[f_] = x.fillna(x.median())
    cuts = [20240101, 20240701, 20250101, 20250701, 20260101, 20260701, 99999999]
    TG = {"win": ("logit", lambda g: g.hit10.to_numpy(float)), "ret": ("ridge", lambda g: np.clip(g.nA10.to_numpy(float), -.5, .5)),
          "big": ("logit", lambda g: (g.mx20 >= .5).to_numpy(float)), "b30": ("logit", lambda g: (g.mx20 >= .3).to_numpy(float)),
          "l50": ("ridge", lambda g: np.clip(g.nL50.to_numpy(float), -.6, .5)), "tr": ("ridge", lambda g: np.clip(g.nTR.to_numpy(float), -.6, 1.0))}
    for k_ in TG:
        S["s_" + k_] = np.nan
    print("\n==== 4) نماذج الترتيب خارج العينة ====")
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        te = S[(S.D >= a_) & (S.D < b_)]
        tr = S[(S.D < a_) & (S.ci <= (te.ci.min() if len(te) else 0) - H)]
        if len(tr) < MIN_TRAIN or len(te) < 100:
            print(f"  {a_}: تدريب {len(tr)} · اختبار {len(te)} (قليل — يتخطى)")
            continue
        X = tr[FEATS].to_numpy(float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Z, Zt = np.clip((X - mu) / sd, -5, 5), np.clip((te[FEATS].to_numpy(float) - mu) / sd, -5, 5)
        msg = []
        for k_, (kind, fy) in TG.items():
            y = fy(tr)
            w = fit_logit(Z, y) if kind == "logit" else fit_ridge(Z, y)
            sc = w[0] + Zt @ w[1:]
            S.loc[te.index, "s_" + k_] = sc
            msg.append(f"{k_} AUC {auc(fy(te), sc):.3f}" if kind == "logit" else f"{k_} corr {np.corrcoef(fy(te), sc)[0, 1]:+.3f}")
        print(f"  {a_}: تدريب {len(tr):,} · اختبار {len(te):,} · " + " · ".join(msg), flush=True)
    O = S[S.s_win.notna() & S["first"]].copy()
    # نفس السهم بنفس اليوم من رادارين = توصية وحدة (نخلي الأعلى)
    print(f"\n  خارج العينة «أول ظهور»: {len(O):,} إشارة بـ {O.D.nunique()} يوم ({O.D.min()} ← {O.D.max()}) · وسيط الإشارات باليوم {O.groupby('D').size().median():.0f}")
    line("كل الإشارات (للمقارنة)", O)
    res = {}
    for k_ in TG:
        Ok = O.sort_values("s_" + k_, ascending=False).drop_duplicates(["D", "t"])
        for K in (1, 3):
            g = Ok[Ok.groupby("D")["s_" + k_].rank(ascending=False, method="first") <= K]
            line(f"أعلى {K} باليوم حسب «{k_}»", g)
            res[f"{k_}:{K}"] = {"n": int(len(g)), **{p_: float(g["n" + p_].mean()) for p_ in PLANS}, "hit10": float(g.hit10.mean()),
                                "b30": float((g.mx20 >= .3).mean()), "big": float((g.mx20 >= .5).mean()), "dbl": float((g.mx40 >= 1).mean()),
                                "src": {s_: int(v) for s_, v in g.src.value_counts().items()},
                                "byA10": {int(y): float(v) for y, v in g.groupby("Y").nA10.mean().items()},
                                "byTR": {int(y): float(v) for y, v in g.groupby("Y").nTR.mean().items()}}
        for q in (0.9, 0.95):
            line(f"أعلى {round((1 - q) * 100)}% حسب «{k_}»", Ok[Ok["s_" + k_] >= O["s_" + k_].quantile(q)])
    print("\n==== 5) قواعد بسيطة شفافة (نفس فترة خارج العينة، أول ظهور) ====")
    RU = {"🩸 < 0.15": O.pb < .15, "🩸 < 0.15 + ✅": (O.pb < .15) & (O.status == "ready"), "🩸 < 0.10": O.pb < .10,
          "🩸 < 0.20 + هدف ≥ 20%": (O.pb < .20) & (O.t1c >= .20), "🧨 ≥ 0.02 + 🩸 < 0.30": (O.p >= .02) & (O.pb < .30),
          "انفجر بسنة + 🩸 < 0.30": (O.x_n_ep252 >= 1) & (O.pb < .30), "سوق فوق متوسطه + 🩸 < 0.20": (O.m_iwm_up >= .5) & (O.pb < .20),
          "اتساع ≥ 45% (سوق صاعد للصغيرة)": O.m_breadth >= .45, "رادارين مع بعض بنفس اليوم": O.duplicated(["D", "t"], keep=False)}
    for nm, m in RU.items():
        line(nm, O[m])
    X = S[FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = np.clip((X - mu) / sd, -5, 5)
    fin = {k_: [float(x) for x in (fit_logit(Z, fy(S)) if kind == "logit" else fit_ridge(Z, fy(S)))] for k_, (kind, fy) in TG.items()}
    for k_ in ("win", "big", "tr"):
        print(f"أوزان «{k_}»: " + " · ".join(f"{f_} {w_:+.3f}" for f_, w_ in sorted(zip(FEATS, fin[k_][1:]), key=lambda z: -abs(z[1]))[:14]))
    qs = {k_: [float(x) for x in O["s_" + k_].quantile([.5, .8, .9, .95, .98])] for k_ in TG}
    js = json.dumps({"version": "rd2-" + datetime.now(timezone.utc).strftime("%Y%m%d"), "feats": FEATS, "mean": [float(x) for x in mu], "std": [float(x) for x in sd],
                     "w": fin, "q": qs, "res": res, "n": int(len(S)), "n_oos_first": int(len(O)), "period": [int(S.D.min()), int(S.D.max())],
                     "feats_b": HR.model()["feats_b"]}, ensure_ascii=False)
    print("RD2_BEGIN")
    for k in range(0, len(js), 900):
        print("R2|" + js[k:k + 900])
    print("RD2_END")
    print(f"\nخلص في {round(time.time() - t0)} ثانية", flush=True)


if __name__ == "__main__":
    main()
