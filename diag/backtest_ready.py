"""
🎯 «الجاهزة» — هل نقدر نطلّع من الرادارات قائمة قصيرة تكسب؟ (تشخيص فقط — طلب المستخدم 2026-11-12:
«شوف لي طريقة من الرادارات تكسبني ولاتوصي على مليون سهم أعطني الأسهم الجاهزة»)

الفكرة: إشارات فيصل المطوّر (✅/🟡، نفس الكود الحي) × «حمض الطيارات» 🧨 (احتمال الانفجار) × كاشف النزيف 🩸،
على كل أسهم ناسداك **مع المشطوبة** (بدون انحياز البقاء)، بالسعر الحقيقي ذاك اليوم، والدخول افتتاح اليوم التالي، **بعد التكاليف**.
- لكل إشارة: نتيجة كذا خطة خروج (الهدف 1/الوقف 10 و20 جلسة، بيع +30/+50/+100%، نص ونص، وقف متحرك) + هل دبّل.
- خط الأساس: أي سهم بالمجال بأي يوم (نفس الدخول) + أي سهم RSI ≤ 40.
- نماذج ترتيب «خارج العينة» (كل نص سنة بنموذج تدرّب على اللي قبلها) ← أعلى 1 و3 كل يوم.
المخرجات: جداول + JSON بين RDY_BEGIN/END (أسطر R1|).
"""
import json
import os
import re
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
import scanner
import faisal_rules as fr
import new_low_rule
import compliance_rule
import confidence as C
import highrisk as HR
from backtest_highrisk import ETF_RX, NOT_COMMON_RX

START = datetime(2022, 6, 1, tzinfo=timezone.utc)
STEP, H = 2, 40
MIN_TRAIN = 800
RSI_PRE = 41.0                 # ✅/🟡 لازم RSI ≤ 40 (شرط بالكود) ← فلتر سريع قبل التحليل الثقيل
PLANS = {"A10": "الهدف 1 / وقف الخطة بالإغلاق / 10 جلسات (الحالية)", "A20": "نفسها 20 جلسة",
         "L30": "بيع +30% / وقف الخطة / 20", "L50": "بيع +50% / وقف الخطة / 20", "L100": "بيع +100% / وقف الخطة / 40",
         "HALF": "نص عند الهدف 1 والباقي +50% ووقفه الدخول / 20", "TR": "بعد الهدف 1: وقف متحرك 20% تحت أعلى إغلاق / 40",
         "H10": "احتفاظ 10 جلسات", "H20": "احتفاظ 20 جلسة"}
HRX = ["l_price", "l_dv20", "n_ep252", "l_days_ep", "atr20p", "dd252", "ret5", "ret20", "vol1_20", "vol5_60", "rs_recent", "pos60"]
_sh, _fsh, _roll = HR._sh, HR._fsh, HR._roll


def universe():
    """أسهم ناسداك العادية من Alpaca (نشط + موقوف/مشطوب) — الرادار الحي ناسداك بس."""
    syms = {}
    for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            for st in ("active", "inactive"):
                r = requests.get(f"{base}/v2/assets", headers=A._headers(), params={"status": st, "asset_class": "us_equity"}, timeout=90)
                r.raise_for_status()
                for a in r.json():
                    s, ex, nm = a.get("symbol", ""), a.get("exchange", ""), a.get("name") or ""
                    if ex != "NASDAQ" or not re.fullmatch(r"[A-Z]{1,5}", s):
                        continue
                    if (len(s) == 5 and s[-1] in "WRUZ") or ETF_RX.search(nm) or NOT_COMMON_RX.search(nm):
                        continue
                    syms.setdefault(s, st)
            break
        except Exception as e:
            print("assets:", base, repr(e)[:120], flush=True)
            syms = {}
    print(f"المجال: {len(syms)} سهم ناسداك عادي (نشط {sum(v == 'active' for v in syms.values())})", flush=True)
    return syms


def score_all(F):
    M = HR.model()
    fb, nE = M["feats_b"], len(M["feats"])
    X = F[fb].to_numpy(float)
    mu, sd = np.array(M["mean"]), np.array(M["std"])
    sd = np.where(sd == 0, 1, sd)
    Z = np.clip((np.where(np.isfinite(X), X, mu) - mu) / sd, -5, 5)
    w, wb = np.array(M["w"]), np.array(M["wb"])
    return 1 / (1 + np.exp(-(w[0] + Z[:, :nE] @ w[1:]))), 1 / (1 + np.exp(-(wb[0] + Z @ wb[1:])))


def sim_plan(e, Hh, Cc, t, stop, days):
    for k in range(days):
        if Hh[k] >= t:
            return t / e - 1
        if stop and Cc[k] < stop:
            return Cc[k] / e - 1
    return Cc[days - 1] / e - 1


def sim_half(e, Hh, Cc, t1, stop, t2, days):
    part, cur = None, stop
    for k in range(days):
        if part is None:
            if Hh[k] >= t1:
                part, cur = 0.5 * (t1 / e - 1), e
                if Hh[k] >= t2:
                    return part + 0.5 * (t2 / e - 1)
                continue
            if cur and Cc[k] < cur:
                return Cc[k] / e - 1
        else:
            if Hh[k] >= t2:
                return part + 0.5 * (t2 / e - 1)
            if Cc[k] < cur:
                return part + 0.5 * (Cc[k] / e - 1)
    last = Cc[days - 1] / e - 1
    return part + 0.5 * last if part is not None else last


def sim_trail(e, Hh, Cc, t1, stop, days, tr=0.2):
    armed, hi = False, e
    for k in range(days):
        if not armed and Hh[k] >= t1:
            armed = True
        if armed:
            hi = max(hi, Cc[k])
            if Cc[k] < max(e, hi * (1 - tr)):
                return Cc[k] / e - 1
        elif stop and Cc[k] < stop:
            return Cc[k] / e - 1
    return Cc[days - 1] / e - 1


def prep_frame(sym, bars, spl):
    d = bars[["Open", "High", "Low", "Close", "Volume"]].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d = d[~d.index.duplicated(keep="last")]
    d = d[(d["Close"] > 0) & d["Close"].notna()]
    d["Stock Splits"] = 0.0
    for dt, ratio in spl or []:
        dt = pd.Timestamp(dt)
        dt = (dt.tz_localize(None) if dt.tzinfo else dt).normalize()
        if dt in d.index:
            d.loc[dt, "Stock Splits"] = ratio
    return d


def masks(d, U):
    """الفلتر السريع: أيام بالمجال السعري الحقيقي + فوليوم كافي (+ RSI ≤ 41 للمرشحين)."""
    c, v = d["Close"].to_numpy(float), d["Volume"].to_numpy(float)
    n = len(c)
    idx = np.arange(n)
    kf = HR.raw_factor(d)
    praw = c * kf
    v30 = pd.Series(v / kf).rolling(30, min_periods=20).mean().to_numpy()
    rsi = fr.rsi(d["Close"]).to_numpy(float)
    base = (idx >= 260) & (idx <= n - H - 2) & (idx % STEP == 0) & (praw >= U["min_price"]) & (praw <= U["max_price"]) & \
           (np.nan_to_num(v30) >= U["min_avg_volume"] * 0.7)
    return kf, praw, rsi, base, base & (np.nan_to_num(rsi, nan=99) <= RSI_PRE)


def job(args):
    sym, d, info, cfg, mkt = args
    U = cfg["universe"]
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    kf, praw, rsi, base, cand = masks(d, U)
    if not base.any():
        return None
    F = HR.features(d)
    p_raw, pb = score_all(F)
    Dint = d.index.strftime("%Y%m%d").astype(int).to_numpy(np.int32)
    o1 = _fsh(o, 1)
    # ---- خط الأساس: أي سهم بالمجال (دخول افتتاح بكرة)
    mx20 = _fsh(_roll(h, 20, "max"), 20) / o1 - 1
    mx40 = _fsh(_roll(h, 40, "max"), 40) / o1 - 1
    mn20 = _fsh(_roll(l, 20, "min"), 20) / o1 - 1

    def generic(T, S, days):
        out, done = np.full(n, np.nan), ~np.isfinite(o1)
        for k in range(1, days + 1):
            hk, ck = _fsh(h, k), _fsh(c, k)
            hit = ~done & (hk >= o1 * (1 + T))
            out[hit] = T
            done |= hit
            st = ~done & (ck <= o1 * (1 - S))
            out[st] = ck[st] / o1[st] - 1
            done |= st
            if k == days:
                rest = ~done & np.isfinite(ck)
                out[rest] = ck[rest] / o1[rest] - 1
        return out

    B = pd.DataFrame({"D": Dint, "praw": praw, "rsi": rsi, "atr": F["atr20p"].to_numpy(float), "p": p_raw, "pb": pb,
                      "H10": _fsh(c, 10) / o1 - 1, "H20": _fsh(c, 20) / o1 - 1, "G30": generic(0.3, 0.15, 20), "G50": generic(0.5, 0.2, 20),
                      "mx20": mx20, "mx40": mx40, "mn20": mn20})[base].astype(np.float32)
    B["D"] = Dint[base]
    # ---- إشارات فيصل (التحليل الثقيل بس للمرشحين)
    rows = []
    for i in np.flatnonzero(cand):
        end, k = i + 1, kf[i]
        hist = d.iloc[max(0, end - 500):end]
        if k != 1:
            hist = hist.copy()
            hist[["Open", "High", "Low", "Close"]] *= k          # الأسعار مثل ما كانت ذاك اليوم (قبل التقسيمات اللي بعده)
            hist["Volume"] /= k
        inf = info
        if info:
            inf = dict(info)
            so, fl = info.get("sharesOutstanding"), info.get("floatShares")
            if so:
                inf["marketCap"] = so * c[i]                        # تقريب القيمة السوقية ذاك اليوم (بدون أثر الطرح)
            if fl:
                inf["floatShares"] = fl / k
        try:
            a = fr.analyze(hist, inf, cfg)
            if not a:
                continue
            new_low_rule.apply(a, hist)
            compliance_rule.apply_faisal(a, hist)
        except Exception:
            continue
        st = a["irtikaz"]["status"]
        if a["gate_fail"] or st not in ("ready", "near"):
            continue
        day = hist.index[-1]
        try:
            f = C.features(a, hist, None, [], [], mkt.get(day))
        except Exception:
            continue
        p = a["plan"]
        stop, t1 = p.get("stop"), p.get("t1")
        e = o[i + 1] * k
        if not (np.isfinite(e) and e > 0) or not t1 or not stop:
            continue
        Hh, Cc = h[i + 1:i + 1 + H] * k, c[i + 1:i + 1 + H] * k
        if len(Cc) < H:
            continue
        skip = 1 if e >= t1 else 2 if e <= stop else 0          # فتح فوق الهدف / تحت الوقف ← ما فيه صفقة
        y = {"A10": sim_plan(e, Hh, Cc, t1, stop, 10), "A20": sim_plan(e, Hh, Cc, t1, stop, 20),
             "L30": sim_plan(e, Hh, Cc, e * 1.3, stop, 20), "L50": sim_plan(e, Hh, Cc, e * 1.5, stop, 20),
             "L100": sim_plan(e, Hh, Cc, e * 2.0, stop, 40), "HALF": sim_half(e, Hh, Cc, t1, stop, e * 1.5, 20),
             "TR": sim_trail(e, Hh, Cc, t1, stop, 40), "H10": Cc[9] / e - 1, "H20": Cc[19] / e - 1,
             "hit10": float(sim_plan(e, Hh, Cc, t1, stop, 10) >= t1 / e - 1 - 1e-9),
             "mx20": float(Hh[:20].max() / e - 1), "mx40": float(Hh.max() / e - 1)}
        rows.append({"t": sym, "D": int(Dint[i]), "status": st, "praw": float(praw[i]), "skip": skip, "gap": float(e / (c[i] * k) - 1),
                     "t1p": float(t1 / e - 1), "stp": float(1 - stop / e), "p": float(p_raw[i]), "pb": float(pb[i]),
                     **{"x_" + q: float(F[q].iat[i]) for q in HRX}, **y, **f})
    return B, rows


def fit_logit(X, y, lam=3.0, iters=40):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        step = np.linalg.solve((Xb * (p * (1 - p))[:, None]).T @ Xb + R, Xb.T @ (y - p) - R @ w)
        w += step
        if np.abs(step).max() < 1e-7:
            break
    return w


def fit_ridge(X, y, lam=30.0):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    R = lam * np.eye(Xb.shape[1])
    R[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)


def auc(y, s):
    y = np.asarray(y)
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def pr(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=0):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def cost_of(px):
    return np.where(px < 1, 0.04, np.where(px < 5, 0.02, 0.01))


def tday(g, col):
    """t-stat على متوسطات الأيام (الصفقات بنفس اليوم مترابطة)."""
    m = g.groupby("D")[col].mean()
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)) + 1e-12)) if len(m) > 5 else float("nan")


def line(nm, g, plans=("A10", "A20", "L50", "HALF", "TR")):
    if len(g) < 20:
        print(f"  {nm}: n={len(g)} (قليل)")
        return
    by = " ".join(f"{int(y) % 100}:{v * 100:+.1f}" for y, v in g.groupby("Y").nA10.mean().items())
    print(f"  {nm}: n={len(g):,} ({g.D.nunique()} يوم) · وصل الهدف 1 خلال 10: {pp(g.hit10.mean())} · "
          + " · ".join(f"{p_} {pr(g['n' + p_].mean())}" for p_ in plans)
          + f" · وسيط A10 {pr(g.nA10.median())} · رابحة {pp((g.nA10 > 0).mean())} · t(A10) {tday(g, 'nA10'):.1f}"
          + f" · +50% خلال 20: {pp((g.mx20 >= .5).mean())} · دبّل خلال 40: {pp((g.mx40 >= 1).mean(), 1)} · [A10 بالسنة {by}]")


def main():
    t0 = time.time()
    cfg = scanner.load_cfg()
    U = cfg["universe"]
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
            if len(b) < 320:
                continue
            d = prep_frame(s, b, spl.get(s, []))
            if len(d) >= 320:
                frames[s] = d
        if (i // 100) % 10 == 0:
            print(f"  جلب {i + 100}/{len(names)} · أسهم {len(frames)} · {round(time.time() - t0)}ث", flush=True)
    # الفلتر السريع بالعملية الرئيسية: نعرف مين عنده مرشحين (عشان بيانات ياهو لهم بس) + اتساع السوق
    ncand, nbase, keep = 0, 0, {}
    for s, d in frames.items():
        kf, praw, rsi, base, cand = masks(d, U)
        if base.any():
            keep[s] = int(cand.sum())
            ncand += int(cand.sum())
            nbase += int(base.sum())
    print(f"أسهم لها أيام بالمجال: {len(keep)} · أيام بالمجال (كل جلستين) {nbase:,} · مرشحين RSI≤{RSI_PRE:.0f}: {ncand:,} · {round(time.time() - t0)}ث", flush=True)
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
    jobs = [(s, frames[s], infos.get(s) or {}, cfg, mkt) for s in keep]
    Bs, rows = [], []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, res in enumerate(ex.map(job, jobs, chunksize=4)):
            if res is None:
                continue
            Bs.append(res[0])
            rows += res[1]
            if i % 300 == 0:
                print(f"  تحليل {i}/{len(jobs)} · إشارات {len(rows):,} · {round(time.time() - t0)}ث", flush=True)
    del frames
    B = pd.concat(Bs, ignore_index=True)
    S = pd.DataFrame(rows)
    del Bs, rows
    for g in (B, S):
        g["Y"] = (g.D // 10000).astype(int)
        g["cost"] = cost_of(g.praw.to_numpy(float))
    print(f"\nالأساس: {len(B):,} سهم-يوم بالمجال · إشارات فيصل ✅/🟡: {len(S):,} (✅ {int((S.status == 'ready').sum()):,}) · من {S.D.min()} لين {S.D.max()} · {round(time.time() - t0)}ث")
    print(f"فتح فوق الهدف (ما فيه صفقة): {pp((S.skip == 1).mean(), 1)} · فتح تحت الوقف: {pp((S.skip == 2).mean(), 1)} · وسيط الهدف 1: {pr(S.t1p.median())} · وسيط الوقف: −{S.stp.median() * 100:.1f}%"
          f" · وسيط السعر {S.praw.median():.2f}$ · وسيط التكلفة {S.cost.median() * 100:.0f}%")
    S = S[S.skip == 0].copy()
    for p_ in PLANS:
        S["n" + p_] = S[p_] - S.cost
    cal = {int(x): k for k, x in enumerate(iw.index.strftime("%Y%m%d").astype(int))}
    S["ci"] = S.D.map(cal).fillna(-1).astype(int)
    S = S.sort_values(["t", "D"]).reset_index(drop=True)
    prev = S.groupby("t").ci.shift(1)
    S["first"] = (prev.isna() | (S.ci - prev > 10)).to_numpy()       # أول ظهور (ما طلع بآخر 10 جلسات) = مثل توصية السجل

    print("\n==== 1) خط الأساس: أي سهم بالمجال (دخول افتتاح بكرة، بعد التكاليف) ====")
    for nm, g in (("أي سهم بالمجال", B), ("RSI ≤ 40", B[B.rsi <= 40]), ("RSI ≤ 30", B[B.rsi <= 30])):
        by = " ".join(f"{int(y) % 100}:{(v - c_) * 100:+.1f}" for (y, v), c_ in zip(g.groupby("Y").H10.mean().items(), g.groupby("Y").cost.mean()))
        print(f"  {nm}: n={len(g):,} · احتفاظ 10: {pr((g.H10 - g.cost).mean())} (وسيط {pr((g.H10 - g.cost).median())}) · 20: {pr((g.H20 - g.cost).mean())}"
              f" · +30%/−15%/20: {pr((g.G30 - g.cost).mean())} · +50%/−20%/20: {pr((g.G50 - g.cost).mean())} · +50% خلال 20: {pp((g.mx20 >= .5).mean())}"
              f" · دبّل خلال 40: {pp((g.mx40 >= 1).mean(), 1)} · لمس −30% خلال 20: {pp((g.mn20 <= -.3).mean())} · [احتفاظ 10 بالسنة {by}]")

    print("\n==== 2) إشارات فيصل المطوّر (بعد التكاليف) — كل الإشارات ثم «أول ظهور» ====")
    line("كل ✅+🟡", S)
    Fi = S[S["first"]]
    line("أول ظهور ✅+🟡", Fi)
    line("أول ظهور ✅ مطابق", Fi[Fi.status == "ready"])
    line("أول ظهور 🟡 قريب", Fi[Fi.status == "near"])
    print("  عدد «أول ظهور» باليوم: وسيط", int(Fi.groupby("D").size().median()), "· أيام فيها ✅:", Fi[Fi.status == "ready"].D.nunique(), "من", S.D.nunique())
    print("  كل الخطط (أول ظهور ✅+🟡): " + " · ".join(f"{p_} {pr(Fi['n' + p_].mean())} (قبل التكاليف {pr(Fi[p_].mean())})" for p_ in PLANS))

    print("\n==== 3) هل 🩸 كاشف النزيف و🧨 حمض الطيارات يفرقون؟ (أول ظهور ✅+🟡) ====")
    for nm, m in (("🩸 أقل من 0.10", Fi.pb < .10), ("🩸 0.10-0.20", (Fi.pb >= .10) & (Fi.pb < .20)), ("🩸 0.20-0.30", (Fi.pb >= .20) & (Fi.pb < .30)), ("🩸 0.30+", Fi.pb >= .30)):
        line(nm, Fi[m])
    for nm, m in (("🧨 p أقل من 0.005", Fi.p < .005), ("🧨 0.005-0.02", (Fi.p >= .005) & (Fi.p < .02)), ("🧨 0.02-0.06", (Fi.p >= .02) & (Fi.p < .0635)), ("🧨 0.0635+ (مرشح قوي)", Fi.p >= .0635)):
        line(nm, Fi[m])
    for nm, m in (("انفجر بسنة (مرة+)", Fi.x_n_ep252 >= 1), ("ما انفجر بسنة", Fi.x_n_ep252 < 1), ("الهدف 1 بعيد ≥ 25%", Fi.t1_pct >= 25), ("الهدف 1 قريب < 12%", Fi.t1_pct < 12),
                  ("السوق فوق متوسطه (IWM)", Fi.iwm_up >= .5), ("السوق تحت متوسطه", Fi.iwm_up < .5), ("اتساع ≥ 45%", Fi.breadth >= .45), ("اتساع < 30%", Fi.breadth < .30),
                  ("تحت 1$", Fi.praw < 1), ("1-3$", (Fi.praw >= 1) & (Fi.praw < 3)), ("3-10$", Fi.praw >= 3)):
        line(nm, Fi[m])

    # ---- 4) نماذج الترتيب خارج العينة: كل نص سنة بنموذج تدرّب على اللي قبلها
    # الميزات = اللي تنعرف وقت الإشارة (بعد الإغلاق) بس: بدون الفجوة وسعر الدخول (ينعرفون عند الافتتاح)
    drop = {"t", "D", "status", "praw", "skip", "Y", "cost", "ci", "first", "hit10", "mx20", "mx40", "p", "pb", "gap", "t1p", "stp",
            *PLANS, *("n" + p_ for p_ in PLANS)}
    FEATS = [c_ for c_ in S.columns if c_ not in drop]
    S["lp"], S["lpb"] = np.log(np.clip(S.p, 1e-6, 1) / (1 - np.clip(S.p, 1e-6, 1 - 1e-6))), np.log(np.clip(S.pb, 1e-6, 1) / (1 - np.clip(S.pb, 1e-6, 1 - 1e-6)))
    FEATS = FEATS + ["lp", "lpb"]
    for f_ in FEATS:
        x = S[f_].replace([np.inf, -np.inf], np.nan).astype(float)
        S[f_] = x.fillna(x.median())
    cuts = [20240101, 20240701, 20250101, 20250701, 20260101, 20260701, 99999999]
    TG = {"win": ("logit", lambda g: g.hit10.to_numpy(float)), "ret": ("ridge", lambda g: np.clip(g.nA10.to_numpy(float), -.5, .5)),
          "big": ("logit", lambda g: (g.mx20 >= .5).to_numpy(float)), "l50": ("ridge", lambda g: np.clip(g.nL50.to_numpy(float), -.6, .5)),
          "tr": ("ridge", lambda g: np.clip(g.nTR.to_numpy(float), -.6, 1.0))}
    for k_ in TG:
        S["s_" + k_] = np.nan
    print("\n==== 4) نماذج الترتيب خارج العينة ====")
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        te = S[(S.D >= a_) & (S.D < b_)]
        tr = S[(S.D < a_) & (S.ci <= (te.ci.min() if len(te) else 0) - H)]     # نتيجة التدريب لازم تكون خلصت قبل بداية الاختبار
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
            if kind == "logit":
                msg.append(f"{k_} AUC {auc(fy(te), sc):.3f}")
            else:
                msg.append(f"{k_} corr {np.corrcoef(fy(te), sc)[0, 1]:+.3f}")
        print(f"  {a_}: تدريب {len(tr):,} · اختبار {len(te):,} · " + " · ".join(msg), flush=True)
    O = S[S.s_win.notna() & S["first"]].copy()
    print(f"\n  خارج العينة «أول ظهور»: {len(O):,} إشارة بـ {O.D.nunique()} يوم ({O.D.min()} ← {O.D.max()})")
    line("كل الإشارات (للمقارنة)", O)
    res = {}
    for k_ in TG:
        for K in (1, 3):
            g = O[O.groupby("D")["s_" + k_].rank(ascending=False, method="first") <= K]
            line(f"أعلى {K} باليوم حسب «{k_}»", g, plans=("A10", "A20", "L30", "L50", "HALF", "TR"))
            res[f"{k_}:{K}"] = {"n": int(len(g)), **{p_: float(g["n" + p_].mean()) for p_ in PLANS}, "hit10": float(g.hit10.mean()),
                                "big": float((g.mx20 >= .5).mean()), "dbl": float((g.mx40 >= 1).mean()),
                                "by": {int(y): float(v) for y, v in g.groupby("Y").nA10.mean().items()}}
        for q in (0.8, 0.9):
            thr = O["s_" + k_].quantile(q)
            line(f"أعلى {round((1 - q) * 100)}% حسب «{k_}»", O[O["s_" + k_] >= thr], plans=("A10", "A20", "L30", "L50", "HALF", "TR"))
    # قواعد بسيطة شفافة (بدون نموذج) — على نفس فترة خارج العينة
    print("\n==== 5) قواعد بسيطة شفافة (نفس الفترة، أول ظهور) ====")
    R = {"✅ + 🩸<0.20": (O.status == "ready") & (O.pb < .20), "✅ + 🩸<0.15": (O.status == "ready") & (O.pb < .15),
         "✅/🟡 + 🩸<0.15": O.pb < .15, "✅/🟡 + 🩸<0.10": O.pb < .10,
         "✅ + هدف ≥ 15%": (O.status == "ready") & (O.t1_pct >= 15), "✅ + 🩸<0.20 + هدف ≥ 15%": (O.status == "ready") & (O.pb < .20) & (O.t1_pct >= 15),
         "✅/🟡 + انفجر بسنة + 🩸<0.25": (O.x_n_ep252 >= 1) & (O.pb < .25), "✅/🟡 + 🧨≥0.02": O.p >= .02,
         "✅/🟡 + 🧨≥0.02 + 🩸<0.25": (O.p >= .02) & (O.pb < .25), "✅/🟡 + سوق فوق متوسطه": O.iwm_up >= .5,
         "✅/🟡 + سوق فوق متوسطه + 🩸<0.20": (O.iwm_up >= .5) & (O.pb < .20)}
    for nm, m in R.items():
        line(nm, O[m], plans=("A10", "A20", "L30", "L50", "HALF", "TR"))
    # النموذج النهائي (كل البيانات) للرادار الحي
    X = S[FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = np.clip((X - mu) / sd, -5, 5)
    fin = {k_: [float(x) for x in (fit_logit(Z, fy(S)) if kind == "logit" else fit_ridge(Z, fy(S)))] for k_, (kind, fy) in TG.items()}
    qs = {k_: [float(x) for x in O["s_" + k_].quantile([.5, .8, .9, .95])] for k_ in TG}
    js = json.dumps({"version": "rdy1-" + datetime.now(timezone.utc).strftime("%Y%m%d"), "feats": FEATS, "mean": [float(x) for x in mu], "std": [float(x) for x in sd],
                     "w": fin, "q": qs, "res": res, "n": int(len(S)), "n_oos_first": int(len(O)), "period": [int(S.D.min()), int(S.D.max())]}, ensure_ascii=False)
    print("RDY_BEGIN")
    for k in range(0, len(js), 900):
        print("R1|" + js[k:k + 900])
    print("RDY_END")
    print(f"\nخلص في {round(time.time() - t0)} ثانية", flush=True)


if __name__ == "__main__":
    main()
