"""
🚀 صيد الطيارات — الجولة 2: طرق الوقف والخروج + مقارنة بعشوائي بنفس المسافات بالضبط (تشخيص فقط).
(نفس شروط الجولة 1، والأسهم الجديدة الإدراج تنحسب من 100 جلسة بدل 260 — مثل QMMM، و V6 = V4 + طار مرتين+ خلال سنة)

الفكرة (تغريدة @xzdixz): سهم انهار ← كوّن قاع ← جلس فترة يكوّن دعوم أعلى من بعض (قيعان متصاعدة) ويختبر المقاومة،
و RSI يسوي قاع أعلى من قاعه (دايفرجنس إيجابي) ← يجي خبر/مضارب ويطير.

- المجال: كل أسهم Alpaca (النشطة + الموقوفة/المشطوبة عشان ما نغش بالناجين)، السعر 0.5-15$،
  سيولة يومية ≥ 50 ألف$ وفوليوم ≥ 20 ألف (أخف من فلتر الرادارات عشان يدخل مثل OMH: فوليومه ~67 ألف).
- كل سهم كل جلستين نحسب الشروط (بالبيانات لين ذاك اليوم بس):
  V0 منهار ويقعد: أعلى سعر في السنة قبل القاع ≥ 3 أضعاف القاع، القاع (أدنى سعر 120 جلسة) عمره ≥ 15 جلسة، السعر 1.05-2 من القاع
  V1 = V0 + قاع أعلى (أدنى سعر آخر 10 جلسات ≥ القاع +8% وبعده بـ 10 جلسات+) + RSI عنده أعلى من RSI القاع بـ 5+
  V2 = V1 + دعوم متصاعدة (أدنى سعر بكل 10 جلسات آخر 30 جلسة ما ينزل)
  V3 = V1 + سبق طار (يوم +50% خلال سنة)        V4 = V2 + سبق طار        V5 = V4 + اختبر المقاومة مرتين
- النتيجة من الإغلاق: أعلى سعر خلال 20/40/60 جلسة (+50% / +100% / +200%)، والوقف = إغلاق تحت القاع −10%.
- المرجع: صفقات عشوائية بنفس شريحة التذبذب والسعر (والطيارات نادرة، فالمقارنة هي الحكم).
- التغطية: من كل الانفجارات (أول يوم +50% بعد 30 جلسة هادية)، كم واحد كان الشرط شغال قبله بـ 10 جلسات؟
"""
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import requests
import alpaca_data as A
import scanner

DAYS = 1180
STEP = 2
RAND_P = 0.06
VARS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
VAR_AR = {"V0": "منهار ويقعد (أساس)", "V1": "+ قاع أعلى + RSI أعلى", "V2": "+ دعوم متصاعدة",
          "V3": "V1 + سبق طار", "V4": "V2 + سبق طار", "V5": "V4 + اختبر المقاومة مرتين",
          "V6": "V4 + طار مرتين+ خلال سنة"}
PLANS = {"P1": ("وقف: إغلاق تحت القاع −10% · هدف +100% · 60 جلسة", "B", 2.0, 60),
         "P2": ("وقف: إغلاق تحت آخر قاع أعلى −5% · هدف +100% · 60 جلسة", "R", 2.0, 60),
         "P3": ("وقف: تحت آخر قاع أعلى −5% · هدف +50% · 60 جلسة", "R", 1.5, 60),
         "P4": ("وقف: تحت آخر قاع أعلى −5% · نص عند +50% ونص عند +150% · 60", "R", None, 60),
         "P5": ("وقف: تحت آخر قاع أعلى −5% · هدف +100% · 30 جلسة", "R", 2.0, 30)}
UG = np.round(np.arange(1.01, 3.0001, 0.01), 4)       # الهدف +1% .. +200%
SG = np.round(1 - np.arange(0.01, 0.6001, 0.01), 4)   # الوقف −1% .. −60%
ETF_RX = re.compile(r"\b(ETF|ETN|Fund|Trust|ProShares|Direxion|iShares|Leveraged|Daily|2X|3X|Bull|Bear|Ultra|Notes|"
                    r"Warrant|Rights?|Units?|Acquisition|SPDR|Invesco|Vanguard|Global X|MicroSectors|GraniteShares|"
                    r"Tradr|Defiance|T-Rex|YieldMax|Roundhill|Volatility|VIX)\b", re.I)


def universe():
    syms, src = {}, "alpaca"
    for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            for st in ("active", "inactive"):
                r = requests.get(f"{base}/v2/assets", headers=A._headers(),
                                 params={"status": st, "asset_class": "us_equity"}, timeout=90)
                r.raise_for_status()
                for a in r.json():
                    s, ex, nm = a.get("symbol", ""), a.get("exchange", ""), a.get("name") or ""
                    if ex not in ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"):
                        continue
                    if not re.fullmatch(r"[A-Z]{1,5}", s) or (len(s) == 5 and s[-1] in "WRU") or ETF_RX.search(nm):
                        continue
                    syms[s] = st
            break
        except Exception as e:
            print("assets:", base, repr(e)[:120], flush=True)
            syms = {}
    if not syms:
        src = "yahoo"
        syms = {s: "active" for s in scanner.get_universe(scanner.load_cfg())}
    for s in ("OMH", "QMMM"):
        syms.setdefault(s, "extra")
    act = sum(v == "active" for v in syms.values())
    print(f"المجال ({src}): {len(syms)} سهم (نشط {act} · موقوف/مشطوب {len(syms) - act})", flush=True)
    return sorted(syms)


def rsi_np(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def outcome(h, c, t, stop, H=60):
    """أعلى ربح خلال 20/40/60 + هل وصل +100% قبل الإغلاق تحت الوقف (60 جلسة) + عائد خطة بسيطة."""
    e = c[t]
    fh, fc = h[t + 1:t + 1 + H], c[t + 1:t + 1 + H]
    av = len(fh)
    res = {"avail": av}
    for k in (20, 40, 60):
        res[f"mfe{k}"] = float(fh[:k].max() / e - 1) if av >= k else np.nan
    hit, ret, kday = 0, np.nan, av
    for i in range(av):
        if fh[i] >= 2 * e:
            hit, ret, kday = 1, 1.0, i + 1
            break
        if stop and fc[i] < stop:
            hit, ret, kday = -1, float(fc[i] / e - 1), i + 1
            break
    if hit == 0 and av >= H:
        ret = float(fc[H - 1] / e - 1)
    res.update({"x100": hit, "ret100": ret, "days100": kday})
    return res


def sim(h, c, t, stop, tgt, H):
    """الهدف يُفحص أول (لمس)، والوقف بإغلاق اليوم تحته — نفس قواعد السجل."""
    e = c[t]
    fh, fc = h[t + 1:t + 1 + H], c[t + 1:t + 1 + H]
    for i in range(len(fh)):
        if fh[i] >= tgt * e:
            return 1, tgt - 1
        if fc[i] < stop:
            return -1, float(fc[i] / e - 1)
    if len(fh) >= H:
        return 0, float(fc[H - 1] / e - 1)
    return None, None


def plans(h, c, t, B, R):
    out = {}
    for k, (_, st, tgt, H) in PLANS.items():
        stop = B * 0.9 if st == "B" else R * 0.95
        out[f"{k}_s"] = 1 - stop / c[t]
        if tgt is None:
            a, ra = sim(h, c, t, stop, 1.5, H)
            b2, rb = sim(h, c, t, stop, 2.5, H)
            out[f"{k}_x"], out[f"{k}_r"] = (None, None) if ra is None or rb is None else (max(a, b2), 0.5 * ra + 0.5 * rb)
        else:
            out[f"{k}_x"], out[f"{k}_r"] = sim(h, c, t, stop, tgt, H)
    return out


def process(job):
    sym, d = job
    if d is None or len(d) < 320:
        return [], [], [], 0
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    dates = d.index
    n = len(c)
    r = rsi_np(c)
    jump = np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)]
    fr = np.cumsum(jump >= 1.5)
    rng = (h - l) / np.maximum(c, 1e-9)
    atr20 = pd.Series(rng).rolling(20).mean().values * 100
    dv20 = pd.Series(c * v).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    rnd = random.Random(int.from_bytes(sym.encode(), "little") % 10_000_019)
    rows, rands, flags, n_eval = [], [], {}, 0
    for t in range(100, n - 1, STEP):
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000):
            continue
        n_eval += 1
        if rnd.random() < RAND_P:
            x = {"t": sym, "date": str(dates[t].date()), "ti": t, "px": px, "atr20": atr20[t], "dv20": dv20[t],
                 **outcome(h, c, t, None)}
            if n - 1 - t >= 60:
                x["ph"] = np.round(h[t + 1:t + 61] / px, 4)
                x["pc"] = np.round(c[t + 1:t + 61] / px, 4)
            rands.append(x)
        w0 = max(0, t - 119)
        seg = l[w0:t + 1]
        b = t - int(np.argmin(seg[::-1]))           # آخر مرة لمس فيها القاع
        age = t - b
        if age < 15:
            continue
        pre0 = max(0, b - 250)
        if b - pre0 < 20:
            continue
        B = l[b]
        crash = float(h[pre0:b].max() / B)
        lift = px / B
        if crash < 3 or not (1.05 <= lift <= 2.0):
            continue
        j = t - 9 + int(np.argmin(l[t - 9:t + 1]))
        R = l[j]
        hl = R >= 1.08 * B and j - b >= 10
        div = r[j] >= r[b] + 5
        v1 = bool(hl and div)
        if t - 29 > b:
            m1, m2, m3 = l[t - 29:t - 19].min(), l[t - 19:t - 9].min(), l[t - 9:t + 1].min()
            rising = m2 >= m1 * 0.99 and m3 >= m2 * 0.99 and m3 >= m1 * 1.03
        else:
            rising = False
        v2 = v1 and rising
        frn = fr[t] - fr[max(0, t - 250)] > 0
        serial = fr[t] - fr[max(0, t - 250)] >= 2
        top = h[b:t + 1].max()
        near = np.where(h[b:t + 1] >= 0.93 * top)[0]
        touches = 0 if len(near) == 0 else 1 + int((np.diff(near) >= 5).sum())
        f = {"V0": True, "V1": v1, "V2": v2, "V3": v1 and frn, "V4": v2 and frn, "V5": v2 and frn and touches >= 2,
             "V6": v2 and serial}
        flags[t] = f
        rows.append({"t": sym, "date": str(dates[t].date()), "px": px, "atr20": atr20[t], "dv20": dv20[t], "v20": v20[t],
                     "age": age, "crash": crash, "lift": lift, "off_top": px / top, "rsi": r[t], "rsi_b": r[b], "rsi_j": r[j],
                     "hl_pct": R / B - 1, "rising": rising, "fr": bool(frn), "touches": touches,
                     **{k: bool(x) for k, x in f.items()}, **outcome(h, c, t, B * 0.9), **plans(h, c, t, B, R)})
    # للمقارنة: هل الشرط كان شغال خلال 10 جلسات قبل يوم عشوائي؟ (نسبة «الصدفة» للتغطية)
    for x in rands:
        ti = x.pop("ti")
        for k in VARS:
            x["act_" + k] = any(flags.get(t, {}).get(k) for t in range(ti - 10, ti))
    # الانفجارات: أول يوم +50% بعد 30 جلسة هادية (السعر قبله 0.5-15 وسيولة)
    exps = []
    for i in range(270, n):
        if jump[i] >= 1.5 and fr[i - 1] - fr[i - 31] == 0 and 0.5 <= c[i - 1] <= 15 and dv20[i - 1] >= 50_000:
            act = {k: any(flags.get(t, {}).get(k) for t in range(i - 10, i)) for k in VARS}
            exps.append({"t": sym, "date": str(dates[i].date()), "prev": c[i - 1], "jump": float(jump[i]),
                         "max20": float(h[i:i + 20].max() / c[i - 1] - 1), **act})
    return rows, rands, exps, n_eval


def collect():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    R, Rr, E, NE = [], [], [], 0
    t0 = time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            batch = syms[i:i + 100]
            try:
                raw = A.bars(batch, "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:120], flush=True)
                continue
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")]))
            for rows, rands, exps, ne in ex.map(process, jobs, chunksize=4):
                R += rows
                Rr += rands
                E += exps
                NE += ne
            if (i // 100) % 10 == 0:
                print(f"  {i + len(batch)}/{len(syms)} · إشارات {len(R)} · عشوائي {len(Rr)} · انفجارات {len(E)} · {round(time.time() - t0)}ث", flush=True)
    print(f"أيام-أسهم مفحوصة (بعد فلتر السعر والسيولة): {NE:,}", flush=True)
    return pd.DataFrame(R), pd.DataFrame(Rr), pd.DataFrame(E)


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.1f}%"


def first_only(g, gap=20):
    g = g.sort_values(["t", "date"])
    keep, last = [], {}
    for idx, s_, dt in zip(g.index, g.t.values, pd.to_datetime(g.date).values):
        if s_ not in last or (dt - last[s_]) > np.timedelta64(gap * 7 // 5, "D"):
            keep.append(idx)
        last[s_] = dt
    return g.loc[keep]


def geo(PH, PC, H):
    """جداول (خطوة 1%) لنسبة الوصول ومتوسط العائد لصفقات عشوائية بكل هدف ووقف — نفس قواعد الخطط."""
    ph, pc = PH[:, :H], PC[:, :H]
    n = len(ph)
    M = np.maximum.accumulate(ph, axis=1)
    mprev = np.minimum.accumulate(np.hstack([np.full((n, 1), np.inf), pc[:, :-1]]), axis=1)
    HIT = np.zeros((len(UG), len(SG)))
    RET = np.zeros_like(HIT)
    for js, sv in enumerate(SG):
        ok = mprev >= sv
        Us = np.max(np.where(ok, M, 0), axis=1)
        stop_mat = pc < sv
        k = np.where(stop_mat.any(1), stop_mat.argmax(1), H)
        nohit = np.where(k < H, pc[np.arange(n), np.minimum(k, H - 1)], pc[:, -1]) - 1
        hit = Us[None, :] >= UG[:, None]
        HIT[:, js] = hit.mean(1)
        RET[:, js] = np.where(hit, (UG - 1)[:, None], nohit[None, :]).mean(1)
    return HIT, RET


def look(T, u, s):
    iu = np.clip((u - UG[0]) / 0.01, 0, len(UG) - 1.0001)
    is_ = np.clip((SG[0] - (1 - s)) / 0.01, 0, len(SG) - 1.0001)
    a, b = iu.astype(int), is_.astype(int)
    fa, fb = iu - a, is_ - b
    return T[a, b] * (1 - fa) * (1 - fb) + T[a + 1, b] * fa * (1 - fb) + T[a, b + 1] * (1 - fa) * fb + T[a + 1, b + 1] * fa * fb


def main():
    S, RND, E = collect()
    os.makedirs("diag_out", exist_ok=True)
    keep = [c for c in RND.columns if c not in ("ph", "pc")]
    S.to_csv("diag_out/runners2_setups.csv.gz", index=False, compression="gzip")
    RND[keep].to_csv("diag_out/runners2_random.csv.gz", index=False, compression="gzip")
    E.to_csv("diag_out/runners2_explosions.csv.gz", index=False, compression="gzip")
    print(f"\nإشارات V0: {len(S)} · عشوائي {len(RND)} · انفجارات {len(E)} · أسهم {S.t.nunique() if len(S) else 0}")
    # مرجع «نفس التذبذب والسعر» لاحتمال الطيران
    qa = np.nanquantile(RND.atr20, [0.2, 0.4, 0.6, 0.8])
    qp = np.nanquantile(RND.px, [1 / 3, 2 / 3])
    cell = lambda df: np.searchsorted(qa, df.atr20.fillna(0).values) * 3 + np.searchsorted(qp, df.px.values)
    RND["cell"], S["cell"] = cell(RND), cell(S)
    for k, x in ((60, 1.0), (60, 2.0)):
        m = RND.dropna(subset=[f"mfe{k}"]).groupby("cell")[f"mfe{k}"].apply(lambda z: (z >= x).mean())
        S[f"exp_r{k}_{int(x*100)}"] = S.cell.map(m)
    # جداول هندسية لكل شريحة تذبذب (60 و 30 جلسة)
    W = RND.dropna(subset=["ph"]) if "ph" in RND else RND.iloc[:0]
    PH, PC = np.vstack(W.ph.values), np.vstack(W.pc.values)
    wq = np.searchsorted(qa, W.atr20.fillna(0).values)
    T = {}
    for H in (60, 30):
        for q in range(5):
            m = wq == q
            T[(H, q)] = geo(PH[m], PC[m], H)
    S["q"] = np.searchsorted(qa, S.atr20.fillna(0).values)
    print(f"صفقات عشوائية بمسار 60 جلسة: {len(W)}")
    print("\n==== الطيران (أول ظهور كل 20 جلسة) مقابل عشوائي بنفس التذبذب والسعر ====")
    for vv in VARS:
        g = first_only(S[S[vv]])
        if len(g) < 30:
            print(f"  {vv} {VAR_AR[vv]:<26} n={len(g)} (قليل)")
            continue
        parts = []
        for k, x in ((60, 1.0), (60, 2.0)):
            gg = g.dropna(subset=[f"mfe{k}"])
            hit, exp = (gg[f"mfe{k}"] >= x).mean(), gg[f"exp_r{k}_{int(x*100)}"].mean()
            per = ((gg[f"mfe{k}"] >= x).astype(float) - gg[f"exp_r{k}_{int(x*100)}"]).groupby(gg.t).mean()
            ci = 1.96 * per.std() / max(1, len(per)) ** 0.5
            parts.append(f"+{int(x*100)}%/{k}: {pct(hit)} (عشوائي {pct(exp)} ← {(hit-exp)*100:+.1f}±{ci*100:.1f})")
        print(f"  {vv} {VAR_AR[vv]:<26} n={len(g):<5} أسهم {g.t.nunique():<4} | " + " · ".join(parts))
    print("\n==== الخطط: النتيجة الحقيقية مقابل عشوائي بنفس الهدف والوقف بالضبط ونفس التذبذب ====")
    for vv in ("V0", "V2", "V4", "V5", "V6"):
        g = first_only(S[S[vv]])
        if len(g) < 30:
            continue
        print(f"  ── {vv} {VAR_AR[vv]} (n={len(g)})")
        for k, (name, st, tgt, H) in PLANS.items():
            gg = g.dropna(subset=[f"{k}_r"])
            if len(gg) < 30:
                continue
            s_ = gg[f"{k}_s"].clip(0.01, 0.6).values
            if tgt is None:
                eh = np.full(len(gg), np.nan)
                er = np.array([0.5 * look(T[(H, q)][1], 1.5, s1) + 0.5 * look(T[(H, q)][1], 2.5, s1) for q, s1 in zip(gg.q, s_)])
            else:
                eh = np.array([look(T[(H, q)][0], tgt, s1) for q, s1 in zip(gg.q, s_)])
                er = np.array([look(T[(H, q)][1], tgt, s1) for q, s1 in zip(gg.q, s_)])
            hit = (gg[f"{k}_x"] == 1).mean()
            per = (gg[f"{k}_r"].values - er)
            ci = 1.96 * pd.Series(per).groupby(gg.t.values).mean().std() / max(1, gg.t.nunique()) ** 0.5
            print(f"     {k} {name}: وصل الهدف {pct(hit)}" + (f" (عشوائي {pct(np.nanmean(eh))})" if tgt else "") +
                  f" · وقف {pct((gg[f'{k}_x'] == -1).mean())} · متوسط العائد {gg[f'{k}_r'].mean()*100:+.1f}% "
                  f"(عشوائي بنفس المسافات {np.nanmean(er)*100:+.1f}% ← {(gg[f'{k}_r'].mean()-np.nanmean(er))*100:+.1f}±{ci*100:.1f}) · "
                  f"وسيط {gg[f'{k}_r'].median()*100:+.1f}% · وسيط الوقف −{np.median(s_)*100:.0f}%")
    # تفصيل V4 حسب السيولة والسعر (الخطة P2)
    g = first_only(S[S.V4]).dropna(subset=["P2_r"])
    if len(g) > 60:
        print("\n==== V4 + الخطة P2 حسب السيولة اليومية والسعر ====")
        for nm, m in (("سيولة < 300 ألف$", g.dv20 < 3e5), ("سيولة 300 ألف-2 مليون$", (g.dv20 >= 3e5) & (g.dv20 < 2e6)),
                      ("سيولة > 2 مليون$", g.dv20 >= 2e6), ("السعر < 1$", g.px < 1), ("السعر 1-3$", (g.px >= 1) & (g.px < 3)),
                      ("السعر ≥ 3$", g.px >= 3)):
            gg = g[m]
            if len(gg) < 25:
                print(f"   {nm}: n={len(gg)} (قليل)")
                continue
            er = np.array([look(T[(60, q)][1], 2.0, s1) for q, s1 in zip(gg.q, gg.P2_s.clip(0.01, 0.6))])
            print(f"   {nm}: n={len(gg)} · وصل +100% {pct((gg.P2_x == 1).mean())} · متوسط {gg.P2_r.mean()*100:+.1f}% "
                  f"(عشوائي {er.mean()*100:+.1f}%) · +100%/60 {pct((gg.mfe60 >= 1).mean())}")
    if len(E):
        print(f"\n==== التغطية: من {len(E)} انفجار (+50% بيوم بعد 30 جلسة هادية) ====")
        for vv in VARS:
            ch = RND["act_" + vv].mean()
            print(f"  {vv} {VAR_AR[vv]:<26} قبل الانفجار {pct(E[vv].mean())} · قبل يوم عشوائي {pct(ch)} · {E[vv].mean() / max(ch, 1e-9):.1f}x")
    for s_ in ("OMH", "QMMM"):
        g = S[S.t == s_]
        print(f"\n{s_}: أيام V0 {len(g)} · " + " · ".join(f"{vv}: {int(g[vv].sum())}" for vv in VARS) if len(g) else f"\n{s_}: ما فيه")
        for _, x in g[g.V1].tail(10).iterrows():
            print(f"   {x.date} {x.px:.3f} · {' '.join(v for v in VARS if x[v])} · قاع أعلى {x.hl_pct*100:.0f}% · RSI {x.rsi_b:.0f}→{x.rsi_j:.0f} · "
                  f"أعلى 60: {pct(x.mfe60)} · P2: {x.P2_x} {pct(x.P2_r)}")
        for _, x in E[E.t == s_].iterrows():
            print(f"   انفجار {x.date} ({x.jump:.1f}x): " + " ".join(v for v in VARS if x[v]))


if __name__ == "__main__":
    main()
