"""
🚀 صيد الطيارات (أمثلة المستخدم: OMH و QMMM) — اختبار تاريخي (تشخيص فقط).

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
VARS = ["V0", "V1", "V2", "V3", "V4", "V5"]
VAR_AR = {"V0": "منهار ويقعد (أساس)", "V1": "+ قاع أعلى + RSI أعلى", "V2": "+ دعوم متصاعدة",
          "V3": "V1 + سبق طار", "V4": "V2 + سبق طار", "V5": "V4 + اختبر المقاومة مرتين"}
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
    for t in range(260, n - 1, STEP):
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000):
            continue
        n_eval += 1
        if rnd.random() < RAND_P:
            rands.append({"t": sym, "date": str(dates[t].date()), "ti": t, "px": px, "atr20": atr20[t], "dv20": dv20[t],
                          **outcome(h, c, t, None)})
        w0 = t - 119
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
        top = h[b:t + 1].max()
        near = np.where(h[b:t + 1] >= 0.93 * top)[0]
        touches = 0 if len(near) == 0 else 1 + int((np.diff(near) >= 5).sum())
        f = {"V0": True, "V1": v1, "V2": v2, "V3": v1 and frn, "V4": v2 and frn, "V5": v2 and frn and touches >= 2}
        flags[t] = f
        rows.append({"t": sym, "date": str(dates[t].date()), "px": px, "atr20": atr20[t], "dv20": dv20[t], "v20": v20[t],
                     "age": age, "crash": crash, "lift": lift, "off_top": px / top, "rsi": r[t], "rsi_b": r[b], "rsi_j": r[j],
                     "hl_pct": R / B - 1, "rising": rising, "fr": bool(frn), "touches": touches,
                     **{k: bool(x) for k, x in f.items()}, **outcome(h, c, t, B * 0.9)})
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
    for idx, s, dt in zip(g.index, g.t.values, pd.to_datetime(g.date).values):
        if s not in last or (dt - last[s]) > np.timedelta64(gap * 7 // 5, "D"):
            keep.append(idx)
        last[s] = dt
    return g.loc[keep]


def main():
    S, RND, E = collect()
    os.makedirs("diag_out", exist_ok=True)
    S.to_csv("diag_out/runners_setups.csv.gz", index=False, compression="gzip")
    RND.to_csv("diag_out/runners_random.csv.gz", index=False, compression="gzip")
    E.to_csv("diag_out/runners_explosions.csv.gz", index=False, compression="gzip")
    print(f"\nإشارات V0 (كل جلستين): {len(S)} · عشوائي {len(RND)} · انفجارات {len(E)} · أسهم {S.t.nunique() if len(S) else 0}")
    # خلايا المقارنة: شريحة التذبذب × شريحة السعر
    qa = np.nanquantile(RND.atr20, [0.2, 0.4, 0.6, 0.8])
    qp = np.nanquantile(RND.px, [1 / 3, 2 / 3])
    cell = lambda df: np.searchsorted(qa, df.atr20.fillna(0).values) * 3 + np.searchsorted(qp, df.px.values)
    RND["cell"], S["cell"] = cell(RND), cell(S)
    print("\nالعشوائي كله: " + " · ".join(f"+{int(x*100)}% خلال {k}: {pct((RND[f'mfe{k}'] >= x).mean())}"
                                         for k in (20, 40, 60) for x in (0.5, 1.0)))
    for k in (40, 60):
        for x in (0.5, 1.0, 2.0):
            key = f"r{k}_{int(x*100)}"
            m = RND.dropna(subset=[f"mfe{k}"]).groupby("cell")[f"mfe{k}"].apply(lambda s: (s >= x).mean())
            S[f"exp_{key}"] = S.cell.map(m)
    print("\n==== نسبة الطيارات (أول ظهور لكل سهم كل 20 جلسة) مقابل عشوائي بنفس التذبذب والسعر ====")
    for vv in VARS:
        g = first_only(S[S[vv]])
        if len(g) < 20:
            print(f"  {vv} {VAR_AR[vv]:<28} n={len(g)} (قليل)")
            continue
        parts = []
        for k, x in ((40, 0.5), (40, 1.0), (60, 1.0), (60, 2.0)):
            gg = g.dropna(subset=[f"mfe{k}"])
            hit = (gg[f"mfe{k}"] >= x).mean()
            exp = gg[f"exp_r{k}_{int(x*100)}"].mean()
            per = ((gg[f"mfe{k}"] >= x).astype(float) - gg[f"exp_r{k}_{int(x*100)}"]).groupby(gg.t).mean()
            ci = 1.96 * per.std() / max(1, len(per)) ** 0.5 if len(per) > 2 else np.nan
            parts.append(f"+{int(x*100)}%/{k}: {pct(hit)} (عشوائي {pct(exp)} ← {(hit-exp)*100:+.1f}±{ci*100:.1f})")
        x100 = g[g.x100 != 0]
        win = (x100.x100 == 1).mean() if len(x100) else np.nan
        print(f"  {vv} {VAR_AR[vv]:<28} n={len(g):<5} أسهم {g.t.nunique():<4} | " + " · ".join(parts))
        print(f"      خطة (+100% أو إغلاق تحت القاع −10% أو 60 جلسة): وصل +100% {pct((g.x100 == 1).mean())} · وقف {pct((g.x100 == -1).mean())} · "
              f"متوسط العائد {g.ret100.mean()*100:+.1f}% · وسيط {g.ret100.median()*100:+.1f}% · وسيط بُعد الوقف {(1 - 0.9 / g.lift).median()*100:.0f}%")
    # التغطية
    if len(E):
        print(f"\n==== التغطية: من {len(E)} انفجار (+50% بيوم بعد 30 جلسة هادية) — كم واحد كان الشرط شغال قبله بـ 10 جلسات ====")
        for vv in VARS:
            ch = RND["act_" + vv].mean()
            print(f"  {vv} {VAR_AR[vv]:<28} قبل الانفجار {pct(E[vv].mean())} · قبل يوم عشوائي {pct(ch)} · "
                  f"أكثر من الصدفة بـ {E[vv].mean() / max(ch, 1e-9):.1f}x")
    # أمثلة المستخدم
    for s in ("OMH", "QMMM"):
        g = S[S.t == s]
        if len(g):
            print(f"\n{s}: أيام فيها V0: {len(g)} · " + " · ".join(f"{vv}: {int(g[vv].sum())}" for vv in VARS))
            for _, x in g[g.V1].tail(12).iterrows():
                print(f"   {x.date} {x.px:.3f} · {''.join(v for v in VARS if x[v])} · قاع أعلى {x.hl_pct*100:.0f}% · RSI {x.rsi_b:.0f}→{x.rsi_j:.0f} · "
                      f"أعلى خلال 60: {pct(x.mfe60)}")
        ee = E[E.t == s]
        for _, x in ee.iterrows():
            print(f"   انفجار {x.date} ({x.jump:.1f}x): " + " ".join(v for v in VARS if x[v]))
    # أكبر الطيارات اللي صادها V4
    g = first_only(S[S.V4]).sort_values("mfe60", ascending=False).head(25)
    print("\nأكبر 25 طيارة صادها V4 (أعلى خلال 60 جلسة):")
    for _, x in g.iterrows():
        print(f"   {x.t} {x.date} {x.px:.3f} → +{x.mfe60*100:.0f}% · انهيار {x.crash:.0f}x · قاع أعلى {x.hl_pct*100:.0f}% · لمسات {x.touches}")


if __name__ == "__main__":
    main()
