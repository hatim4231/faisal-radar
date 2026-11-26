"""
⚓ نظرية «الارتكاز» (منشور نايف @Naifo9x، طلب أبوفرح 2026-11-26: «شوف لنا نظرية الآن وسمها الارتكاز وخذ وقتك في فهمها وتحليلها»).
شروط المنشور: RSI أقل من 35 · قيمة سوقية 2-50 مليون (بيني) · المؤشرات تشبّع بيعي · ثبات على الدعم · حقق دعم أدنى سنوي وثبت عليه ·
قسّم تقسيم عكسي · الشورت أقل من 20 ألف · رجع للمتوسطات 20/30/50 بعد سحب السيولة.
القابل للاختبار تاريخياً (القيمة السوقية والشورت تاريخهم مو متوفر — يُطبّقان بالرادار الحي بس):
  RSI14<35 · تشبع (Stoch14 %K<20) · قاع سنوي (أدنى 252 جلسة) انعمل خلال آخر 20 جلسة وصار له 3+ جلسات بدون قاع أدنى والسعر ضمن 15% فوقه ·
  تقسيم عكسي خلال سنة · سحب سيولة خلال 10 جلسات (نزل تحت قاع 20 جلسة السابق وأغلق فوقه) · إغلاق ≥ متوسط 20.
المنهج نفسه للطيارات/الصيد: كل يوم (خطوة 2) لكل أسهم Alpaca مع المشطوبة 3 سنين، سعر 0.5-10$، فوليوم 20 ≥ 60 ألف؛
المرجع = كل الأيام المؤهلة بنفس السنة + فئة السعر + شريحة التذبذب؛ أول ظهور كل 20 جلسة؛ هامش 95% مجمّع بالسهم؛ نصين بالتاريخ. تشخيص فقط.
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import alpaca_data as A
from faisal_rules import rsi
from backtest_runners2 import universe, first_only

DAYS, STEP = 1180, 2
COST = lambda p: 0.04 if p < 1 else (0.02 if p < 5 else 0.01)


def process(job):
    sym, d, spl = job
    if d is None or len(d) < 300:
        return None
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    C = pd.Series(c)
    r14 = rsi(C, 14).values
    ll14, hh14 = pd.Series(l).rolling(14).min().values, pd.Series(h).rolling(14).max().values
    st = (c - ll14) / np.maximum(hh14 - ll14, 1e-9) * 100
    ma20 = C.rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    atr = (pd.Series((h - l) / np.maximum(c, 1e-9)).rolling(20).mean() * 100).values
    low20prev = pd.Series(l).shift(1).rolling(20).min().values
    sweep = (l < low20prev) & (c > low20prev)
    sweep10 = pd.Series(sweep.astype(float)).rolling(10, min_periods=1).max().values > 0
    rs_days = [d.index.get_indexer([pd.Timestamp(x)], method="bfill")[0] for x, ratio in spl if ratio < 1]
    rs_days = [x for x in rs_days if x >= 0]
    rows = []
    for t in range(260, n - 1, STEP):
        px = c[t]
        if not (0.5 <= px <= 10) or not (v20[t] >= 60_000) or not np.isfinite(atr[t]) or not np.isfinite(r14[t]):
            continue
        w = l[t - 251:t + 1]
        k = int(np.argmin(w)); lo = w[k]; since = 251 - k
        annual = (3 <= since <= 20) and px <= lo * 1.15
        rec = {"t": sym, "date": str(d.index[t].date()), "px": px, "atr20": atr[t],
               "rsi": r14[t] < 35, "stoch": st[t] < 20, "annual": annual,
               "rs": any(0 <= t - x <= 252 for x in rs_days), "sweep": bool(sweep10[t]), "ma20": c[t] >= ma20[t]}
        fh, fl, fc = h[t + 1:], l[t + 1:], c[t + 1:]
        for k_ in (10, 20):
            rec[f"mfe{k_}"] = float(fh[:k_].max() / px - 1) if len(fh) >= k_ else np.nan
        rec["mae20"] = float(fl[:20].min() / px - 1) if len(fl) >= 20 else np.nan
        rec["ret20"] = float(fc[19] / px - 1 - COST(px)) if len(fc) >= 20 else np.nan
        rec["ret10"] = float(fc[9] / px - 1 - COST(px)) if len(fc) >= 10 else np.nan
        rows.append(rec)
    return rows


def collect():
    syms = sorted(universe())
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            b = syms[i:i + 100]
            try:
                raw = A.bars(b, "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:90], flush=True); continue
            try:
                sp = A.splits(b, start)
            except Exception:
                sp = {}
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")], [(pd.Timestamp(x).tz_localize(None) if pd.Timestamp(x).tz else pd.Timestamp(x), r) for x, r in sp.get(s, [])]))
            for rows in ex.map(process, jobs, chunksize=4):
                if rows:
                    out += rows
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · أيام {len(out):,} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(out)


def analyze(df):
    df = df.copy()
    df["year"] = df.date.str[:4]; df["sub1"] = df.px < 1
    qa = np.nanquantile(df.atr20, [.2, .4, .6, .8]); df["q"] = np.searchsorted(qa, df.atr20.values)
    df["cell"] = df.year + "|" + df.sub1.astype(str) + "|" + df.q.astype(str)
    M = {"h20_10": ("mfe10", .2), "h50_20": ("mfe20", .5), "h100_20": ("mfe20", 1.0)}
    for k, (c, x) in M.items():
        df[k] = np.where(df[c].isna(), np.nan, (df[c] >= x).astype(float))
    df["bleed"] = np.where(df.mae20.isna(), np.nan, (df.mae20 <= -.30).astype(float))
    cols = list(M) + ["bleed", "ret10", "ret20"]
    for c in cols:
        df["b_" + c] = df.groupby("cell")[c].transform("mean")
    mid = df.date.sort_values().iloc[len(df) // 2]
    print(f"\nأيام مؤهلة: {len(df):,} · أسهم {df.t.nunique():,} · {df.date.min()} → {df.date.max()} · المنتصف {mid}")
    print("انتشار الشروط بكل الأيام: " + " · ".join(f"{k} {df[k].mean()*100:.1f}%" for k in ("rsi", "stoch", "annual", "rs", "sweep", "ma20")))
    print(f"(تداخل RSI<35 مع إغلاق ≥ متوسط 20: {(df.rsi & df.ma20).mean()*100:.2f}% من الأيام — شبه متعارضين)")

    def stat(g, c):
        x = g[g[c].notna() & g["b_" + c].notna()]
        if len(x) < 20:
            return None
        dd = (x[c] - x["b_" + c]).groupby(x.t).mean()
        return x[c].mean() * 100, x["b_" + c].mean() * 100, (x[c] - x["b_" + c]).mean() * 100, 1.96 * dd.std() / max(1, len(dd)) ** .5 * 100

    def f(s):
        return "        -        " if s is None else f"{s[0]:5.1f}|{s[1]:5.1f}|{s[2]:+5.1f}±{s[3]:.1f}"

    def line(title, m):
        g = first_only(df[m])
        if len(g) < 25:
            print(f"  {title:<46} n={len(g)} (قليل)"); return
        print(f"  {title:<46} n={len(g):>5} ({g.t.nunique()} سهم)")
        print(f"      +20%/10ج {f(stat(g,'h20_10'))}  +50%/20ج {f(stat(g,'h50_20'))}  +100%/20ج {f(stat(g,'h100_20'))}")
        print(f"      نزيف −30% {f(stat(g,'bleed'))}  عائد 10ج صافي {f(stat(g,'ret10'))}  عائد 20ج صافي {f(stat(g,'ret20'))}")
        for nm, mm in (("نص1", m & (df.date < mid)), ("نص2", m & (df.date >= mid))):
            gg = first_only(df[mm])
            s1, s2 = stat(gg, "h50_20"), stat(gg, "ret20")
            print(f"      └ {nm}: n={len(gg)} · +50%/20ج {f(s1)} · عائد 20ج {f(s2)}")

    R, S, AN, RS, SW, MA = df.rsi, df.stoch, df.annual, df.rs, df.sweep, df.ma20
    print("\n(إشارة | مرجع مطابق | الفرق ± هامش) — كل الأرقام %")
    groups = [
        ("A) RSI أقل من 35", R),
        ("B) RSI<35 + تشبّع Stoch<20", R & S),
        ("C) قاع سنوي ثابت (3-20 جلسة، ضمن 15%)", AN),
        ("D) قاع سنوي ثابت + RSI<35", AN & R),
        ("E) D + تشبّع (الارتكاز الأساسي)", AN & R & S),
        ("F) E + تقسيم عكسي خلال سنة", AN & R & S & RS),
        ("G) D + سحب سيولة خلال 10 جلسات", AN & R & SW),
        ("H) G + تقسيم عكسي", AN & R & SW & RS),
        ("I) قاع سنوي ثابت + تقسيم عكسي", AN & RS),
        ("J) قاع سنوي + سحب سيولة + رجع فوق متوسط 20", AN & SW & MA),
        ("K) تقسيم عكسي خلال سنة (لحاله)", RS),
    ]
    for t, m in groups:
        line(t, m)
    print("\n⚠ تشخيص — القيمة السوقية والشورت ما انختبروا (تاريخهم مو متوفر). بيانات Alpaca معدّلة للتقسيم (الأسعار القديمة لسهم قسّم عكسي تطلع كبيرة فتخرج من مجال 0.5-10$).")


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/irtikaz_naif_days.csv.gz", index=False, compression="gzip")
    analyze(df)


if __name__ == "__main__":
    main()
