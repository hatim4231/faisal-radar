"""
🚀 الطيارات مقابل 🏹 الصيد (ودمجهم) — أيهم أحسن مرشّح للطيران؟ (طلب أبوفرح: «شغل الدراسة»)

الفرق بين الدراستين السابقتين: الطيارات انقاست بعشوائي مطابق (+4.1 ±2.5 لاحتمال +50% خلال 20 جلسة للجاهز)، والصيد انقاس
بعشوائي مو مطابق (×3.3 لاحتمال ×2 خلال 20 جلسة: 13.5% مقابل 4.1%) — ما نقدر نقارنهم. هنا الاثنين على نفس الأساس بالضبط:

- نفس المجال والأيام: كل يوم (كل 3 جلسات) لكل أسهم Alpaca مع المشطوبة، سعر 0.5-15$، سيولة يومية ≥ 50 ألف$ وفوليوم 20 ≥ 20 ألف، 3 سنين.
- نفس الكاشفين الحيين: runners.detect (جاهز=V5/V6، قريب=V4) و hunt.detect (جاهز/يتجمّع/مبكر) على بيانات لين ذاك اليوم بس.
- نفس المقياس: أعلى سعر خلال 10/20/60 جلسة من إغلاق الإشارة ≥ +50% / +100%، وخطر النزيف (أدنى سعر خلال 20 جلسة ≤ −30%).
- نفس المرجع: كل الأيام المؤهلة بنفس السنة + فئة السعر (تحت/فوق 1$) + شريحة التذبذب (5) — مو عشوائي عام.
الفرق = الإشارة − المرجع (±95%، مجمّع بالسهم؛ أول ظهور كل 20 جلسة). تشخيص فقط.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import alpaca_data as A  # noqa: E402
import runners as RN  # noqa: E402
import hunt as HU  # noqa: E402
from backtest_runners2 import universe, first_only  # noqa: E402

DAYS, STEP, WIN = 1180, 3, 420
HZ = (10, 20, 60)
CODE = {None: 0, "watch": 1, "near": 2, "ready": 3}


def process(job):
    sym, d = job
    if d is None or len(d) < 300:
        return None
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    rng = (h - l) / np.maximum(c, 1e-9)
    atr20 = pd.Series(rng).rolling(20).mean().values * 100
    dv20 = pd.Series(c * v).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    rows = []
    for t in range(270, n - 1, STEP):
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000) or not np.isfinite(atr20[t]):
            continue
        w = d.iloc[max(0, t - WIN + 1):t + 1]
        try:
            r = RN.detect(w)
        except Exception:
            r = None
        try:
            hh = HU.detect(w, {})
        except Exception:
            hh = None
        rec = {"t": sym, "date": str(d.index[t].date()), "px": px, "atr20": atr20[t],
               "R": CODE[r["status"]] if r else 0, "H": CODE[hh["status"]] if hh else 0}
        fut_h, fut_l = h[t + 1:], l[t + 1:]
        for k in HZ:
            rec[f"mfe{k}"] = float(fut_h[:k].max() / px - 1) if len(fut_h) >= k else np.nan
        rec["mae20"] = float(fut_l[:20].min() / px - 1) if len(fut_l) >= 20 else np.nan
        rows.append(rec)
    return rows


def collect():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            try:
                raw = A.bars(syms[i:i + 100], "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:100], flush=True)
                continue
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")]))
            for rows in ex.map(process, jobs, chunksize=4):
                if rows:
                    out += rows
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · أيام مؤهلة {len(out):,} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(out)


def analyze(df):
    df = df.copy()
    df["year"] = df.date.str[:4]
    df["sub1"] = df.px < 1
    qa = np.nanquantile(df.atr20, [0.2, 0.4, 0.6, 0.8])
    df["q"] = np.searchsorted(qa, df.atr20.values)
    df["cell"] = df.year + "|" + df.sub1.astype(str) + "|" + df.q.astype(str)
    cols = []
    for k in HZ:
        for x in (0.5, 1.0):
            c = f"h{k}_{int(x*100)}"
            df[c] = np.where(df[f"mfe{k}"].isna(), np.nan, (df[f"mfe{k}"] >= x).astype(float))
            cols.append(c)
    df["risk20"] = np.where(df.mae20.isna(), np.nan, (df.mae20 <= -0.30).astype(float))
    cols.append("risk20")
    for c in cols:
        df["b_" + c] = df.groupby("cell")[c].transform("mean")
    df["i"] = np.arange(len(df))
    print(f"\nأيام مؤهلة: {len(df):,} · أسهم {df.t.nunique():,} · الفترة {df.date.min()} → {df.date.max()}", flush=True)
    R, H = df.R, df.H
    groups = {
        "🚀 الطيارات جاهز (V5/V6)": R == 3,
        "🚀 الطيارات جاهز + قريب (V4+)": R >= 2,
        "🏹 الصيد 🎯 جاهز": H == 3,
        "🏹 الصيد 🎯+🟡 (جاهز/يتجمّع)": H >= 2,
        "🏹 الصيد كل البصمة (حتى 👀 مبكر)": H >= 1,
        "🤝 الاثنين: طيارات V4+ و صيد 🎯/🟡": (R >= 2) & (H >= 2),
        "🚀 طيارات V4+ بس (مو صيد 🎯/🟡)": (R >= 2) & (H < 2),
        "🏹 صيد 🎯/🟡 بس (مو طيارات V4+)": (H >= 2) & (R < 2),
        "🤝 الاثنين: طيارات جاهز و صيد (أي حالة)": (R == 3) & (H >= 1),
    }

    def stat(sig, base, st):
        ok = np.isfinite(sig) & np.isfinite(base)
        if ok.sum() < 20:
            return np.nan, np.nan, np.nan, np.nan
        d = pd.Series(sig[ok] - base[ok]).groupby(st[ok]).mean()
        return sig[ok].mean() * 100, base[ok].mean() * 100, (sig[ok] - base[ok]).mean() * 100, \
            1.96 * d.std() / max(1, len(d)) ** 0.5 * 100

    def fmt(s):
        return "      -      " if not np.isfinite(s[0]) else f"{s[0]:4.1f}|{s[1]:4.1f}|{s[2]:+5.1f}±{s[3]:.1f}"

    def report(mask, title, hs=HZ):
        g = first_only(df[mask])
        if len(g) < 25:
            print(f"\n── {title}: n={len(g)} (قليل)")
            return
        ix, st = g.i.values, g.t.values
        print(f"\n══ {title} — n={len(g)} · {g.t.nunique()} سهم ══   (إشارة | مرجع مطابق | الفرق ± هامش)")
        for k in hs:
            a = stat(df[f"h{k}_50"].values[ix], df[f"b_h{k}_50"].values[ix], st)
            b = stat(df[f"h{k}_100"].values[ix], df[f"b_h{k}_100"].values[ix], st)
            print(f"   {k:>2} جلسة   +50%: {fmt(a)}    +100%: {fmt(b)}")
        r = stat(df["risk20"].values[ix], df["b_risk20"].values[ix], st)
        print(f"   خطر النزيف (أدنى سعر ≤ −30% خلال 20 جلسة): {fmt(r)}")

    for nm, m in groups.items():
        report(m, nm)
    print("\n" + "#" * 72 + "\n# ثبات بالسنين (20 جلسة)\n" + "#" * 72)
    for nm in ("🚀 الطيارات جاهز + قريب (V4+)", "🏹 الصيد 🎯+🟡 (جاهز/يتجمّع)", "🏹 الصيد كل البصمة (حتى 👀 مبكر)",
               "🤝 الاثنين: طيارات V4+ و صيد 🎯/🟡"):
        for y in sorted(df.year.unique()):
            report(groups[nm] & (df.year == y), f"{nm} · سنة {y}", hs=(20,))
    print("\n" + "#" * 72 + "\n# تحت 1$ / فوق 1$ (20 جلسة)\n" + "#" * 72)
    for nm in ("🚀 الطيارات جاهز + قريب (V4+)", "🏹 الصيد 🎯+🟡 (جاهز/يتجمّع)", "🏹 الصيد كل البصمة (حتى 👀 مبكر)"):
        report(groups[nm] & df.sub1, f"{nm} · تحت 1$", hs=(20,))
        report(groups[nm] & ~df.sub1, f"{nm} · فوق 1$", hs=(20,))
    print("\n⚠ تشخيص فقط — ليس توصية. المرجع = كل الأيام المؤهلة بنفس السنة وفئة السعر وشريحة التذبذب (بيانات معدّلة للتقسيم، فالأرقام المطلقة مبالغ فيها).")


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/runners_vs_hunt_days.csv.gz", index=False, compression="gzip")
    analyze(df)


if __name__ == "__main__":
    main()
