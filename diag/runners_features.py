"""
🚀 الطيارات — وش يحسّن اختيار السهم؟ (طلب أبوفرح: «ركز كثير على الطيارات حط كل جهدك هناك»)

السؤال: داخل إشارات الطيارات نفسها (أي حالة: مراقبة/قريب/جاهز)، أي خاصية ترفع احتمال الطيران (+50% خلال 20 جلسة) أو تخفّض النزيف
(أدنى سعر ≤ −30% خلال 20 جلسة) **بشكل يصمد خارج العينة**؟

المنهج (عشان ما ننخدع بتجربة 15 خاصية وناخذ الأحسن صدفة):
  1) كل يوم مؤهل (سعر 0.5-15$، سيولة ≥ 50 ألف$، 3 جلسات خطوة، مع المشطوبة) نطبّق runners.detect؛ الإشارة = أي حالة غير فاضية.
  2) لكل إشارة نحسب 15 خاصية معروفة بإغلاق اليوم.
  3) المرجع: كل الأيام المؤهلة بنفس السنة + فئة السعر + شريحة التذبذب. الفرق = الإشارة − المرجع (±95%، مجمّع بالسهم، أول ظهور كل 20 جلسة).
  4) نقسم الفترة نصين بالتاريخ. نختار أفضل 3 قواعد (خاصية ≥ الثلث الأعلى أو ≤ الثلث الأدنى) على نص، ونختبرها على النص الثاني (وبالعكس).
     اللي ما يصمد في النصين = صدفة.
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
CODE = {None: 0, "watch": 1, "near": 2, "ready": 3}
FEATS = {  # اسم: وصف عربي
    "crash": "عمق الانهيار %", "age": "عمر القاع (جلسة)", "lift": "الارتفاع عن القاع %", "hl": "القاع الأعلى فوق القاع %",
    "rsi_gain": "تحسن RSI (قاع أخير − قاع كبير)", "rsi_now": "RSI الحالي", "njump": "عدد أيام الطيران بسنة",
    "maxjump": "أكبر قفزة سابقة %", "since_jump": "جلسات منذ آخر قفزة", "pos": "موقع السعر بين القاع والمقاومة",
    "off_top": "السعر ÷ المقاومة", "contract": "انكماش التذبذب (atr5/atr20)", "vol_dry": "جفاف الفوليوم (5ج/60ج)",
    "upv": "فوليوم صعود/هبوط 10ج", "logdv": "السيولة اليومية (log)", "stop_pct": "بعد الوقف %",
}


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
    tr = np.r_[h[0] - l[0], np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]
    jump = np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)]
    up = c > np.r_[c[0], c[:-1]]
    dn = c < np.r_[c[0], c[:-1]]
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
        rec = {"t": sym, "date": str(d.index[t].date()), "px": px, "atr20": atr20[t], "R": CODE[r["status"]] if r else 0}
        fh, fl = h[t + 1:], l[t + 1:]
        for k in (20,):
            rec[f"mfe{k}"] = float(fh[:k].max() / px - 1) if len(fh) >= k else np.nan
        rec["mae20"] = float(fl[:20].min() / px - 1) if len(fl) >= 20 else np.nan
        if r:
            try:
                hh = HU.detect(w, {})
            except Exception:
                hh = None
            B, top = r["bottom"], r["top"]
            js = [i for i in range(max(1, t - 250), t + 1) if jump[i] >= 1.5]
            a5, a20 = tr[max(0, t - 4):t + 1].mean(), tr[max(0, t - 19):t + 1].mean()
            upv = v[max(0, t - 9):t + 1][up[max(0, t - 9):t + 1]].sum()
            dnv = v[max(0, t - 9):t + 1][dn[max(0, t - 9):t + 1]].sum()
            rec.update({
                "crash": r["crash_pct"], "age": r["age"], "lift": r["lift_pct"], "hl": r["hl_pct"],
                "rsi_gain": r["rsi_last_low"] - r["rsi_bottom"], "rsi_now": r["rsi"], "njump": r["n_jumps"],
                "maxjump": max((j["pct"] for j in r["jumps"]), default=np.nan),
                "since_jump": (t - js[-1]) if js else np.nan,
                "pos": (px - B) / max(top - B, 1e-9), "off_top": px / top if top else np.nan,
                "contract": a5 / a20 if a20 > 0 else np.nan,
                "vol_dry": v[max(0, t - 4):t + 1].mean() / max(v[max(0, t - 59):t + 1].mean(), 1),
                "upv": upv / max(dnv, 1), "logdv": np.log10(max(dv20[t], 1)), "stop_pct": r["plan"]["stop_pct"],
                "hunt_fp": int(hh is not None),
            })
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
                print(f"  {i + 100}/{len(syms)} · أيام {len(out):,} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(out)


def cl(sig, st):
    """متوسط ± هامش 95% مجمّع بالسهم، وscore = متوسط/خطأ معياري."""
    ok = np.isfinite(sig)
    if ok.sum() < 20:
        return np.nan, np.nan, np.nan
    s = pd.Series(sig[ok]).groupby(st[ok]).mean()
    se = s.std() / max(1, len(s)) ** 0.5
    m = float(np.nanmean(sig[ok]))
    return m * 100, 1.96 * se * 100, (m / se if se > 0 else 0.0)


def analyze(df):
    df = df.copy()
    df["year"] = df.date.str[:4]
    df["sub1"] = df.px < 1
    qa = np.nanquantile(df.atr20, [0.2, 0.4, 0.6, 0.8])
    df["q"] = np.searchsorted(qa, df.atr20.values)
    df["cell"] = df.year + "|" + df.sub1.astype(str) + "|" + df.q.astype(str)
    df["f50"] = np.where(df.mfe20.isna(), np.nan, (df.mfe20 >= 0.5).astype(float))
    df["f100"] = np.where(df.mfe20.isna(), np.nan, (df.mfe20 >= 1.0).astype(float))
    df["bleed"] = np.where(df.mae20.isna(), np.nan, (df.mae20 <= -0.30).astype(float))
    for c in ("f50", "f100", "bleed"):
        df["d_" + c] = df[c] - df.groupby("cell")[c].transform("mean")        # الفرق عن المرجع المطابق لكل يوم
    cand = first_only(df[df.R > 0]).copy()
    print(f"\nأيام مؤهلة: {len(df):,} · أسهم {df.t.nunique():,} · {df.date.min()} → {df.date.max()}", flush=True)
    print(f"إشارات الطيارات (أي حالة، أول ظهور كل 20 جلسة): {len(cand):,} · أسهم {cand.t.nunique():,}", flush=True)
    split = cand.date.sort_values().iloc[len(cand) // 2]
    cand["half"] = np.where(cand.date < split, "A", "B")
    print(f"تقسيم الفترة: A قبل {split} ({(cand.half == 'A').sum()}) · B بعدها ({(cand.half == 'B').sum()})", flush=True)

    def line(g, title):
        if len(g) < 25:
            return f"   {title:<44} n={len(g):<4} (قليل)"
        st = g.t.values
        a, b, c_ = cl(g.d_f50.values, st), cl(g.d_f100.values, st), cl(g.d_bleed.values, st)
        return (f"   {title:<44} n={len(g):<5} +50%: {np.nanmean(g.f50)*100:4.1f} ({a[0]:+5.1f}±{a[1]:.1f}) "
                f"· +100%: {np.nanmean(g.f100)*100:4.1f} ({b[0]:+5.1f}±{b[1]:.1f}) · نزيف: {np.nanmean(g.bleed)*100:4.1f} ({c_[0]:+5.1f}±{c_[1]:.1f})")

    print("\n══ كل إشارات الطيارات (المرجع الأساس) — 20 جلسة، بين قوسين الفرق عن المرجع المطابق ══")
    print(line(cand, "الكل"))
    for nm, v in (("مراقبة 👀", 1), ("قريب 🟡", 2), ("جاهز 🚀", 3)):
        print(line(cand[cand.R == v], nm))
    for h_ in ("A", "B"):
        print(line(cand[cand.half == h_], f"نص {h_}"))

    print("\n══ كل خاصية: ثلث أدنى | وسط | ثلث أعلى (فرق +50% عن المرجع ± هامش) — نص A ثم نص B ══")
    for f, ar in FEATS.items():
        x = cand[f]
        q1, q2 = x.quantile([1 / 3, 2 / 3])
        row = [f"{ar:<34}"]
        for h_ in ("A", "B"):
            seg = []
            for nm, m in (("أدنى", x <= q1), ("وسط", (x > q1) & (x < q2)), ("أعلى", x >= q2)):
                g = cand[m & (cand.half == h_)]
                a = cl(g.d_f50.values, g.t.values) if len(g) >= 25 else (np.nan, np.nan, np.nan)
                seg.append(f"{nm} {a[0]:+5.1f}±{a[1]:<4.1f}" if np.isfinite(a[0]) else f"{nm}   -   ")
            row.append(f"{h_}: " + " | ".join(seg))
        print("   " + "  ┃  ".join(row))
    xs = cand["hunt_fp"]
    print("   " + line(cand[xs == 1], "بصمة الصيد موجودة (منهار 70%+ وطيّار ×2)"))
    print("   " + line(cand[xs == 0], "بصمة الصيد غايبة"))

    # اختيار القواعد على نص واختبارها على الثاني
    for tr_h, te_h in (("A", "B"), ("B", "A")):
        tr, te = cand[cand.half == tr_h], cand[cand.half == te_h]
        rules = []
        for f in FEATS:
            x = tr[f].dropna()
            if len(x) < 150:
                continue
            q1, q2 = x.quantile([1 / 3, 2 / 3])
            for side, thr in (("≥", q2), ("≤", q1)):
                m = (tr[f] >= thr) if side == "≥" else (tr[f] <= thr)
                g = tr[m]
                if len(g) < 100:
                    continue
                a = cl(g.d_f50.values, g.t.values)
                if a[0] > 0:
                    rules.append((a[2], f, side, thr))
        rules.sort(reverse=True)
        top = rules[:3]
        print(f"\n══ اختيار على نص {tr_h} ← اختبار على نص {te_h} (اللي ما شافه) ══")
        if not top:
            print("   ما طلعت قواعد موجبة"); continue

        def mask(df_, rs):
            m = np.ones(len(df_), bool)
            for _, f, side, thr in rs:
                m &= (df_[f] >= thr).values if side == "≥" else (df_[f] <= thr).values
            return m
        for k, (sc, f, side, thr) in enumerate(top, 1):
            ds = f"{FEATS[f]} {side} {thr:.3g}"
            print(f"  قاعدة {k}: {ds}   (score على التدريب {sc:.1f})")
            print("     تدريب:" + line(tr[mask(tr, [top[k - 1]])], "").strip())
            print("     اختبار:" + line(te[mask(te, [top[k - 1]])], "").strip())
        for k in (2, 3):
            if len(top) >= k:
                print(f"  أول {k} قواعد معاً (AND):")
                print("     تدريب:" + line(tr[mask(tr, top[:k])], "").strip())
                print("     اختبار:" + line(te[mask(te, top[:k])], "").strip())
        print("  بدون قاعدة (كل إشارات نص الاختبار):")
        print("     اختبار:" + line(te, "").strip())
    print("\n⚠ تشخيص فقط — ليس توصية. الفرق = الإشارة − مرجع مطابق (السنة+فئة السعر+التذبذب)، ±95% مجمّع بالسهم. القواعد المختارة على نص وتنختبر على الثاني.")


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/runners_features_days.csv.gz", index=False, compression="gzip")
    analyze(df)


if __name__ == "__main__":
    main()
