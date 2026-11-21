"""
🚀 الطيارات — دراسة الحدث اللحظي (طلب أبوفرح: «ابدأ الدراسة»): البري والأفتر مو الشارت اليومي.

السؤال: سهم على قائمة الطيارات (أي حالة) وطلع له حدث بين إغلاق اليوم وافتتاح بكرة (أفتر اليوم + بري بكرة): قفزة ≥ +15% عن الإغلاق بفوليوم حقيقي.
  1) كم من الأسهم اللي طارت فعلاً (+100% خلال 5 جلسات) كان فيها هالحدث (التغطية)؟ وكم من الأحداث انتهت بطيران (الدقة)؟
  2) لو دخلت وقت الحدث (بسعر واقعي: الزناد + 2% انزلاق، ما يزيد عن أعلى الشمعة)، وش يصير؟ قواعد البيع بنفس اليوم/الجلسة:
     A هدف +20% / وقف −10% · B هدف +30% / وقف −15% · C هدف +50% / وقف −15% · D أمسك لإغلاق بكرة الرسمي · E أمسك لافتتاح بعد بكرة · F أمسك 5 جلسات.
     الهدف ما ينحسب إلا لو تعدّاه بـ 2% (الشمعة الخفيفة ما تنحسب)، والوقف يتفحص قبل الهدف بنفس الشمعة (الأحوط).
  3) بعد التكاليف: السبريد بالبري/الأفتر أعرض (تحت 1$: 6% · تحت 5$: 4% · غير كذا: 2% للصفقة كاملة).
القائمة = runners.detect بأي حالة (مراقبة/قريب/جاهز) على إغلاق اليوم. مجال: سعر 0.5-15$، سيولة ≥ 50 ألف$، من 2025-01. المرجع: صفر (ما فيه عشوائي هنا)،
والهامش ±95% مجمّع بالسهم، وأول حدث لكل سهم كل 5 جلسات. تشخيص فقط.
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
from backtest_runners2 import universe, first_only  # noqa: E402

DAYS, WIN = 1180, 420
FROM = "2025-01-01"
TRIG, SLIP, MIN_SH, MIN_USD = 0.15, 0.02, 100_000, 50_000
EXITS = {"A": (0.20, 0.10), "B": (0.30, 0.15), "C": (0.50, 0.15)}
EXIT_AR = {"A": "A هدف +20% / وقف −10%", "B": "B هدف +30% / وقف −15%", "C": "C هدف +50% / وقف −15%",
           "D": "D أمسك لإغلاق بكرة الرسمي", "E": "E أمسك لافتتاح بعد بكرة", "F": "F أمسك 5 جلسات"}
CODE = {"watch": 1, "near": 2, "ready": 3}
COSTX = lambda px: 0.06 if px < 1 else (0.04 if px < 5 else 0.02)
NY = "America/New_York"


def watch_days(job):
    """مرحلة 1 (يومي): أيام القائمة (أي حالة طيارات) على الإغلاق + معلومات اليوم التالي."""
    sym, d = job
    if d is None or len(d) < 300:
        return []
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    dv20 = pd.Series(c * v).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    dates = [str(x.date()) for x in d.index]
    out = []
    for t in range(270, n - 2):
        if dates[t] < FROM:
            continue
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000):
            continue
        try:
            r = RN.detect(d.iloc[max(0, t - WIN + 1):t + 1])
        except Exception:
            r = None
        if not r:
            continue
        fut = h[t + 1:t + 6]
        out.append({"t": sym, "date": dates[t], "nxt": dates[t + 1], "close": px, "R": CODE[r["status"]], "v20": v20[t],
                    "flight5": float(fut.max() >= 2 * px) if len(fut) >= 5 else np.nan,
                    "o2": o[t + 2], "c5": c[t + 5] if t + 5 < n else np.nan})
    return out


def sim_exit(o, h, l, c, entry, x, s):
    """بعد الدخول: وقف أول (بالشمعة) ثم هدف (يتعدّاه 2%)؛ وإلا نهاية البيانات. يرجع العائد."""
    stop_lv, tgt = entry * (1 - s), entry * (1 + x)
    for i in range(len(c)):
        if l[i] <= stop_lv:
            return min(o[i], stop_lv) / entry - 1, -1
        if h[i] >= tgt * 1.02:
            return x, 1
    return (c[-1] / entry - 1, 0) if len(c) else (0.0, 0)


def event(rec, H):
    """مرحلة 2: حدث البري/الأفتر لسجل قائمة واحد. H = مصفوفات الساعة للسهم."""
    ts = H["ts"]
    t0 = pd.Timestamp(rec["date"] + " 16:00", tz=NY).value
    t1 = pd.Timestamp(rec["nxt"] + " 09:30", tz=NY).value
    tr = pd.Timestamp(rec["nxt"] + " 16:00", tz=NY).value
    te = pd.Timestamp(rec["nxt"] + " 20:00", tz=NY).value
    i0, i1, ir, ie = (int(np.searchsorted(ts, x)) for x in (t0, t1, tr, te))
    res = {"trig": False, "has_ext": i1 > i0}
    if i1 <= i0:
        return res
    P0 = rec["close"]
    hh, ll, oo, cc, vv = (H[k][i0:i1] for k in ("h", "l", "o", "c", "v"))
    cumv, cumu = np.cumsum(vv), np.cumsum(vv * cc)
    cond = (hh >= P0 * (1 + TRIG)) & (cumv >= MIN_SH) & (cumu >= MIN_USD)
    if not cond.any():
        return res
    k = int(np.argmax(cond))
    j = i0 + k
    entry = min(hh[k], max(ll[k], P0 * (1 + TRIG) * (1 + SLIP)))
    res.update(trig=True, entry=entry, sess="AH" if ts[j] < pd.Timestamp(rec["nxt"], tz=NY).value else "PM",
               volratio=cumv[k] / max(rec["v20"], 1), gap_at_trig=entry / P0 - 1)
    rest = slice(j + 1, ie)
    ro, rh, rl, rc = (H[x][rest] for x in ("o", "h", "l", "c"))
    for nm, (x, s) in EXITS.items():
        res["r" + nm], res["k" + nm] = sim_exit(ro, rh, rl, rc, entry, x, s)
    res["rD"] = (H["c"][ir - 1] / entry - 1) if ir - 1 > j else 0.0
    res["rE"] = rec["o2"] / entry - 1
    res["rF"] = rec["c5"] / entry - 1 if np.isfinite(rec["c5"]) else np.nan
    res["mfe_day"] = (rh.max() / entry - 1) if len(rh) else 0.0
    return res


def to_arrays(df):
    # نانوثانية صريحة (pandas الجديد يحفظ µs، و Timestamp.value دايماً ns)
    return {"ts": df.index.tz_convert("UTC").values.astype("datetime64[ns]").astype("int64"), "o": df.Open.to_numpy(float), "h": df.High.to_numpy(float),
            "l": df.Low.to_numpy(float), "c": df.Close.to_numpy(float), "v": df.Volume.to_numpy(float)}


def collect():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    recs, t0 = [], time.time()
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
            for r in ex.map(watch_days, jobs, chunksize=4):
                recs += r
            if (i // 100) % 10 == 0:
                print(f"  يومي {i + 100}/{len(syms)} · أيام قائمة {len(recs):,} · {round(time.time() - t0)}ث", flush=True)
    df = pd.DataFrame(recs)
    print(f"أيام القائمة (من {FROM}): {len(df):,} · أسهم {df.t.nunique():,}", flush=True)
    # شموع الساعة لأسهم القائمة
    syms2 = sorted(df.t.unique())
    out = []
    hs = datetime.fromisoformat(FROM).replace(tzinfo=timezone.utc) - timedelta(days=3)
    for i in range(0, len(syms2), 40):
        b = syms2[i:i + 40]
        try:
            hb = A.bars(b, "1Hour", hs)
        except Exception as e:
            print("hourly err", repr(e)[:100], flush=True)
            continue
        for s in b:
            if s not in hb:
                continue
            H = to_arrays(hb[s])
            for r in df[df.t == s].to_dict("records"):
                out.append({**r, **event(r, H)})
        if (i // 40) % 5 == 0:
            print(f"  ساعة {i + 40}/{len(syms2)} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(out)


def cl(x, st):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 15:
        return np.nan, np.nan
    s = pd.Series(x[ok]).groupby(np.asarray(st)[ok]).mean()
    return float(np.mean(x[ok])) * 100, 1.96 * s.std() / max(1, len(s)) ** 0.5 * 100


def analyze(df):
    df = df.copy()
    df["trig"] = df["trig"].fillna(False).astype(bool)
    df["has_ext"] = df["has_ext"].fillna(False).astype(bool)
    print(f"\nأيام القائمة: {len(df):,} · فيها شموع ساعة {int(df.has_ext.sum()):,} · أحداث (قفزة ≥+15% بفوليوم): {int(df.trig.sum()):,}")
    fl = df[df.flight5.notna()]
    print(f"\n══ 1) التغطية والدقة (طيران = +100% خلال 5 جلسات من إغلاق يوم القائمة) ══")
    base = fl.flight5.mean() * 100
    tr = fl[fl.trig]
    print(f"   نسبة الطيران بكل أيام القائمة: {base:.2f}% (n={len(fl):,}) · عدد الطيارات: {int(fl.flight5.sum())}")
    print(f"   أيام فيها حدث: {len(tr):,} ({len(tr)/len(fl)*100:.1f}% من الأيام) · نسبة الطيران بعد الحدث (الدقة): {tr.flight5.mean()*100:.1f}%")
    print(f"   التغطية: من {int(fl.flight5.sum())} طيران، {int(tr.flight5.sum())} كان فيها حدث بري/أفتر ≥+15% ({tr.flight5.sum()/max(fl.flight5.sum(),1)*100:.0f}%)")
    print(f"   بدون حدث: نسبة الطيران {fl[~fl.trig].flight5.mean()*100:.2f}%")
    for nm, m in (("مراقبة 👀", df.R == 1), ("قريب 🟡", df.R == 2), ("جاهز 🚀", df.R == 3)):
        g = fl[m.reindex(fl.index)]
        gt = g[g.trig]
        print(f"   {nm}: أيام {len(g):,} · طيران {g.flight5.mean()*100:.2f}% · أحداث {len(gt)} · دقة الحدث {gt.flight5.mean()*100 if len(gt) else float('nan'):.1f}%")

    ev = first_only(df[df.trig]).copy()
    ev["cost"] = ev.entry.map(COSTX)
    print(f"\n══ 2) قواعد البيع بعد الحدث (أول حدث لكل سهم كل 20 جلسة — n={len(ev)} · {ev.t.nunique()} سهم) ══")

    def table(g, title):
        if len(g) < 25:
            print(f"\n   ── {title}: n={len(g)} (قليل)")
            return
        print(f"\n   ── {title} — n={len(g)} · {g.t.nunique()} سهم · متوسط أعلى مدى باليوم +{g.mfe_day.mean()*100:.0f}% ──")
        print(f"   {'القاعدة':<30}{'خام':>14}{'بعد التكاليف':>16}{'وسيط صافي':>11}{'هدف':>6}{'وقف':>6}")
        for k in EXIT_AR:
            r = g["r" + k].values
            net = r - g.cost.values
            a, b = cl(r, g.t.values), cl(net, g.t.values)
            hit = (g["k" + k] == 1).mean() * 100 if "k" + k in g else float("nan")
            stp = (g["k" + k] == -1).mean() * 100 if "k" + k in g else float("nan")
            print(f"   {EXIT_AR[k]:<30}{a[0]:+7.1f}±{a[1]:<5.1f}{b[0]:+9.1f}±{b[1]:<5.1f}{np.nanmedian(net)*100:+10.1f}%"
                  + (f"{hit:5.0f}%{stp:5.0f}%" if k in EXITS else "     -     -"))

    table(ev, "كل الأحداث")
    table(ev[ev.sess == "AH"], "حدث في الأفتر (بعد إغلاق يوم القائمة)")
    table(ev[ev.sess == "PM"], "حدث في البري (قبل افتتاح بكرة)")
    for nm, m in (("فوليوم الحدث < 0.5× متوسط اليوم", ev.volratio < 0.5),
                  ("0.5-2×", (ev.volratio >= 0.5) & (ev.volratio < 2)), ("≥ 2× (فوليوم ضخم)", ev.volratio >= 2)):
        table(ev[m], nm)
    table(ev[ev.entry < 1], "تحت 1$")
    table(ev[ev.entry >= 1], "فوق 1$")
    for nm, v in (("مراقبة 👀", 1), ("قريب 🟡", 2), ("جاهز 🚀", 3)):
        table(ev[ev.R == v], f"حالة الطيارات: {nm}")
    for y in ("2025", "2026"):
        table(ev[ev.date.str[:4] == y], f"سنة {y}")
    print("\n⚠ تشخيص فقط — ليس توصية. المرجع صفر (مو عشوائي)؛ الدخول بسعر الزناد +2% (≤ أعلى الشمعة)، والتكاليف 6%/4%/2% حسب السعر. السبريد الحقيقي بالبري ممكن أسوأ.")


def main():
    df = collect()
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/runners_events.csv.gz", index=False, compression="gzip")
    analyze(df)


if __name__ == "__main__":
    main()
