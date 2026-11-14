"""الحمض النووي لكل سهم ارتفع ≥100% خلال أسبوعين (2–13 نوفمبر 2026).
مرحلة 1: فحص خفيف لكل السوق ← نلقى الموفرين (إغلاق أو أعلى ≥ 2× أمس، تداول ≥ 300 ألف$).
مرحلة 2: لكل موفر، نحلل حالته **قبل القفزة بيوم** بقوانين فيصل + معلومات ياهو، ونطلع القاسم المشترك بالأرقام.
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "highrisk_radar")); sys.path.insert(0, str(ROOT / "faisal_original"))

import numpy as np
import pandas as pd
import alpaca_data as A
import highrisk_scanner as HS
import faisal_rules as fr

cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
WEEKS = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06",
         "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
MIN_DV, MIN_JUMP = 300_000.0, 1.00

syms = HS.alpaca_universe()
print(f"المجال: {len(syms)} سهم نشط")

# ---- مرحلة 1: كشف الموفرين (خفيف)
start = datetime(2026, 10, 13, tzinfo=timezone.utc)
first_jump = {}   # رمز ← (يوم أول قفزة، قبل، إغلاق، أعلى، قفزة إغلاق%، قفزة أعلى%، فوليوم، تداول$)
for i in range(0, len(syms), 200):
    try:
        got = A.bars(syms[i:i + 200], "1Day", start)
    except Exception:
        continue
    for s, d in got.items():
        if d is None or len(d) < 3:
            continue
        d = d.sort_index()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        c, h, v = d["Close"].astype(float), d["High"].astype(float), d["Volume"].astype(float)
        for ds in WEEKS:
            t = pd.Timestamp(ds)
            if t not in d.index:
                continue
            j = d.index.get_loc(t)
            if j == 0:
                continue
            prev = float(c.iloc[j - 1]); cl = float(c.iloc[j]); hi = float(h.iloc[j]); vol = float(v.iloc[j])
            if prev <= 0:
                continue
            dv = cl * vol
            jc, jh = cl / prev - 1, hi / prev - 1
            if max(jc, jh) >= MIN_JUMP and dv >= MIN_DV and s not in first_jump:
                first_jump[s] = (ds, prev, cl, hi, jc * 100, jh * 100, vol, dv)
                break   # أول قفزة بس

hits = sorted(first_jump.keys())
print(f"موفرون ≥100% خلال الأسبوعين: {len(hits)} سهم")

# ---- مرحلة 2: الحمض النووي (ثقيل لكن للموفرين بس)
daily = A.daily(hits, days=730)


def info_of(s):
    try:
        import yfinance as yf
        inf = yf.Ticker(s).info or {}
        return {"country": inf.get("country"), "floatShares": inf.get("floatShares"),
                "marketCap": inf.get("marketCap"), "sharesShort": inf.get("sharesShort")}
    except Exception:
        return {}


rows = []
print("\n=== الحمض النووي قبل القفزة بيوم ===")
hdr = f"{'رمز':6} {'يوم القفزة':11} {'قفزة%':>6} {'سعر-قبل':>8} {'منهار%':>7} {'RSI':>5} {'بعد-دعم%':>8} {'انفجار-سابق%':>12} {'فلووت':>8} {'حالة-فيصل':>10} {'نقاط':>5} {'الدولة':>10}"
print(hdr)
for s in sorted(hits, key=lambda x: -first_jump[x][5]):
    ds, prev, cl, hi, jc, jh, vol, dv = first_jump[s]
    d = daily.get(s)
    if d is None or len(d) < 60:
        print(f"{s:6} {ds:11} {jh:>5.0f}%  (بيانات يومية ناقصة)")
        continue
    d = d.sort_index()
    before = d[d.index < pd.Timestamp(ds)]
    if len(before) < 60:
        print(f"{s:6} {ds:11} {jh:>5.0f}%  (تاريخ قصير)")
        continue
    info = info_of(s)
    a = fr.analyze(before.copy(), info, cfg)
    if not a:
        print(f"{s:6} {ds:11} {jh:>5.0f}%  (ما تحلل)")
        continue
    flt = a.get("float")
    row = {"s": s, "day": ds, "jh": jh, "jc": jc, "prev": prev,
           "dd": a["drawdown_pct"], "rsi": a["rsi"], "dist": a["plan"]["dist_support_pct"],
           "fr": a["former_runner"]["max_pct"], "flt": flt, "status": a["irtikaz"]["status"],
           "score": a["irtikaz"]["score"], "country": info.get("country") or "?",
           "below1": a["below_1_streak"], "rs": len(a["reverse_splits"] or []),
           "up20": a["up_from_low20_pct"], "gate": bool(a["gate_fail"])}
    rows.append(row)
    fltx = f"{flt/1e6:.2f}M" if flt else "?"
    print(f"{s:6} {ds:11} {jh:>5.0f}% {prev:>8.3f} {row['dd']:>6.0f}% {str(row['rsi']):>5} {str(row['dist']):>8} {row['fr']:>11.0f}% {fltx:>8} {row['status']:>10} {row['score']:>5.0f} {str(row['country'])[:10]:>10}")

# ---- القاسم المشترك (إحصائيات)
if rows:
    n = len(rows)
    def frac(f): return 100 * sum(1 for r in rows if f(r)) / n
    def med(key):
        xs = [r[key] for r in rows if r[key] is not None]
        return float(np.median(xs)) if xs else None
    print(f"\n=== القاسم المشترك ({n} سهم) ===")
    print(f"منهار ≥70% من قمة السنة: {frac(lambda r: r['dd'] is not None and r['dd'] <= -70):.0f}%  (وسيط الانهيار {med('dd'):.0f}%)")
    print(f"former runner (سبق طار ≥100%): {frac(lambda r: (r['fr'] or 0) >= 100):.0f}%  (وسيط أكبر انفجار {med('fr'):.0f}%)")
    print(f"RSI ≤ 35 (تشبع بيعي): {frac(lambda r: r['rsi'] is not None and r['rsi'] <= 35):.0f}%  (وسيط RSI {med('rsi'):.0f})")
    print(f"قريب من الدعم ≤10%: {frac(lambda r: r['dist'] is not None and abs(r['dist']) <= 10):.0f}%  (وسيط البعد {med('dist'):.1f}%)")
    fl = [r['flt'] for r in rows if r['flt']]
    print(f"فلووت معروف: {len(fl)}/{n}  · فلووت ≤10M: {100*sum(1 for x in fl if x<=1e7)/len(fl) if fl else 0:.0f}%  (وسيط {np.median(fl)/1e6:.2f}M)" if fl else "فلووت: غير معروف")
    print(f"سعر-قبل ≤3$: {frac(lambda r: r['prev'] <= 3):.0f}%  · ≤1$: {frac(lambda r: r['prev'] <= 1):.0f}%")
    print(f"تقسيم عكسي بالتاريخ: {frac(lambda r: r['rs'] > 0):.0f}%")
    print(f"أجنبي (مو أمريكا): {frac(lambda r: r['country'] not in ('United States','?')):.0f}%")
    print(f"حالة فيصل قبل القفزة: ready {frac(lambda r: r['status']=='ready'):.0f}% · near {frac(lambda r: r['status']=='near'):.0f}% · radar {frac(lambda r: r['status']=='radar'):.0f}% · no {frac(lambda r: r['status']=='no'):.0f}%")
    print(f"كان بيُرفض من فلتر السعر 0.70-10$: {frac(lambda r: r['gate']):.0f}%")
