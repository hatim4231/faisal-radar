"""
🚀 الطيارات — جدول «العشوائي بنفس الهدف والوقف» لـ 60 جلسة (للرادار الحي، تشخيص فقط).

صفقات عشوائية (كل أسهم Alpaca مع المشطوبة، السعر 0.5-15$، سيولة ≥ 50 ألف$ وفوليوم ≥ 20 ألف) مع مسار 60 جلسة.
لكل شريحة تذبذب (5) ولكل هدف (+50% / +100% / +200%) ووقف (−1% .. −60%، إغلاق تحت الوقف):
نسبة الوصول للهدف قبل الوقف خلال 60 جلسة + متوسط العائد. يطبع JSON على أسطر قصيرة (RB|).
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
from backtest_runners2 import universe, geo, UG, SG

H = 60
RAND_P = 0.05
TARGETS = (1.5, 2.0, 3.0)


def sample(job):
    sym, d = job
    if d is None or len(d) < 100:
        return []
    h, l, c, v = (d[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    n = len(c)
    atr20 = pd.Series((h - l) / np.maximum(c, 1e-9)).rolling(20).mean().values * 100
    dv20 = pd.Series(c * v).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    rnd = random.Random(int.from_bytes(sym.encode(), "little") % 10_000_019)
    out = []
    for t in range(40, n - H - 1):
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000):
            continue
        if rnd.random() < RAND_P:
            out.append((atr20[t], np.round(h[t + 1:t + 1 + H] / px, 4), np.round(c[t + 1:t + 1 + H] / px, 4)))
    return out


def main():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=1180)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            try:
                raw = A.bars(syms[i:i + 100], "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:100], flush=True)
                continue
            jobs = []
            for s, dd in raw.items():
                dd = dd[["High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")]))
            for r in ex.map(sample, jobs, chunksize=4):
                rows += r
            if (i // 100) % 15 == 0:
                print(f"  {i + 100}/{len(syms)} · صفقات {len(rows)} · {round(time.time() - t0)}ث", flush=True)
    atr = np.array([r[0] for r in rows])
    PH, PC = np.vstack([r[1] for r in rows]), np.vstack([r[2] for r in rows])
    qs = np.nanquantile(atr, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(qs, np.nan_to_num(atr))
    iu = [int(round((u - UG[0]) / 0.01)) for u in TARGETS]
    hit, ret, ns = [], [], []
    for k in range(5):
        m = q == k
        HIT, RET = geo(PH[m], PC[m], H)
        hit.append([np.rint(HIT[i] * 1000).astype(int).tolist() for i in iu])
        ret.append([np.round(RET[i], 4).tolist() for i in iu])
        ns.append(int(m.sum()))
    print(f"صفقات عشوائية: {len(PH)} · شرائح {ns}")
    for k in range(5):
        print(f"  شريحة {k + 1}: +100% قبل وقف −20%: {hit[k][1][19] / 10:.1f}% · قبل −40%: {hit[k][1][39] / 10:.1f}% · "
              f"متوسط عائد (+100%/−40%): {ret[k][1][39] * 100:+.1f}%")
    out = {"version": datetime.now().strftime("%Y-%m-%d"), "horizon": H, "targets": list(TARGETS),
           "rule": "target touched before close<stop within 60 sessions (random entries, same volatility quintile)",
           "s_pct": list(range(1, 61)), "atr20_q": [round(float(x), 4) for x in qs], "n": ns,
           "hit_permille": hit, "ret": ret}
    js = json.dumps(out, separators=(",", ":"))
    print("RUNNERS_BASE_BEGIN", len(js))
    for i in range(0, len(js), 900):
        print("RB|" + js[i:i + 900])
    print("RUNNERS_BASE_END", flush=True)


if __name__ == "__main__":
    main()
