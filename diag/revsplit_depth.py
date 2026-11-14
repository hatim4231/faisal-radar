"""كم ينزل السهم المقسّم (تقسيم عكسي) من سعره بعد التقسيم لين يبني القاعدة؟ (طلب أبوفرح: مو شرط 60%).
نقيس توزيع عمق النزول (قاع ما بعد التقسيم ÷ سعر ما بعد التقسيم) + نسبة الانفجار ≥+70% حسب العمق.
3 سنين مع المشطوبة.
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "diag"))

import numpy as np
import pandas as pd
import alpaca_data as A
import highrisk as HR
from backtest_highrisk import universe

START = datetime(2021, 6, 1, tzinfo=timezone.utc)
W = 160
EV_FROM, EV_TO = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-04-01")
BASE_MIN = 10
SPIKE = 0.70


def build(sb):
    raw = A.bars(sb, "1Day", START)
    try: spl = A.splits(sb, START)
    except Exception: spl = {}
    out = {}
    for s, d in raw.items():
        if d is None or len(d) < 60: continue
        d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d["Stock Splits"] = 0.0
        for dt, r in spl.get(s, []):
            dt = pd.Timestamp(dt).normalize()
            if dt in d.index: d.loc[dt, "Stock Splits"] = r
        out[s] = d
    return out


drops = []                 # عمق النزول للقاع (% تحت سعر ما بعد التقسيم)
# دلاء العمق ← [عدد, انفجر ≥+70%, عدد اللي بنى قاعدة]
BUCKETS = [(0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 90), (90, 101)]
bk = {b: [0, 0, 0] for b in BUCKETS}
n_ev = 0


def bucket(dp):
    for b in BUCKETS:
        if b[0] <= dp < b[1]: return b
    return BUCKETS[-1]


syms = universe()
B = 180
for i in range(0, len(syms), B):
    try: data = build(syms[i:i + B])
    except Exception as e: print("دفعة خطأ:", repr(e)[:70], flush=True); continue
    for s, d in data.items():
        n = len(d)
        if n < 60: continue
        c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); lo = d["Low"].to_numpy(float)
        sp = d["Stock Splits"].to_numpy(float); idxdt = d.index
        for j in np.flatnonzero((sp > 0) & (sp < 1)):
            dt = idxdt[j]
            if dt < EV_FROM or dt > EV_TO or j + W >= n: continue
            n_ev += 1
            P0 = float(h[j:min(j + 4, n)].max())
            if P0 <= 0: continue
            win_c = c[j + 1:j + 1 + W]; win_h = h[j + 1:j + 1 + W]; win_l = lo[j + 1:j + 1 + W]
            lowk = int(np.argmin(win_c)); bottom = float(win_c[lowk])
            dp = (1 - bottom / P0) * 100            # كم نزل % تحت سعر ما بعد التقسيم
            drops.append(dp)
            b = bucket(dp); bk[b][0] += 1
            # بنى قاعدة بعد القاع؟ (10ج بدون كسر القاع ضمن 3%)
            based = False
            for k in range(lowk + BASE_MIN, len(win_c)):
                seg = win_l[max(0, k - BASE_MIN):k + 1]
                if len(seg) and win_l[k] >= seg.min() * 0.97:
                    based = True; be = k; break
            if based:
                bk[b][2] += 1
                after = win_h[be + 1:be + 41]
                cprev = np.r_[win_c[be], win_c[be + 1:be + 40]]
                if len(after) and (after / np.maximum(cprev[:len(after)], 1e-9) - 1 >= SPIKE).any():
                    bk[b][1] += 1
    if (i // B) % 6 == 0: print(f"  ... {min(i+B,len(syms))}/{len(syms)} · أحداث {n_ev}", flush=True)


def q(p): return float(np.percentile(drops, p)) if drops else 0.0
print("\n============ عمق النزول للقاعدة (تقسيم عكسي) ============")
print(f"أحداث: {n_ev:,}")
print(f"\nتوزيع نسبة النزول من سعر ما بعد التقسيم لين القاع:")
print(f"  الوسيط {q(50):.0f}% · الربع الأول {q(25):.0f}% · الربع الثالث {q(75):.0f}%")
print(f"  أدنى 10% ينزلون {q(10):.0f}% · أعلى 10% ينزلون {q(90):.0f}% · (أقل نزول {q(5):.0f}% · أعمق {q(95):.0f}%)")
print(f"\nحسب عمق النزول (كم نزل% ← كم منهم بنى قاعدة ← منهم انفجر ≥+70% داخل اليوم خلال 40ج):")
print(f"  {'النزول':10} {'عدد':>6} {'بنى قاعدة%':>11} {'انفجر≥70%':>11}")
for b in BUCKETS:
    cnt, sp70, based = bk[b]
    lab = f"{b[0]}–{b[1] if b[1]<=100 else '+'}%"
    pc_base = 100 * based / cnt if cnt else 0
    pc_sp = 100 * sp70 / based if based else 0
    print(f"  {lab:10} {cnt:>6} {pc_base:>10.0f}% {pc_sp:>10.0f}%")
print("\nالخلاصة: نشوف من أي عمق تبدأ القاعدة غالباً، وأي عمق يعطي أعلى انفجار — عشان نضبط حد الرادار (مو 60% ثابت).")
