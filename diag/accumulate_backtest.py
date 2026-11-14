"""طريقة فيصل الصح: تجميع في القاعدة قبل الانطلاق + بيع داخل القفزة + احتفاظ 3-7 أيام.
نصحّح غلط الاختبار السابق (شراء أعمى بوقف -30% = شيك خاطئ). الحين:

الصيد (بصمة القاعدة/التجميع — معروفة بإغلاق اليوم):
  طيّار سابق (n_ep252≥1) · منهار ≥70% (dd252) · وقف صناعة قيعان (الإغلاق فوق أدنى 20ج بـ3%+) ·
  انكماش تذبذب (atr5_60 ≤ 0.9) · بصمة تجميع (فوليوم صعود≥هبوط upv10≥1 أو OBV صاعد أو قيعان أعلى) ·
  RSI 20-50 (مضغوط لكن مو ساقط ميت) · سعر حقيقي 0.1-10$ · سيولة20 ≥ 30 ألف$ · مو يوم انفجار.

التنفيذ (نبيع داخل القفزة بأمر Limit، احتفاظ H أيام):
  دخول = إغلاق يوم الإشارة (تجميع). لكل H∈{3,5,7} ولكل هدف قفزة {+15,+20,+30%}:
    لو أعلى أي يوم خلال H لمس الهدف ← بيع عند الهدف (Limit داخل القفزة)، وإلا بيع بإغلاق اليوم H.
    نسخة بوقف واسع (إغلاق تحت -25%) ونسخة بدون وقف. كله بعد التكاليف.
  نقارن بالعشوائي **بنفس التنفيذ بالضبط** (نفس الأسهم البيني القابلة للتداول) — عشان نعزل: هل الصيد يضيف فوق التنفيذ؟
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

START = datetime(2022, 11, 1, tzinfo=timezone.utc)
LIQ, PX_LO, PX_HI = 30_000.0, 0.10, 10.0
HORIZONS = [3, 5, 7]
SPIKES = [0.15, 0.20, 0.30]


def cost_of(px):
    return 0.04 if px < 1 else 0.02 if px < 5 else 0.01


def build(sym_batch):
    raw = A.bars(sym_batch, "1Day", START)
    try:
        spl = A.splits(sym_batch, START)
    except Exception:
        spl = {}
    out = {}
    for s, d in raw.items():
        if d is None or len(d) < 160:
            continue
        d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d["Stock Splits"] = 0.0
        for dt, ratio in spl.get(s, []):
            dt = pd.Timestamp(dt).normalize()
            if dt in d.index:
                d.loc[dt, "Stock Splits"] = ratio
        out[s] = d
    return out


def simulate(entry, fh, fc_last, spike, use_stop, fl=None):
    """يرجع عائد الصفقة (قبل التكاليف). fh=أعلى الأيام القادمة, fc_last=إغلاق آخر يوم, fl=أدنى الأيام (للوقف)."""
    tgt = entry * (1 + spike)
    if use_stop and fl is not None:
        stp = entry * 0.75
        for k in range(len(fh)):
            if fl[k] <= stp:        # إغلاق/أدنى تحت الوقف
                return -0.25
            if fh[k] >= tgt:
                return spike
        return fc_last / entry - 1
    else:
        if fh.max() >= tgt:
            return spike
        return fc_last / entry - 1


# عدّادات: (نوع, H, spike, stop) ← [n, sum_net, wins, hit_spike, list_net]
from collections import defaultdict
res = defaultdict(lambda: [0, 0.0, 0, 0])
net_lists = defaultdict(list)
fp_total = 0
base_total = 0

syms = universe()
BATCH = 180
for i in range(0, len(syms), BATCH):
    try:
        data = build(syms[i:i + BATCH])
    except Exception as e:
        print("  دفعة خطأ:", repr(e)[:80], flush=True); continue
    for s, d in data.items():
        n = len(d)
        if n < 160:
            continue
        F = HR.features(d)
        c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); lo = d["Low"].to_numpy(float)
        praw = F["praw"].to_numpy(float); dd = F["dd252"].to_numpy(float); nep = F["n_ep252"].to_numpy(float)
        rsi = F["rsi"].to_numpy(float); dv = F["dv20"].to_numpy(float); ep = F["ep"].to_numpy(float); jump = F["jump"].to_numpy(float)
        lo20 = F["lo20"].to_numpy(float); atr5_60 = F["atr5_60"].to_numpy(float); upv10 = F["upv10"].to_numpy(float)
        obv20 = F["obv20"].to_numpy(float); hl_up = F["hl_up"].to_numpy(float)
        tradeable = (praw >= PX_LO) & (praw <= PX_HI) & (dv >= LIQ) & (ep < 1) & (jump < 1)
        accum = ((upv10 >= 1) | (obv20 > 0) | (hl_up >= 1))
        base = (c > lo20 * 1.03) & (atr5_60 <= 0.9)
        fp = tradeable & (dd <= -0.70) & (nep >= 1) & (rsi >= 20) & (rsi <= 50) & base & accum
        maxH = max(HORIZONS)
        last = n - 1
        for j in np.flatnonzero(tradeable):
            if j + maxH > last:
                continue
            entry = c[j]
            is_fp = bool(fp[j])
            if is_fp:
                fp_total += 1
            base_total += 1
            for H in HORIZONS:
                fh = h[j + 1:j + 1 + H]; fl = lo[j + 1:j + 1 + H]; fcl = c[j + H]
                for spike in SPIKES:
                    for use_stop in (False, True):
                        r = simulate(entry, fh, fcl, spike, use_stop, fl)
                        net = r - cost_of(entry)
                        for grp in (("fp" if is_fp else "base"),):
                            key = (grp, H, spike, use_stop)
                            res[key][0] += 1
                            res[key][1] += net
                            res[key][2] += int(net > 0)
                            res[key][3] += int(r >= spike - 1e-9)
                            if grp == "fp":
                                net_lists[key].append(net)
    done = min(i + BATCH, len(syms))
    if (i // BATCH) % 6 == 0:
        print(f"  ... {done}/{len(syms)} · إشارات بصمة {fp_total:,}", flush=True)

print(f"\n================ طريقة التجميع — النتيجة ================")
print(f"إشارات البصمة (قاعدة/تجميع): {fp_total:,}  ·  أيام بيني قابلة للتداول (عشوائي): {base_total:,}")
print("\nH=أيام الاحتفاظ · هدف=القفزة · وقف؟ | متوسط صافي% | وسيط% | رابحة% | لمس القفزة% | (عشوائي متوسط%)")
for H in HORIZONS:
    for spike in SPIKES:
        for use_stop in (False, True):
            fk = ("fp", H, spike, use_stop); bk = ("base", H, spike, use_stop)
            fn = res[fk]; bn = res[bk]
            if fn[0] == 0:
                continue
            fmean = 100 * fn[1] / fn[0]; fwin = 100 * fn[2] / fn[0]; fhit = 100 * fn[3] / fn[0]
            med = 100 * float(np.median(net_lists[fk])) if net_lists[fk] else 0
            bmean = 100 * bn[1] / bn[0] if bn[0] else 0
            stopt = "وقف" if use_stop else "بدون"
            print(f"  H={H} +{int(spike*100):>2}% {stopt:4} | {fmean:+6.1f}% | {med:+6.1f}% | {fwin:5.1f}% | {fhit:5.1f}% | (عشوائي {bmean:+.1f}%)")
print("\nقراءة: لو متوسط البصمة موجب وأعلى من العشوائي = التجميع+البيع بالقفزة يضيف فعلاً. السالب = حتى بالتنفيذ الصح ما تكفي بيانات اليوم (الباقي تنفيذ لحظي بشري).")
