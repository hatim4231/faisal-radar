"""اختبار «الحمض النووي» بصدق: هل بصمة (منهار ≥70% + طيّار سابق + RSI مضغوط + بيني) تكسب ولا خداع؟
على كل أسهم Alpaca (نشط + مشطوب) 3 سنين، نقطة-بنقطة (ما نشوف المستقبل)، مقابل العشوائي وبعد التكاليف.

البصمة (معروفة بإغلاق اليوم): dd252 ≤ -70% · n_ep252 ≥ 1 (سبق ×2 بسنة) · RSI ≤ 35 · سعر حقيقي 0.1-10$ · تداول20 ≥ 30 ألف$ · مو يوم انفجار.
النتيجة خلال 20 جلسة قدّام: طار ×2؟ · طار ×1.5؟ · لمس -30%؟ · اختفى/شطب؟ · صفقة (دخول افتتاح بكرة، هدف +100%، وقف -30%) بعد التكاليف.
العشوائي = كل أيام الأسهم البيني القابلة للتداول (نفس فلتر السعر/السيولة) بدون شرط البصمة.
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
H = 20                    # أفق النتيجة (جلسات)
LIQ = 30_000.0
PX_LO, PX_HI = 0.10, 10.0


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


# عدّادات
agg = {"base_n": 0, "base_j100": 0, "base_j50": 0,
       "fp_n": 0, "fp_j100": 0, "fp_j50": 0, "fp_bleed30": 0, "fp_delist": 0,
       "win": 0, "stop": 0, "flat": 0}
fp_maxgain, fp_ret20, fp_net = [], [], []
by_year = {}   # سنة ← [fp_n, fp_j100, base_n, base_j100]

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
        c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); lo = d["Low"].to_numpy(float); o = d["Open"].to_numpy(float)
        praw = F["praw"].to_numpy(float); dd = F["dd252"].to_numpy(float); nep = F["n_ep252"].to_numpy(float)
        rsi = F["rsi"].to_numpy(float); dv = F["dv20"].to_numpy(float); ep = F["ep"].to_numpy(float); jump = F["jump"].to_numpy(float)
        yrs = d.index.year.to_numpy()
        # قابل للتداول (بيني): سعر حقيقي بالمجال + سيولة + مو يوم انفجار اليوم
        tradeable = (praw >= PX_LO) & (praw <= PX_HI) & (dv >= LIQ) & (ep < 1) & (jump < 1)
        # البصمة
        fp = tradeable & (dd <= -0.70) & (nep >= 1) & (rsi <= 35)
        last = n - 1
        for j in np.flatnonzero(tradeable):
            if j + H > last:        # نحتاج 20 جلسة قدّام مكتملة
                # لو اختفى السهم خلال المدة والبصمة متحققة = شطب/موت (إشارة نزيف قوية)
                if fp[j] and (last - j) < H:
                    agg["fp_delist"] += 1
                continue
            entry_c = c[j]
            fh = h[j + 1:j + 1 + H]; fl = lo[j + 1:j + 1 + H]; fc = c[j + H]
            j100 = fh.max() >= 2.0 * entry_c
            j50 = fh.max() >= 1.5 * entry_c
            yr = int(yrs[j])
            by_year.setdefault(yr, [0, 0, 0, 0])
            agg["base_n"] += 1; agg["base_j100"] += int(j100); agg["base_j50"] += int(j50)
            by_year[yr][2] += 1; by_year[yr][3] += int(j100)
            if not fp[j]:
                continue
            # البصمة
            agg["fp_n"] += 1; agg["fp_j100"] += int(j100); agg["fp_j50"] += int(j50)
            by_year[yr][0] += 1; by_year[yr][1] += int(j100)
            if fl.min() <= 0.70 * entry_c:
                agg["fp_bleed30"] += 1
            fp_maxgain.append(fh.max() / entry_c - 1)
            fp_ret20.append(fc / entry_c - 1)
            # صفقة: دخول افتتاح بكرة، هدف +100%، وقف -30%، أول لمس (نفس اليوم الاثنين = وقف)
            entry = o[j + 1]
            if entry <= 0:
                entry = entry_c
            tgt, stp = entry * 2.0, entry * 0.70
            res, ret = "flat", fc / entry - 1
            for k in range(j + 1, j + 1 + H):
                if lo[k] <= stp:
                    res, ret = "stop", -0.30; break
                if h[k] >= tgt:
                    res, ret = "win", 1.00; break
            agg[res] += 1
            fp_net.append(ret - cost_of(entry))
    done = min(i + BATCH, len(syms))
    if (i // BATCH) % 5 == 0:
        print(f"  ... {done}/{len(syms)} سهم · بصمة حتى الآن {agg['fp_n']:,}", flush=True)

# ---- النتائج
def pct(a, b): return 100 * a / b if b else 0.0
print("\n================ نتيجة اختبار الحمض النووي ================")
print(f"أيام قابلة للتداول (بيني، العشوائي): {agg['base_n']:,}")
print(f"أيام البصمة: {agg['fp_n']:,}  (+ اختفى/شطب خلال 20ج: {agg['fp_delist']:,})")
print()
print(f"طار ×2 (+100%) خلال 20ج:  البصمة {pct(agg['fp_j100'], agg['fp_n']):.1f}%  مقابل العشوائي {pct(agg['base_j100'], agg['base_n']):.1f}%  (الرافعة ×{(pct(agg['fp_j100'],agg['fp_n'])/pct(agg['base_j100'],agg['base_n'])) if agg['base_j100'] else 0:.1f})")
print(f"طار ×1.5 (+50%) خلال 20ج: البصمة {pct(agg['fp_j50'], agg['fp_n']):.1f}%  مقابل العشوائي {pct(agg['base_j50'], agg['base_n']):.1f}%")
print(f"لمس -30% خلال 20ج (نزيف): البصمة {pct(agg['fp_bleed30'], agg['fp_n']):.1f}%")
print()
tr = agg["win"] + agg["stop"] + agg["flat"]
print(f"الصفقة (هدف +100% · وقف -30% · 20ج · افتتاح بكرة): {tr:,} صفقة")
print(f"  وصل الهدف: {pct(agg['win'], tr):.1f}%  ·  ضرب الوقف: {pct(agg['stop'], tr):.1f}%  ·  خرج بالأفق: {pct(agg['flat'], tr):.1f}%")
if fp_net:
    net = np.array(fp_net)
    print(f"  متوسط عائد الصفقة بعد التكاليف: {net.mean()*100:+.1f}%  ·  الوسيط: {np.median(net)*100:+.1f}%")
    print(f"  أعلى ربح وصله (وسيط): {np.median(fp_maxgain)*100:+.0f}%  ·  عائد 20ج بدون هدف/وقف (وسيط): {np.median(fp_ret20)*100:+.0f}%")
print()
print("حسب السنة (طار ×2): بصمة مقابل عشوائي")
for y in sorted(by_year):
    fn, fj, bn, bj = by_year[y]
    print(f"  {y}: البصمة {pct(fj,fn):4.1f}% ({fn:,} يوم) · العشوائي {pct(bj,bn):4.1f}% ({bn:,} يوم)")
print("\nملاحظة صدق: الفلووت مو داخل الاختبار التاريخي (Alpaca ما يعطيه قديماً) — البصمة هنا = منهار+طيّار+مضغوط+بيني+سيولة. الفلووت الذري يقوّيها نظرياً بس ما نقدر نثبته تاريخياً.")
