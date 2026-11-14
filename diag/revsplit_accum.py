"""تحقق من نمط أبوفرح بالضبط: تقسيم عكسي ← ينزل لـ≤60% من سعره بعد التقسيم ← يبني قاعدة ويجمّع (ما يكسر قاعه)
← ينفجر مضاربياً (قفزة داخل اليوم). نقيس: كم يوصل 60%، كم منهم يبني قاعدة، كم منهم ينفجر ≥+70%/+100% داخل اليوم،
وهل «البيع بالقفزة» (+40% Limit) يكسب — مقابل عشوائي. 3 سنين مع المشطوبة.
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
W = 160                       # نافذة ما بعد التقسيم
DROP = 0.60                   # ينزل لـ ≤60% من سعر ما بعد التقسيم
BASE_MIN = 10                 # جلسات القاعدة بدون كسر القاع
SPIKE = 0.70                  # قفزة جنونية
SELL = 0.40                   # نبيع بالقفزة عند +40% Limit
EV_FROM, EV_TO = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-04-01")


def cost_of(px): return 0.04 if px < 1 else 0.02 if px < 5 else 0.01


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


n_ev = 0
n_drop60 = 0            # نزل ≤60%
n_based = 0            # نزل 60% + بنى قاعدة 10ج بدون كسر القاع
spike70 = spike100 = 0  # من اللي بنى قاعدة: انفجر ≥+70% / +100% داخل اليوم خلال 40ج بعد القاعدة
same_day_dump = 0       # القفزة رجعت نفس اليوم (إغلاق ≤ 1.2× قبلها)
sell_ret = []           # عائد البيع بالقفزة (+40% Limit خلال 40ج)
days_drop, days_base_to_spike = [], []
# مقارنة: نسبة كل أحداث التقسيم تنفجر ≥+70% داخل اليوم خلال نفس النافذة (بدون شرط 60/قاعدة)
all_spike70 = 0
# عشوائي بيني: قفزة ≥+70% داخل اليوم خلال 40ج
rnd_n = rnd_spike = 0

syms = universe()
B = 180
for i in range(0, len(syms), B):
    try: data = build(syms[i:i + B])
    except Exception as e: print("دفعة خطأ:", repr(e)[:70], flush=True); continue
    for s, d in data.items():
        n = len(d)
        if n < 60: continue
        c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); lo = d["Low"].to_numpy(float); o = d["Open"].to_numpy(float)
        spl = d["Stock Splits"].to_numpy(float); praw = c * HR.raw_factor(d); idxdt = d.index
        for j in range(60, n - 80, 30):
            if 0.5 <= praw[j] <= 15:
                rnd_n += 1
                rnd_spike += int(h[j + 1:j + 41].max() >= 1.70 * c[j])
        for j in np.flatnonzero((spl > 0) & (spl < 1)):
            dt = idxdt[j]
            if dt < EV_FROM or dt > EV_TO or j + W >= n: continue
            n_ev += 1
            P0 = float(h[j:min(j + 4, n)].max())          # سعر ما بعد التقسيم
            win = slice(j + 1, j + 1 + W)
            cw, hw, lw = c[win], h[win], lo[win]
            # أي قفزة ≥+70% داخل اليوم خلال النافذة (للمقارنة العامة)
            if (hw / np.r_[c[j], cw[:-1]] - 1 >= SPIKE).any(): all_spike70 += 1
            # نزل ≤60% من P0
            below = np.flatnonzero(cw <= DROP * P0)
            if not len(below): continue
            n_drop60 += 1
            k0 = int(below[0]); days_drop.append(k0 + 1)
            gk = j + 1 + k0                                # مؤشر أول نزول 60%
            # قاعدة: 10ج بدون كسر القاع (ضمن 3%)
            be = None
            for k in range(gk + BASE_MIN, min(j + W, n)):
                seg = lo[max(0, k - BASE_MIN):k + 1]
                if not len(seg): continue
                base_low = seg.min()
                if lo[k] >= base_low * 0.97 and c[k] > base_low:   # ما كسر قاعه
                    be = k; base_low_v = base_low; break
            if be is None: continue
            n_based += 1
            days_base_to_spike_added = False
            entry = c[be]
            fh = h[be + 1:be + 41]; fc = c[be + 1:be + 41]; fcprev = np.r_[c[be], fc[:-1]]
            if not len(fh): continue
            # انفجار داخل اليوم ≥+70% / +100% (أعلى اليوم ÷ إغلاق أمس)
            intr = fh / np.maximum(fcprev, 1e-9) - 1
            if (intr >= SPIKE).any():
                spike70 += 1
                sidx = int(np.flatnonzero(intr >= SPIKE)[0])
                days_base_to_spike.append(sidx + 1)
                # تصريف نفس اليوم؟ إغلاق ذاك اليوم ≤ 1.2× إغلاق أمسه
                if fc[sidx] <= 1.2 * fcprev[sidx]: same_day_dump += 1
            if (intr >= 1.00).any(): spike100 += 1
            # البيع بالقفزة: Limit +40% من الدخول خلال 40ج، وإلا إغلاق آخر يوم
            e = o[be + 1] if be + 1 < n and o[be + 1] > 0 else entry
            tgt = e * (1 + SELL)
            ret = SELL if fh.max() >= tgt else (fc[-1] / e - 1)
            sell_ret.append(ret - cost_of(praw[be] if praw[be] > 0 else e))
    if (i // B) % 6 == 0: print(f"  ... {min(i+B,len(syms))}/{len(syms)} · تقسيم {n_ev} · نزل60 {n_drop60} · قاعدة {n_based}", flush=True)


def pc(a, b): return 100 * a / b if b else 0.0
def q(a, p): return float(np.percentile(a, p)) if a else 0.0
print("\n============ تحقق نمط التقسيم العكسي (نزول 60% ← قاعدة ← انفجار) ============")
print(f"أحداث تقسيم عكسي: {n_ev:,}")
print(f"نزل ≤60% من سعر ما بعد التقسيم: {n_drop60:,} ({pc(n_drop60,n_ev):.0f}%) · وسيط الأيام للنزول {q(days_drop,50):.0f}")
print(f"ثم بنى قاعدة {BASE_MIN}ج بدون كسر قاعه: {n_based:,} ({pc(n_based,n_drop60):.0f}% من اللي نزل · {pc(n_based,n_ev):.0f}% من الكل)")
print()
print(f"🚀 من اللي بنى قاعدة — انفجر داخل اليوم خلال 40ج:")
print(f"   ≥+70%: {pc(spike70,n_based):.0f}% · ≥+100%: {pc(spike100,n_based):.0f}% · وسيط أيام القاعدة→الانفجار {q(days_base_to_spike,50):.0f}")
print(f"   منهم تصريف نفس اليوم (رجع قريب سعره): {pc(same_day_dump,spike70):.0f}% ← يأكد: بيع بالقفزة")
print(f"   🎲 عشوائي بيني (قفزة ≥+70% داخل اليوم خلال 40ج): {pc(rnd_spike,rnd_n):.0f}%  ← الرافعة ×{(pc(spike70,n_based)/pc(rnd_spike,rnd_n)) if rnd_spike else 0:.1f}")
print(f"   كل أحداث التقسيم (بدون شرط القاعدة، قفزة خلال 160ج): {pc(all_spike70,n_ev):.0f}%")
print()
if sell_ret:
    sr = np.array(sell_ret)
    print(f"💰 «ادخل بالقاعدة، بيع بالقفزة +40% Limit» (40ج، بعد التكاليف): متوسط {sr.mean()*100:+.1f}% · وسيط {np.median(sr)*100:+.1f}% · رابحة {pc((sr>0).sum(),len(sr)):.0f}%")
print("\nصدق: المقارنة بعشوائي بيني بنفس الفترة. التكاليف مطروحة. البيع الحقيقي بالقفزة يعتمد على تنفيذك اللحظي.")
