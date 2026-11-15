"""سؤال أبوفرح: سهم مثل NUWE (فلووت صغير + تقسيم عكسي حديث + أرباح فيها إيجابية) — تاريخياً المتوقع ارتفاع أو هبوط؟
نجيب وضع NUWE بالضبط، ثم نقارن بكل الحالات المشابهة (تقسيم عكسي حديث + نزل، نفس وضعه) على 3 سنين مع المشطوبة.
ملاحظة صدق: الفلووت التاريخي مو متوفر (Alpaca) — الدراسة على «تقسيم حديث + نزل»، والفلووت الصغير نوعياً يضخّم الحركة (صعود وهبوط).
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
from backtest_highrisk import universe

START = datetime(2021, 6, 1, tzinfo=timezone.utc)
ASOF = 20                 # نقيس «بعد 20 جلسة من التقسيم» (قريب وضع NUWE)
DROP_MIN = 0.35           # نزل ≥35% من سعر ما بعد التقسيم (مثل NUWE)
EV_FROM, EV_TO = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-08-01")


def build(sb):
    raw = A.bars(sb, "1Day", START)
    try: spl = A.splits(sb, START)
    except Exception: spl = {}
    out = {}
    for s, d in raw.items():
        if d is None or len(d) < 40: continue
        d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d["Stock Splits"] = 0.0
        for dt, r in spl.get(s, []):
            dt = pd.Timestamp(dt).normalize()
            if dt in d.index: d.loc[dt, "Stock Splits"] = r
        out[s] = d
    return out


# ---- وضع NUWE الحالي
print("========== وضع NUWE الحالي ==========")
try:
    nd = A.daily(["NUWE"], days=400).get("NUWE")
    if nd is not None and len(nd):
        nd = nd.sort_index()
        sp = nd["Stock Splits"].to_numpy(float); ri = np.flatnonzero((sp > 0) & (sp < 1))
        c = nd["Close"].to_numpy(float); h = nd["High"].to_numpy(float); t = len(c) - 1
        print(f"السعر الحالي {c[t]:.3f} · التاريخ {str(nd.index[t])[:10]}")
        if len(ri):
            rj = int(ri[-1]); p0 = float(h[rj:rj+4].max())
            print(f"آخر تقسيم عكسي: 1:{round(1/sp[rj])} يوم {str(nd.index[rj])[:10]} (قبل {t-rj} جلسة) · سعر ما بعد التقسيم {p0:.3f}")
            print(f"نزل {( c[t]/p0 - 1)*100:.0f}% من سعر ما بعد التقسيم · قاع ما بعده {float(c[rj+1:].min()):.3f}")
        print("آخر 8 جلسات (إغلاق · أعلى · فوليوم):")
        for i in range(max(0, t-7), t+1):
            print(f"  {str(nd.index[i])[:10]} {c[i]:.3f} · {h[i]:.3f} · {int(nd['Volume'].iloc[i]):,}")
    try:
        import yfinance as yf
        info = yf.Ticker("NUWE").info or {}
        print(f"فلووت {info.get('floatShares')} · أسهم {info.get('sharesOutstanding')} · قيمة {info.get('marketCap')} · شورت {info.get('sharesShort')} · دولة {info.get('country')}")
    except Exception as e:
        print("ياهو:", repr(e)[:80])
except Exception as e:
    print("NUWE تعذّر:", repr(e)[:120])

# ---- الدراسة التاريخية: تقسيم عكسي حديث + نزل، المتوقع بعدها
# عند «20 جلسة بعد التقسيم» إذا نزل ≥35% ← نقيس 20 و40 جلسة قدّام
fwd = {20: [], 40: []}      # عائد الإغلاق
mg = {20: [], 40: []}       # أعلى ارتفاع
md = {20: [], 40: []}       # أقصى هبوط
hit = {"+30": 0, "+50": 0, "+100": 0, "-30": 0, "-50": 0}
n = 0
# خط الأساس: أيام بيني عشوائية نفس الفترة (للمقارنة)
b_fwd20, b_n = [], 0

syms = universe()
B = 180
for i in range(0, len(syms), B):
    try: data = build(syms[i:i+B])
    except Exception as e: print("دفعة خطأ:", repr(e)[:70], flush=True); continue
    for s, d in data.items():
        c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); lo = d["Low"].to_numpy(float)
        sp = d["Stock Splits"].to_numpy(float); idx = d.index; N = len(c)
        for j in np.flatnonzero((sp > 0) & (sp < 1)):
            dt = idx[j]
            if dt < EV_FROM or dt > EV_TO: continue
            k = j + ASOF                    # 20 جلسة بعد التقسيم
            if k + 40 >= N: continue
            p0 = float(h[j:j+4].max())
            if p0 <= 0 or not (0.3 <= c[k] <= 20): continue
            if c[k] / p0 - 1 > -DROP_MIN:   # لازم يكون نزل ≥35% (مثل NUWE)
                continue
            n += 1
            e = c[k]
            for H in (20, 40):
                seg_h = h[k+1:k+1+H]; seg_l = lo[k+1:k+1+H]; fc = c[k+H]
                fwd[H].append(fc/e - 1); mg[H].append(seg_h.max()/e - 1); md[H].append(seg_l.min()/e - 1)
            sh20 = h[k+1:k+41]; sl20 = lo[k+1:k+41]
            hit["+30"] += int(sh20.max() >= 1.30*e); hit["+50"] += int(sh20.max() >= 1.50*e); hit["+100"] += int(sh20.max() >= 2.0*e)
            hit["-30"] += int(sl20.min() <= 0.70*e); hit["-50"] += int(sl20.min() <= 0.50*e)
        for j in range(60, N-45, 30):       # عشوائي بيني
            if 0.3 <= c[j] <= 20:
                b_n += 1; b_fwd20.append(c[j+40]/c[j]-1)
    if (i//B) % 6 == 0: print(f"  ... {min(i+B,len(syms))}/{len(syms)} · حالات {n}", flush=True)


def pct(a, b): return 100*a/b if b else 0
def med(x): return float(np.median(x))*100 if x else 0
def mean(x): return float(np.mean(x))*100 if x else 0
print(f"\n========== الدراسة: تقسيم عكسي حديث + نزل ≥35% (مثل NUWE)، المتوقع بعدها ==========")
print(f"عدد الحالات (3 سنين، مع المشطوبة): {n:,}")
print(f"\nخلال 20 جلسة قدّام:  وسيط العائد {med(fwd[20]):+.0f}% · متوسط {mean(fwd[20]):+.0f}% · وسيط أعلى ارتفاع {med(mg[20]):+.0f}% · وسيط أقصى هبوط {med(md[20]):+.0f}%")
print(f"خلال 40 جلسة قدّام:  وسيط العائد {med(fwd[40]):+.0f}% · متوسط {mean(fwd[40]):+.0f}% · وسيط أعلى ارتفاع {med(mg[40]):+.0f}% · وسيط أقصى هبوط {med(md[40]):+.0f}%")
print(f"\nخلال 40 جلسة — كم مرة لمس:")
print(f"  ارتفاع +30%: {pct(hit['+30'],n):.0f}% · +50%: {pct(hit['+50'],n):.0f}% · +100%: {pct(hit['+100'],n):.0f}%")
print(f"  هبوط  -30%: {pct(hit['-30'],n):.0f}% · -50%: {pct(hit['-50'],n):.0f}%")
print(f"\n🎲 العشوائي البيني (نفس الفترة، 40 جلسة): وسيط العائد {med(b_fwd20):+.0f}% · متوسط {mean(b_fwd20):+.0f}% (n={b_n:,})")
print("\nصدق: الفلووت مو داخل (Alpaca). NUWE فلوته ~384 ألف = يضخّم الحركة (صعود وهبوط). الأرباح الإيجابية = محفّز، بس الأرقام فوق هي خط الأساس بدون محفّز.")
