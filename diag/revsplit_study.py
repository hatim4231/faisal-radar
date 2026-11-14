"""📊 دراسة الأسهم المقسّمة (تقسيم عكسي) — طلب أبوفرح: متى تثبت ثم ترتفع +70%+، وكيف نجمّع.
فكرة فيصل: سهم يسوي تقسيم عكسي ← يهبط/ينجر ← يثبت (قاعدة) ← يرتفع ارتفاع جميل. الميزة: بعد التقسيم سعره فوق 1$
فمؤقتاً بعيد عن الشطب. الخطر: خبر طرح/تخفيف ← هبوط قوي.

3 سنين مع المشطوبة، بالسعر الحقيقي ذاك اليوم وبعد التكاليف. نقيس:
  1. كم جلسة من التقسيم للقاع (متى يثبت)؟
  2. من قاع ما بعد التقسيم: كم % يرتفع ≥70% خلال 20/40/60 جلسة؟ (مقابل العشوائي)
  3. أفضل طريقة دخول/تجميع: (أ) يوم التقسيم (ب) بعد 20 جلسة (ج) عند تأكد القاعدة (10 جلسات بدون قاع 20ج جديد).
  4. نافذة الأمان: متى يرجع تحت 1$ (يرجع خطر الشطب)؟
  5. خطر الطرح: كم % فيه هبوط يوم واحد ≥30% بعد التقسيم؟
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
W = 120               # نافذة ما بعد التقسيم (جلسات)
RISE = 0.70           # ارتفاع جميل
EVENT_FROM, EVENT_TO = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-05-20")  # عشان يكون قدّامها 120ج


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
        if d is None or len(d) < 60:
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


days_to_bottom, maxgain_from_low = [], []
hit = {20: 0, 40: 0, 60: 0}
entry_hit = {"split": [0, 0], "d20": [0, 0], "base": [0, 0]}   # [وصل+70%, عدد]
entry_ret = {"split": [], "d20": [], "base": []}
back_under1_days, back_under1 = [], 0
offering_drop = 0
n_events = 0
by_ratio = {}   # bucket ← [عدد, وصل+70% من القاع]
by_year = {}
base_days_list = []

# العشوائي: % يرتفع ≥70% خلال 60 من يوم بيني عشوائي
base_rand_n, base_rand_hit = 0, 0

syms = universe()
BATCH = 180
for i in range(0, len(syms), BATCH):
    try:
        data = build(syms[i:i + BATCH])
    except Exception as e:
        print("  دفعة خطأ:", repr(e)[:80], flush=True); continue
    for s, d in data.items():
        n = len(d)
        if n < 60:
            continue
        c = d["Close"].to_numpy(float); h = d["High"].to_numpy(float); lo = d["Low"].to_numpy(float); o = d["Open"].to_numpy(float)
        spl = d["Stock Splits"].to_numpy(float)
        praw = c * HR.raw_factor(d)
        idxdt = d.index
        # عينة عشوائية خفيفة للعشوائي (كل 25 يوم، سعر حقيقي بيني وسيولة)
        for j in range(60, n - 60, 25):
            if 0.5 <= praw[j] <= 15 and c[j] > 0:
                base_rand_n += 1
                base_rand_hit += int(h[j + 1:j + 61].max() >= 1.70 * c[j])
        # أحداث التقسيم العكسي
        ev = np.flatnonzero((spl > 0) & (spl < 1))
        for j in ev:
            dt = idxdt[j]
            if dt < EVENT_FROM or dt > EVENT_TO or j + W >= n:
                continue
            n_events += 1
            ratio = spl[j]
            rbucket = "1:≥10" if ratio <= 0.1 else "1:5–10" if ratio <= 0.2 else "1:<5"
            by_ratio.setdefault(rbucket, [0, 0]); by_ratio[rbucket][0] += 1
            yr = int(dt.year); by_year.setdefault(yr, [0, 0]); by_year[yr][0] += 1
            win_c = c[j + 1:j + 1 + W]; win_h = h[j + 1:j + 1 + W]
            lowk = int(np.argmin(win_c))
            low = float(win_c[lowk]); dtb = lowk + 1
            days_to_bottom.append(dtb)
            base_days_list.append(dtb)
            # من القاع: أعلى ارتفاع خلال 60 بعده
            after = win_h[lowk + 1: lowk + 1 + 60] if lowk + 1 < len(win_h) else np.array([low])
            mg = (after.max() / low - 1) if len(after) else 0.0
            maxgain_from_low.append(mg)
            got70 = mg >= RISE
            by_ratio[rbucket][1] += int(got70); by_year[yr][1] += int(got70)
            for wk in (20, 40, 60):
                seg = win_h[lowk + 1: lowk + 1 + wk]
                if len(seg) and seg.max() >= (1 + RISE) * low:
                    hit[wk] += 1
            # طرق الدخول (عائد بعد التكاليف، هدف +70% Limit خلال 60، وإلا إغلاق)
            def sim(entry_idx):
                if entry_idx >= n - 1:
                    return None
                e = o[entry_idx] if o[entry_idx] > 0 else c[entry_idx]
                seg_h = h[entry_idx + 1: entry_idx + 61]; seg_c = c[entry_idx + 1: entry_idx + 61]
                if not len(seg_c):
                    return None
                tgt = e * (1 + RISE)
                hit_t = seg_h.max() >= tgt
                ret = RISE if hit_t else (seg_c[-1] / e - 1)
                return hit_t, ret - cost_of(praw[entry_idx] if praw[entry_idx] > 0 else e)
            r = sim(j)                       # (أ) يوم التقسيم
            if r: entry_hit["split"][0] += int(r[0]); entry_hit["split"][1] += 1; entry_ret["split"].append(r[1])
            r = sim(j + 20)                  # (ب) بعد 20 جلسة
            if r: entry_hit["d20"][0] += int(r[0]); entry_hit["d20"][1] += 1; entry_ret["d20"].append(r[1])
            # (ج) تأكد القاعدة: أول يوم بعد التقسيم فيه 10 جلسات بدون قاع 20ج جديد
            be = None
            for k in range(j + 11, j + W):
                lo20 = lo[k - 19:k + 1].min()
                if lo[k - 9:k + 1].min() > lo20 * 1.0 and c[k] > lo20 * 1.03:
                    be = k; break
            if be is not None:
                base_days_list.append(be - j)
                r = sim(be)
                if r: entry_hit["base"][0] += int(r[0]); entry_hit["base"][1] += 1; entry_ret["base"].append(r[1])
            # نافذة الأمان: متى يرجع السعر الحقيقي تحت 1$
            prw = praw[j + 1:j + 1 + W]
            under = np.flatnonzero(prw < 1.0)
            if len(under):
                back_under1 += 1; back_under1_days.append(int(under[0]) + 1)
            # خطر الطرح: هبوط يوم ≥30%
            dd1 = win_c / np.r_[c[j], win_c[:-1]] - 1
            if (dd1 <= -0.30).any():
                offering_drop += 1
    if (i // BATCH) % 6 == 0:
        print(f"  ... {min(i+BATCH,len(syms))}/{len(syms)} · أحداث تقسيم {n_events}", flush=True)


def pct(a, b): return 100 * a / b if b else 0.0
def q(arr, p): return float(np.percentile(arr, p)) if arr else 0.0
print("\n================ دراسة الأسهم المقسّمة (تقسيم عكسي) ================")
print(f"عدد أحداث التقسيم العكسي ({EVENT_FROM.date()} → {EVENT_TO.date()}): {n_events:,}")
print()
print(f"⏱ متى تثبت (جلسات من التقسيم للقاع): الوسيط {q(days_to_bottom,50):.0f} · الربع الأول {q(days_to_bottom,25):.0f} · الربع الثالث {q(days_to_bottom,75):.0f}")
print(f"   (يعني نص الأسهم تقيعّ خلال أول ~{q(days_to_bottom,50):.0f} جلسة بعد التقسيم)")
print()
print(f"📈 ترتفع ≥70% من قاع ما بعد التقسيم خلال: 20ج {pct(hit[20],n_events):.0f}% · 40ج {pct(hit[40],n_events):.0f}% · 60ج {pct(hit[60],n_events):.0f}%")
print(f"   وسيط أعلى ارتفاع من القاع (60ج): +{q(maxgain_from_low,50)*100:.0f}% · الربع الثالث +{q(maxgain_from_low,75)*100:.0f}%")
print(f"   🎲 العشوائي (بيني، ≥70% خلال 60): {pct(base_rand_hit,base_rand_n):.0f}% (n={base_rand_n:,})")
print()
print("🎯 طرق الدخول (هدف +70% Limit خلال 60ج، بعد التكاليف):")
for k, lab in (("split","يوم التقسيم"),("d20","بعد 20 جلسة"),("base","عند تأكد القاعدة (10ج بدون قاع جديد)")):
    hh, nn = entry_hit[k]; rr = entry_ret[k]
    print(f"   {lab:38}: وصل +70% {pct(hh,nn):4.0f}% · متوسط العائد {np.mean(rr)*100 if rr else 0:+5.1f}% · وسيط {np.median(rr)*100 if rr else 0:+5.1f}% (n={nn:,})")
print()
print(f"🛡 نافذة الأمان من الشطب: {pct(back_under1,n_events):.0f}% رجعوا تحت 1$ خلال 120ج · وسيط الأيام للرجوع {q(back_under1_days,50):.0f}")
print(f"⚠ خطر الطرح: {pct(offering_drop,n_events):.0f}% فيهم هبوط يوم واحد ≥30% بعد التقسيم")
print()
print("حسب نسبة التقسيم (ترتفع ≥70% من القاع):")
for b in sorted(by_ratio):
    nn, hh = by_ratio[b]; print(f"   {b:7}: {pct(hh,nn):4.0f}% ({nn:,} حدث)")
print("حسب السنة:")
for y in sorted(by_year):
    nn, hh = by_year[y]; print(f"   {y}: {pct(hh,nn):4.0f}% ({nn:,} حدث)")
print("\nصدق: المقارنة بالعشوائي البيني بنفس الفترة. التكاليف مطروحة. الفلووت مو داخل (Alpaca ما يعطيه قديماً).")
