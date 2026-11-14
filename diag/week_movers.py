"""كل الأسهم اللي ارتفعت ≥100% هذا الأسبوع (9–13 نوفمبر 2026) — من كل أسهم السوق (Alpaca).
نطبع: الرمز · الاسم · يوم القفزة · إغلاق أمس → إغلاق اليوم · القفزة% · أعلى داخل اليوم% · الفوليوم · تداول$ · السعر قبل.
المعيار: إغلاق أو أعلى ≥ 2× إغلاق اليوم اللي قبله، وتداول اليوم ≥ 300 ألف$ (عشان ما نحسب صفقة شاذة بسهم ميت).
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "highrisk_radar"))

import numpy as np
import pandas as pd
import alpaca_data as A
import highrisk_scanner as HS

WEEK = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
MIN_DV = 300_000.0   # تداول$ اليوم
MIN_JUMP = 1.00      # ≥ +100%

syms = HS.alpaca_universe()
print(f"المجال: {len(syms)} سهم نشط من Alpaca")

start = datetime(2026, 10, 20, tzinfo=timezone.utc)  # ~17 جلسة تكفي
hits = []
BATCH = 200
for i in range(0, len(syms), BATCH):
    chunk = syms[i:i + BATCH]
    try:
        got = A.bars(chunk, "1Day", start)
    except Exception as e:
        print("  خطأ دفعة:", repr(e)[:80]); continue
    for s, d in got.items():
        if d is None or len(d) < 3:
            continue
        d = d.sort_index()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        c = d["Close"].astype(float)
        h = d["High"].astype(float)
        v = d["Volume"].astype(float)
        for ds in WEEK:
            t = pd.Timestamp(ds)
            if t not in d.index:
                continue
            j = d.index.get_loc(t)
            if j == 0:
                continue
            prev = float(c.iloc[j - 1])
            if prev <= 0:
                continue
            cl = float(c.iloc[j]); hi = float(h.iloc[j]); vol = float(v.iloc[j])
            dv = cl * vol
            jump_c = cl / prev - 1
            jump_h = hi / prev - 1
            if max(jump_c, jump_h) >= MIN_JUMP and dv >= MIN_DV:
                hits.append({"s": s, "name": HS.NAMES.get(s, "")[:26], "day": ds, "prev": prev,
                             "close": cl, "high": hi, "jc": jump_c * 100, "jh": jump_h * 100,
                             "vol": vol, "dv": dv})

hits.sort(key=lambda x: -x["jh"])
print(f"\n=== {len(hits)} سهم ارتفع ≥100% (إغلاق أو أعلى) هذا الأسبوع ===")
print(f"{'الرمز':6} {'اليوم':11} {'قبل':>7} {'إغلاق':>7} {'أعلى':>7} {'قفزة إغلاق':>10} {'قفزة أعلى':>10} {'فوليوم':>13} {'تداول$':>12}  الاسم")
for x in hits:
    print(f"{x['s']:6} {x['day']:11} {x['prev']:>7.3f} {x['close']:>7.3f} {x['high']:>7.3f} {x['jc']:>+9.0f}% {x['jh']:>+9.0f}% {int(x['vol']):>13,} {x['dv']/1e6:>10.1f}M  {x['name']}")

# ملخص بالرمز (لو طار أكثر من يوم)
uniq = {}
for x in hits:
    uniq.setdefault(x["s"], x)
print(f"\nعدد الرموز الفريدة: {len(uniq)}")
