"""
رادار الأفتر — متى نبيع؟ وهل فوليوم البري شرط مفيد؟ (تشخيص فقط، بيانات Alpaca، سنة)

نفس إشارات backtest_ah_alpaca (قفز 15%+ بالأفتر، حافظ 70%+، فوليوم أفتر 50K+، 0.5-5$).
الدخول: Limit عند فواب الأفتر، ينفذ إذا البري نزل له (ولو الفواب فوق آخر سعر ← الدخول بآخر سعر عند 4:00 البري).
طرق البيع (كلها بوقف −10%، ونفس الشمعة = وقف):
  OPEN    = بيع عند افتتاح السوق 9:30
  M945    = بيع 9:45 (أول 15 دقيقة)
  T5o     = هدف +5% وإلا بيع عند الافتتاح
  T10o    = هدف +10% وإلا بيع عند الافتتاح
  T15o    = هدف +15% وإلا بيع عند الافتتاح
  T10c    = هدف +10% وإلا بيع عند إغلاق اليوم (الطريقة الحالية)
  TRAIL   = بعد +8% وقف متحرك 5% من أعلى سعر، وإلا إغلاق اليوم
المقياس الأهم: متوسط ربح الصفقة (مو بس نسبة النجاح).
فوليوم البري: نحسبه لين لحظة التنفيذ (بدون نظر للمستقبل).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
from backtest_ah_alpaca import find_signals

NY = "America/New_York"
STOP = 0.10


def sim(seq, entry, tg=None, exit_at=None, trail=False):
    """seq: الشموع من بعد التنفيذ. exit_at: وقت البيع (Timestamp) أو None = إغلاق اليوم."""
    peak = entry
    for ts, r in seq.iterrows():
        if exit_at is not None and ts >= exit_at:
            return float(r["Open"]) / entry - 1
        if r["Low"] <= entry * (1 - STOP):
            return -STOP
        if trail:
            peak = max(peak, float(r["High"]))
            if peak >= entry * 1.08 and r["Low"] <= peak * 0.95:
                return peak * 0.95 / entry - 1
        elif tg and r["High"] >= entry * (1 + tg):
            return tg
    return float(seq["Close"].iloc[-1]) / entry - 1


def detail(sig):
    try:
        return _detail(sig)
    except Exception as e:
        return {**sig, "err": repr(e)[:60]}


def _detail(sig):
    s = pd.Timestamp(sig["D"]).tz_localize(NY) + pd.Timedelta(hours=16)
    e = pd.Timestamp(sig["N"]).tz_localize(NY) + pd.Timedelta(hours=16)
    b = A.bars([sig["t"]], "5Min", s.tz_convert("UTC").to_pydatetime(), e.tz_convert("UTC").to_pydatetime()).get(sig["t"])
    if b is None or b.empty:
        return {**sig, "err": "no 5m"}
    b["sess"] = A.session_of(b.index).values
    ah = b[(b.index.date == sig["D"]) & (b.sess == "ah")]
    nxt = b[b.index.date == sig["N"]]
    pm, rth = nxt[nxt.sess == "pm"], nxt[nxt.sess == "rth"]
    if ah.empty or pm.empty or rth.empty:
        return {**sig, "err": "empty"}
    ah_last = float(ah["Close"].iloc[-1])
    v = ah["Volume"].sum()
    vwap = float((ah["Close"] * ah["Volume"]).sum() / v) if v else ah_last
    if vwap < ah_last * 0.995:
        dip = pm[pm["Low"] <= vwap]
        if dip.empty:
            return {**sig, "fill": False}
        fi, entry = dip.index[0], vwap
    else:
        fi, entry = pm.index[0], float(pm["Open"].iloc[0])
    seq = pd.concat([pm, rth]).loc[fi:]
    seq_after = seq.iloc[1:]
    first = seq.iloc[0]
    # شمعة التنفيذ: نفحص الوقف بس
    if first["Low"] <= entry * (1 - STOP) or seq_after.empty:
        r = {k: -STOP for k in ("OPEN", "M945", "T5o", "T10o", "T15o", "T10c", "TRAIL")}
    else:
        o930 = rth.index[0]
        r = {"OPEN": sim(seq_after, entry, exit_at=o930),
             "M945": sim(seq_after, entry, exit_at=o930 + pd.Timedelta(minutes=15)),
             "T5o": sim(seq_after, entry, 0.05, exit_at=o930),
             "T10o": sim(seq_after, entry, 0.10, exit_at=o930),
             "T15o": sim(seq_after, entry, 0.15, exit_at=o930),
             "T10c": sim(seq_after, entry, 0.10),
             "TRAIL": sim(seq_after, entry, trail=True)}
    pm_before = float(pm.loc[:fi]["Volume"].sum())
    return {**sig, "fill": True, **r, "fill_min": int((fi.hour * 60 + fi.minute) - 240),
            "pm_vol_fill": pm_before, "pm_ratio_fill": pm_before / max(sig["ah_vol"], 1),
            "pm_ratio_all": float(pm["Volume"].sum()) / max(sig["ah_vol"], 1)}


EXITS = ("OPEN", "M945", "T5o", "T10o", "T15o", "T10c", "TRAIL")
NAMES = {"OPEN": "بيع عند الافتتاح", "M945": "بيع 9:45", "T5o": "+5% وإلا الافتتاح", "T10o": "+10% وإلا الافتتاح",
         "T15o": "+15% وإلا الافتتاح", "T10c": "+10% وإلا الإغلاق (الحالي)", "TRAIL": "وقف متحرك بعد +8%"}


def summ(sub, name):
    print(f"\n## {name} (n={len(sub)})")
    if len(sub) < 5:
        return
    for k in EXITS:
        x = sub[k] * 100
        print(f"   {NAMES[k]:<26} متوسط {x.mean():+5.2f}% · وسيط {x.median():+5.1f}% · رابحة {(x > 0).mean() * 100:3.0f}% · وقف {(x <= -9.99).mean() * 100:3.0f}%")


def main():
    try:
        A.bars(["AAPL"], "1Day", datetime.now(timezone.utc) - timedelta(days=10))
        print("✅ مفاتيح Alpaca شغالة", flush=True)
    except Exception as e:
        print("⛔ Alpaca:", repr(e)[:150]); sys.exit(1)
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    start = datetime.now(timezone.utc) - timedelta(days=365)
    sigs, t0 = [], time.time()
    for i in range(0, len(tickers), 40):
        try:
            hb = A.bars(tickers[i:i + 40], "1Hour", start)
        except Exception as e:
            print("err", repr(e)[:120], flush=True); continue
        for s, h in hb.items():
            sigs += [x for x in find_signals(s, h) if x["hold"] >= 0.7]
    print("إشارات (قفز وحافظ):", len(sigs), f"{round(time.time() - t0)}ث", flush=True)
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(detail, sigs))
    df = pd.DataFrame([r for r in rows if "err" not in r])
    F = df[df["fill"] == True].copy()  # noqa: E712
    print(f"تنفذ الـ Limit: {len(F)} من {len(df)}", flush=True)
    print("\n==== طرق البيع (وقف −10% للكل) ====")
    summ(F, "كل الصفقات المنفذة")
    q = F["pm_ratio_fill"].median()
    print(f"\n(وسيط فوليوم البري لين التنفيذ ÷ فوليوم الأفتر = {q:.2f})")
    summ(F[F.pm_ratio_fill >= q], "فوليوم البري لين التنفيذ قوي (فوق الوسيط) — يُعرف وقت الدخول")
    summ(F[F.pm_ratio_fill < q], "فوليوم البري لين التنفيذ ضعيف")
    summ(F[F.pm_ratio_all >= 0.5], "فوليوم البري الكامل ≥ نص الأفتر (فيه نظر للمستقبل — للمقارنة)")
    summ(F[F.fill_min <= 120], "التنفيذ قبل 6:00 صباحاً")
    summ(F[F.fill_min > 120], "التنفيذ بعد 6:00")
    summ(F[F.jump >= 0.30], "قفزة 30%+")
    summ(F[F.close < 1], "سعر تحت 1$")
    summ(F[F.close >= 1], "سعر 1-5$")
    mid = F["D"].astype(str).sort_values().iloc[len(F) // 2]
    summ(F[F.D.astype(str) < mid], "النص الأول من السنة")
    summ(F[F.D.astype(str) >= mid], "النص الثاني من السنة")


if __name__ == "__main__":
    main()
