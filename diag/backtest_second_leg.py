"""
🔥 الموجة الثانية (OMH + منشور فيصل) — اختبار تاريخي (تشخيص فقط).

السؤال: سهم انفجر (أعلى سعر ≥ ضعف إغلاق أمس) ورجع — متى تجي موجة ثانية؟ وهل قفزة الأفتر + شروط فيصل تفرق؟
- المجال: كل أسهم Alpaca مع المشطوبة (يومي 3 سنين)، إغلاق قبل الانفجار 0.3-50$، سيولة ≥ 100 ألف$ يوم الانفجار.
- الانفجار E: يوم ≥ ×2 بعد 20 جلسة بدون انفجار. الأيام بعده t = E+1..E+10 (القرار بعد إغلاق t / آخر الأفتر).
- المحفز (شموع الساعة): قفزة أفتر يوم t ≥ +15% فوق الإغلاق، محافظ 70%+، فوليوم أفتر ≥ 100 ألف (مثل OMH: +130%).
- شروط فيصل (من منشوره عن OMH): (1) ما رجع للقاعدة: إغلاق t ≥ 1.5× إغلاق ما قبل الانفجار
  (2) RSI الأسبوعي ≤ 45 (الصورة الكبيرة لسا منهارة) (3) طيّار متكرر: انفجارين+ خلال سنة
  (4) مساحة صعود: أعلى إغلاق أسبوعي خلال سنة قبل الانفجار ≥ 3× سعر الدخول (شموع ساقطة فوق).
- النتيجة من الدخول (آخر سعر بالأفتر للمحفز، وإلا إغلاق t): أعلى سعر خلال 1 / 5 / 10 جلسات، كسر قمة الانفجار، إغلاق بعد 5.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
import alpaca_data as A
from backtest_runners2 import universe, rsi_np

DAYS = 1180
NY = "America/New_York"


def weekly_rsi(d, upto):
    w = d["Close"][d.index <= upto].resample("W-FRI").last().dropna()
    if len(w) < 16:
        return np.nan
    return float(rsi_np(w.to_numpy(float))[-1])


def scan_daily(job):
    """أحداث الانفجار + أيام ما بعده (ميزات يومية + نتايج من الإغلاق)."""
    sym, d = job
    if d is None or len(d) < 60:
        return []
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    jump = np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)]
    big = jump >= 2.0
    j50 = np.cumsum(np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)] >= 2.0)
    out = []
    last_e = -99
    for E in range(21, n - 2):
        if not big[E] or E - last_e <= 20:
            if big[E]:
                last_e = E
            continue
        last_e = E
        pre = c[E - 1]
        if not (0.3 <= pre <= 50) or h[E] * v[E] < 100_000:
            continue
        hiE = h[E]
        serial = int(j50[E] - j50[max(0, E - 250)])          # عدد الانفجارات خلال سنة (يشمل E)
        wk = d["Close"].iloc[max(0, E - 260):E].resample("W-FRI").last().dropna()
        room_ref = float(wk.max()) if len(wk) else np.nan
        for k in range(1, 11):
            t = E + k
            if t >= n - 1:
                break
            hi_since = h[E:t + 1].max()
            fut_h = h[t + 1:t + 11]
            fut_c = c[t + 1:t + 11]
            row = {"t": sym, "E": str(d.index[E].date()), "D": str(d.index[t].date()), "k": k, "pre": pre, "hiE": hiE,
                   "jumpE": float(jump[E]), "close": c[t], "low_since": float(l[E + 1:t + 1].min()),
                   "pull": float(c[t] / hi_since - 1), "above_base": float(c[t] / pre), "serial": serial,
                   "wrsi": weekly_rsi(d, d.index[t]), "room": float(room_ref / c[t]) if np.isfinite(room_ref) else np.nan,
                   "vol_t": float(v[t]), "rsi_d": float(rsi_np(c[:t + 1])[-1])}
            for m in (1, 5, 10):
                row[f"mx{m}"] = float(fut_h[:m].max() / c[t] - 1) if len(fut_h) >= m else np.nan
            row["newhigh10"] = float(fut_h[:10].max() > hiE) if len(fut_h) else np.nan
            row["c5"] = float(fut_c[4] / c[t] - 1) if len(fut_c) >= 5 else np.nan
            out.append(row)
    return out


def ah_features(ev):
    """شموع الساعة لنافذة الحدث: قفزة الأفتر لكل يوم t + النتيجة من آخر سعر بالأفتر."""
    sym, days = ev["t"], ev["days"]
    s = pd.Timestamp(days[0]).tz_localize(NY) - pd.Timedelta(days=1)
    e = pd.Timestamp(days[-1]).tz_localize(NY) + pd.Timedelta(days=16)
    try:
        b = A.bars([sym], "1Hour", s.tz_convert("UTC").to_pydatetime(), e.tz_convert("UTC").to_pydatetime()).get(sym)
    except Exception:
        return {}
    if b is None or b.empty:
        return {}
    b = b.copy()
    b["sess"] = A.session_of(b.index).values
    b["day"] = b.index.date
    alld = sorted(b["day"].unique())
    res = {}
    for D in days:
        Dd = pd.Timestamp(D).date()
        x = b[b.day == Dd]
        rth, ah = x[x.sess == "rth"], x[x.sess == "ah"]
        if rth.empty or ah.empty:
            continue
        close = float(rth["Close"].iloc[-1])
        ah_hi, ah_last, ah_vol = float(ah["High"].max()), float(ah["Close"].iloc[-1]), float(ah["Volume"].sum())
        jmp = ah_hi / close - 1
        hold = (ah_last - close) / (ah_hi - close) if ah_hi > close else 0.0
        later = b[(b.index > ah.index[-1])]
        nxt_days = [z for z in alld if z > Dd][:10]
        r = {"ah_jump": jmp, "ah_hold": hold, "ah_vol": ah_vol, "ah_last": ah_last}
        for m in (1, 5, 10):
            seg = later[later.day.isin(nxt_days[:m])]
            r[f"amx{m}"] = float(seg["High"].max() / ah_last - 1) if len(nxt_days) >= m and not seg.empty else np.nan
        res[(sym, D)] = r
    return res


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def summ(g, name, ah=False):
    if len(g) < 15:
        print(f"  {name:<46} n={len(g)} (قليل)")
        return
    p = "amx" if ah else "mx"
    parts = [f"n={len(g)} ({g.t.nunique()} سهم)",
             f"+50% باليوم التالي {pct((g[f'{p}1'] >= 0.5).mean())}",
             f"+100% خلال 5 {pct((g[f'{p}5'] >= 1.0).mean())}",
             f"+200% خلال 10 {pct((g[f'{p}10'] >= 2.0).mean())}",
             f"قمة جديدة فوق الانفجار {pct(g.newhigh10.mean())}",
             f"وسيط أعلى ربح 5 جلسات {g[f'{p}5'].median() * 100:+.0f}%",
             f"إغلاق بعد 5 (وسيط) {g.c5.median() * 100:+.0f}%"]
    print(f"  {name:<46} " + " · ".join(parts))


def main():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
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
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")]))
            for r in ex.map(scan_daily, jobs, chunksize=4):
                rows += r
            if (i // 100) % 15 == 0:
                print(f"  يومي {i + 100}/{len(syms)} · أيام بعد انفجار {len(rows)} · {round(time.time() - t0)}ث", flush=True)
    df = pd.DataFrame(rows)
    print(f"\nأحداث انفجار: {df.groupby(['t', 'E']).ngroups} · أيام بعدها: {len(df)} · أسهم {df.t.nunique()}", flush=True)
    # شموع الساعة لكل حدث (للمحفز)
    evs = [{"t": t, "days": sorted(g.D.unique())} for (t, E), g in df.groupby(["t", "E"])]
    AH = {}
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(ah_features, evs)):
            AH.update(r)
            if i % 300 == 0:
                print(f"  ساعة {i}/{len(evs)} · {round(time.time() - t0)}ث", flush=True)
    for col in ("ah_jump", "ah_hold", "ah_vol", "ah_last", "amx1", "amx5", "amx10"):
        df[col] = [AH.get((t, D), {}).get(col, np.nan) for t, D in zip(df.t, df.D)]
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/second_leg.csv.gz", index=False, compression="gzip")
    trig = (df.ah_jump >= 0.15) & (df.ah_hold >= 0.7) & (df.ah_vol >= 100_000)
    fais = (df.above_base >= 1.5) & (df.wrsi <= 45) & (df.serial >= 2) & (df.room >= 3)
    print("\n==== بعد الانفجار (الدخول = إغلاق اليوم) ====")
    summ(df, "كل أيام ما بعد الانفجار (1-10)")
    summ(df[df.k <= 3], "أول 3 أيام بعده")
    summ(df[fais], "+ شروط فيصل الأربعة (بدون محفز)")
    summ(df[df.above_base >= 1.5], "  شرط 1 بس: ما رجع للقاعدة (≥1.5×)")
    summ(df[df.wrsi <= 45], "  شرط 2 بس: RSI أسبوعي ≤ 45")
    summ(df[df.serial >= 2], "  شرط 3 بس: طيّار متكرر")
    summ(df[df.room >= 3], "  شرط 4 بس: مساحة صعود ×3")
    print("\n==== المحفز: قفزة أفتر محافظة (الدخول = آخر سعر بالأفتر) ====")
    has_ah = df.ah_jump.notna()
    summ(df[has_ah & ~trig], "أيام ما بعد الانفجار بدون قفزة أفتر", ah=True)
    summ(df[trig], "قفزة أفتر محافظة (15%+ · 70%+ · 100K+)", ah=True)
    summ(df[trig & (df.ah_jump >= 0.5)], "  قفزة أفتر 50%+", ah=True)
    summ(df[trig & fais], "قفزة أفتر + شروط فيصل الأربعة", ah=True)
    summ(df[trig & (df.above_base >= 1.5) & (df.serial >= 2)], "قفزة أفتر + ما رجع للقاعدة + طيّار متكرر", ah=True)
    summ(df[trig & (df.k <= 3)], "قفزة أفتر خلال أول 3 أيام", ah=True)
    print("\nOMH:")
    for _, x in df[df.t == "OMH"].iterrows():
        print(f"  E {x.E} t {x.D} (k={x.k}) إغلاق {x.close:.2f} · فوق القاعدة {x.above_base:.1f}× · RSI أسبوعي {x.wrsi:.0f} · "
              f"متكرر {x.serial} · مساحة {x.room:.1f}× · أفتر {pct(x.ah_jump)} محافظ {pct(x.ah_hold)} · بعدها 1/5/10: "
              f"{pct(x.amx1 if np.isfinite(x.amx1) else x.mx1)} / {pct(x.amx5 if np.isfinite(x.amx5) else x.mx5)}")
    top = df[trig & fais].sort_values("amx5", ascending=False).head(15)
    print("\nأكبر 15 (قفزة أفتر + شروط فيصل):")
    for _, x in top.iterrows():
        print(f"  {x.t} انفجار {x.E} · يوم {x.D} · أفتر +{x.ah_jump * 100:.0f}% · بعدها خلال 5: {pct(x.amx5)} · خلال 10: {pct(x.amx10)}")


if __name__ == "__main__":
    main()
