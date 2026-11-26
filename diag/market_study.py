"""
🕘 دراسة «سهم الماركت» (نظرية عبدالإله، طلب أبوفرح 2026-11-26) — هل «ثبات دعم البري بعد الاختبار» يعطي +20% فعلاً؟
1) كل قفزات الأفتر ≥ +15% (فوليوم أفتر ≥ 50 ألف، الجلسة ≥ 100 ألف، سعر 0.5-10$) آخر سنة، مع المشطوبة (أسهم Alpaca النشطة والموقوفة).
2) لكل قفزة: شموع 5 دقايق ليوم الأفتر واليوم التالي ← market.classify (نفس كود الرادار بالضبط).
3) المقياس من سعر 10:00 (بعد نص ساعة الاختبار) لكل الأحداث — عشان نقارن «ثبت الدعم» بكل قفزات الأفتر بنفس الدخول:
   +20% بنفس اليوم · +20% خلال يومين · الهدف قبل الإغلاق تحت دعم الأفتر · العائد بعد التكاليف (4% تحت 1$ · 2% تحت 5$ · 1%).
تشخيص فقط. هامش 95% مجمّع بالسهم.
"""
import os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import alpaca_data as A
import market as M
from backtest_runners2 import universe

DAYS = int(os.environ.get("DATA_RUN") or 365)
COST = lambda p: 0.04 if p < 1 else (0.02 if p < 5 else 0.01)


def candidates():
    syms = sorted(universe())
    start = datetime.now(timezone.utc) - timedelta(days=DAYS + 10)
    keep = []
    for i in range(0, len(syms), 100):
        try:
            raw = A.bars(syms[i:i + 100], "1Day", start)
        except Exception as e:
            print("daily err", repr(e)[:90], flush=True); continue
        for s, d in raw.items():
            c, v = d["Close"].values, d["Volume"].values
            if len(c) and ((c >= 0.5) & (c <= 10) & (v >= M.MIN_RTH_VOL)).any():
                keep.append(s)
    print(f"أسهم بالمجال: {len(keep)} من {len(syms)}", flush=True)
    return keep


def ah_days(syms):
    """أيام فيها قفزة أفتر (من شموع الساعة) — تقريب أولي، والتأكيد بشموع 5 دقايق."""
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    ev, t0 = [], time.time()
    for i in range(0, len(syms), 40):
        try:
            hb = A.bars(syms[i:i + 40], "1Hour", start)
        except Exception as e:
            print("hourly err", repr(e)[:90], flush=True); continue
        for s, h in hb.items():
            h = h.copy(); h.index = h.index.tz_convert(M.NY)
            m = h.index.hour * 60 + h.index.minute
            day = h.index.normalize()
            rth = h[(m >= 570) & (m < 960)]; ah = h[(m >= 960) & (m < 1200)]
            if rth.empty or ah.empty:
                continue
            cl = rth.groupby(rth.index.normalize())["Close"].last()
            rv = rth.groupby(rth.index.normalize())["Volume"].sum()
            ahh = ah.groupby(ah.index.normalize())["High"].max()
            ahv = ah.groupby(ah.index.normalize())["Volume"].sum()
            j = pd.concat([cl, rv, ahh, ahv], axis=1, keys=["c", "rv", "hh", "hv"]).dropna()
            j = j[(j.hh / j.c - 1 >= M.MIN_JUMP) & (j.hv >= M.MIN_AH_VOL) & (j.rv >= M.MIN_RTH_VOL) & (j.c >= 0.5) & (j.c <= 10)]
            ev += [(s, d) for d in j.index]
        if (i // 40) % 10 == 0:
            print(f"  ساعة {i + 40}/{len(syms)} · أحداث {len(ev)} · {round(time.time() - t0)}ث", flush=True)
    return ev


def study(ev):
    rows, t0 = [], time.time()
    for k, (s, d) in enumerate(ev):
        st = (d + pd.Timedelta(hours=9)).tz_convert("UTC").to_pydatetime()
        en = (d + pd.Timedelta(days=5)).tz_convert("UTC").to_pydatetime()
        if en > datetime.now(timezone.utc) - timedelta(minutes=20):
            continue
        try:
            b = A.bars([s], "5Min", st, en).get(s)
        except Exception:
            continue
        if b is None or b.empty:
            continue
        b = b.copy(); b.index = b.index.tz_convert(M.NY)
        days = sorted(set(b.index.date))
        if len(days) < 3 or days[0] != d.date():
            continue
        DD = b[b.index.date == days[0]]; N = b[b.index.date == days[1]]; N2 = b[b.index.date == days[2]]
        e = M.ah_event(DD)
        if not e:
            continue
        c = M.classify(e, N)
        pm, rth, _ = M.sessions(N)
        r = {"t": s, "D": str(days[0]), "status": c["status"], "jump": e["jump_pct"], "give": e["giveback_pct"], "px": e["close"]}
        pre = rth[[(i.hour, i.minute) < (10, 0) for i in rth.index]]
        post = rth[[(i.hour, i.minute) >= (10, 0) for i in rth.index]]
        _, rth2, _ = M.sessions(N2)
        if len(pre) and len(post):
            e10 = float(pre["Close"].iloc[-1]); r["e10"] = e10
            r["hit_day"] = float(post["High"].max()) >= e10 * 1.2
            r["hit_2d"] = r["hit_day"] or (len(rth2) and float(rth2["High"].max()) >= e10 * 1.2)
            seq = pd.concat([post, rth2])
            ra = M.result_after(e, {"entry": e10}, seq)
            r["res"], r["ret"] = ra["res"], ra["ret"] - COST(e10)
            r["ret_close"] = float(post["Close"].iloc[-1]) / e10 - 1 - COST(e10)
            r["tested"] = c.get("tested")
        rows.append(r)
        if k % 300 == 0:
            print(f"  5د {k}/{len(ev)} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(rows)


def ci(g, col):
    x = g.dropna(subset=[col])
    if len(x) < 10:
        return "  -  "
    m = x.groupby("t")[col].mean()
    return f"{x[col].mean()*100:5.1f}±{1.96*m.std()/max(1,len(m))**.5*100:.1f}"


def report(df):
    df = df[df.e10.notna()] if "e10" in df else df
    print(f"\nأحداث مكتملة: {len(df):,} · أسهم {df.t.nunique():,} · {df.D.min()} → {df.D.max()}")
    lab = {"held": "✅ ثبت دعم البري (سهم الماركت + اختبار)", "fail": "❌ فشل باختبار الماركت", "cancel_pm": "❌ البري كسر الدعم (قاعدة 2)",
           "cancel_ah": "❌ جنى وهبط بالأفتر (قاعدة 1)", "market": "🟡 ماركت بدون بيانات اختبار"}
    print(f"{'المجموعة':<44} {'n':>5}  {'+20% نفس اليوم':>14} {'+20% خلال يومين':>15} {'هدف قبل وقف':>12} {'عائد صافي (هدف/وقف/يومين)':>26} {'إغلاق اليوم صافي':>16}")
    def line(name, g):
        tgt = (g.res == "target").mean() * 100 if len(g) else 0
        print(f"{name:<44} {len(g):>5}  {ci(g.assign(h=g.hit_day.astype(float)),'h'):>14} {ci(g.assign(h=g.hit_2d.astype(float)),'h'):>15} {tgt:11.0f}% {ci(g,'ret'):>26} {ci(g,'ret_close'):>16}")
    line("كل قفزات الأفتر (الدخول 10:00 لأي حدث)", df)
    for k, nm in lab.items():
        line(nm, df[df.status == k])
    h = df[df.status == "held"]
    line("   ✅ منها: نزل واختبر دعم البري", h[h.tested == True])
    line("   ✅ منها: ما نزل يختبره", h[h.tested == False])
    mid = df.D.sort_values().iloc[len(df) // 2]
    for nm, part in (("النص الأول", df[df.D < mid]), ("النص الثاني", df[df.D >= mid])):
        line(f"{nm}: الكل", part); line(f"{nm}: ✅ ثبت", part[part.status == "held"])
    for nm, part in (("تحت 1$", df[df.px < 1]), ("فوق 1$", df[df.px >= 1])):
        line(f"{nm}: الكل", part); line(f"{nm}: ✅ ثبت", part[part.status == "held"])
    print("\n⚠ تشخيص — بيانات Alpaca (تأخير 15 دقيقة مجاناً؛ معدّلة للتقسيم). المرجع = كل قفزات الأفتر بنفس الدخول 10:00، مو عشوائي مطابق.")


def main():
    os.makedirs("diag_out", exist_ok=True)
    ev = ah_days(candidates())
    print(f"قفزات أفتر (ساعة): {len(ev):,}", flush=True)
    df = study(ev)
    df.to_csv("diag_out/market_study.csv.gz", index=False, compression="gzip")
    report(df)


if __name__ == "__main__":
    main()
