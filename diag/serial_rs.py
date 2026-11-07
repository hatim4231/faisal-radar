"""
شركات «مثل NUWE»: وش يصير لها بعد إعلان الأرباح؟ (تشخيص فقط)
- الأسهم: كل أسهمنا (0.5-5$) اللي سوت تقسيمين عكسيين أو أكثر خلال 24 شهر (مثل NUWE: 3 تقسيمات).
- الأحداث: إعلانات الأرباح (8-K بند 2.02) خلال آخر سنتين (اللي مر عليها 40 جلسة على الأقل).
- لكل إعلان: هل جاه تمويل خلال 30 / 60 يوم (424B1/424B4/424B5، S-1/F-1 جديد، أو 8-K بند 3.02)؟
  والعائد: يوم الإعلان (+1 جلسة)، +5، +20، +40، وأقصى صعود/نزول خلال 20 جلسة.
- المقارنة: نفس الشي لعينة من الشركات اللي ما سوت تقسيم عكسي.
- مجموعة «الأقرب لـ NUWE»: إعلان أرباح خلال 45 يوم بعد تقسيم عكسي.
تنبيه: الأسهم الحالية بس (اللي انشطبت ما تظهر) ← النتايج متفائلة شوي.
"""
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import sec_tools

FIN_FORMS = ("424B1", "424B4", "424B5", "S-1", "F-1")


def filings(t):
    cik = sec_tools._cik(t)
    if not cik:
        return None
    js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
    df = pd.DataFrame({k: js[k] for k in ("form", "filingDate", "items")})
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    return df


def events(t, d, f, rs_dates):
    out = []
    earn = f[(f.form == "8-K") & f["items"].fillna("").str.contains("2.02")]["filingDate"]
    fin = f[f.form.isin(FIN_FORMS) | ((f.form == "8-K") & f["items"].fillna("").str.contains("3.02"))]["filingDate"]
    idx = d.index
    for e in earn:
        if e < idx[0] + pd.Timedelta(days=5):
            continue
        p = idx.searchsorted(e)
        if p < 1 or p + 40 >= len(d):
            continue
        c0 = float(d["Close"].iloc[p - 1])
        if not c0:
            continue
        fut = d.iloc[p:p + 20]
        rs_before = [r for r in rs_dates if pd.Timedelta(0) <= e - r <= pd.Timedelta(days=45)]
        out.append({"t": t, "date": e, "px": c0,
                    "fin30": int(any((x > e) and (x <= e + pd.Timedelta(days=30)) for x in fin)),
                    "fin60": int(any((x > e) and (x <= e + pd.Timedelta(days=60)) for x in fin)),
                    "fin_before30": int(any((x <= e) and (x > e - pd.Timedelta(days=30)) for x in fin)),
                    "r1": float(d["Close"].iloc[p + 1]) / c0 - 1, "r5": float(d["Close"].iloc[p + 5]) / c0 - 1,
                    "r20": float(d["Close"].iloc[p + 20]) / c0 - 1, "r40": float(d["Close"].iloc[p + 40]) / c0 - 1,
                    "up20": float(fut["High"].max()) / c0 - 1, "dn20": float(fut["Low"].min()) / c0 - 1,
                    "up2": float(d["High"].iloc[p:p + 2].max()) / c0 - 1,
                    "after_rs": int(bool(rs_before))})
    return out


def summary(df, name):
    if df.empty:
        print(f"\n## {name}: لا يوجد"); return
    print(f"\n## {name}  (إعلانات={len(df)} · شركات={df.t.nunique()})")
    print(f"   تمويل خلال 30 يوم: {df.fin30.mean():.0%} · خلال 60 يوم: {df.fin60.mean():.0%}")
    for k, lab in (("r1", "يوم الإعلان (+1)"), ("r5", "+5 جلسات"), ("r20", "+20 جلسة"), ("r40", "+40 جلسة")):
        x = df[k]
        print(f"   {lab:<16} وسيط {x.median():+.1%} · متوسط {x.mean():+.1%} · طلع {(x > 0).mean():.0%} · "
              f"نزل 20%+ {(x <= -0.2).mean():.0%} · نزل 50%+ {(x <= -0.5).mean():.0%}")
    print(f"   أقصى صعود خلال 20 جلسة: وسيط {df.up20.median():+.0%} · وصل +30% {(df.up20 >= 0.3).mean():.0%} · "
          f"وصل +100% {(df.up20 >= 1).mean():.0%}")
    print(f"   أقصى صعود أول جلستين: وسيط {df.up2.median():+.0%} · وصل +20% {(df.up2 >= 0.2).mean():.0%}")
    print(f"   أقصى نزول خلال 20 جلسة: وسيط {df.dn20.median():+.0%} · نزل 30%+ {(df.dn20 <= -0.3).mean():.0%}")
    a, b = df[df.fin30 == 1], df[df.fin30 == 0]
    if len(a) >= 5 and len(b) >= 5:
        print(f"   مع تمويل خلال 30 يوم: +20 جلسة وسيط {a.r20.median():+.1%} (n={len(a)}) · بدون: {b.r20.median():+.1%} (n={len(b)})")


def main():
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    if "NUWE" not in tickers:
        tickers.append("NUWE")
    start = datetime.now(timezone.utc) - timedelta(days=740)
    spl = {}
    for i in range(0, len(tickers), 100):
        try:
            spl.update(A.splits(tickers[i:i + 100], start))
        except Exception as e:
            print("splits err", repr(e)[:100], flush=True)
    rs = {t: sorted(d for d, r in v if r < 1) for t, v in spl.items()}
    serial = [t for t, v in rs.items() if len(v) >= 2]
    any_rs = [t for t, v in rs.items() if len(v) >= 1]
    none_rs = [t for t in tickers if t not in any_rs]
    print(f"أسهم: {len(tickers)} · سوت تقسيم عكسي: {len(any_rs)} · تقسيمين+: {len(serial)}", flush=True)
    print("التقسيمات المتكررة:", ", ".join(f"{t}({len(rs[t])})" for t in sorted(serial)), flush=True)
    random.seed(5)
    control = random.sample(none_rs, min(150, len(none_rs)))
    groups = {"serial": serial, "one_rs": [t for t in any_rs if t not in serial], "control": control}
    rows = []
    allt = sorted(set(serial) | set(groups["one_rs"]) | set(control))
    daily = {}
    for i in range(0, len(allt), 100):
        try:
            daily.update(A.bars(allt[i:i + 100], "1Day", start))
        except Exception as e:
            print("daily err", repr(e)[:100], flush=True)
    t0 = time.time()
    for g, ts in groups.items():
        for t in ts:
            d = daily.get(t)
            if d is None or len(d) < 80:
                continue
            d = d.copy()
            d.index = d.index.tz_localize(None).normalize()
            try:
                f = filings(t)
            except Exception:
                f = None
            if f is None:
                continue
            for e in events(t, d, f, rs.get(t, [])):
                e["group"] = g
                rows.append(e)
        print(g, "خلص", round(time.time() - t0), "ث", flush=True)
    df = pd.DataFrame(rows)
    print("\n==== النتائج ====")
    summary(df[df.group == "serial"], "شركات بتقسيمين عكسيين أو أكثر (مثل NUWE)")
    summary(df[(df.group == "serial") & (df.after_rs == 1)], "⭐ الأقرب لـ NUWE: تقسيمين+ والإعلان خلال 45 يوم بعد تقسيم")
    summary(df[(df.group == "serial") & (df.fin_before30 == 0)], "تقسيمين+ وما جاها تمويل في الـ 30 يوم قبل الإعلان")
    summary(df[(df.group == "serial") & (df.px < 10)], "تقسيمين+ والسعر قبل الإعلان (معدّل) تحت 10$")
    summary(df[df.group == "one_rs"], "شركات بتقسيم عكسي واحد")
    summary(df[df.group == "control"], "مقارنة: شركات بدون تقسيم عكسي")
    s = df[(df.group == "serial") & (df.after_rs == 1)].sort_values("date")
    print("\nأمثلة الأقرب لـ NUWE:")
    for r in s.itertuples():
        print(f"  {r.t} {r.date.date()} · +1 {r.r1:+.0%} · +5 {r.r5:+.0%} · +20 {r.r20:+.0%} · أقصى صعود20 {r.up20:+.0%} · "
              f"تمويل30 {'نعم' if r.fin30 else 'لا'}")
    n = df[df.t == "NUWE"]
    print("\nNUWE نفسها:")
    for r in n.itertuples():
        print(f"  {r.date.date()} · +1 {r.r1:+.0%} · +5 {r.r5:+.0%} · +20 {r.r20:+.0%} · تمويل30 {'نعم' if r.fin30 else 'لا'} · تمويل60 {'نعم' if r.fin60 else 'لا'}")


if __name__ == "__main__":
    main()
