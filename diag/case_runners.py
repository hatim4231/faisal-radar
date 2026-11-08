"""
دراسة حالة: OMH و QMMM (أمثلة المستخدم من تغريدة @xzdixz) — وش صار قبل الانفجار؟ (تشخيص فقط)

- يومي Alpaca ~3 سنين + التقسيمات.
- أيام الانفجار: أعلى سعر ≥ +50% فوق إغلاق اليوم اللي قبله (أول يوم بكل موجة).
- القاعدة قبل كل انفجار: القاع، كم جلسة من القاع، الهبوط قبل القاع، مدى القاعدة، القيعان المتصاعدة،
  RSI عند القيعان (دايفرجنس)، لمسات المقاومة، الفوليوم، وهل يعدّي فلتر المجال (السعر/الفوليوم).
- 4 ساعات حول قاع QMMM (يونيو 2025) — RSI والقيعان.
- رادراتنا على البيانات لين قبل الانفجار بـ 1/5/10/20 جلسة.
- معلومات الشركة من ياهو + الشورت المتاح الحين من IB.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "faisal_original"))
sys.path.insert(0, str(ROOT / "abdullah"))
sys.path.insert(0, str(ROOT / "accumulation"))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import faisal_rules as FR
import faisal_rules_original as FO
import abdullah_rules as AR
import accum_rules as ACC
import schools as SC
import new_low_rule
import compliance_rule
import split_rule

SYMS = ["OMH", "QMMM"]
cfg = scanner.load_cfg()
cfg_abd = json.loads((ROOT / "abdullah" / "config.json").read_text(encoding="utf-8"))
U = cfg["universe"]


def f(x, n=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{n}f}" if abs(x) < 1000 else f"{x:,.0f}")


def rsi(c):
    return FR.rsi(c)


def base_report(t, d, E):
    c, h, l, v = (d[k] for k in ("Close", "High", "Low", "Volume"))
    r = rsi(c)
    e_day = d.index[E]
    w0 = max(0, E - 150)
    win = d.iloc[w0:E]
    b = int(np.argmin(win["Low"].values)) + w0
    pre = d.iloc[max(0, b - 120):b + 1]
    peak_i = int(np.argmax(pre["High"].values)) + max(0, b - 120)
    base = d.iloc[b:E]
    top = float(base["High"].max())
    print(f"\n  ── الانفجار {e_day.date()}: إغلاق قبله {f(c.iloc[E-1])} ← أعلى {f(h.iloc[E])} ({(h.iloc[E]/c.iloc[E-1]-1)*100:+.0f}%) "
          f"· إغلاق اليوم {f(c.iloc[E])} · فوليوم {v.iloc[E]:,.0f}")
    print(f"  القاع: {d.index[b].date()} عند {f(l.iloc[b])} · قبل الانفجار بـ {E-b} جلسة · "
          f"الهبوط قبله: من {f(h.iloc[peak_i])} ({d.index[peak_i].date()}) = {(l.iloc[b]/h.iloc[peak_i]-1)*100:.0f}%")
    print(f"  القاعدة: {len(base)} جلسة · أعلى سعر فيها {f(top)} (+{(top/l.iloc[b]-1)*100:.0f}% فوق القاع) · "
          f"إغلاق قبل الانفجار {f(c.iloc[E-1])} (+{(c.iloc[E-1]/l.iloc[b]-1)*100:.0f}% فوق القاع، {(c.iloc[E-1]/top-1)*100:.0f}% عن السقف)")
    # القيعان: أدنى سعر بكل أسبوع (5 جلسات) داخل القاعدة + zigzag
    blk = [(base.index[i].date(), float(base["Low"].iloc[i:i+5].min()), float(r.loc[base["Low"].iloc[i:i+5].idxmin()]))
           for i in range(0, len(base), 5)]
    print("  أدنى سعر كل أسبوع (RSI عنده): " + " · ".join(f"{dd}:{f(p)}({f(rr,0)})" for dd, p, rr in blk))
    hh, ll = base["High"].values, base["Low"].values
    piv, _ = SC.zigzag(hh, ll, 0.08)
    lows = [(base.index[i].date(), p, float(r.iloc[b + i])) for i, p, k, _ in piv if k == "L"]
    highs = [(base.index[i].date(), p) for i, p, k, _ in piv if k == "H"]
    print("  قيعان zigzag 8% (السعر/RSI): " + (" → ".join(f"{dd}:{f(p)}/{rr:.0f}" for dd, p, rr in lows) or "-"))
    print("  قمم zigzag 8%: " + (" → ".join(f"{dd}:{f(p)}" for dd, p in highs) or "-"))
    touches = int((base["High"] >= top * 0.95).sum())
    print(f"  لمسات المقاومة (ضمن 5% من السقف): {touches} · RSI: عند القاع {f(r.iloc[b],0)} · أقل RSI بالقاعدة {f(r.iloc[b:E].min(),0)} "
          f"· قبل الانفجار {f(r.iloc[E-1],0)} · أعلى RSI بالقاعدة {f(r.iloc[b:E].max(),0)}")
    vb = float(v.iloc[b:E].median())
    vp = float(v.iloc[max(0, b-60):b].median()) if b > 5 else float("nan")
    up = base[base["Close"] > base["Close"].shift(1)]["Volume"].mean()
    dn = base[base["Close"] < base["Close"].shift(1)]["Volume"].mean()
    spikes = [(x.date(), int(vv)) for x, vv in base["Volume"].items() if vv >= 3 * vb]
    print(f"  الفوليوم: وسيط القاعدة {vb:,.0f} (قبل القاع {vp:,.0f}) · صعود/هبوط {up/max(1,dn):.2f}x · "
          f"قفزات فوليوم (3x+): {spikes[:8]}")
    dv = float((c.iloc[E-30:E] * v.iloc[E-30:E]).mean())
    av = float(v.iloc[E-30:E].mean())
    ok = U["min_price"] <= c.iloc[E-1] <= U["max_price"] and av >= U["min_avg_volume"] * 0.7
    print(f"  فلتر المجال قبل الانفجار: السعر {f(c.iloc[E-1])} · متوسط فوليوم 30 يوم {av:,.0f} (الحد {U['min_avg_volume']*0.7:,.0f}) "
          f"· سيولة يومية ${dv:,.0f} ← {'يعدّي ✅' if ok else 'ما يعدّي ❌'}")
    # بعد الانفجار
    after = d.iloc[E:E+20]
    print(f"  بعده: أعلى سعر خلال 20 جلسة {f(after['High'].max())} · إغلاق بعد 5 جلسات {f(c.iloc[min(E+5, len(d)-1)])} "
          f"· بعد 20 {f(c.iloc[min(E+20, len(d)-1)])}")
    return b


def radars(t, d, E, inf):
    print("  رادراتنا قبل الانفجار:")
    for k in (1, 3, 5, 10, 20, 30):
        if E - k < 120:
            continue
        dc = d.iloc[:E - k + 1].copy()
        out = []
        try:
            a = FR.analyze(dc, dict(inf), cfg)
            if a:
                split_rule.apply(a, dc)
                new_low_rule.apply(a, dc)
                compliance_rule.apply_faisal(a, dc)
                I = a["irtikaz"]
                miss = [x["title"][:18] for x in I["checks"] if not x["ok"]][:3]
                out.append(f"فيصل المطوّر {I['status']} {I['score']:.0f}%{' ⛔gate' if a.get('gate_fail') else ''} (ناقص: {'، '.join(miss)})")
            else:
                out.append("فيصل المطوّر: -")
        except Exception as e:
            out.append(f"فيصل المطوّر خطأ {repr(e)[:60]}")
        try:
            a = FO.analyze(dc, dict(inf), cfg)
            out.append(f"الأصلي {a['irtikaz']['status']} {a['irtikaz']['score']:.0f}%" if a else "الأصلي: -")
        except Exception as e:
            out.append(f"الأصلي خطأ {repr(e)[:40]}")
        try:
            a = AR.analyze(dc, dict(inf), cfg_abd, None)
            out.append(f"عبدالله {a['status']} {a.get('score', 0):.0f}" if a else "عبدالله: -")
        except Exception as e:
            out.append(f"عبدالله خطأ {repr(e)[:40]}")
        try:
            a = ACC.analyze(dc, dict(inf), cfg, None)
            out.append(f"التجميع {a['status']} {a.get('score', 0):.0f}" if a else "التجميع: -")
        except Exception as e:
            out.append(f"التجميع خطأ {repr(e)[:40]}")
        try:
            s = SC.detect(dc)
            out.append(f"المدارس {'+'.join(x['setup'] for x in s['setups'])} توافق {s['confluence']}" if s else "المدارس: -")
        except Exception as e:
            out.append(f"المدارس خطأ {repr(e)[:40]}")
        print(f"   قبل {k:>2} جلسة ({dc.index[-1].date()}): " + " | ".join(out))


def four_hour(t, start, end):
    try:
        hb = A.bars([t], "1Hour", start, end).get(t)
    except Exception as e:
        print("  ساعة: خطأ", repr(e)[:100])
        return
    if hb is None or hb.empty:
        print("  ساعة: ما فيه بيانات")
        return
    d4 = hb.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    r4 = rsi(d4["Close"])
    piv, _ = SC.zigzag(d4["High"].values, d4["Low"].values, 0.06)
    lows = [(d4.index[i], p, float(r4.iloc[i])) for i, p, k, _ in piv if k == "L"]
    print(f"  4 ساعات ({len(d4)} شمعة): قيعان zigzag 6% (السعر/RSI4h): " +
          " → ".join(f"{x.strftime('%m-%d %H')}:{f(p)}/{rr:.0f}" for x, p, rr in lows[:14]))
    i0 = int(np.argmin(d4["Low"].values))
    print(f"  أدنى 4 ساعات: {d4.index[i0]} عند {f(d4['Low'].iloc[i0])} RSI {f(r4.iloc[i0],0)}")


def main():
    D = A.daily(SYMS, days=1150)
    try:
        spl = A.splits(SYMS, datetime(2023, 1, 1, tzinfo=timezone.utc))
    except Exception:
        spl = {}
    try:
        ib = scanner.load_ib_short()
    except Exception:
        ib = {}
    import yfinance as yf
    for t in SYMS:
        d = D.get(t)
        print(f"\n================ {t} ================")
        if d is None or d.empty:
            print("ما فيه بيانات من Alpaca")
            continue
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        inf = {k: info.get(k) for k in ("shortName", "country", "sector", "floatShares", "sharesOutstanding", "sharesShort",
                                        "marketCap", "heldPercentInsiders", "heldPercentInstitutions")}
        print("ياهو:", inf)
        print("الشورت المتاح الحين (IB):", ib.get(t))
        print("التقسيمات:", [(str(x.date()), r) for x, r in spl.get(t, [])])
        print(f"البيانات: {d.index[0].date()} → {d.index[-1].date()} ({len(d)} جلسة)")
        m = d.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        print("شهري (أعلى/أدنى/إغلاق/فوليوم):")
        for ix, row in m.iterrows():
            print(f"   {ix.strftime('%Y-%m')}: {f(row.High)} / {f(row.Low)} / {f(row.Close)} / {row.Volume:,.0f}")
        jump = d["High"] / d["Close"].shift(1)
        top = jump.sort_values(ascending=False).head(10)
        print("أكبر قفزات يومية (أعلى ÷ إغلاق أمس):")
        for ix, j in top.items():
            i = d.index.get_loc(ix)
            print(f"   {ix.date()}: {j:.2f}x · إغلاق أمس {f(d['Close'].iloc[i-1])} · أعلى {f(d['High'].iloc[i])} · إغلاق {f(d['Close'].iloc[i])} · فوليوم {d['Volume'].iloc[i]:,.0f}")
        # أول يوم بكل موجة انفجار (+50%)
        E_list, last = [], -99
        for i in range(130, len(d)):
            if jump.iloc[i] >= 1.5 and i - last > 30:
                E_list.append(i)
            if jump.iloc[i] >= 1.5:
                last = i
        # QMMM: موجة الصعود من قاع يونيو 2025 (ما كانت يوم واحد)
        if t == "QMMM":
            r1 = d.loc["2025-06-01":"2025-07-31"]
            if len(r1):
                bi = d.index.get_loc(r1["Low"].idxmin())
                for i in range(bi + 1, len(d)):
                    if d["Close"].iloc[i] >= d["Low"].iloc[bi] * 1.6:
                        E_list.append(i)
                        print(f"QMMM: أول إغلاق +60% فوق قاع يونيو ({d.index[bi].date()} {f(d['Low'].iloc[bi])}) = {d.index[i].date()}")
                        break
        for E in sorted(set(E_list))[:4]:
            base_report(t, d, E)
            radars(t, d, E, inf)
        if t == "QMMM":
            four_hour(t, datetime(2025, 5, 20, tzinfo=timezone.utc), datetime(2025, 8, 20, tzinfo=timezone.utc))
        if t == "OMH":
            four_hour(t, datetime(2026, 8, 25, tzinfo=timezone.utc), datetime(2026, 11, 6, tzinfo=timezone.utc))


if __name__ == "__main__":
    main()
