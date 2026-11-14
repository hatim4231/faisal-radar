"""دراسة AKAN: هل فيصل طبّق قواعده قبل ما يطير؟
نجيب اليومي من Alpaca، نلقى يوم القفزة (أعلى سعر ≥ 1.8× إغلاق أمس)، ونشغّل محرّك فيصل
(المطوّر + الأصلي) على البيانات **قبل القفزة بيوم** + **اليوم اللي أعلن فيه** — ونطبع كل شرط.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "faisal_original"))

import numpy as np
import pandas as pd

import alpaca_data as A
import faisal_rules as fr
import faisal_rules_original as fo

SYM = (sys.argv[1] if len(sys.argv) > 1 else "AKAN").upper()
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def get_info(sym):
    try:
        import yfinance as yf
        t = yf.Ticker(sym)
        fi = t.fast_info or {}
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        return {
            "shortName": info.get("shortName") or info.get("longName") or sym,
            "country": info.get("country"),
            "floatShares": info.get("floatShares"),
            "marketCap": info.get("marketCap") or fi.get("market_cap"),
            "sharesOutstanding": info.get("sharesOutstanding") or fi.get("shares"),
            "sharesShort": info.get("sharesShort"),
            "shortPercentOfFloat": info.get("shortPercentOfFloat"),
        }
    except Exception as e:
        print("  (ياهو للمعلومات تعذّر:", e, ")")
        return {}


def show(label, a):
    if not a:
        print(f"\n### {label}: ما قدر يحلل (بيانات ناقصة)")
        return
    ir = a["irtikaz"]
    print(f"\n### {label}")
    print(f"  السعر {a['price']} | تغير اليوم {a['change_pct']}% | التاريخ {a['date']}")
    print(f"  RSI {a['rsi']} (أدنى 10 جلسات {a['rsi_min10']}) | هبوط من القمة {a['drawdown_pct']}% | صعد من قاع 20ج {a['up_from_low20_pct']}%")
    pl = a["plan"]
    print(f"  الدعم {pl['support']} ({pl['support_kind']}، لمسات {pl['support_touches']}) | البعد عن الدعم {pl['dist_support_pct']}%")
    print(f"  الوقف {pl['stop']} | هدف1 {pl['t1']} | الهدف الرئيسي {pl['main_target']} | مساحة الصعود {pl['upside_pct']}% | R/R {pl['rr']}")
    print(f"  **النتيجة = {ir['score']} (أساس {ir['base']}) → الحالة = {ir['status'].upper()}**")
    print("  الشروط:")
    for c in ir["checks"]:
        mark = "✅" if c["ok"] else "❌"
        print(f"    {mark} {c['title']}: {c['got']:.1f}/{c['weight']} — {c['detail']}" + (f"  [ناقص: {c['remaining']}]" if c.get("remaining") else ""))
    if ir["adjustments"]:
        print("  تعديلات السلوك:")
        for ad in ir["adjustments"]:
            print(f"    {ad['pts']:+d} {ad['name']} — {ad['detail']}")
    if a["gate_fail"]:
        print("  ⛔ فلتر القائمة يرفضه:", " · ".join(a["gate_fail"]))
    print("  وسوم:", ", ".join(a["tags"]) or "—")


def main():
    print(f"========== دراسة {SYM} ==========")
    d = A.daily([SYM], days=730).get(SYM)
    if d is None or len(d) < 60:
        print("ما فيه بيانات كافية من Alpaca لـ", SYM)
        return
    d = d.sort_index()
    print(f"بيانات Alpaca: {len(d)} جلسة، من {str(d.index[0])[:10]} إلى {str(d.index[-1])[:10]}")

    info = get_info(SYM)
    print("المعلومات:", {k: v for k, v in info.items() if v is not None})

    # آخر ~15 جلسة سعر/أعلى/فوليوم
    print("\nآخر 15 جلسة (إغلاق · أعلى · التغير% · فوليوم):")
    c = d["Close"].to_numpy(float)
    for i in range(max(0, len(d) - 15), len(d)):
        ch = (c[i] / c[i - 1] - 1) * 100 if i else 0
        print(f"  {str(d.index[i])[:10]}  {c[i]:>7.3f}  أعلى {d['High'].iloc[i]:>7.3f}  {ch:>+6.1f}%  {int(d['Volume'].iloc[i]):>12,}")

    # يوم القفزة: أعلى ≥ 1.8× إغلاق أمس
    jump = None
    for i in range(len(d) - 1, 0, -1):
        if d["High"].iloc[i] >= c[i - 1] * 1.8 or c[i] >= c[i - 1] * 1.8:
            jump = i
            break
    if jump is None:
        # ما فيه قفزة ×1.8 — نحلل آخر جلستين فقط
        print("\n(ما لقيت قفزة ≥ ×1.8؛ نحلل آخر جلسة واللي قبلها)")
        jump = len(d) - 1

    jday = str(d.index[jump])[:10]
    bday = str(d.index[jump - 1])[:10]
    print(f"\n>>> يوم القفزة = {jday} (إغلاق {c[jump]:.3f}، أعلى {d['High'].iloc[jump]:.3f}) | القفزة من إغلاق {bday} = {c[jump-1]:.3f}")
    print(f">>> نحلل البيانات **حتى {bday}** = اللي شافه فيصل قبل ما يطير")

    before = d.iloc[: jump]   # حتى اليوم اللي قبل القفزة (ضمناً)
    show("فيصل المطوّر — قبل القفزة", fr.analyze(before.copy(), info, cfg))
    show("فيصل الأصلي — قبل القفزة", fo.analyze(before.copy(), info, cfg))

    # الثقة (المطوّر)
    try:
        import confidence as CF
        a = fr.analyze(before.copy(), info, cfg)
        if a:
            a["symbol"] = SYM
            CF.apply([a], {SYM: before.copy()}, log=lambda *x: None)
            conf = a.get("confidence")
            if conf:
                print(f"\n### الثقة (فيصل المطوّر، قبل القفزة): {conf.get('level')} — احتمال {conf.get('prob')}")
                for r in (conf.get("reasons_for") or []):
                    print("   + ", r)
                for r in (conf.get("reasons_against") or []):
                    print("   − ", r)
    except Exception as e:
        print("(الثقة تعذّرت:", e, ")")

    # لو القفزة مو آخر جلسة، نبين إيش صار بعدها
    if jump < len(d) - 1:
        print(f"\n>>> بعد القفزة (للعلم): من {jday} لين {str(d.index[-1])[:10]}")
        print(f"    أعلى سعر وصله = {d['High'].iloc[jump:].max():.3f} | آخر إغلاق = {c[-1]:.3f}")


if __name__ == "__main__":
    main()
