"""
OMH يوم 9 نوفمبر (توصية فيصل): وش سوى السهم؟ وليش راداراتنا ما طلعته؟ (تشخيص فقط)
1) الأسعار: يومي آخر 40 جلسة + شموع الساعة (فيها البري والأفتر) من 4 نوفمبر.
2) خط المسح خطوة خطوة: قائمة ياهو، فلتر السعر/الفوليوم، معلومات ياهو، الشورت (IB/ياهو)، HRT، ملفات SEC والتخفيف المتكرر،
   وحالة كل رادار على البيانات لين 5 و6 و9 نوفمبر.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "faisal_original", ROOT / "abdullah", ROOT / "accumulation"):
    sys.path.insert(0, str(p))
import numpy as np
import pandas as pd
import alpaca_data as A
import scanner
import faisal_rules as FR
import faisal_rules_original as FO
import abdullah_rules as AR
import accum_rules as ACC
import schools as SC
import runners as RN
import new_low_rule
import compliance_rule
import split_rule
import sec_tools
import holders

T = "OMH"
cfg = scanner.load_cfg()
cfg_abd = json.loads((ROOT / "abdullah" / "config.json").read_text(encoding="utf-8"))
U = cfg["universe"]


def main():
    d = A.daily([T], days=400).get(T)
    print("يومي آخر 25 جلسة (فتح/أعلى/أدنى/إغلاق/فوليوم):")
    for ix, r in d.tail(25).iterrows():
        print(f"  {ix.date()}  {r.Open:.3f} / {r.High:.3f} / {r.Low:.3f} / {r.Close:.3f} / {r.Volume:,.0f}  تقسيم={r['Stock Splits']}")
    hb = A.bars([T], "1Hour", datetime(2026, 11, 4, tzinfo=timezone.utc)).get(T)
    if hb is not None:
        ses = A.session_of(hb.index)
        print("\nشموع الساعة من 6 نوفمبر (الوقت نيويورك · الجلسة · فتح/أعلى/أدنى/إغلاق · فوليوم):")
        for ix, r in hb[hb.index >= "2026-11-06"].iterrows():
            print(f"  {ix.strftime('%m-%d %H:%M')} {ses.loc[ix]:>3}  {r.Open:.3f}/{r.High:.3f}/{r.Low:.3f}/{r.Close:.3f}  {r.Volume:,.0f}")
    # 1) قائمة ياهو
    try:
        uni = scanner.get_universe(cfg)
        print(f"\nقائمة ياهو العادية: {len(uni)} سهم · OMH فيها؟ {'✅' if T in uni else '❌'}")
    except Exception as e:
        print("قائمة ياهو خطأ:", repr(e)[:100])
    # 2) معلومات ياهو + الشورت
    infos = scanner.get_info([T], cfg)
    inf = infos.get(T) or {}
    print("ياهو:", {k: inf.get(k) for k in ("shortName", "country", "marketCap", "floatShares", "sharesOutstanding", "sharesShort",
                                           "sharesShortPriorMonth", "dateShortInterest", "averageVolume", "averageVolume10days")})
    try:
        ib = scanner.load_ib_short()
        print("IB الشورت المتاح:", ib.get(T), "· IB فيه", len(ib), "سهم")
    except Exception as e:
        ib = {}
        print("IB خطأ:", repr(e)[:100])
    sa = (ib.get(T) or {}).get("available")
    sv = sa if sa is not None else inf.get("sharesShort")
    print(f"الشورت المستخدم: {sv} ({'IB' if sa is not None else 'Short Interest ياهو'}) · حد الرادارات {cfg['behavior'].get('max_short_shares')} "
          f"· حد الطيارات {(cfg.get('runners') or {}).get('max_short_shares')}")
    # 3) HRT
    try:
        rows = [{"symbol": T, "name": inf.get("shortName")}]
        holders.annotate(rows, print)
        print("HRT:", rows[0].get("hrt"))
    except Exception as e:
        print("HRT خطأ:", repr(e)[:100])
    # 4) SEC
    try:
        row = {"symbol": T, "name": inf.get("shortName"), "news": []}
        sec_tools.annotate([row], print)
        print("SEC:", (row.get("dilution") or {}).get("level"), [(h["form"], h["date"]) for h in (row.get("dilution") or {}).get("hits", [])][:8])
        print("أحداث تخفيف (90 يوم):", sec_tools.dilution_events(row))
    except Exception as e:
        print("SEC خطأ:", repr(e)[:100])
    # 5) الرادارات على كل تاريخ
    for cut in ("2026-11-05", "2026-11-06", "2026-11-09"):
        dc = d[d.index <= cut].copy()
        if dc.empty:
            continue
        px = float(dc["Close"].iloc[-1])
        av = float(dc["Volume"].tail(30).mean())
        print(f"\n── البيانات لين {dc.index[-1].date()} · السعر {px:.3f} · متوسط فوليوم 30 جلسة {av:,.0f} "
              f"· فلتر السعر {'✅' if U['min_price'] <= px <= U['max_price'] else '❌'} · فلتر الفوليوم {'✅' if av >= U['min_avg_volume'] * 0.7 else '❌'}")
        try:
            a = FR.analyze(dc, dict(inf, shortAvailable=sa) if sa is not None else dict(inf), cfg)
            if a:
                split_rule.apply(a, dc)
                new_low_rule.apply(a, dc)
                compliance_rule.apply_faisal(a, dc)
                I = a["irtikaz"]
                print(f"  فيصل المطوّر: {I['status']} {I['score']:.0f}% · gate={a.get('gate_fail')} · RSI {a.get('rsi')} · عن الدعم {a['plan'].get('dist_support_pct')}% "
                      f"· الدعم {a['plan'].get('support')} · وسوم {a.get('tags')}")
                for c in I["checks"]:
                    print(f"     {'✓' if c['ok'] else '✗'} {c['title'][:40]} ({c['got']:.1f}/{c['weight']}) — {c.get('detail', '')[:80]}")
                for x in I.get("adjustments", []):
                    print(f"     ± {x['name']} {x['pts']:+} — {x['detail'][:80]}")
            a = FO.analyze(dc, dict(inf), cfg)
            print(f"  فيصل الأصلي: {a['irtikaz']['status']} {a['irtikaz']['score']:.0f}%" if a else "  فيصل الأصلي: -")
        except Exception as e:
            print("  فيصل خطأ:", repr(e)[:150])
        for name, fn in (("عبدالله", lambda: AR.analyze(dc, dict(inf), cfg_abd, None)), ("التجميع", lambda: ACC.analyze(dc, dict(inf), cfg, None))):
            try:
                a = fn()
                print(f"  {name}: {a['status']} {a.get('score', 0):.0f}" if a else f"  {name}: -")
            except Exception as e:
                print(f"  {name} خطأ:", repr(e)[:100])
        try:
            s = SC.detect(dc)
            print("  المدارس:", ("+".join(x["setup"] for x in s["setups"]) + f" توافق {s['confluence']}") if s else "-")
        except Exception as e:
            print("  المدارس خطأ:", repr(e)[:100])
        try:
            r = RN.detect(dc)
            print("  الطيارات:", (f"{r['status']} {r['level']} · ارتفع {r['lift_pct']}% فوق القاع") if r else "- (بعد الانفجار: السعر فوق ضعف القاع = «طار» فما يدخل)")
        except Exception as e:
            print("  الطيارات خطأ:", repr(e)[:100])


if __name__ == "__main__":
    main()
