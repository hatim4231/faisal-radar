"""فحص خفيف لأي سهم (DATA_RUN = الرمز أو رموز مفصولة بفاصلة): وش صار + حالته بكل الرادارات.
بدون مسح المجال كامل (سريع). الأسعار من Alpaca + معلومات الشركة من ياهو + حالة فيصل/الطيارات/عالي مخاطرة.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "faisal_original", ROOT / "runners_radar", ROOT / "highrisk_radar"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import alpaca_data as A  # noqa: E402
import scanner  # noqa: E402
import faisal_rules as FR  # noqa: E402
import runners as RN  # noqa: E402
import highrisk as HR  # noqa: E402

cfg = scanner.load_cfg()
SYMS = [s.strip().upper() for s in (os.environ.get("DATA_RUN") or "WETO").split(",") if s.strip()]


def look(sym):
    print(f"\n{'='*70}\n== {sym} ==")
    d = A.daily([sym], days=400).get(sym)
    if d is None or len(d) < 30:
        print("  ما فيه بيانات كافية من Alpaca (سهم جديد أو منشطب أو الرمز غلط)"); return
    d = d.dropna(subset=["Close"])
    print("  آخر 20 جلسة (إغلاق · أعلى · أدنى · فوليوم · تقسيم):")
    for ix, r in d.tail(20).iterrows():
        sp = "" if r.get("Stock Splits", 0) in (0, 0.0) else f"  ✂️{r['Stock Splits']}"
        print(f"    {str(ix)[:10]}  {r.Close:.4f}  H {r.High:.4f}  L {r.Low:.4f}  v {r.Volume:,.0f}{sp}")
    c = d["Close"].to_numpy(float)
    now, prev = float(c[-1]), float(c[-2])
    hi252, lo252 = float(d["High"].tail(252).max()), float(d["Low"].tail(252).min())
    print(f"\n  السعر الحالي {now:.4f} ({(now/prev-1)*100:+.1f}% عن أمس) · قمة سنة {hi252:.4f} · قاع سنة {lo252:.4f} · من القمة {(now/hi252-1)*100:+.0f}%")

    infos = scanner.get_info([sym], cfg)
    inf = infos.get(sym) or {}
    print(f"  ياهو: {inf.get('shortName')} · {inf.get('country')} · فلووت {inf.get('floatShares')} · قيمة {inf.get('marketCap')} · شورت ياهو {inf.get('sharesShort')}")
    try:
        ib = scanner.load_ib_short(); sa = (ib.get(sym) or {}).get("available")
        print(f"  الشورت المتاح IB: {sa if sa is not None else 'مو بقائمة IB (=0 متاح)'}")
    except Exception as e:
        print("  IB:", repr(e)[:80])

    # فيصل (المطوّر)
    a = FR.analyze(d, inf, cfg)
    if a:
        I, p = a["irtikaz"], a["plan"]
        print(f"\n  📡 فيصل المطوّر: {I['status']} ({I['score']}%) · عن الدعم {p['dist_support_pct']}% · دعم {p['support']} · وقف {p['stop']} · هدف1 {p['t1']} · RSI {a['rsi']}")
        if a.get("gate_fail"):
            print(f"     ⚠ خارج فلتر فيصل: {'; '.join(a['gate_fail'])}")
        fr = a.get("former_runner") or {}
        print(f"     Former Runner: +{fr.get('max_pct')}% ({fr.get('date')}) · تقسيم عكسي: {a.get('reverse_splits') or 'لا'}")
    else:
        print("\n  📡 فيصل: ما قدر يحلل (بيانات ناقصة)")

    # الطيارات
    rn = RN.detect(d)
    if rn:
        pl = rn["plan"]
        print(f"  🚀 الطيارات: {rn['status']} {rn.get('level_ar','')} · منطقة دخول {pl['zone'][0]}–{pl['zone'][1]} · وقف {pl['stop']} · هدف {pl['t1']}/{pl['t2']} · طار {rn['n_jumps']} مرة · انهار {rn['crash_pct']}%")
    else:
        print("  🚀 الطيارات: ما فيه إشارة (ما يطابق نمط القاعدة/الدعم المتصاعد)")

    # عالي مخاطرة (احتمال الانفجار + النزيف)
    try:
        F = HR.features(d); sc = HR.score(F.iloc[-1])
        if sc:
            print(f"  🧨 عالي مخاطرة: احتمال انفجار (خام) {sc['p_raw']*100:.0f}% · 🩸 خطر النزيف (−30% خلال 20ج) {sc['pb']*100:.0f}% [{HR.band(sc['pb'])}]")
    except Exception as e:
        print("  🧨 عالي مخاطرة:", repr(e)[:80])


if __name__ == "__main__":
    for s in SYMS:
        look(s)
    print("\n⚠ ليس توصية — معلومة فقط.")
