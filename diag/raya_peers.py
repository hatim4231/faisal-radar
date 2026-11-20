"""RAYA: وش صار + ليش انشال من رادار الطيارات + وين الدخول الصح — وأسهم مشابهة الحين بدخول صح.

1) يتتبّع RAYA: آخر يوم كانت فيه إشارة طيارات صالحة (منطقة الدخول اللي عطاها الرادار)، مقابل دخول المستخدم 0.560،
   والسعر الحالي ونسبة الارتفاع من القاع (ليش انشالت الحين = طارت فوق السقف).
2) يشغّل رادار الطيارات الحقيقي على كل المجال ويطلع المرشحين الحاليين (🚀 جاهز / 🟡 قريب) مع منطقة الدخول (Limit) والوقف والأهداف.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "faisal_original", ROOT / "runners_radar", ROOT / "highrisk_radar"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import alpaca_data as A  # noqa: E402
import scanner  # noqa: E402
import runners as RN  # noqa: E402

cfg = scanner.load_cfg()
USER_ENTRY = 0.560          # دخول أبوفرح الخطأ
EXTRA = [s.strip().upper() for s in (os.environ.get("DATA_RUN") or "RAYA").split(",") if s.strip()]


def trace(sym):
    print(f"\n{'='*70}\n== {sym} ==")
    d = A.daily([sym], days=400).get(sym)
    if d is None or len(d) < 60:
        print("  ما فيه بيانات كافية من Alpaca"); return
    d = d.dropna(subset=["Close"])
    c = d["Close"].to_numpy(float)
    print("  آخر 15 جلسة (إغلاق · أعلى · أدنى · فوليوم):")
    for ix, r in d.tail(15).iterrows():
        print(f"    {str(ix)[:10]}  {r.Close:.3f}  H {r.High:.3f}  L {r.Low:.3f}  v {r.Volume:,.0f}")
    now = float(c[-1])
    # الإشارة الحالية؟
    cur = RN.detect(d)
    print(f"\n  السعر الحالي: {now:.3f}")
    if cur:
        p = cur["plan"]
        print(f"  ✅ لساها إشارة طيارات: {cur['status']} {cur.get('level_ar','')} · منطقة دخول Limit {p['zone'][0]:.3f}–{p['zone'][1]:.3f} "
              f"· وقف {p['stop']:.3f} (−{p['stop_pct']}%) · هدف {p['t1']:.3f} ثم {p['t2']:.3f}")
    else:
        print("  ❌ ما عادت إشارة طيارات الحين. ليش؟ نفحص الشرط:")
        # أعد تطبيق منطق detect يدوياً لنعرف السبب الأساسي (طارت فوق السقف؟)
        h, l = d["High"].to_numpy(float), d["Low"].to_numpy(float)
        t = len(c) - 1
        w0 = max(0, t - 119)
        b = t - int(np.argmin(l[w0:t + 1][::-1]))
        B = l[b]
        lift = now / B
        pre0 = max(0, b - 250)
        peak_i = pre0 + int(np.argmax(h[pre0:b]))
        crash = h[peak_i] / B
        print(f"     قاع القاعدة {B:.3f} (يوم {str(d.index[b])[:10]}) · السعر الحالي أعلى منه بـ ×{lift:.2f} "
              f"(+{(lift-1)*100:.0f}%) · حد «ما طارت» {RN.LIFT[0]}–{RN.LIFT[1]}×")
        if lift > RN.LIFT[1]:
            print(f"     ← السبب: السهم **طار** (ارتفع ×{lift:.2f} من القاع فوق سقف {RN.LIFT[1]}×). الرادار يصيد قبل الطيران،"
                  f" فأول ما تطير تنشال تلقائياً. (هذا تصميم مو خطأ — بس نتابعها كـ former runner لما تهدأ وتبني قاعدة جديدة.)")
        elif crash < RN.CRASH_X:
            print(f"     ← السبب: ما عاد انهيارها كافي (×{crash:.2f} < {RN.CRASH_X}).")
        else:
            print("     ← السبب: نقص أحد شروط القاعدة/الدعم المتصاعد/الدايفرجنس.")
    # آخر يوم كانت فيه إشارة صالحة (ready/near) — نطلع منطقة الدخول اللي عطاها الرادار وقتها
    last_sig = None
    for k in range(len(d) - 1, 59, -1):
        s = RN.detect(d.iloc[:k + 1])
        if s and s["status"] in ("ready", "near"):
            last_sig = (d.index[k], s, float(c[k]))
            break
    if last_sig:
        dt, s, px = last_sig
        p = s["plan"]
        print(f"\n  📍 آخر يوم عطا فيه الرادار إشارة صالحة: {str(dt)[:10]} ({s['status']} {s.get('level_ar','')})")
        print(f"     منطقة الدخول اللي عطاها (Limit): {p['zone'][0]:.3f} – {p['zone'][1]:.3f}  ·  وقف {p['stop']:.3f}  ·  هدف1 {p['t1']:.3f}  هدف2 {p['t2']:.3f}")
        z0, z1 = p["zone"]
        verdict = ("داخل المنطقة ✅" if z0 <= USER_ENTRY <= z1 else
                   f"فوق المنطقة بـ +{(USER_ENTRY/z1-1)*100:.0f}% (مطاردة ❌)" if USER_ENTRY > z1 else
                   f"تحت المنطقة بـ {(USER_ENTRY/z0-1)*100:.0f}%")
        print(f"     دخولك 0.560 كان: {verdict}")
        print(f"     لو دخلت بأعلى المنطقة {z1:.3f} وبعت نص عند الهدف {p['t1']:.3f}: الربح كان +{(p['t1']/z1-1)*100:.0f}%")
        print(f"     السعر الحالي {now:.3f} = +{(now/z1-1)*100:.0f}% من أعلى المنطقة · +{(now/USER_ENTRY-1)*100:.0f}% من دخولك")


def peers():
    print(f"\n\n{'#'*70}\n# مرشحو رادار الطيارات الحاليون (دخول صح — Limit داخل المنطقة):\n{'#'*70}")
    pay = __import__("runners_scanner").run(cfg)
    rows = pay.get("stocks", [])
    rd = [r for r in rows if r.get("status") == "ready"]
    nr = [r for r in rows if r.get("status") == "near"]
    print(f"\n🚀 جاهز: {len(rd)} · 🟡 قريب: {len(nr)}")
    for lab, grp in (("🚀 جاهز", rd), ("🟡 قريب", nr)):
        for r in grp[:12]:
            p = r.get("plan", {})
            z = p.get("zone", [None, None])
            print(f"\n  {lab}  {r['symbol']}  ({r.get('level_ar','')}) — السعر {r.get('price')}  فلووت {r.get('float')}")
            print(f"      منطقة دخول Limit: {z[0]}–{z[1]}  ·  وقف {p.get('stop')} (−{p.get('stop_pct')}%)  ·  هدف1 {p.get('t1')} (+100%)  هدف2 {p.get('t2')} (+200%)  ·  مدة {p.get('horizon')}ج")
            for w in (r.get("why") or [])[:4]:
                print(f"      • {w}")


if __name__ == "__main__":
    for s in EXTRA:
        trace(s)
    peers()
    print("\n\n⚠ ليس توصية شراء أو بيع — الأفضلية في اختيار السهم (×3.3 احتمال طيران)، مو في ضمان الربح. Limit داخل المنطقة + بيع داخل القفزة + حجم صغير + وقف واسع.")
