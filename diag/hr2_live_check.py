"""فحص حي لرادار عالي مخاطرة 2 **بدون نشر**: يشغل المسح على بيانات السوق الحقيقية ويطبع ملخص للتأكد قبل الدمج.
+ يجرب «سجل الصدق» بسجل مزروع (يومين قديمين برموز حقيقية) عشان نتأكد من التقييم وشموع الساعة (البري والأفتر)."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "highrisk_radar"))
import numpy as np

import highrisk as HR
import highrisk_scanner as S

# سجل مزروع: 2026-10-07 (نضج 20 جلسة) و 2026-10-28 (نضج 10 جلسات؛ OMH انفجر 11-04، وRUBI طار بالبري 11-10)
(ROOT / "trackdata").mkdir(exist_ok=True)
seed = {"days": [
    {"date": "2026-10-07", "picks": [{"s": t, "c": None} for t in ("OMH", "RUBI", "NUWE")], "picks_s": ["OMH", "RUBI", "NUWE"],
     "ready": ["OMH", "RUBI", "LICN", "NXXT"], "b25": ["NUWE", "RAYA", "FBGL"], "b40": ["RUBI", "LICN"]},
    {"date": "2026-10-28", "picks": [{"s": t, "c": None} for t in ("OMH", "RUBI", "QMMM")], "picks_s": ["OMH", "RUBI", "QMMM"],
     "ready": ["OMH", "RUBI", "LICN", "NXXT"], "b25": ["NUWE", "RAYA", "FBGL"], "b40": ["RUBI", "LICN"]}]}
S.STORE.write_text(json.dumps(seed), encoding="utf-8")

cfg = S.data.load_cfg()
t0 = time.time()
p = S.run(cfg)
html = (ROOT / "highrisk_radar" / "template.html").read_text(encoding="utf-8")
page = html.replace("/*__DATA__*/null", json.dumps(p, ensure_ascii=False, default=str).replace("</", "<\\/"))
out = ROOT / "diag_out"
out.mkdir(exist_ok=True)
(out / "highrisk.html").write_text(page, encoding="utf-8")
A = {r[0]: dict(zip(p["all"]["cols"], r)) for r in p["all"]["rows"]}
M = HR.model()
cal = M["cal"]
print(f"\n==== ملخص ==== وقت {round(time.time() - t0)}ث · الصفحة {len(page.encode()) / 1e6:.2f}MB (منها كل الأسهم {len(json.dumps(p['all'], ensure_ascii=False).encode()) / 1e6:.2f}MB)"
      f" · أسهم {len(A)} · آخر جلسة {p.get('sig_date')} · وقت السوق؟ {p.get('rth')}")
print("الحالات (الكل):", p["totals"], "· المعروض:", p["counts"], "· أقوى 3:", p["picks"], "· متكرر:", p["serial_total"])


def info(t):
    a = A.get(t)
    if not a:
        return "— مو موجود"
    rad = 0.7 <= (a["c"] or 0) <= 10 and a["dv"] >= M["liq"]
    b = HR.band(cal["pb_radar" if rad else "pb_all"], a["pb"]) if a["ac"] else None
    e = HR.band(cal["p_liq" if a["dv"] >= M["liq"] else "p_all"], a["p"]) if a["ac"] else None
    return (f"{a['c']} ({a['ch']}%) · p={a['p']} ← انفجار خلال 10: {e['e10'] * 100:.1f}% · pb={a['pb']} ← يلمس −30%: {b['rate'] * 100:.0f}% (وسيط {b['r20_med'] * 100:.1f}%)"
            f" · انفجر {a['ny']} بسنة (قبل {a['ds']}) · dd {a['dd']} · dv {a['dv']:,} · حالة {a['st']} · نشط {a['ac']} · {a['n']}" if b and e
            else f"{a['c']} · غير نشط/بدون تقييم · dv {a['dv']:,} · ac {a['ac']} · {a['n']}")


print("\n==== 🔎 فحص أسهم معروفة ====")
for t in ("OMH", "RUBI", "NUWE", "LICN", "NXXT", "RAYA", "QMMM", "FBGL", "VRAX", "ELPW", "AAPL", "TSLA", "NVDA", "GME", "AMC", "F", "PLTR", "SOFI"):
    print(f"  {t}: {info(t)}")
act = [a for a in A.values() if a["ac"] == 1 and a["pb"] is not None]
pb = np.array([a["pb"] for a in act])
pp = np.array([a["p"] for a in act])
print(f"\nنشط: {len(act)} · غير نشط {sum(1 for a in A.values() if a['ac'] == 0)} · ما تداول بآخر جلسة {sum(1 for a in A.values() if a['ac'] == 2)}")
print("pb كل النشط — نسب مئوية 50/75/90/95/99:", [round(float(x), 3) for x in np.quantile(pb, [.5, .75, .9, .95, .99])], "· الدراسة:", cal["all_q"])
print("p  كل النشط — 90/95/99:", [round(float(x), 4) for x in np.quantile(pp, [.9, .95, .99])], "· حدود الدراسة:", M["thresholds"])
rr = [a for a in act if 0.7 <= a["c"] <= 10 and a["dv"] >= M["liq"]]
lv = lambda x: 4 if x >= .4 else 3 if x >= .25 else 2 if x >= .15 else 1 if x >= .06 else 0
cnt = np.bincount([lv(a["pb"]) for a in rr], minlength=5)
print(f"مجال راداراتنا (0.7-10$ وسيولة): {len(rr)} سهم · منخفض {cnt[0]} · متوسط {cnt[1]} · مرتفع {cnt[2]} · عالي {cnt[3]} · شديد {cnt[4]}"
      f" · نسب pb 50/90/95: {[round(float(x), 3) for x in np.quantile([a['pb'] for a in rr], [.5, .9, .95])]} · الدراسة: {cal['radar_q']}")
print("\n==== المعروض (أول 30) ====")
for s in p["stocks"][:30]:
    print(f"  {s['status']:5s} {s['symbol']:5s} {s['price']} · p {s['p_raw']} · pb {s['pb']} · سيولة {s['dv20']:,.0f} · liq {s['liq']} · انفجر {s['n_year']} (قبل {s['days_since']})"
          f" · pick {s['pick']} · أعلام {len(s['flags'])} · أخبار {len(s.get('news') or [])} · {s['name'][:28]}")
print("\n==== 📒 سجل الصدق (مزروع) ====")
print(json.dumps(p["score"], ensure_ascii=False)[:1500])
L = json.loads(S.STORE.read_text(encoding="utf-8"))
for d in L["days"]:
    print({k: (v if k in ("date", "picks_s", "res10", "res20", "tickets", "done") else (len(v) if isinstance(v, list) else v)) for k, v in d.items()})
print("others:", len(p["others"]))
