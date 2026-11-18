"""تحليل SVRE كما هو يوم أمس (آخر بيانات) — ليش راداراتنا ماصادته؟"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"faisal_original"))
import numpy as np, pandas as pd
import alpaca_data as A, faisal_rules as fr, faisal_rules_original as fo
import hunt as HU, rahib as RB, prior_rise as PR
cfg = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
def info_of(s):
    try:
        import yfinance as yf; i=yf.Ticker(s).info or {}
        return {"shortName":i.get("shortName"),"country":i.get("country"),"floatShares":i.get("floatShares"),
                "marketCap":i.get("marketCap"),"sharesOutstanding":i.get("sharesOutstanding")}
    except Exception as e: print("yf:",e); return {}
d = A.daily(["SVRE"],days=730)["SVRE"].sort_index()
info = info_of("SVRE")
px=float(d["Close"].iloc[-1]); avgv=float(d["Volume"].iloc[-31:-1].mean())
dv20=float((d["Close"].tail(20)*d["Volume"].tail(20)).mean())
print(f"SVRE يوم {str(d.index[-1])[:10]}: سعر {px:.3f} · متوسط فوليوم {avgv:,.0f} سهم · تداول$20ج {dv20:,.0f}$ · فلووت {info.get('floatShares')} · قيمة {info.get('marketCap')}")
U=cfg["universe"]
print(f"\n=== فلاتر رادارات فيصل الرئيسية ===")
print(f"  السعر {px:.3f} ضمن {U['min_price']}-{U['max_price']}$؟ {'✅' if U['min_price']<=px<=U['max_price'] else '❌ خارج المجال'}")
print(f"  متوسط الفوليوم {avgv:,.0f} ≥ {U['min_avg_volume']:,}؟ {'✅' if avgv>=U['min_avg_volume'] else '❌ أقل من الحد'}")
ok,inf = PR.has_prior_rise(d); print(f"  ارتفاع سابق ≥50%/6شهور؟ {'✅' if ok else '❌'} {inf}")
a=fr.analyze(d.copy(),info,cfg); b=fo.analyze(d.copy(),info,cfg)
print(f"\n=== فيصل المطوّر: {('حالة '+a['irtikaz']['status']+' نقاط '+str(a['irtikaz']['score'])) if a else 'ما حلل'} · فلتر: {a['gate_fail'] if a else '-'}")
print(f"=== فيصل الأصلي: {('حالة '+b['irtikaz']['status']+' نقاط '+str(b['irtikaz']['score'])) if b else 'ما حلل'} · فلتر: {b['gate_fail'] if b else '-'}")
h=HU.detect(d.copy(),info); print(f"\n=== 🏹 الصيد: {(h['status_ar']+' نقاط '+str(h['score'])) if h else 'ما ترشح (ما مر الكشف)'}")
r=RB.detect(d.copy(),info,cfg); print(f"=== 🔥 الرهيب: {(r['status_ar']+' نقاط '+str(r['score'])) if r else 'ما ترشح (ما مر فلتر فيصل أو ما فيه إعداد)'}")
fr_run=fr.former_runner(d); print(f"\nFormer Runner: أكبر انفجار سابق +{fr_run['max_pct']:.0f}% ({fr_run['date']})")
print("آخر 6 جلسات:", [(str(i)[:10],round(float(c),3)) for i,c in d['Close'].tail(6).items()])
