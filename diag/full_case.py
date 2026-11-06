"""تشخيص شامل لسهم واحد في كل الرادارات والفلاتر (ما يغير شي)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "abdullah", ROOT / "faisal_original", ROOT / "insiders"):
    sys.path.insert(0, str(p))
import pandas as pd
import scanner, faisal_rules as fr, faisal_rules_original as fro, abdullah_rules as ar
import split_rule, new_low_rule, sec_tools, news_tools
import insiders_scanner as ins

T = sys.argv[1]
cfg = scanner.load_cfg()
acfg = json.loads((ROOT / "abdullah" / "config.json").read_text(encoding="utf-8"))
d = scanner.download_daily([T], cfg)[T]
info = dict(scanner.get_info([T], cfg)[T] or {})
try:
    ib = scanner.load_ib_short().get(T)
except Exception as e:
    ib = None
print("== INFO", json.dumps({k: info.get(k) for k in ("shortName", "marketCap", "floatShares", "sharesOutstanding", "sharesShort", "country", "sector")}, default=str))
print("== IB SHORT AVAILABLE", ib)
if ib and ib.get("available") is not None:
    info["shortAvailable"] = ib["available"]
pd.set_option("display.width", 220)
print("== LAST 30 DAYS")
print(d[["Open", "High", "Low", "Close", "Volume", "Stock Splits"]].tail(30).round(3).to_string())
U = cfg["universe"]
px = float(d["Close"].iloc[-1])
print("== GATE: price", px, "avg vol 30d", int(d["Volume"].tail(30).mean()))

for name, mod in (("فيصل المطوّر", fr), ("فيصل الأصلي", fro)):
    a = mod.analyze(d, info, cfg)
    split_rule.apply(a, d)
    new_low_rule.apply(a, d)
    I, M, P = a["irtikaz"], a["momentum"], a["plan"]
    print(f"\n######## {name}: ارتكاز {I['status']} {I['score']} | زخم {M['status']} {M['score']} | gate {a['gate_fail']}")
    for c in I["checks"]:
        print(f"   [{'OK' if c['ok'] else '--'}] {c['got']:.0f}/{c['weight']} {c['title']} :: {c['detail']} :: {c['remaining']}")
    print("   تعديلات:", [(x["name"], x["pts"], x["detail"][:90]) for x in I["adjustments"]])
    print("   خطة: دعم", P["support"], P["support_kind"], "| منطقة طلب", P["entry_zone"], "| وقف", P["stop"], "| هدف1", P["t1"], "| 4 أهداف", P["four_targets"], "| مقاومات", P["resistances"][:5])
    print("   RSI", a["rsi"], "| MA", a["ma"], "| RVOL", a["rel_volume"], "| tags", a["tags"])
    if a.get("split_zone"):
        print("   التقسيم:", json.dumps(a["split_zone"], ensure_ascii=False))
    if a.get("new_low"):
        print("   قاع جديد:", a["new_low"]["why"])

b = ar.analyze(d, info, acfg)
print("\n######## عبدالله:", "لا توجد موجة تدبيل" if not b else f"{b['status']} {b['score']}")
if b:
    for c in b["checks"]:
        print(f"   [{'OK' if c['ok'] else '--'}] {c['got']:.0f}/{c['weight']} {c['title']} :: {c['detail']} :: {c['remaining']}")
    print("   موجة", b["wave"], "| دعم", b["support"], "| تماسك", b["consolidation"], "| إبطال", b["invalidation"])

print("\n######## SEC")
r = sec_tools.check(T)
print("   level:", r and r["level"])
for h in (r or {}).get("hits", []):
    print(f"   {h['date']} {h['form']} — {h['label']}")
row = {"symbol": T, "dilution": r, "news": []}
try:
    import yfinance as yf
    n = scanner.parse_news(yf.Ticker(T).news)
    news_tools.enrich(n, scanner.translate_ar, full_text_count=0)
    row["news"] = n
    print("\n######## الأخبار")
    for x in n:
        print(f"   [{x.get('sentiment')}] {(x.get('time') or '')[:10]} {x['title']} | {x.get('title_ar')}")
except Exception as e:
    print("news err", e)
print("   أحداث تخفيف:", sec_tools.dilution_events(row))

print("\n######## المطّلعين")
try:
    buys = ins.insider_buys(T, {})
    for x in buys:
        print(f"   {x['date']} {x['name']} {x['roles']} {x['shares']:.0f} @ {x['price']} = ${x['value']:,.0f}")
    print("   المجموع:", sum(x["value"] for x in buys))
except Exception as e:
    print("   err", e)
