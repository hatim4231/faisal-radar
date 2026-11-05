"""تشخيص فقط: يطبع كل تفاصيل تحليل سهم واحد في رادار فيصل. ما يغير شي."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import scanner, faisal_rules as fr

T = sys.argv[1] if len(sys.argv) > 1 else "FLYE"
cfg = scanner.load_cfg()
d = scanner.download_daily([T], cfg)[T]
info = scanner.get_info([T], cfg)[T]
ib = scanner.load_ib_short().get(T)
print("== INFO", json.dumps(info, default=str))
print("== IB SHORT", ib)
pd.set_option("display.width", 200)
print("== LAST 45 DAYS")
print(d[["Open", "High", "Low", "Close", "Volume", "Stock Splits"]].tail(45).round(3).to_string())
a = fr.analyze(d, info, cfg)
a.pop("chart", None)
print("== STATUS irtikaz:", a["irtikaz"]["status"], a["irtikaz"]["score"], "| momentum:", a["momentum"]["status"], a["momentum"]["score"])
print("== GATE FAIL:", a["gate_fail"])
print("== TAGS:", a["tags"])
for c in a["irtikaz"]["checks"]:
    print(f"  [{'OK' if c['ok'] else '--'}] {c['got']}/{c['weight']} {c['title']} :: {c['detail']} :: {c['remaining']}")
print("== ADJUSTMENTS")
for x in a["irtikaz"]["adjustments"]:
    print("  ", x)
print("== MOMENTUM CHECKS")
for c in a["momentum"]["checks"]:
    print(f"  [{'OK' if c['ok'] else '--'}] {c['got']}/{c['weight']} {c['title']} :: {c['detail']}")
print("== MOMENTUM WARNINGS", a["momentum"]["warnings"])
for k in ("price", "rsi", "rsi_min10", "ma", "rel_volume", "change_pct", "gap_pct", "float", "market_cap",
          "short_shares", "short_pct_float", "up_from_low20_pct", "former_runner", "reverse_splits",
          "pump_candles", "w", "sweep", "median_range_pct", "drawdown_pct"):
    print(f"  {k}: {a.get(k)}")
print("== PLAN", json.dumps(a["plan"], default=str))

print("\n\n######## AS OF PREVIOUS SESSION (before last candle) ########")
for cut in (1, 2, 5):
    dd = d.iloc[:-cut]
    b = fr.analyze(dd, info, cfg)
    print(f"--- date {dd.index[-1].date()} close {dd['Close'].iloc[-1]:.2f}: irtikaz {b['irtikaz']['status']} {b['irtikaz']['score']} | RSI {b['rsi']} min10 {b['rsi_min10']}")
    print("    support", b["plan"]["support"], b["plan"]["support_kind"], "touches", b["plan"]["support_touches"], "dist", b["plan"]["dist_support_pct"], "t1", b["plan"]["t1"])
    for c in b["irtikaz"]["checks"]:
        print(f"    [{'OK' if c['ok'] else '--'}] {c['got']}/{c['weight']} {c['title']} :: {c['detail']}")
    print("    adj", [(x['name'], x['pts']) for x in b["irtikaz"]["adjustments"]])
