"""تشخيص: وش قال كل رادار عن أسهم أوصى عليها فيصل، على كل جلسة من آخر 6 جلسات."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "abdullah"))
import scanner, faisal_rules as fr, abdullah_rules as ar

T = sys.argv[1].split(",")
cfg = scanner.load_cfg()
acfg = json.loads((Path(__file__).resolve().parent.parent / "abdullah" / "config.json").read_text(encoding="utf-8"))
daily = scanner.download_daily(T, cfg)
infos = scanner.get_info(T, cfg)
try:
    ib = scanner.load_ib_short()
except Exception:
    ib = {}
for t in T:
    print("\n" + "#" * 70 + f"\n# {t}")
    d = daily.get(t)
    if d is None:
        print("  لا توجد بيانات"); continue
    inf = dict(infos.get(t) or {})
    sa = (ib.get(t) or {}).get("available")
    if sa is not None:
        inf["shortAvailable"] = sa
    print(f"  name={inf.get('shortName')} mcap={inf.get('marketCap')} float={inf.get('floatShares')} SI={inf.get('sharesShort')} IB_avail={sa}")
    print(d[["Open", "High", "Low", "Close", "Volume"]].tail(12).round(3).to_string())
    for cut in (6, 5, 4, 3, 2, 1, 0):
        dd = d.iloc[:len(d) - cut] if cut else d
        date, px = dd.index[-1].date(), float(dd["Close"].iloc[-1])
        a = fr.analyze(dd, inf, cfg)
        line = f"  {date} close {px:.2f}"
        if a:
            I, M = a["irtikaz"], a["momentum"]
            miss = [c["title"][:22] for c in I["checks"] if not c["ok"]]
            adj = [f"{x['name']}{x['pts']:+d}" for x in I["adjustments"]]
            line += (f" | فيصل ارتكاز {I['status']} {I['score']:.0f} · زخم {M['status']} {M['score']:.0f}"
                     f" | RSI {a['rsi']} | gate {a['gate_fail']}\n      ناقص: {miss}\n      تعديلات: {adj}")
        b = ar.analyze(dd, inf, acfg)
        if b:
            k = b["consolidation"]
            line += (f"\n      عبدالله: {b['status']} {b['score']:.0f} | موجة {b['wave']['low']}→{b['wave']['peak']} x{b['wave']['mult']}"
                     f" | دعم {b['support']} اختبارات {k['touches']} تماسك {k['len']}ج")
        else:
            line += "\n      عبدالله: لا توجد موجة تدبيل"
        print(line)
