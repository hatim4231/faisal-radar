"""كم سهم من أسهمنا HRT من ملاكه؟ (المجال الجديد 0.7-10$ والقديم 0.5-5$) + تحقق من ياهو لعينة."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scanner
import holders as Hd

t = Hd.hrt_table(max_age_days=0)
print("المقدِّم:", t.get("name"), "· النموذج", t.get("form"), "· الفترة", t.get("period"), "· تاريخ التقديم", t.get("filed"),
      "· عدد الأسهم في التقرير", t.get("n"), "· خطأ:", t.get("error"))
for rng in ((0.5, 5.0), (0.7, 10.0)):
    cfg = scanner.load_cfg()
    cfg["universe"]["min_price"], cfg["universe"]["max_price"] = rng
    tick = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    hits = []
    for s in tick:
        m = Hd.match(s, [Hd.sec_title(s)], t)
        if m:
            hits.append((s, Hd.sec_title(s), m["name"], int(m["shares"])))
    print(f"\n== المجال {rng}: أسهم {len(tick)} · HRT يملك {len(hits)} ({len(hits) / max(1, len(tick)) * 100:.0f}%)")
    for h in hits[:40]:
        print("  ", h)
    if rng == (0.7, 10.0):
        import yfinance as yf
        random.seed(3)
        sample = random.sample([h[0] for h in hits], min(12, len(hits))) + \
            random.sample([s for s in tick if s not in {h[0] for h in hits}], 12)
        print("\n== تحقق من ياهو (أكبر 10 مؤسسات):")
        for s in sample:
            try:
                ih = yf.Ticker(s).institutional_holders
                names = [] if ih is None or ih.empty else list(ih.iloc[:, 0].astype(str))
                hrt = [n for n in names if "HRT" in n.upper() or "HUDSON RIVER" in n.upper()]
                print(f"  {s:<6} مطابقتنا={'نعم' if s in {h[0] for h in hits} else 'لا'} · ياهو={'نعم ' + str(hrt) if hrt else 'لا'} · {names[:4]}")
            except Exception as e:
                print("  ", s, "خطأ", repr(e)[:80])
