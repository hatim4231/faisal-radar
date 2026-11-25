"""تحدي 11 نوفمبر: LICN/NXXT/RAYA — كل شموع 6→24 نوفمبر (معدّلة للتقسيم) + أكبر قفزة يومية (أعلى ÷ إغلاق أمس) خلال 11→24 + أعلى سعر مقابل إغلاق 10 نوفمبر."""
import sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, ".")
import alpaca_data as A
S = ["LICN", "NXXT", "RAYA"]
raw = A.bars(S, "1Day", datetime.now(timezone.utc) - timedelta(days=40))
for s in S:
    d = raw.get(s)
    d.index = d.index.tz_convert("America/New_York").normalize().tz_localize(None)
    w = d.loc["2026-11-05":"2026-11-24"]
    print(f"\n== {s} ==")
    for ix, r in w.iterrows():
        print(f"  {ix.date()}  O {r.Open:.4f} H {r.High:.4f} L {r.Low:.4f} C {r.Close:.4f} V {int(r.Volume):,}")
    c10 = float(d.loc["2026-11-10", "Close"])
    win = d.loc["2026-11-11":"2026-11-24"]
    prev = d["Close"].shift(1).loc["2026-11-11":"2026-11-24"]
    jump = (win.High / prev)
    print(f"  إغلاق 10 نوفمبر (معدّل) {c10:.4f} · أعلى سعر 11→24 {win.High.max():.4f} يوم {win.High.idxmax().date()} = {win.High.max()/c10*100-100:+.0f}% · "
          f"أكبر قفزة يومية ×{jump.max():.2f} يوم {jump.idxmax().date()} · إغلاق 24 نوفمبر {win.Close.iloc[-1]:.4f} ({win.Close.iloc[-1]/c10*100-100:+.0f}%)")
