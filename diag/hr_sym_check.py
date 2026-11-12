"""فحص سريع لرموز معيّنة: هل هي بقائمة Alpaca النشطة؟ وكم شمعة يومية لها وآخر تاريخ؟ (DATA_RUN = الرموز بفاصلة)
يفيد لما سهم يختفي من الرادار (تغيير رمز / شطب / إيقاف)."""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import requests

import alpaca_data as A

syms = [s.strip().upper() for s in (os.environ.get("DATA_RUN") or "NXXT,QMMM,LICN,RAYA").split(",") if s.strip()]
for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
    ok = False
    for s in syms:
        try:
            r = requests.get(f"{base}/v2/assets/{s}", headers=A._headers(), timeout=30)
            if r.status_code == 200:
                a = r.json()
                ok = True
                print(f"{s}: status={a.get('status')} · tradable={a.get('tradable')} · exchange={a.get('exchange')} · name={a.get('name')}")
            else:
                print(f"{s}: assets HTTP {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(s, "assets err", repr(e)[:100])
    if ok:
        break
bars = A.bars(syms, "1Day", datetime.now(timezone.utc) - timedelta(days=420))
for s in syms:
    b = bars.get(s)
    if b is None or b.empty:
        print(f"{s}: ما فيه شموع يومية بآخر 420 يوم")
        continue
    t = b.tail(6)
    print(f"{s}: شموع {len(b)} · من {str(b.index[0])[:10]} لين {str(b.index[-1])[:10]}")
    for ix, r in t.iterrows():
        print(f"    {str(ix)[:10]}  O {r.Open:.4f}  H {r.High:.4f}  L {r.Low:.4f}  C {r.Close:.4f}  V {int(r.Volume):,}")
try:
    for s, L in A.splits(syms, datetime.now(timezone.utc) - timedelta(days=420)).items():
        print(f"{s}: تقسيمات {[(str(d)[:10], r) for d, r in L]}")
except Exception as e:
    print("splits err", repr(e)[:100])
