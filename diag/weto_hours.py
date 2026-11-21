"""إثبات: شموع الساعة لـ WETO (أو DATA_RUN) مع تمييز الجلسة PRE/RTH/AFTER — عشان نشوف الأفتر والبري مو بس اليومي."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import alpaca_data as A  # noqa: E402

SYMS = [s.strip().upper() for s in (os.environ.get("DATA_RUN") or "WETO").split(",") if s.strip()]
SINCE = datetime(2026, 11, 17, tzinfo=timezone.utc)
SES_AR = {"PRE": "بري", "RTH": "رسمي", "AFTER": "أفتر", "pre": "بري", "rth": "رسمي", "after": "أفتر",
          "premarket": "بري", "regular": "رسمي", "afterhours": "أفتر", "post": "أفتر"}


def run(sym):
    print(f"\n{'='*72}\n== {sym} — شموع الساعة (توقيت نيويورك) من 17 نوفمبر ==")
    hb = A.bars([sym], "1Hour", SINCE).get(sym)
    if hb is None or not len(hb):
        print("  ما رجّع Alpaca شموع ساعة لهالسهم"); return
    try:
        ses = A.session_of(hb.index)
    except Exception as e:
        ses = None
        print("  (ما قدرت أميّز الجلسة:", repr(e)[:70], ")")
    print("  الوقت NY    | الجلسة | فتح   أعلى   أدنى   إغلاق | فوليوم")
    day = None
    for ix, r in hb.iterrows():
        try:
            ny = ix.tz_convert("America/New_York")
        except Exception:
            ny = ix
        dd = ny.strftime("%Y-%m-%d")
        if dd != day:
            day = dd
            print(f"  --- {dd} ---")
        lab = ""
        if ses is not None:
            raw = str(ses.loc[ix]) if ix in ses.index else ""
            lab = SES_AR.get(raw, raw)
        print(f"  {ny.strftime('%m-%d %H:%M')} | {lab:>5} | {r.Open:.4f} {r.High:.4f} {r.Low:.4f} {r.Close:.4f} | {r.Volume:,.0f}")
    # أعلى سعر بالأفتر/البري مقابل إغلاق الرسمي
    print(f"\n  أعلى سعر خلال الفترة كلها: {hb['High'].max():.4f} · آخر سعر: {hb['Close'].iloc[-1]:.4f} · آخر شمعة: {hb.index[-1]}")


if __name__ == "__main__":
    for s in SYMS:
        run(s)
