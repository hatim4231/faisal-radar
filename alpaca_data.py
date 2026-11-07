"""
بيانات Alpaca (حساب مجاني): شموع تاريخية من كل البورصات (SIP) وفيها البري والأفتر بالفوليوم الصحيح.
القيد في الخطة المجانية: ما نقدر نطلب آخر 15 دقيقة.
المفاتيح من Secrets: ALPACA_KEY_ID و ALPACA_SECRET_KEY (ما تنكتب في الكود أبداً).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

BASE = "https://data.alpaca.markets/v2/stocks/bars"


def _headers():
    k, s = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not k or not s:
        raise RuntimeError("ALPACA_KEY_ID / ALPACA_SECRET_KEY مو موجودة")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def available() -> bool:
    return bool(os.environ.get("ALPACA_KEY_ID") and os.environ.get("ALPACA_SECRET_KEY"))


def bars(symbols: list[str], timeframe: str, start: datetime, end: datetime | None = None) -> dict[str, pd.DataFrame]:
    """شموع لعدة أسهم (مع الصفحات). الأوقات بتوقيت نيويورك."""
    import requests
    end = end or (datetime.now(timezone.utc) - timedelta(minutes=16))
    params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start.isoformat(),
              "end": end.isoformat(), "limit": 10000, "adjustment": "split", "feed": "sip"}
    out: dict[str, list] = {}
    token = None
    while True:
        if token:
            params["page_token"] = token
        for attempt in range(5):
            r = requests.get(BASE, headers=_headers(), params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            break
        r.raise_for_status()
        js = r.json()
        for sym, rows in (js.get("bars") or {}).items():
            out.setdefault(sym, []).extend(rows)
        token = js.get("next_page_token")
        if not token:
            break
    res = {}
    for sym, rows in out.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
        df = df.set_index("t").rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "vw": "VWAP", "n": "Trades"})
        res[sym] = df[["Open", "High", "Low", "Close", "Volume"] + [c for c in ("VWAP", "Trades") if c in df]]
    return res


def session_of(idx) -> pd.Series:
    """pm: 4:00-9:30 · rth: 9:30-16:00 · ah: 16:00-20:00 (توقيت نيويورك)."""
    hm = idx.hour * 60 + idx.minute
    return pd.Series(["pm" if m < 570 else "rth" if m < 960 else "ah" for m in hm], index=idx)
