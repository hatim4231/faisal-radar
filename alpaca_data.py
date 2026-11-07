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


# ----------------------------------------------------------------------------
# الشارت اليومي + التقسيمات (بديل ياهو في الرادارات)
# ----------------------------------------------------------------------------

def splits(symbols: list[str], start: datetime) -> dict[str, list[tuple]]:
    """التقسيمات (العكسي والعادي) من corporate-actions: {سهم: [(تاريخ, نسبة مثل 0.1 لتقسيم 1:10)]}."""
    import requests
    out: dict[str, list] = {}
    params = {"symbols": ",".join(symbols), "types": "reverse_split,forward_split",
              "start": start.date().isoformat(), "end": datetime.now(timezone.utc).date().isoformat(), "limit": 1000}
    token = None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get("https://data.alpaca.markets/v1/corporate-actions", headers=_headers(), params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        ca = js.get("corporate_actions") or {}
        for kind in ("reverse_splits", "forward_splits"):
            for x in ca.get(kind) or []:
                try:
                    ratio = float(x["new_rate"]) / float(x["old_rate"])
                    out.setdefault(x["symbol"], []).append((pd.Timestamp(x["ex_date"]), ratio))
                except Exception:
                    pass
        token = js.get("next_page_token")
        if not token:
            break
    return out


def daily(symbols: list[str], days: int = 730) -> dict[str, pd.DataFrame]:
    """شارت يومي (جلسة السوق) معدّل للتقسيم + عمود Stock Splits مثل ياهو.
    إذا اليوم الحالي ما طلع كشمعة يومية (وقت السوق)، نبنيه من شموع 15 دقيقة (بتأخير 15 دقيقة)."""
    start = datetime.now(timezone.utc) - timedelta(days=days)
    raw = bars(symbols, "1Day", start)
    try:
        spl = splits(symbols, start)
    except Exception:
        spl = {}
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    need_today = [s for s, d in raw.items() if not d.empty and d.index[-1].normalize() < today]
    now_ny = pd.Timestamp.now(tz="America/New_York")
    if need_today and now_ny.weekday() < 5 and now_ny.hour * 60 + now_ny.minute >= 9 * 60 + 50:
        try:
            m15 = bars(need_today, "15Min", (today + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC").to_pydatetime())
        except Exception:
            m15 = {}
        for s, m in m15.items():
            m = m[(m.index >= today + pd.Timedelta(hours=9, minutes=30)) & (m.index < today + pd.Timedelta(hours=16))]
            if m.empty:
                continue
            row = pd.DataFrame({"Open": [m["Open"].iloc[0]], "High": [m["High"].max()], "Low": [m["Low"].min()],
                                "Close": [m["Close"].iloc[-1]], "Volume": [m["Volume"].sum()]}, index=[today])
            raw[s] = pd.concat([raw[s][["Open", "High", "Low", "Close", "Volume"]], row])
    out = {}
    for s, d in raw.items():
        d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize() if d.index.tz is not None else pd.to_datetime(d.index).normalize()
        d = d[~d.index.duplicated(keep="last")]
        d["Stock Splits"] = 0.0
        for dt, ratio in spl.get(s, []):
            dt = dt.normalize()
            if dt in d.index:
                d.loc[dt, "Stock Splits"] = ratio
        out[s] = d
    return out


def intraday_frame(symbols: list[str], timeframe: str, days: int) -> pd.DataFrame:
    """شموع داخل اليوم (فيها البري والأفتر) بنفس شكل yf.download(group_by='ticker')."""
    start = datetime.now(timezone.utc) - timedelta(days=days)
    got = bars(symbols, timeframe, start)
    frames = {s: d[["Open", "High", "Low", "Close", "Volume"]] for s, d in got.items() if not d.empty}
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)
