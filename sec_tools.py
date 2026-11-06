"""
فحص خطر التخفيف من ملفات SEC الرسمية (EDGAR) — مشترك بين الرادارات الثلاثة.

لكل سهم مرشح: هل عنده خلال آخر 90 يوم ملفات طرح/تسجيل أسهم؟
  - 424B1..424B8      ← نشرة طرح تسعّرت (الطرح صار أو جاري)  [خطر عالي إذا خلال 30 يوم]
  - S-1 / F-1         ← تسجيل طرح جديد أو إعادة بيع (Resale)
  - S-3 / F-3         ← رف طرح (Shelf) — تقدر تطرح أي وقت، ومنه ATM
  - EFFECT            ← SEC فعّلت التسجيل = الأسهم صارت جاهزة للبيع
  - S-8               ← أسهم موظفين (خطر خفيف)

SEC تطلب تعريف البرنامج بإيميل تواصل (User-Agent)، وحد أقصى 10 طلبات بالثانية.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

USER_AGENT = "FaisalRadar hatim423@hotmail.com"
CACHE_FILE = Path(__file__).resolve().parent / "cache" / "sec_cache.json"
LOOKBACK_DAYS = 90
HIGH_DAYS = 30

FORMS = {
    "424B": ("نشرة طرح تسعّرت", 3),
    "S-1": ("تسجيل طرح / إعادة بيع (S-1)", 2),
    "F-1": ("تسجيل طرح / إعادة بيع (F-1)", 2),
    "S-3": ("رف طرح Shelf (S-3) — ممكن ATM", 2),
    "F-3": ("رف طرح Shelf (F-3) — ممكن ATM", 2),
    "EFFECT": ("SEC فعّلت التسجيل — الأسهم جاهزة للبيع", 3),
    "S-8": ("أسهم موظفين (S-8)", 1),
}

_lock = threading.Lock()
_last = [0.0]
_tickers: dict | None = None


def _get(url: str):
    import requests
    with _lock:  # حد SEC: ~10 طلبات بالثانية
        wait = 0.12 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
    r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}, timeout=20)
    r.raise_for_status()
    return r.json()


def _cik(ticker: str) -> int | None:
    global _tickers
    if _tickers is None:
        js = _get("https://www.sec.gov/files/company_tickers.json")
        _tickers = {v["ticker"].upper(): int(v["cik_str"]) for v in js.values()}
    return _tickers.get(ticker.upper().replace("-", "."))


def _classify(form: str):
    f = form.upper()
    for key, (label, sev) in FORMS.items():
        if f.startswith(key):
            return key, label, sev
    return None


def check(ticker: str) -> dict | None:
    cik = _cik(ticker)
    if not cik:
        return None
    js = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    rec = js.get("filings", {}).get("recent", {})
    since = date.today() - timedelta(days=LOOKBACK_DAYS)
    hits = []
    for form, fdate, acc, doc in zip(rec.get("form", []), rec.get("filingDate", []),
                                     rec.get("accessionNumber", []), rec.get("primaryDocument", [])):
        try:
            d = datetime.strptime(fdate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < since:
            break  # مرتبة من الأحدث
        c = _classify(form)
        if not c:
            continue
        key, label, sev = c
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        hits.append({"form": form, "date": fdate, "label": label, "sev": sev, "url": url,
                     "days_ago": (date.today() - d).days})
    if not hits:
        return {"level": "none", "hits": [], "cik": cik}
    recent_high = any(h["sev"] >= 3 and h["days_ago"] <= HIGH_DAYS for h in hits)
    level = "high" if recent_high else ("medium" if any(h["sev"] >= 2 for h in hits) else "low")
    return {"level": level, "hits": hits[:8], "cik": cik}


def _load():
    try:
        c = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return c if c.get("_day") == str(date.today()) else {"_day": str(date.today())}
    except Exception:
        return {"_day": str(date.today())}


def annotate(rows: list[dict], log=print) -> None:
    """يضيف r['dilution'] ووسم للأسهم المرشحة. ما يغير النقاط ولا الحالة."""
    cache = _load()
    ok = fail = 0
    for r in rows:
        t = r["symbol"]
        if t in cache:
            res = cache[t]
        else:
            try:
                res = check(t)
                ok += 1
            except Exception:
                res = None
                fail += 1
            cache[t] = res
        if not res:
            continue
        r["dilution"] = res
        tags = r.setdefault("tags", [])
        if res["level"] == "high":
            tags.append("⛔ تخفيف SEC")
        elif res["level"] == "medium":
            tags.append("⚠ تخفيف محتمل SEC")
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass
    lv = [r.get("dilution", {}).get("level") for r in rows]
    log(f"  SEC: فُحص {ok} (فشل {fail}) · عالي {lv.count('high')} · متوسط {lv.count('medium')} · منخفض {lv.count('low')}")
