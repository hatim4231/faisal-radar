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
CACHE_FILE = Path(__file__).resolve().parent / "cache" / "sec_cache_v2.json"
LOOKBACK_DAYS = 90
HIGH_DAYS = 30

FORMS = {
    "424B3": ("نشرة إعادة بيع — مساهمين يقدرون يبيعون أسهمهم", 2),
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
    hits += fts_hits(cik)
    hits.sort(key=lambda h: h["date"], reverse=True)
    if not hits:
        return {"level": "none", "hits": [], "cik": cik}
    recent_high = any(h["sev"] >= 3 and h["days_ago"] <= HIGH_DAYS for h in hits)
    level = "high" if recent_high else ("medium" if any(h["sev"] >= 2 for h in hits) else "low")
    return {"level": level, "hits": hits[:8], "cik": cik}


EXHIBITS = {
    "EX-1": "اتفاقية اكتتاب / وكيل طرح (Underwriting / Placement)",
    "EX-4": "Warrants أو سندات قابلة للتحويل",
    "EX-10": "اتفاقية شراء أسهم / تمويل (Securities Purchase)",
    "EX-99": "بيان صحفي عن تمويل",
}
FTS_QUERY = ('"registered direct" OR "private placement" OR "securities purchase agreement" OR "public offering" '
             'OR "pre-funded" OR "warrant inducement" OR "convertible note" OR "equity line" OR "at-the-market" '
             'OR "placement agent" OR "underwriting agreement"')


FTS_STATS = {"ok": 0, "err": 0, "hits": 0, "last_err": ""}


def fts_hits(cik: int) -> list[dict]:
    """بحث في نص ملفات 6-K و 8-K (الشركات الأجنبية تعلن الطرح والتمويل في 6-K)."""
    from urllib.parse import quote
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    url = (f"https://efts.sec.gov/LATEST/search-index?q={quote(FTS_QUERY)}&ciks={cik:010d}"
           f"&forms=6-K,8-K,6-K/A,8-K/A&dateRange=custom&startdt={start}&enddt={date.today().isoformat()}")
    try:
        js = _get(url)
        FTS_STATS["ok"] += 1
    except Exception as e:
        FTS_STATS["err"] += 1
        FTS_STATS["last_err"] = repr(e)[:160]
        return []
    FTS_STATS["hits"] += len(js.get("hits", {}).get("hits", []))
    out, seen = [], set()
    for h in js.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        fdate = src.get("file_date") or ""
        acc, _, fname = (h.get("_id") or "").partition(":")
        desc = (src.get("file_description") or src.get("file_type") or "").upper()
        if (acc, desc) in seen or not fdate:
            continue
        seen.add((acc, desc))
        ex = desc.split(".")[0]  # "EX-10.1" -> "EX-10"
        label = EXHIBITS.get(ex, "إعلان فيه طرح أو تمويل")
        try:
            days = (date.today() - datetime.strptime(fdate, "%Y-%m-%d").date()).days
        except ValueError:
            continue
        form = (src.get("form") or (src.get("root_forms") or ["6-K"])[0])
        out.append({"form": f"{form} {desc}".strip(), "date": fdate, "label": label, "sev": 3 if ex in ("EX-1", "EX-4", "EX-10") else 2,
                    "days_ago": days, "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{fname}"})
    # ملف واحد فيه عدة مرفقات: نخلي أهم مرفق لكل ملف
    best = {}
    for h in out:
        k = h["date"] + h["form"].split()[0]
        if k not in best or h["sev"] > best[k]["sev"]:
            best[k] = h
    return list(best.values())


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
    for r in rows:
        dl = r.get("dilution")
        if dl and dl["level"] != "none":
            log(f"    SEC {r['symbol']}: {dl['level']} · " + " | ".join(f"{h['date']} {h['form']}" for h in dl["hits"][:4]))
    lv = [r.get("dilution", {}).get("level") for r in rows]
    log(f"  SEC بحث 6-K/8-K: نجح {FTS_STATS['ok']} فشل {FTS_STATS['err']} نتائج {FTS_STATS['hits']} {FTS_STATS['last_err']}")
    log(f"  SEC: فُحص {ok} (فشل {fail}) · عالي {lv.count('high')} · متوسط {lv.count('medium')} · منخفض {lv.count('low')}")


# ----------------------------------------------------------------------------
# قرار المستخدم: الشركة اللي عندها طرح/تخفيف متكرر خلال شهور قليلة ما تترشح أبداً (كل الرادارات)
# ----------------------------------------------------------------------------
REPEAT_MIN_EVENTS = 2     # حدثين تخفيف منفصلين أو أكثر خلال 90 يوم = متكرر
MERGE_DAYS = 3            # ملفات بنفس الأيام (مثلاً EFFECT ثم 424B بعد يوم) = حدث واحد
NEWS_DILUTION = ("طرح أسهم (تخفيف)", "Warrants (تخفيف)", "تخفيف", "تمويل بتخفيف محتمل")


def dilution_events(r: dict) -> list[dict]:
    """أحداث التخفيف المنفصلة: من ملفات SEC (بدون S-8) + الأخبار السلبية عن طرح/Warrants."""
    pts = []
    for h in (r.get("dilution") or {}).get("hits", []):
        if h["sev"] >= 2 and not h["form"].upper().startswith("S-8"):
            pts.append((h["date"], f"SEC {h['form']}"))
    for n in r.get("news") or []:
        if n.get("sentiment") == "negative" and any(w in NEWS_DILUTION for w in n.get("why_neg") or []):
            d = (n.get("time") or "")[:10]
            try:
                if d and (date.today() - datetime.strptime(d, "%Y-%m-%d").date()).days <= LOOKBACK_DAYS:
                    pts.append((d, "خبر: " + (n.get("title_ar") or n.get("title") or "")[:60]))
            except ValueError:
                pass
    pts.sort()
    events = []
    for d, what in pts:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        if events and (dt - events[-1]["_d"]).days <= MERGE_DAYS:
            events[-1]["what"].append(what)
            continue
        events.append({"_d": dt, "date": d, "what": [what]})
    for e in events:
        e.pop("_d")
    return events


def _status_score(r: dict) -> tuple:
    """حالة السهم ونقاطه بأي رادار (فيصل: irtikaz، الباقي: status/score/quality)."""
    I = r.get("irtikaz")
    if isinstance(I, dict):
        return I.get("status"), I.get("score")
    return r.get("status"), r.get("score", r.get("quality"))


def exclude_repeat_diluters(rows: list[dict], log=print, full: list | None = None) -> tuple[list[dict], list[dict]]:
    """قرار المستخدم: الطرح المتكرر ما يترشح. الخيار 2 (2026-11-10 بعد RUBI): ما ينشال بصمت — الملخص فيه حالته ونقاطه
    عشان الصفحة تعرضه بقسم «⚠ طرح متكرر — مخاطرة عالية». لو مررت قائمة `full` تنضاف لها الأسطر الكاملة (للبطاقات)."""
    keep, out = [], []
    for r in rows:
        ev = dilution_events(r)
        if len(ev) >= REPEAT_MIN_EVENTS:
            st, sc = _status_score(r)
            r["dilution_events"] = [{"date": e["date"], "what": " + ".join(e["what"][:3])} for e in ev]
            out.append({"symbol": r["symbol"], "name": r.get("name"), "price": r.get("price"), "status": st,
                        "score": round(float(sc), 1) if isinstance(sc, (int, float)) else sc,
                        "events": r["dilution_events"]})
            if full is not None:
                full.append(r)
        else:
            keep.append(r)
    if out:
        log(f"  مستبعد لتخفيف متكرر ({len(out)}): " + ", ".join(f"{x['symbol']}({len(x['events'])})" for x in out))
    return keep, out
