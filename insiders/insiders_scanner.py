#!/usr/bin/env python3
"""
🟢 رادار المطّلعين (Insider Buying) — رادار مستقل.

الفكرة: المدير أو عضو مجلس الإدارة أو مالك 10%+ يشتري أسهم شركته من السوق المفتوح بفلوسه
(Form 4، رمز العملية P). "المطّلع يبيع لأسباب كثيرة، بس يشتري لسبب واحد."

الفلاتر العامة نفس باقي الرادارات: 1-5$، فوليوم، قيمة سوقية، فلووت، بدون Warrants،
الشورت 50 ألف وأقل (قرار المستخدم)، واستبعاد شركات الطرح/التخفيف المتكرر.
الحد الأدنى: مجموع الشراء 25 ألف دولار خلال آخر 30 يوم (يتجاهل الشراء الرمزي).

ملاحظة: الشركات الأجنبية (أغلب الصينية) ما تقدم Form 4، فالرادار يغطي الشركات الأمريكية فقط.

    python insiders/insiders_scanner.py --out site/insiders/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import scanner as data  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
CACHE = data.CACHE / "insiders_form4.json"   # كاش ملفات Form 4 المقروءة (ما تتغير)

LOOKBACK = 30
MIN_TOTAL = 25_000
TOP_ROLES = re.compile(r"\b(CEO|Chief Executive|President|CFO|Chief Financial|Chairman|COO|Chief Operating)\b", re.I)


# ----------------------------------------------------------------------------
# SEC
# ----------------------------------------------------------------------------

def _get_text(url: str) -> str:
    import requests
    with sec_tools._lock:
        wait = 0.12 - (time.time() - sec_tools._last[0])
        if wait > 0:
            time.sleep(wait)
        sec_tools._last[0] = time.time()
    r = requests.get(url, headers={"User-Agent": sec_tools.USER_AGENT}, timeout=20)
    r.raise_for_status()
    return r.text


def form4_list(cik: int) -> list[dict]:
    """ملفات Form 4 خلال آخر 30 يوم للشركة (المصدرة)."""
    js = sec_tools._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    rec = js.get("filings", {}).get("recent", {})
    since = date.today() - timedelta(days=LOOKBACK)
    out = []
    for form, fdate, acc, doc in zip(rec.get("form", []), rec.get("filingDate", []),
                                     rec.get("accessionNumber", []), rec.get("primaryDocument", [])):
        try:
            if datetime.strptime(fdate, "%Y-%m-%d").date() < since:
                break
        except ValueError:
            continue
        if form not in ("4", "4/A"):
            continue
        raw = doc.split("/")[-1]  # "xslF345X05/form4.xml" -> "form4.xml" (الملف الخام)
        out.append({"date": fdate, "acc": acc,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{raw}"})
    return out


def _t(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_form4(xml: str) -> list[dict]:
    """يرجع عمليات الشراء من السوق المفتوح (P) فقط."""
    root = ET.fromstring(xml)
    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        roles = []
        if rel is not None:
            if _t(rel, "isDirector") in ("1", "true"):
                roles.append("عضو مجلس إدارة")
            if _t(rel, "isOfficer") in ("1", "true"):
                roles.append(_t(rel, "officerTitle") or "مسؤول تنفيذي")
            if _t(rel, "isTenPercentOwner") in ("1", "true"):
                roles.append("مالك 10%+")
        owners.append({"name": _t(ro, "reportingOwnerId/rptOwnerName") or "?", "roles": roles})
    buys = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _t(tx, "transactionCoding/transactionCode")
        ad = _t(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
        if code != "P" or ad != "A":
            continue
        try:
            sh = float(_t(tx, "transactionAmounts/transactionShares/value") or 0)
            px = float(_t(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        if sh <= 0 or px <= 0:
            continue
        for o in owners[:1]:  # أول مالك مبلّغ (غالباً واحد)
            buys.append({"name": o["name"], "roles": o["roles"], "date": _t(tx, "transactionDate/value"),
                         "shares": sh, "price": px, "value": sh * px})
    return buys


def insider_buys(ticker: str, cache: dict) -> list[dict]:
    cik = sec_tools._cik(ticker)
    if not cik:
        return []
    buys = []
    for f in form4_list(cik):
        if f["acc"] in cache:
            got = cache[f["acc"]]
        else:
            try:
                got = parse_form4(_get_text(f["url"]))
            except Exception:
                got = []
            cache[f["acc"]] = got
        for b in got:
            buys.append({**b, "filed": f["date"]})
    return buys


# ----------------------------------------------------------------------------
# التقييم
# ----------------------------------------------------------------------------

def score(buys: list[dict], price: float, df: pd.DataFrame) -> dict:
    total = sum(b["value"] for b in buys)
    shares = sum(b["shares"] for b in buys)
    avg = total / shares if shares else None
    people = {}
    for b in buys:
        p = people.setdefault(b["name"], {"name": b["name"], "roles": b["roles"], "value": 0, "shares": 0, "last": b["date"]})
        p["value"] += b["value"]
        p["shares"] += b["shares"]
        p["last"] = max(p["last"] or "", b["date"] or "")
    n = len(people)
    top = any(TOP_ROLES.search(" ".join(p["roles"])) for p in people.values())
    director = any("عضو مجلس" in " ".join(p["roles"]) for p in people.values())
    last = max((b["date"] or b["filed"]) for b in buys)
    days = (date.today() - datetime.strptime(last[:10], "%Y-%m-%d").date()).days
    above = (price / avg - 1) * 100 if avg else None

    # الأوزان من الاختبار التاريخي (ملفات SEC، نوفمبر 2024 - أبريل 2026، قرار المستخدم):
    # المنصب أقوى شرط (CEO/CFO/الرئيس)، العدد والمبلغ وقرب السعر من متوسطهم فرقهم بسيط
    pts, why = 0, []
    r = 40 if top else 15 if director else 5
    pts += r
    why.append("المدير التنفيذي/المالي/الرئيس اشترى" if top else "عضو مجلس إدارة" if director else "مالك كبير")
    c = 20 if n >= 3 else 15 if n == 2 else 8
    pts += c
    why.append(f"{n} مطّلع اشترى" + (" (Cluster)" if n >= 2 else ""))
    a = 10 if total >= 100_000 else 6
    pts += a
    rec = 10 if days <= 7 else 6 if days <= 14 else 3
    pts += rec
    near = 5 if above is not None and above <= 25 else 0
    pts += near
    # الشارت: فوق قاع 20 يوم وما كسر قاع جديد = أفضل
    low20 = float(df["Low"].tail(20).min())
    stop = round(max(price * 0.85, low20 * 0.97), 4)
    status = "ready" if pts >= 65 and top and (above is None or above <= 20) else "near" if pts >= 45 else "radar"
    if pts >= 65 and not top:
        why.append("«شراء قوي» يحتاج المدير التنفيذي أو المالي أو الرئيس")
    if above is not None and above > 30:
        status = "radar"
        why.append(f"السعر فوق متوسط شرائهم بـ {above:.0f}% — لا تطارد")
    return {"score": pts, "status": status, "total": round(total), "avg_price": round(avg, 4) if avg else None,
            "above_avg_pct": round(above, 1) if above is not None else None, "insiders": n, "top_role": top,
            "people": sorted(people.values(), key=lambda p: -p["value"]), "last_buy": last[:10], "days_ago": days,
            "why": why, "stop": stop, "targets": [round(price * 1.2, 4), round(price * 1.4, 4)]}


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    explicit = tickers is not None
    if tickers is None:
        tickers = data.get_universe(cfg)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not explicit:
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = data.download_daily(tickers, cfg)
    U = dict(cfg["universe"])
    # قرار المستخدم (نوفمبر 2026): كل الرادارات 0.7$ إلى 10$ (كان المطّلعين من 1$)
    pre = {t: d for t, d in daily.items()
           if explicit or (U["min_price"] <= float(d["Close"].iloc[-1]) <= U["max_price"]
                           and float(d["Volume"].tail(30).mean()) >= U["min_avg_volume"] * 0.7)}
    log(f"[المطّلعين] فحص Form 4 لـ {len(pre)} سهم...")
    cache = {}
    try:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    found, errs = {}, 0
    for i, t in enumerate(pre):
        try:
            b = insider_buys(t, cache)
        except Exception:
            errs += 1
            continue
        if b:
            px_now = float(pre[t]["Close"].iloc[-1])
            mcap_guess = px_now * float(pre[t]["Volume"].tail(30).mean()) * 200
            ok_b = [x for x in b if 0.25 * px_now <= x["price"] <= 4 * px_now and x["value"] <= max(5e6, mcap_guess)]
            if len(ok_b) < len(b):
                log(f"  {t}: تجاهلت {len(b) - len(ok_b)} عملية أرقامها غير منطقية (سعر/مبلغ)")
            b = ok_b
        if b and sum(x["value"] for x in b) >= MIN_TOTAL:
            found[t] = b
        if i % 100 == 0:
            log(f"  {i}/{len(pre)} · لقيت {len(found)}")
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    log(f"[المطّلعين] أسهم فيها شراء داخلي ≥ ${MIN_TOTAL:,}: {len(found)} (أخطاء {errs})")

    infos = data.get_info(list(found), cfg)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    max_sh = cfg["behavior"].get("max_short_shares")  # 50 ألف (None = مفتوح)
    rows = []
    for t, buys in found.items():
        d = pre[t]
        inf = infos.get(t) or {}
        price = float(d["Close"].iloc[-1])
        mcap, flt = inf.get("marketCap"), inf.get("floatShares")
        if not explicit and ((mcap and mcap > U["max_market_cap"]) or (flt and flt > U["max_float"])):
            continue
        sa = (ib.get(t) or {}).get("available")
        sv, src = (sa, "المتاح للاقتراض") if sa is not None else (inf.get("sharesShort"), "Short Interest")
        if sv is not None and max_sh and sv > max_sh and not explicit:
            continue
        if mcap:
            buys = [x for x in buys if x["value"] <= mcap * 0.5]
            if sum(x["value"] for x in buys) < MIN_TOTAL:
                continue
        s = score(buys, price, d)
        prev = float(d["Close"].iloc[-2])
        tail = d.tail(90)
        rows.append({
            "symbol": t, "name": inf.get("shortName") or inf.get("longName") or t,
            "sector": inf.get("sector") or inf.get("industry"), "price": round(price, 4),
            "change_pct": round((price / prev - 1) * 100, 2), "date": str(d.index[-1])[:10],
            "market_cap": mcap, "float": flt, "short_used": {"value": sv, "source": src}, **s,
            "chart": {"d": [str(i)[:10] for i in tail.index], "c": [round(float(x), 4) for x in tail["Close"]]},
            "tags": [], "flags": [] if (sv is not None or not max_sh) else ["الشورت غير معروف"],
            "country": inf.get("country"),
        })
        if sv is None and max_sh and rows[-1]["status"] in ("ready", "near"):
            rows[-1]["status"] = "radar"

    # الأخبار + التخفيف + استبعاد التخفيف المتكرر
    import yfinance as yf

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n
    with ThreadPoolExecutor(6) as ex:
        nm = dict(ex.map(news, [r["symbol"] for r in rows]))
    for r in rows:
        r["news"] = nm.get(r["symbol"], [])
    news_tools.save_cache()
    sec_tools.annotate(rows, log)
    rows, excluded = sec_tools.exclude_repeat_diluters(rows, log)
    import flags
    import holders
    flags.annotate(rows)
    excluded_hrt = []
    if cfg["behavior"].get("exclude_hrt"):
        try:
            holders.annotate(rows, log)
            rows, excluded_hrt = holders.exclude(rows, log)
        except Exception as e:
            log("فحص HRT تعطل:", repr(e)[:150])
    order = {"ready": 0, "near": 1, "radar": 2}
    rows.sort(key=lambda r: (order[r["status"]], -r["score"]))
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "SEC EDGAR (Form 4) + الأسعار: Alpaca + الشركة: Yahoo", "scanned": len(pre), "found": len(found),
               "excluded_hrt": excluded_hrt,
               "min_total": MIN_TOTAL, "lookback": LOOKBACK, "seconds": round(time.time() - t0),
               "stocks": rows, "excluded_dilution": excluded}
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    log(f"## رادار المطّلعين: شراء قوي {c['ready']} · شراء داخلي {c['near']} · مراقبة {c['radar']} · "
        f"خلص في {payload['seconds']} ثانية")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status']} {r['score']} · {r['insiders']} مطّلع · ${r['total']:,} · "
            f"متوسط {r['avg_price']} · آخر شراء {r['last_buy']}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار المطّلعين")
    ap.add_argument("--out", default=str(OUT / "insiders.html"))
    ap.add_argument("--tickers")
    a = ap.parse_args()
    cfg = data.load_cfg()
    payload = run(cfg, a.tickers.split(",") if a.tickers else None)
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    html = (HERE / "template.html").read_text(encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")),
                   encoding="utf-8")
    (OUT / "sig_insiders.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[المطّلعين] التقرير: {out}")


if __name__ == "__main__":
    main()
