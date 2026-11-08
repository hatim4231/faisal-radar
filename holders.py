"""
فحص الملاك: هل صندوق HRT من ضمن ملاك السهم؟ (طلب المستخدم)

HRT = Hudson River Trading (صانع سوق / تداول خوارزمي). يقدم تقرير ملكياته الربعي لـ SEC (نموذج 13F-HR)
باسم «HRT Financial LP» (كان اسمه HRT Financial LLC) — رقم SEC (CIK) = 1475597. Hudson River Trading LLC تملك 98% منه.
نقرا آخر تقرير 13F (الأسهم بس، بدون عقود الأوبشن Put/Call) ونطابق اسم الشركة مع أسهمنا.
تنبيه: تقرير 13F يطلع بعد نهاية الربع بـ 45 يوم ← الملكية ممكن تكون تغيرت.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "hrt_13f.json"
HRT_CIK = 1475597
HRT_NAME = "HRT Financial LP (Hudson River Trading)"

STOP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "SA", "NV", "AG", "SE",
        "LLC", "LP", "THE", "CL", "CLASS", "A", "B", "COM", "ORD", "SHS", "ADR", "ADS", "NEW", "HLDGS", "HOLDINGS",
        "HOLDING", "GROUP", "GRP", "INTL", "INTERNATIONAL", "DE", "NY", "NV", "BVI", "CAYMAN", "TR", "TRUST", "SPONSORED",
        "PAR", "USD", "VALUE", "OF", "AND", "N", "V", "S"}
ABBR = {"TECHNOLOGIES": "TECH", "TECHNOLOGY": "TECH", "TECHNOLOGS": "TECH", "PHARMACEUTICALS": "PHARMA",
        "PHARMACEUTICAL": "PHARMA", "PHARMACEUTICA": "PHARMA", "THERAPEUTICS": "THERAP", "THERAPEUTIC": "THERAP",
        "BIOSCIENCES": "BIOSCI", "BIOSCIENCE": "BIOSCI", "INDUSTRIES": "INDS", "SYSTEMS": "SYS", "SOLUTIONS": "SOLUT",
        "ENTERPRISES": "ENTPR", "INTERNATIONAL": "INTL", "MEDICAL": "MED", "ENERGY": "ENRGY", "RESOURCES": "RES",
        "COMMUNICATIONS": "COMMUN", "ACQUISITION": "ACQ", "AMERICA": "AMER", "AMERICAN": "AMER", "FINANCIAL": "FINL"}


def norm(name: str) -> list[str]:
    s = re.sub(r"[^A-Z0-9 ]", " ", (name or "").upper().replace("&", " AND "))
    toks = [ABBR.get(t, t) for t in s.split()]
    return [t for t in toks if t not in STOP]


def _get_json(url):
    import sec_tools
    return sec_tools._get(url)


def _get_text(url):
    import requests
    import sec_tools
    for i in range(3):
        r = requests.get(url, headers={"User-Agent": sec_tools.USER_AGENT}, timeout=60)
        if r.status_code == 200:
            return r.text
        time.sleep(2 * (i + 1))
    r.raise_for_status()


def latest_filing():
    js = _get_json(f"https://data.sec.gov/submissions/CIK{HRT_CIK:010d}.json")
    rec = js["filings"]["recent"]
    for form, acc, filed, period in zip(rec["form"], rec["accessionNumber"], rec["filingDate"], rec["reportDate"]):
        if form in ("13F-HR", "13F-HR/A"):
            return {"acc": acc, "filed": filed, "period": period, "form": form, "name": js.get("name")}
    return None


def parse_info_table(xml: str) -> list[dict]:
    out = []
    for blk in re.findall(r"<(?:\w+:)?infoTable>(.*?)</(?:\w+:)?infoTable>", xml, re.S):
        def g(tag):
            m = re.search(rf"<(?:\w+:)?{tag}>\s*([^<]*?)\s*</(?:\w+:)?{tag}>", blk, re.S)
            return m.group(1) if m else None
        if g("putCall"):
            continue  # عقود أوبشن، مو ملكية أسهم
        try:
            sh = float(g("sshPrnamt") or 0)
        except ValueError:
            sh = 0
        out.append({"name": g("nameOfIssuer"), "cls": g("titleOfClass"), "cusip": g("cusip"),
                    "shares": sh, "value": g("value")})
    return out


def hrt_table(max_age_days: float = 3) -> dict:
    """آخر تقرير 13F لـ HRT (كاش، يتحدث كل كم يوم أو لما يطلع تقرير جديد)."""
    try:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    if cache.get("checked") and (time.time() - cache["checked"]) < max_age_days * 86400:
        return cache
    try:
        f = latest_filing()
        if not f:
            return cache
        if cache.get("acc") != f["acc"]:
            base = f"https://www.sec.gov/Archives/edgar/data/{HRT_CIK}/{f['acc'].replace('-', '')}"
            idx = _get_json(f"{base}/index.json")
            xmls = [it for it in idx["directory"]["item"] if it["name"].lower().endswith(".xml")
                    and "primary_doc" not in it["name"].lower()]
            xmls.sort(key=lambda it: -int(it.get("size") or 0))
            rows = parse_info_table(_get_text(f"{base}/{xmls[0]['name']}")) if xmls else []
            agg = {}
            for r in rows:
                k = " ".join(norm(r["name"]))
                if not k:
                    continue
                a = agg.setdefault(k, {"name": r["name"], "cusip": r["cusip"], "shares": 0.0})
                a["shares"] += r["shares"]
            cache = {**f, "filer": HRT_NAME, "holdings": agg, "n": len(agg)}
        cache["checked"] = time.time()
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        cache.setdefault("error", repr(e)[:160])
    return cache


_SEC_TITLES = None


def sec_title(ticker: str) -> str | None:
    """اسم الشركة الرسمي في SEC (أقرب لأسماء تقارير 13F من اسم ياهو)."""
    global _SEC_TITLES
    if _SEC_TITLES is None:
        try:
            js = _get_json("https://www.sec.gov/files/company_tickers.json")
            _SEC_TITLES = {v["ticker"].upper(): v["title"] for v in js.values()}
        except Exception:
            _SEC_TITLES = {}
    return _SEC_TITLES.get(ticker.upper().replace("-", "."))


def match(ticker: str, names: list[str], table: dict) -> dict | None:
    """يطابق أسماء الشركة (SEC/ياهو) مع ملكيات HRT. يرجع الملكية أو None."""
    H = table.get("holdings") or {}
    if not H:
        return None
    for nm in [n for n in names if n]:
        toks = norm(nm)
        if not toks:
            continue
        k = " ".join(toks)
        if k in H:
            return H[k]
        # أسماء 13F أحياناً مقطوعة: أول كلمتين متطابقة (أو كلمة وحدة مميزة 6 أحرف+)
        for hk, hv in H.items():
            ht = hk.split()
            if len(toks) >= 2 and len(ht) >= 2 and toks[:2] == ht[:2]:
                return hv
            if len(toks) == 1 and len(ht) >= 1 and ht[0] == toks[0] and len(toks[0]) >= 6:
                return hv
    return None


def annotate(rows: list[dict], log=print) -> dict:
    """يضيف r["hrt"] للأسهم اللي HRT من ملاكها. يرجع معلومات التقرير."""
    t = hrt_table()
    if not t.get("holdings"):
        log("HRT: ما قدرت أقرا تقرير 13F —", t.get("error"))
        return t
    n = 0
    for r in rows:
        m = match(r["symbol"], [sec_title(r["symbol"]), r.get("name")], t)
        if m:
            r["hrt"] = {"shares": int(m["shares"]), "name13f": m["name"], "period": t.get("period"), "filed": t.get("filed")}
            n += 1
    log(f"HRT ({t.get('name')}): تقرير {t.get('period')} ({t.get('n')} سهم) · من أسهمنا: {n} من {len(rows)}")
    return t


def exclude(rows: list[dict], log=print) -> tuple[list[dict], list[dict]]:
    """قرار المستخدم: الأسهم اللي HRT من ملاكها ما تترشح — تنشال وتنعرض في صندوق المستبعد."""
    kept, out = [], []
    for r in rows:
        if r.get("hrt"):
            out.append({"symbol": r["symbol"], "name": r.get("name"), "price": r.get("price"),
                        "shares": r["hrt"]["shares"], "period": r["hrt"]["period"], "name13f": r["hrt"]["name13f"]})
        else:
            kept.append(r)
    if out:
        log(f"HRT: استبعاد {len(out)} سهم")
    return kept, out
