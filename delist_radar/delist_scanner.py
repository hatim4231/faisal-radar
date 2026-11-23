#!/usr/bin/env python3
"""
⛔ رادار الشطب — الأسهم اللي استلمت إنذار أو قرار شطب من البورصة (طلب أبوفرح 2026-11-23:
«رادار بخصوص الأسهم اللي راح تنشطب بسبب مخالفاتها وترسلها لي على التليجرام أول بأول»).

المصدر الرسمي (مو تخمين من السعر): ملفات SEC اللي الشركة مُلزمة تقدمها خلال 4 أيام عمل من استلام الإنذار:
  · 8-K بند Item 3.01 (الشركات الأمريكية: إشعار شطب أو عدم استيفاء شرط إدراج)
  · 6-K (الأجنبية) فيه كلمات إنذار ناسداك/NYSE (الحد الأدنى للسعر، حقوق المساهمين، تأخر التقارير، قرار الشطب، لجنة الاستماع)
  · Form 25-NSE (البورصة نفسها تقدم طلب الشطب = نهائي)
نقرا نص كل ملف ونصنّف السبب والخطورة والمهلة، ونجيب السعر من Alpaca. التنبيه بتيليجرام لكل ملف جديد مرة وحدة
(الحالة في trackdata/delist_sent.json؛ أول تشغيل ما يرسل القديم — يرسل ملخص بس).
⚠ معلومة بس — ليس توصية شراء أو بيع. الإنذار مو شطب: كثير شركات ترجع ممتثلة (تقسيم عكسي، رفع حقوق المساهمين، تقديم التقرير).

    python delist_radar/delist_scanner.py --out site/delisting/index.html [--no-telegram] [--days 45]
"""
from __future__ import annotations

import argparse
import html as H
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import sec_tools  # noqa: E402  (طلبات SEC بنفس User-Agent وحد السرعة)

OUT = ROOT / "output"
STATE = Path(os.environ.get("TRACK_DIR", ROOT / "trackdata")) / "delist_sent.json"
SITE = "https://hatim4231.github.io/faisal-radar/delisting/"

# بحث نص SEC ما يدعم OR بين العبارات (جرّبناها: 3 نتايج بس) ← كل عبارة استعلام لحاله ونجمعها
PHRASES = ['"minimum bid price"', '"Listing Rule 5550"', '"Listing Rule 5450"', '"Listing Rule 5250"', '"Listing Rule 5810"',
           '"delisting determination"', '"Hearings Panel"', '"deficiency letter"', '"Notice of Delisting"', '"regained compliance"',
           '"continued listing standard"', '"Section 802.01"', '"Section 1003"']
QUERIES = [(f"{f}:{q}", q, f) for q in PHRASES for f in ("8-K", "6-K")]
# لازم النص يذكر مخالفة فعلية (Item 3.01 يشمل نقل الإدراج والاندماج — مو مخالفة)
VIOL = (r"deficiency|not in compliance|no longer (?:in )?compl|fail(?:ed|ure|s)? to (?:satisfy|comply|meet)|did not (?:comply|meet)|"
        r"below the minimum|delisting determination|determined to delist|hearings? panel|regain(?:ed)? compliance|"
        r"5550\(|5450\(|5250\(|5810\(|802\.01|1003\(")
# السبب: (مفتاح، عربي، أنماط)
REASONS = [
    ("bid", "السعر تحت 1$ لـ 30 جلسة (الحد الأدنى للسعر)", r"minimum bid price|bid price requirement|5550\(a\)\(2\)|5450\(a\)\(1\)|802\.01C|\$1\.00 per share"),
    ("equity", "حقوق المساهمين أقل من المطلوب", r"stockholders.? equity|shareholders.? equity|5550\(b\)\(1\)|5450\(b\)\(1\)|1003\(a\)"),
    ("mvls", "القيمة السوقية للأسهم المدرجة أقل من المطلوب", r"market value of listed securities|MVLS|5550\(b\)\(2\)|5450\(b\)\(2\)"),
    ("mvphs", "قيمة الأسهم بيد الجمهور أقل من المطلوب", r"market value of publicly held shares|MVPHS|5550\(a\)\(5\)|5450\(a\)\(2\)"),
    ("late", "تأخر تقديم التقارير المالية (10-Q/10-K/20-F)", r"5250\(c\)\(1\)|timely file|periodic report|delinquen"),
    ("holders", "عدد المساهمين/الأسهم المتاحة أقل من المطلوب", r"round lot holders|publicly held shares|5550\(a\)\(3\)|5550\(a\)\(4\)"),
    ("gov", "مخالفة حوكمة (لجنة تدقيق/أعضاء مستقلين)", r"audit committee|independent director|5605"),
    ("fee", "رسوم الإدراج ما انسدّدت", r"listing fee|5250\(f\)"),
    ("rs_noperiod", "سوّى تقسيم عكسي خلال سنة/سنتين — بدون مهلة 180 يوم", r"5810\(c\)\(3\)\(A\)\(iv\)|reverse stock split.{0,120}(prior|within).{0,30}(one|1|two|2)[- ]year"),
    ("public_interest", "قرار البورصة لمصلحة عامة (أخطر الأسباب)", r"public interest|5101"),
]
# الخطورة: 3 = قرار شطب/تعليق/طلب Form 25 · 2 = جلسة استماع أو بدون مهلة · 1 = إنذار مع مهلة · 0 = رجع ممتثل
SEV3 = (r"(?:received|receipt of|issued) (?:a |the )?(?:written )?(?:staff )?(?:delisting determination|determination letter)|determined to delist|"
        r"(?:trading|securities|shares)[^.]{0,80}will be suspended|suspended[^.]{0,40}(?:effective|at the opening)|panel[^.]{0,40}(?:denied|determined to delist)|"
        r"has been delisted|will be delisted (?:from|effective)")
SEV2 = (r"(?:requested|has requested|intends to request|will request|timely requested) a hearing|hearing (?:is |was |has been )?scheduled|"
        r"not eligible for (?:an? )?(?:additional|second|any) [^.]{0,30}(?:compliance )?period|ineligible for [^.]{0,30}compliance period|"
        r"reverse stock split[^.]{0,120}(?:prior|within)[^.]{0,30}(?:one|1|two|2)[- ]year|5810\(c\)\(3\)\(A\)\(iv\)")
REGAIN = r"regained compliance|has regained|evidenced compliance|now in compliance"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DEADLINE = re.compile(rf"(?:until|through|by|on or before)\s+((?:{MONTHS})\s+\d{{1,2}},\s+20\d\d)", re.I)
HEADING = re.compile(r"notice of delisting or failure to satisfy a continued listing rule or standard;?\s*(?:transfer of listing)?", re.I)
SEV_AR = {3: "🔴 قرار شطب / تعليق", 2: "🟠 جلسة استماع / بدون مهلة", 1: "🟡 إنذار مع مهلة", 0: "✅ رجع ممتثل"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def search(q: str, forms: str, start: str, end: str, max_pages: int = 8) -> list[dict]:
    out = []
    for page in range(max_pages):
        url = (f"https://efts.sec.gov/LATEST/search-index?q={quote(q)}&forms={forms}&dateRange=custom"
               f"&startdt={start}&enddt={end}&from={page * 100}")
        try:
            js = sec_tools._get(url)
        except Exception as e:
            log(f"  بحث SEC فشل ({forms}):", repr(e)[:160])
            break
        hits = js.get("hits", {}).get("hits", [])
        out += hits
        total = (js.get("hits", {}).get("total") or {}).get("value", 0)
        if len(hits) < 100 or len(out) >= total:
            break
    return out


def _ticker(src: dict) -> str | None:
    for n in src.get("display_names") or []:
        m = re.search(r"\(([A-Z][A-Z0-9.\-]{0,6}(?:,\s*[A-Z][A-Z0-9.\-]{0,6})*)\)\s*\(CIK", n)
        if m:
            return m.group(1).split(",")[0].strip()
    return None


def _name(src: dict) -> str:
    n = (src.get("display_names") or [""])[0]
    return re.sub(r"\s*\(.*$", "", n).strip()


def fetch_text(url: str) -> str:
    import requests
    try:
        with sec_tools._lock:
            wait = 0.12 - (time.time() - sec_tools._last[0])
            if wait > 0:
                time.sleep(wait)
            sec_tools._last[0] = time.time()
        r = requests.get(url, headers={"User-Agent": sec_tools.USER_AGENT}, timeout=25)
        if not r.ok:
            return ""
        t = re.sub(r"(?is)<(script|style).*?</\1>", " ", r.text)
        t = H.unescape(re.sub(r"<[^>]+>", " ", t))
        return re.sub(r"\s+", " ", t)[:60000]
    except Exception:
        return ""


def classify(text: str, form: str) -> dict:
    tl = text.lower()
    reasons = [(k, ar) for k, ar, pat in REASONS if re.search(pat, text, re.I)]
    if form.startswith("25"):
        sev = 3
    elif re.search(REGAIN, tl) and not re.search(r"(?<!not )(?<!has not )notif(ied|ication).{0,80}(not|no longer) (in )?compl", tl):
        sev = 0
    elif re.search(SEV3, tl):
        sev = 3
    elif re.search(SEV2, tl):
        sev = 2
    else:
        sev = 1
    dl = DEADLINE.search(text)
    deadline = None
    if dl:
        try:
            deadline = datetime.strptime(re.sub(r"\s+", " ", dl.group(1)), "%B %d, %Y").date().isoformat()
        except ValueError:
            deadline = None
    m = re.search(r"(?:Nasdaq|NYSE)[^.]{0,250}(?:letter|notice|notif)[^.]{0,300}\.", text, re.I)
    snippet = (m.group(0) if m else "")[:420]
    exch = "NYSE" if re.search(r"\bNYSE\b|New York Stock Exchange", text) and not re.search(r"\bNasdaq\b", text, re.I) else (
        "Nasdaq" if re.search(r"\bNasdaq\b", text, re.I) else "؟")
    return {"sev": sev, "reasons": reasons, "deadline": deadline, "snippet": snippet, "exchange": exch}


def collect(days: int) -> list[dict]:
    end, start = date.today(), date.today() - timedelta(days=days)
    seen, filings = set(), []
    for kind, q, forms in QUERIES:
        hits = search(q, forms, start.isoformat(), end.isoformat(), max_pages=5)
        log(f"  SEC {kind}: {len(hits)} ملف")
        for h in hits:
            s = h.get("_source") or {}
            adsh = s.get("adsh") or (h.get("_id") or "").split(":")[0]
            if not adsh or adsh in seen:
                continue
            seen.add(adsh)
            cik = int((s.get("ciks") or ["0"])[0])
            fn = (h.get("_id") or ":").split(":", 1)[1]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fn}"
            filings.append({"adsh": adsh, "cik": cik, "form": s.get("form") or s.get("file_type") or kind, "date": s.get("file_date"),
                            "ticker": _ticker(s), "name": _name(s), "url": url})
    # Form 25-NSE (البورصة قدمت الشطب) — يشمل سندات واندماجات، فناخذه بس تأكيد لشركة عندها مخالفة بالقائمة
    f25 = []
    for h in search('"25-NSE" OR delist', "25-NSE", start.isoformat(), end.isoformat(), max_pages=5) + \
            search("Nasdaq", "25-NSE", start.isoformat(), end.isoformat(), max_pages=5):
        s = h.get("_source") or {}
        adsh = s.get("adsh") or (h.get("_id") or "").split(":")[0]
        if adsh and adsh not in seen:
            seen.add(adsh)
            cik = int((s.get("ciks") or ["0"])[0])
            fn = (h.get("_id") or ":").split(":", 1)[1]
            f25.append({"adsh": adsh, "cik": cik, "form": "25-NSE", "date": s.get("file_date"), "ticker": _ticker(s), "name": _name(s),
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fn}",
                        "sev": 3, "reasons": [], "deadline": None, "snippet": "البورصة قدمت طلب الشطب (Form 25-NSE)", "exchange": "؟"})
    log(f"  Form 25-NSE: {len(f25)} (ينحسب بس للشركات اللي عندها مخالفة)")
    log(f"  ملفات فريدة: {len(filings)} — نقرا نصوصها")
    with ThreadPoolExecutor(4) as ex:
        texts = list(ex.map(lambda f: fetch_text(f["url"]), filings))
    out = []
    for f, t in zip(filings, texts):
        if not t:
            f.update({"sev": 1, "reasons": [], "deadline": None, "snippet": "", "exchange": "؟", "unread": True})
        else:
            t = re.sub(HEADING, " ", t)                     # عنوان بند 3.01 فيه «Failure to Satisfy» بكل ملف — نشيله
            if not re.search(r"Nasdaq|NYSE|New York Stock Exchange", t, re.I) or not re.search(VIOL, t, re.I):
                continue                                   # ما له علاقة بمخالفة إدراج (اندماج/نقل إدراج/خبر عام)
            c = classify(t, str(f["form"]))
            if not c["reasons"] and c["sev"] != 0:
                continue                                   # شطب بسبب اندماج/استحواذ أو نقل إدراج = مو مخالفة
            f.update(c)
        out.append(f)
    ciks = {f["cik"] for f in out}
    out += [f for f in f25 if f["cik"] in ciks]
    return out


def by_company(filings: list[dict]) -> list[dict]:
    """سهم واحد لكل شركة: آخر حالة (لو آخر ملف «رجع ممتثل» = ممتثل) + كل الأحداث."""
    g = {}
    for f in sorted(filings, key=lambda x: (x["date"] or "", x["sev"])):
        k = f["ticker"] or f"CIK{f['cik']}"
        g.setdefault(k, []).append(f)
    rows = []
    for k, ev in g.items():
        last = ev[-1]
        worst = max(e["sev"] for e in ev)
        sev = 0 if last["sev"] == 0 else worst
        reasons = {}
        for e in ev:
            for rk, ar in e.get("reasons") or []:
                reasons[rk] = ar
        dls = sorted(e["deadline"] for e in ev if e.get("deadline"))
        rows.append({"symbol": last["ticker"], "cik": last["cik"], "name": last["name"], "sev": sev, "sev_ar": SEV_AR[sev],
                     "reasons": list(reasons.values()), "deadline": dls[-1] if dls else None, "exchange": last["exchange"],
                     "last_date": last["date"], "events": ev[::-1]})
    rows.sort(key=lambda r: (-r["sev"] if r["sev"] else 9, r["last_date"] or ""), reverse=False)
    rows.sort(key=lambda r: (0 if r["sev"] == 3 else 1 if r["sev"] == 2 else 2 if r["sev"] == 1 else 3,
                             -(datetime.strptime(r["last_date"], "%Y-%m-%d").toordinal() if r["last_date"] else 0)))
    return rows


def prices(rows: list[dict]) -> None:
    syms = sorted({r["symbol"] for r in rows if r["symbol"] and re.fullmatch(r"[A-Z]{1,5}", r["symbol"])})
    if not syms:
        return
    import alpaca_data as A
    if not A.available():
        return
    raw, start, bad = {}, datetime.now() - timedelta(days=60), []

    def get(batch):                       # رمز مو موجود بـ Alpaca (OTC) يخرب الدفعة كلها (400) ← نقسمها
        try:
            raw.update(A.bars(batch, "1Day", start))
        except Exception:
            if len(batch) == 1:
                bad.append(batch[0])
            else:
                get(batch[:len(batch) // 2]); get(batch[len(batch) // 2:])
    for i in range(0, len(syms), 50):
        get(syms[i:i + 50])
    if bad:
        log(f"  ما لها سعر بـ Alpaca ({len(bad)}): {', '.join(bad[:15])}")
    for r in rows:
        d = raw.get(r["symbol"])
        if d is None or not len(d):
            continue
        c = d["Close"]
        r["price"] = round(float(c.iloc[-1]), 4)
        r["chg30"] = round((float(c.iloc[-1]) / float(c.iloc[-min(len(c), 22)]) - 1) * 100, 1)
        r["under1"] = int((c < 1).iloc[::-1].cumprod().sum())          # كم جلسة متتالية تحت 1$
        r["avg_dv"] = int((c * d["Volume"]).tail(20).mean())


def alerts(rows: list[dict], send: bool) -> None:
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        st = None
    first = st is None
    st = st or {"sent": []}
    sent = set(st["sent"])
    new = [(r, e) for r in rows for e in r["events"] if e["adsh"] not in sent]
    msgs = []
    if first:
        n3, n2, n1 = (sum(1 for r in rows if r["sev"] == k) for k in (3, 2, 1))
        msgs.append(f"⛔ <b>رادار الشطب اشتغل</b>\nحالياً من ملفات SEC آخر 45 يوم: 🔴 قرار شطب/تعليق {n3} · 🟠 جلسة استماع/بدون مهلة {n2} · 🟡 إنذار مع مهلة {n1}"
                    f"\nمن الحين بيوصلك كل إنذار جديد أول بأول.\n{SITE}")
    else:
        minor = [(r, e) for r, e in new if e["sev"] <= 1]
        if minor:
            ln = [f"{'✅' if e['sev'] == 0 else '🟡'} <b>{H.escape(r['symbol'] or r['name'][:20])}</b>"
                  + (f" ${r['price']}" if r.get("price") is not None else "")
                  + f" — {'رجع ممتثل' if e['sev'] == 0 else ('، '.join(a for _, a in (e.get('reasons') or [])[:1]) or 'إنذار')}"
                  + (f" · مهلة {e['deadline']}" if e.get("deadline") and e["sev"] else "") for r, e in minor[:30]]
            msgs.append("📋 <b>إنذارات شطب جديدة (مع مهلة) / رجع ممتثل</b>\n" + "\n".join(ln)
                        + (f"\n… و {len(minor) - 30} ثاني" if len(minor) > 30 else "") + f"\n{SITE}")
        for r, e in sorted([x for x in new if x[1]["sev"] >= 2], key=lambda x: -x[1]["sev"]):
            sym = H.escape(r["symbol"] or r["name"])
            px = f" · السعر ${r['price']}" if r.get("price") is not None else ""
            why = "، ".join(a for _, a in (e.get("reasons") or [])[:2]) or "السبب ما انقرا آلياً — افتح الملف"
            dl = f"\n⏳ المهلة: {e['deadline']}" if e.get("deadline") else ""
            msgs.append(f"{SEV_AR[e['sev']]} — <b>{sym}</b> ({H.escape(r['name'])[:40]}){px}\n📄 {e['form']} بتاريخ {e['date']} · {e.get('exchange','')}"
                        f"\n❓ {why}{dl}\n<a href=\"{e['url']}\">الملف الرسمي</a> · {SITE}\n⚠ معلومة بس — الإنذار مو شطب أكيد")
    sent |= {e["adsh"] for r in rows for e in r["events"]}
    log(f"  تيليجرام الشطب: {len(msgs)} رسالة ({'أول تشغيل — ملخص بس' if first else f'{len(new)} ملف جديد'})")
    if send and msgs:
        import tracker
        if len(msgs) > 12:
            head = msgs[:10]
            head.append(f"… و {len(msgs) - 10} إنذار ثاني — شوف الصفحة\n{SITE}")
            msgs = head
        tracker.telegram(msgs)
    if send:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        st["sent"] = sorted(sent)[-5000:]
        STATE.write_text(json.dumps(st), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="رادار الشطب")
    ap.add_argument("--out", default=str(OUT / "delisting.html"))
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    filings = collect(a.days)
    rows = by_company(filings)
    prices(rows)
    cnt = {k: sum(1 for r in rows if r["sev"] == k) for k in (3, 2, 1, 0)}
    log(f"## رادار الشطب: 🔴 {cnt[3]} · 🟠 {cnt[2]} · 🟡 {cnt[1]} · ✅ {cnt[0]} (من {len(filings)} ملف) · {round(time.time() - t0)}ث")
    for r in rows[:25]:
        log(f"  {r['symbol']} {r['sev_ar']} · {r['last_date']} · {', '.join(r['reasons'][:2])} · مهلة {r['deadline']} · سعر {r.get('price')}")
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "days": a.days, "counts": cnt, "stocks": rows,
               "seconds": round(time.time() - t0)}
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    OUT.mkdir(exist_ok=True)
    (OUT / "sig_delist.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")), encoding="utf-8")
    log(f"[الشطب] التقرير: {out}")
    alerts(rows, send=not a.no_telegram)


if __name__ == "__main__":
    main()
