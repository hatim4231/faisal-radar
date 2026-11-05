"""
أدوات الأخبار المشتركة بين رادار فيصل ورادار عبدالله:
  - جلب نص الخبر كامل من صفحته
  - ترجمته للعربي (على أجزاء)
  - تقدير آلي: ✅ إيجابي / ❌ سلبي / ➖ محايد (بالكلمات المفتاحية للأخبار المالية)
  - كاش عشان نفس الخبر ما ينجلب ولا يترجم مرتين
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

CACHE_FILE = Path(__file__).resolve().parent / "cache" / "news_cache.json"
_lock = threading.Lock()
_cache: dict | None = None

MAX_CHARS = 3500  # حد طول الخبر اللي نترجمه


# ----------------------------------------------------------------------------
# التقدير: إيجابي / سلبي
# ----------------------------------------------------------------------------
# (نمط، وزن، سبب بالعربي)
POSITIVE = [
    (r"\bFDA (approv|clear)|\bapproval\b|\bapproved\b|\bclearance\b", 3, "موافقة"),
    (r"\b(contract|agreement|order|purchase order)s? (award|win|won|signed|secur)|\bawarded\b|\bwins?\b .*contract|\bsecures?\b", 3, "عقد/طلبية"),
    (r"\bpartnership\b|\bcollaborat|\bstrategic alliance\b|\bjoint venture\b|\bteams? up\b", 2, "شراكة"),
    (r"\brecord (revenue|sales|quarter)|\brevenue (grew|growth|increase|up)|\bbeats?\b|\bexceed(s|ed)? (expectations|estimates)|\bprofit(able|ability)\b", 3, "نتائج قوية"),
    (r"\bpositive (results|data|topline)|\bmet (its )?primary endpoint|\bstatistically significant\b|\bsuccessful(ly)?\b", 3, "نتائج إيجابية"),
    (r"\bpatent (grant|issued|allow)|\breceives? .*patent\b", 2, "براءة اختراع"),
    (r"\bregain(s|ed)? compliance\b|\bcompliance (regained|restored)\b", 3, "استعاد الامتثال"),
    (r"\b(share )?(buyback|repurchase)\b|\binsider (buy|purchas)|\bdirector .* (buy|purchas)", 3, "إعادة شراء/شراء إداري"),
    (r"\bupgrade[sd]?\b|\braises? (guidance|outlook)|\bprice target (raised|increase)", 2, "رفع تقييم/توقعات"),
    (r"\bacquir(es|ed|ing)\b|\bacquisition of\b|\bmerger agreement\b|\bto be acquired\b|\bbuyout\b", 2, "استحواذ"),
    (r"\blaunch(es|ed)?\b|\bexpan(ds|sion)\b|\bmilestone\b|\bbreakthrough\b|\bfirst (commercial|shipment)", 1, "توسع/إطلاق"),
    (r"\buplist", 2, "انتقال لسوق أكبر"),
    (r"\bdebt (eliminat|reduc|repaid)|\bdebt[- ]free\b", 2, "تخفيض ديون"),
]
NEGATIVE = [
    (r"\bpublic offering\b|\bregistered direct\b|\bprivate placement\b|\bat[- ]the[- ]market\b|\bpriced? .*offering|\bproposed offering\b|\bunderwritten\b", 4, "طرح أسهم (تخفيف)"),
    (r"\bwarrants?\b.*(exercise|issu|inducement)|\bwarrant inducement\b", 3, "Warrants (تخفيف)"),
    (r"\bdilut", 3, "تخفيف"),
    (r"\breverse (stock )?split\b", 2, "تقسيم عكسي"),
    (r"\bdelist|\bdeficiency (notice|letter)\b|\bnon-?compliance\b|\bminimum bid price\b", 3, "خطر الشطب/عدم الامتثال"),
    (r"\bgoing concern\b|\bbankrupt|\bchapter 11\b|\binsolven|\bdefault(s|ed)? on\b", 4, "خطر الإفلاس"),
    (r"\blawsuit\b|\bclass action\b|\bsued\b|\binvestigation\b|\bsubpoena\b|\bSEC charges\b|\bfraud\b", 3, "قضايا/تحقيق"),
    (r"\bmiss(es|ed)? (estimates|expectations)|\bnet loss widen|\brevenue (declin|decreas|fell|drop)|\bloss (widen|increase)", 2, "نتائج ضعيفة"),
    (r"\bcomplete response letter\b|\bCRL\b|\bFDA (reject|declin)|\bfailed to meet\b|\bdid not meet\b|\bclinical hold\b|\bdiscontinu", 4, "فشل/رفض"),
    (r"\bresign(s|ed|ation)\b|\bdeparture of\b|\bterminat(es|ed|ion)\b|\bsuspend(s|ed)?\b|\bhalt(s|ed)?\b", 2, "استقالة/إنهاء/إيقاف"),
    (r"\bdowngrade[sd]?\b|\bcuts? (guidance|outlook)|\blowers? (guidance|outlook)|\bprice target (cut|lower)", 2, "خفض تقييم/توقعات"),
    (r"\brecall(s|ed)?\b|\bdata breach\b|\bcyberattack\b", 2, "سحب منتج/اختراق"),
    (r"\bconvertible (note|debenture)s?\b|\bequity line\b|\bELOC\b|\bS-1\b|\bS-3\b|\bshelf registration\b", 3, "تمويل بتخفيف محتمل"),
]


def sentiment(title: str, text: str) -> dict:
    """تقدير آلي بالكلمات المفتاحية. العنوان وزنه مضاعف."""
    pos, neg, why_p, why_n = 0, 0, [], []
    for blob, mult in ((title or "", 2), (text or "", 1)):
        for pat, w, why in POSITIVE:
            if re.search(pat, blob, re.I):
                pos += w * mult
                if why not in why_p:
                    why_p.append(why)
        for pat, w, why in NEGATIVE:
            if re.search(pat, blob, re.I):
                neg += w * mult
                if why not in why_n:
                    why_n.append(why)
    if neg >= pos + 2:
        label = "negative"
    elif pos >= neg + 2:
        label = "positive"
    else:
        label = "neutral"
    return {"label": label, "pos": pos, "neg": neg, "why_pos": why_p[:3], "why_neg": why_n[:3]}


# ----------------------------------------------------------------------------
# جلب نص الخبر
# ----------------------------------------------------------------------------

def fetch_article(url: str) -> str:
    if not url:
        return ""
    import requests
    from bs4 import BeautifulSoup
    try:
        r = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "figure"]):
            tag.decompose()
        body = soup.find("div", class_=re.compile(r"(caas-body|atoms-wrapper|article-body|body)")) or soup.find("article") or soup
        paras = []
        for p in body.find_all(["p", "li"]):
            t = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
            if len(t) < 40:
                continue
            if re.search(r"cookie|subscribe|sign up|newsletter|all rights reserved|javascript", t, re.I):
                continue
            if t not in paras:
                paras.append(t)
        text = "\n\n".join(paras)
        # قص "حول الشركة" و "البيانات التطلعية" (مو من الخبر)
        cut = re.search(r"\n\n(About [A-Z][^\n]{0,80}\n|Forward[- ]Looking Statements|Safe Harbor)", text)
        if cut and cut.start() > 300:
            text = text[:cut.start()]
        return text[:MAX_CHARS]
    except Exception:
        return ""


def translate_long(text: str, translate) -> str:
    """يترجم نص طويل على أجزاء (كل جزء ~1500 حرف) عشان حدود خدمة الترجمة."""
    if not text:
        return ""
    chunks, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > 1500 and cur:
            chunks.append(cur)
            cur = ""
        cur += (("\n\n" if cur else "") + para)
    if cur:
        chunks.append(cur)
    return "\n\n".join(translate(chunks))


# ----------------------------------------------------------------------------
# الكاش + التجميع
# ----------------------------------------------------------------------------

def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def save_cache(keep: int = 3000):
    with _lock:
        c = _load()
        if len(c) > keep:
            for k in list(c)[:-keep]:
                c.pop(k, None)
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")


def enrich(items: list[dict], translate, full_text_count: int = 3) -> list[dict]:
    """
    لكل خبر: نص كامل مترجم (لأحدث full_text_count أخبار) + تقدير إيجابي/سلبي.
    translate = دالة تترجم قائمة نصوص للعربي.
    """
    for i, x in enumerate(items):
        key = x.get("url") or x["title"]
        with _lock:
            hit = _load().get(key)
        if hit:
            x.update(hit)
            continue
        text = fetch_article(x.get("url")) if i < full_text_count else ""
        if not text:
            text = x.get("summary") or ""
        title_ar, = translate([x["title"]])
        text_ar = translate_long(text, translate)
        s = sentiment(x["title"], text)
        rec = {"title_ar": title_ar, "text": text, "text_ar": text_ar, "full": bool(text and text != x.get("summary")),
               "sentiment": s["label"], "why_pos": s["why_pos"], "why_neg": s["why_neg"],
               "summary_ar": text_ar[:400] if text_ar else ""}
        x.update(rec)
        with _lock:
            _load()[key] = rec
    return items
