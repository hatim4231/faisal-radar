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
    (r"\b(distribution|supply|licens\w*|exclusive) (agreement|deal)\b|\bdeal with\b|\bsigns?\b .*(agreement|deal|MOU)", 3, "اتفاقية/صفقة"),
    (r"\b(surpass|top)s? .*estimates|\bbeat(s)? .*estimates|\btops? (Q\d|earnings|revenue)", 3, "فاق التوقعات"),
    (r"\bshares? (double|triple|soar|surge|jump|rall)|\bstock (double|triple|soar|surge|jump|rall)|\b(soars?|surges?|jumps?|skyrockets?)\b", 2, "صعود قوي للسهم"),
    (r"\bup \d+%|\bnarrow(s|ed|ing)? (net )?loss", 1, "تحسن"),
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
    (r"\b(falls?|plunges?|tumbles?|sinks?|slumps?|drops?|crash(es)?)\b \d|\bshares? (fall|plunge|tumble|sink|drop)|\bstock (falls|plunges|tumbles|sinks|drops)", 2, "هبوط قوي للسهم"),
    (r"\bmiss(es|ed)? (estimates|expectations)|\bsales decline|\bnet loss widen|\brevenue (declin|decreas|fell|drop)|\bloss (widen|increase)", 2, "نتائج ضعيفة"),
    (r"\bcomplete response letter\b|\bCRL\b|\bFDA (reject|declin)|\bfailed to meet\b|\bdid not meet\b|\bclinical hold\b|\bdiscontinu", 4, "فشل/رفض"),
    (r"\bresign(s|ed|ation)\b|\bdeparture of\b|\bterminat(es|ed|ion)\b|\bsuspend(s|ed)?\b|\bhalt(s|ed)?\b", 2, "استقالة/إنهاء/إيقاف"),
    (r"\bdowngrade[sd]?\b|\bcuts? (guidance|outlook)|\blowers? (guidance|outlook)|\bprice target (cut|lower)", 2, "خفض تقييم/توقعات"),
    (r"\brecall(s|ed)?\b|\bdata breach\b|\bcyberattack\b", 2, "سحب منتج/اختراق"),
    (r"\bconvertible (note|debenture)s?\b|\bequity line\b|\bELOC\b|\bS-1\b|\bS-3\b|\bshelf registration\b", 3, "تمويل بتخفيف محتمل"),
]


def _score(blob: str):
    pos, neg, wp, wn = 0, 0, [], []
    for pat, w, why in POSITIVE:
        if re.search(pat, blob, re.I):
            pos += w
            wp.append(why)
    for pat, w, why in NEGATIVE:
        if re.search(pat, blob, re.I):
            neg += w
            wn.append(why)
    return pos, neg, wp, wn


def sentiment(title: str, text: str) -> dict:
    """
    تقدير آلي بالكلمات المفتاحية.
    العنوان يحكم أولاً؛ النص يُستخدم فقط إذا العنوان ما فيه إشارة واضحة (وبشرط أقوى)،
    لأن بيانات الشركات الصغيرة مليانة كلمات ترويجية.
    """
    tp, tn, twp, twn = _score(title or "")
    bp, bn, bwp, bwn = _score(text or "")
    if tp or tn:
        if tn > tp:
            label = "negative"
        elif tp > tn:
            label = "positive"
        else:
            label = "neutral"
        why_p, why_n = twp, twn
        # خبر تخفيف/طرح في النص يقلب الخبر لسلبي حتى لو العنوان إيجابي
        if label == "positive" and any(w in bwn for w in ("طرح أسهم (تخفيف)", "خطر الإفلاس")):
            label, why_n = "negative", twn + [w for w in bwn if w in ("طرح أسهم (تخفيف)", "خطر الإفلاس")]
    else:
        if bn >= 4 and bn >= bp + 3:
            label = "negative"
        elif bp >= 7 and bp >= bn + 3:
            label = "positive"
        else:
            label = "neutral"
        why_p, why_n = bwp, bwn
    return {"label": label, "pos": tp * 2 + bp, "neg": tn * 2 + bn,
            "why_pos": list(dict.fromkeys(why_p))[:3], "why_neg": list(dict.fromkeys(why_n))[:3]}


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
            s = sentiment(x["title"], hit.get("text", ""))  # التقييم ينحسب من جديد دائماً
            x.update({"sentiment": s["label"], "why_pos": s["why_pos"], "why_neg": s["why_neg"]})
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
