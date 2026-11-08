"""
علم الدولة لكل سهم (طلب المستخدم). الدولة من ياهو (مقر الشركة الرئيسي: info["country"]).
"""
from __future__ import annotations

# الاسم في ياهو ← (رمز الدولة ISO، الاسم بالعربي)
COUNTRIES = {
    "United States": ("US", "أمريكا"), "China": ("CN", "الصين"), "Hong Kong": ("HK", "هونغ كونغ"),
    "Taiwan": ("TW", "تايوان"), "Macau": ("MO", "ماكاو"), "Singapore": ("SG", "سنغافورة"), "Malaysia": ("MY", "ماليزيا"),
    "Japan": ("JP", "اليابان"), "South Korea": ("KR", "كوريا الجنوبية"), "Korea, Republic of": ("KR", "كوريا الجنوبية"),
    "India": ("IN", "الهند"), "Thailand": ("TH", "تايلاند"), "Indonesia": ("ID", "إندونيسيا"), "Philippines": ("PH", "الفلبين"),
    "Vietnam": ("VN", "فيتنام"), "Cambodia": ("KH", "كمبوديا"), "Mongolia": ("MN", "منغوليا"), "Kazakhstan": ("KZ", "كازاخستان"),
    "Israel": ("IL", "إسرائيل"), "United Arab Emirates": ("AE", "الإمارات"), "Saudi Arabia": ("SA", "السعودية"),
    "Qatar": ("QA", "قطر"), "Kuwait": ("KW", "الكويت"), "Bahrain": ("BH", "البحرين"), "Oman": ("OM", "عُمان"),
    "Jordan": ("JO", "الأردن"), "Egypt": ("EG", "مصر"), "Lebanon": ("LB", "لبنان"), "Turkey": ("TR", "تركيا"),
    "Türkiye": ("TR", "تركيا"), "Morocco": ("MA", "المغرب"), "South Africa": ("ZA", "جنوب أفريقيا"), "Nigeria": ("NG", "نيجيريا"),
    "Kenya": ("KE", "كينيا"), "Ghana": ("GH", "غانا"), "Mauritius": ("MU", "موريشيوس"),
    "Canada": ("CA", "كندا"), "Mexico": ("MX", "المكسيك"), "Brazil": ("BR", "البرازيل"), "Argentina": ("AR", "الأرجنتين"),
    "Chile": ("CL", "تشيلي"), "Colombia": ("CO", "كولومبيا"), "Peru": ("PE", "بيرو"), "Uruguay": ("UY", "الأوروغواي"),
    "Panama": ("PA", "بنما"), "Costa Rica": ("CR", "كوستاريكا"), "Puerto Rico": ("PR", "بورتوريكو"), "Bahamas": ("BS", "البهاما"),
    "Bermuda": ("BM", "برمودا"), "Cayman Islands": ("KY", "جزر كايمان"), "British Virgin Islands": ("VG", "جزر العذراء البريطانية"),
    "Virgin Islands, British": ("VG", "جزر العذراء البريطانية"), "Marshall Islands": ("MH", "جزر مارشال"),
    "Curacao": ("CW", "كوراساو"), "Jamaica": ("JM", "جامايكا"), "Barbados": ("BB", "باربادوس"), "Belize": ("BZ", "بليز"),
    "United Kingdom": ("GB", "بريطانيا"), "Ireland": ("IE", "أيرلندا"), "Jersey": ("JE", "جيرسي"), "Guernsey": ("GG", "غيرنزي"),
    "Isle of Man": ("IM", "جزيرة مان"), "Gibraltar": ("GI", "جبل طارق"), "Germany": ("DE", "ألمانيا"), "France": ("FR", "فرنسا"),
    "Netherlands": ("NL", "هولندا"), "Belgium": ("BE", "بلجيكا"), "Luxembourg": ("LU", "لوكسمبورغ"), "Switzerland": ("CH", "سويسرا"),
    "Liechtenstein": ("LI", "ليختنشتاين"), "Austria": ("AT", "النمسا"), "Italy": ("IT", "إيطاليا"), "Spain": ("ES", "إسبانيا"),
    "Portugal": ("PT", "البرتغال"), "Greece": ("GR", "اليونان"), "Cyprus": ("CY", "قبرص"), "Malta": ("MT", "مالطا"),
    "Monaco": ("MC", "موناكو"), "Sweden": ("SE", "السويد"), "Norway": ("NO", "النرويج"), "Denmark": ("DK", "الدنمارك"),
    "Finland": ("FI", "فنلندا"), "Iceland": ("IS", "آيسلندا"), "Estonia": ("EE", "إستونيا"), "Latvia": ("LV", "لاتفيا"),
    "Lithuania": ("LT", "ليتوانيا"), "Poland": ("PL", "بولندا"), "Czech Republic": ("CZ", "التشيك"), "Hungary": ("HU", "المجر"),
    "Romania": ("RO", "رومانيا"), "Bulgaria": ("BG", "بلغاريا"), "Slovenia": ("SI", "سلوفينيا"), "Croatia": ("HR", "كرواتيا"),
    "Serbia": ("RS", "صربيا"), "Ukraine": ("UA", "أوكرانيا"), "Russia": ("RU", "روسيا"), "Georgia": ("GE", "جورجيا"),
    "Armenia": ("AM", "أرمينيا"), "Australia": ("AU", "أستراليا"), "New Zealand": ("NZ", "نيوزيلندا"),
}


def emoji(iso2: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in iso2.upper() if "A" <= c <= "Z")


def flag_for(country: str | None) -> dict | None:
    if not country:
        return None
    iso, ar = COUNTRIES.get(country.strip(), (None, None))
    if not iso:
        return {"flag": "🏳️", "country": country, "country_ar": country}
    return {"flag": emoji(iso), "country": country, "country_ar": ar, "iso": iso}


def annotate(rows: list[dict], infos: dict | None = None) -> None:
    """يضيف r["flag"] و r["country_ar"] (من r["country"] أو من معلومات ياهو)."""
    for r in rows:
        c = r.get("country") or ((infos or {}).get(r.get("symbol")) or {}).get("country")
        f = flag_for(c)
        if f:
            r.update({"country": f["country"], "flag": f["flag"], "country_ar": f["country_ar"]})
