"""
NUWE — تحليل معمّق (تشخيص فقط).
1) بيانات XBRL الرسمية لكل ربع: الإيرادات، الربح الإجمالي، المصاريف، الخسارة، النقد، حقوق المساهمين، التدفق التشغيلي، عدد الأسهم.
2) نصوص 10-Q الربع الثاني: السيولة، كفاية النقد، الأحداث اللاحقة، الـ Warrants، Nasdaq، الـ ATM.
3) شروط صفقات يونيو وأغسطس: منع الإصدار بعد الصفقة، حق المشاركة، Series D (بدون دفع)، حدود الملكية.
4) كبار الملاك (13G) + نماذج 3/4 للمطّلعين.
5) نص 8-K التقسيم (أكتوبر) و 8-K أغسطس (7.01) و 8-K يناير (3.03).
6) الأسعار اليومية حول كل إعلان أرباح وكل تمويل: من الإعلان لين تسعير الطرح.
"""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
import alpaca_data as A
import sec_tools

SYM, CIK = "NUWE", 1506492
H = {"User-Agent": sec_tools.USER_AGENT}


def get(url):
    r = requests.get(url, headers=H, timeout=60)
    r.raise_for_status()
    return r


def clean(html):
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in (("&nbsp;", " "), ("&#160;", " "), ("&amp;", "&"), ("&#8217;", "'"), ("&#8220;", '"'), ("&#8221;", '"'),
                 ("&#8212;", "—"), ("&#8211;", "–"), ("&#8226;", "•"), ("&#36;", "$")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t)


def doc(acc, name):
    return clean(get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{acc.replace('-', '')}/{name}").text)


def snip(t, pats, w=450, n=3, title=""):
    if title:
        print(f"\n######## {title} ########")
    seen = set()
    for p in pats:
        for m in list(re.finditer(p, t, re.I))[:n]:
            k = m.start() // 300
            if k in seen:
                continue
            seen.add(k)
            s = max(0, m.start() - w)
            print(f"[{p}] …{t[s:m.end() + w]}…\n")


def xbrl():
    js = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK:010d}.json").json()
    facts = js.get("facts", {})
    want = {
        "us-gaap": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "GrossProfit", "OperatingExpenses",
                    "OperatingIncomeLoss", "NetIncomeLoss", "CashAndCashEquivalentsAtCarryingValue",
                    "StockholdersEquity", "NetCashProvidedByUsedInOperatingActivities",
                    "NetCashProvidedByUsedInFinancingActivities", "ProceedsFromIssuanceOfCommonStock",
                    "FairValueAdjustmentOfWarrants", "DerivativeLiabilitiesCurrent", "Liabilities", "Assets",
                    "WeightedAverageNumberOfSharesOutstandingBasic", "ClassOfWarrantOrRightOutstanding",
                    "AccountsReceivableNetCurrent", "InventoryNet", "SellingGeneralAndAdministrativeExpense",
                    "ResearchAndDevelopmentExpense"],
        "dei": ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"],
    }
    print("\n######## XBRL (القيم الرسمية من التقارير) ########")
    for ns, keys in want.items():
        for k in keys:
            f = facts.get(ns, {}).get(k)
            if not f:
                continue
            for unit, vals in f["units"].items():
                rows = []
                for v in vals:
                    if v.get("end", "") < "2025-01-01":
                        continue
                    dur = ""
                    if v.get("start"):
                        d = (pd.Timestamp(v["end"]) - pd.Timestamp(v["start"])).days
                        dur = "Q" if d < 100 else "6M" if d < 200 else "9M" if d < 290 else "Y"
                    rows.append((v["end"], dur, v["val"], v.get("form"), v.get("filed"), v.get("frame", "")))
                rows = sorted(set(rows))
                if rows:
                    print(f"\n{k} [{unit}]")
                    for r in rows[-14:]:
                        print("  ", r)


def main():
    xbrl()
    subs = get(f"https://data.sec.gov/submissions/CIK{CIK:010d}.json").json()["filings"]["recent"]
    F = pd.DataFrame({k: subs[k] for k in ("form", "filingDate", "accessionNumber", "primaryDocument", "items")})
    F = F[F.filingDate >= "2025-06-01"]

    def one(form, date):
        x = F[(F.form == form) & (F.filingDate == date)]
        return x.iloc[0] if len(x) else None

    # 10-Q الربع الثاني
    r = one("10-Q", "2026-08-13")
    if r is not None:
        t = doc(r.accessionNumber, r.primaryDocument)
        print("\n10-Q chars", len(t))
        snip(t, [r"Liquidity and Capital Resources", r"sufficient to fund", r"runway", r"Subsequent Event",
                 r"warrant liabilit", r"Series D Warrants", r"zero[- ]c(ash|heck)", r"inducement", r"Nasdaq",
                 r"publicly held", r"minimum bid", r"at[- ]the[- ]market", r"consoles?", r"Rendiatech",
                 r"shares of common stock outstanding", r"outstanding warrants", r"weighted[- ]average exercise",
                 r"cost reduction|reduction in force|headcount", r"cash used in operating"], 500, 2, "10-Q Q2 2026")
        # جدول الـ Warrants
        i = t.find("Warrants outstanding")
        if i > 0:
            print("\n-- جدول Warrants --\n", t[i - 300:i + 2500])
    # شروط صفقة يونيو (424B4) وأغسطس (424B5)
    for form, date, title in (("424B4", "2026-06-08", "طرح يونيو 2026 (424B4)"), ("424B5", "2026-08-03", "طرح أغسطس 2026 (424B5)")):
        r = one(form, date)
        if r is None:
            continue
        t = doc(r.accessionNumber, r.primaryDocument)
        snip(t, [r"Series D Warrants?.{0,200}zero", r"zero exercise price|zero[- ]cash exercise|without payment",
                 r"shall not (issue|enter)", r"Subsequent Equity Sales", r"Variable Rate Transaction",
                 r"(right|option) to participate|participation right", r"lock[- ]up", r"Beneficial Ownership Limitation",
                 r"public offering price", r"Dilution", r"net tangible book value", r"offering price per share",
                 r"as adjusted", r"use of proceeds"], 380, 2, title)
    # 8-K يناير (3.03) و أغسطس 7.01 و أكتوبر (تقسيم)
    for date in ("2026-01-30", "2026-08-26", "2026-10-23", "2026-06-10", "2026-06-18", "2026-03-17", "2026-04-24"):
        x = F[(F.form == "8-K") & (F.filingDate == date)]
        for _, r in x.iterrows():
            t = doc(r.accessionNumber, r.primaryDocument)
            print(f"\n######## 8-K {date} items={r['items']} ########\n", t[:3500])
            # الملحقات (EX-99)
            try:
                idx = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{r.accessionNumber.replace('-', '')}/index.json").json()
                for it in idx["directory"]["item"]:
                    nm = it["name"]
                    if re.search(r"ex99|ex-99|ex10|ex4", nm, re.I) and nm.endswith(".htm"):
                        tx = doc(r.accessionNumber, nm)
                        print(f"\n   -- ملحق {nm} ({len(tx)} حرف) --\n", tx[:3000])
            except Exception as e:
                print("ملحقات خطأ", e)
    # 13G
    print("\n######## كبار الملاك (13G) ########")
    for _, r in F[F.form.str.startswith("SCHEDULE 13G")].iterrows():
        try:
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{r.accessionNumber.replace('-', '')}/index.json").json()
            xmlname = next((it["name"] for it in idx["directory"]["item"] if it["name"].endswith(".xml")), None)
            if xmlname:
                x = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{r.accessionNumber.replace('-', '')}/{xmlname}").text
                names = re.findall(r"<(?:reportingPersonName|filerName|nameOfReportingPerson)>([^<]+)<", x)
                pcts = re.findall(r"<(?:classPercent|percentOfClass|aggregatePercent)>([^<]+)<", x)
                amts = re.findall(r"<(?:aggregateAmountOwned|reportingPersonBeneficiallyOwnedAggregateNumberOfShares|aggregateAmountBeneficiallyOwned)>([^<]+)<", x)
                evd = re.findall(r"<(?:eventDateRequiresFilingThisStatement|dateOfEvent)>([^<]+)<", x)
                print(r.filingDate, r.form, "|", names[:3], "| %", pcts[:3], "| أسهم", amts[:3], "| حدث", evd[:1])
                if not names:
                    print("   ", clean(x)[:600])
        except Exception as e:
            print(r.filingDate, "خطأ", e)
    # نماذج 3/4
    print("\n######## نماذج المطّلعين (3/4) ########")
    for _, r in F[F.form.isin(["3", "4"])].iterrows():
        try:
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{r.accessionNumber.replace('-', '')}/index.json").json()
            xmlname = next((it["name"] for it in idx["directory"]["item"] if it["name"].endswith(".xml")), None)
            x = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{r.accessionNumber.replace('-', '')}/{xmlname}").text
            name = re.findall(r"<rptOwnerName>([^<]+)<", x)
            title = re.findall(r"<officerTitle>([^<]+)<", x)
            codes = re.findall(r"<transactionCode>([^<]+)<", x)
            sh = re.findall(r"<transactionShares>\s*<value>([^<]+)<", x)
            px = re.findall(r"<transactionPricePerShare>\s*<value>([^<]*)<", x)
            print(r.filingDate, "Form", r.form, name[:1], title[:1], "codes", codes, "shares", sh[:4], "px", px[:4])
        except Exception as e:
            print(r.filingDate, "خطأ", e)
    # الأسعار حول الأحداث
    now = datetime.now(timezone.utc)
    d = A.bars([SYM], "1Day", now - timedelta(days=520)).get(SYM)
    d.index = d.index.tz_localize(None).normalize()
    print("\n######## الأسعار الأسبوعية (معدّلة للتقسيمات) ########")
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    print(w.tail(70).round(3).to_string())
    print("\n######## من الإعلان/الملف لين التمويل ########")
    ev = [("أرباح Q2-25", "2025-08-14"), ("424B5 ATM", "2025-09-03"), ("أرباح Q3-25", "2025-11-12"), ("DRS", "2025-11-25"),
          ("8-K تمويل يناير", "2026-01-30"), ("أرباح Q4-25", "2026-03-10"), ("أرباح Q1-26", "2026-05-12"), ("S-1", "2026-05-22"),
          ("تسعير يونيو", "2026-06-04"), ("424B4", "2026-06-08"), ("تقسيم 1:35", "2026-06-26"), ("RDO أغسطس", "2026-07-31"),
          ("أرباح Q2-26", "2026-08-13"), ("S-3 إعادة بيع", "2026-08-18"), ("قرار تقسيم", "2026-10-13"), ("8-K تقسيم", "2026-10-23"),
          ("تقسيم 1:9", "2026-10-27"), ("طلب FDA", "2026-11-05")]
    for nm, dt in ev:
        p = d.index.searchsorted(pd.Timestamp(dt))
        if p >= len(d):
            continue
        c0 = float(d["Close"].iloc[max(0, p - 1)])
        def ch(k):
            j = min(p + k, len(d) - 1)
            return f"{float(d['Close'].iloc[j]) / c0 - 1:+.0%}"
        print(f"{dt} {nm:<16} إغلاق قبل {c0:10.3f} · يوم {ch(0)} · +5 {ch(5)} · +10 {ch(10)} · +20 {ch(20)} · +40 {ch(40)}")


if __name__ == "__main__":
    main()
