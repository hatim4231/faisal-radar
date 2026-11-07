"""
اختبار تاريخي لرادار المطّلعين (تشخيص فقط).
المصدر: ملفات SEC الربعية «Insider Transactions Data Sets» (Form 3/4/5 كاملة) + أسعار Alpaca اليومية.
الإشارة: يوم التداول اللي بعد تاريخ تقديم Form 4 فيه شراء من السوق المفتوح (P/A).
نجمع شراء آخر 30 يوم لين تاريخ التقديم، ونحسب النقاط بنفس قواعد insiders_scanner.score.
الدخول = إغلاق يوم الإشارة (بعد ما الملف صار علني) · السعر 1-5$ · مجموع ≥ 25 ألف$.
النتيجة: +20% قبل −10% خلال 10 جلسات · طار 30% خلال 20 · العائد بعد 10 و 20. المرجع: عشوائي من نفس الأسهم.
"""
import io
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import requests
import scanner
import sec_tools
import alpaca_data as A

TOP = re.compile(r"\b(CEO|Chief Executive|President|CFO|Chief Financial|Chairman|COO|Chief Operating)\b", re.I)
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"


def quarters():
    now = datetime.now()
    out = []
    y, q = now.year - 2, 1
    while (y, q) <= (now.year, (now.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def load_q(q):
    r = requests.get(URL.format(q=q), headers={"User-Agent": sec_tools.USER_AGENT}, timeout=120)
    if r.status_code != 200:
        print(q, "غير متاح", r.status_code, flush=True)
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = {n.split("/")[-1].upper(): n for n in z.namelist()}

    def rd(name, cols):
        with z.open(names[name]) as f:
            df = pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in cols, on_bad_lines="skip")
        return df
    sub = rd("SUBMISSION.TSV", {"ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"})
    own = rd("REPORTINGOWNER.TSV", {"ACCESSION_NUMBER", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"})
    tr = rd("NONDERIV_TRANS.TSV", {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                   "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"})
    tr = tr[(tr.TRANS_CODE == "P") & (tr.TRANS_ACQUIRED_DISP_CD == "A")].copy()
    tr["sh"] = pd.to_numeric(tr.TRANS_SHARES, errors="coerce")
    tr["px"] = pd.to_numeric(tr.TRANS_PRICEPERSHARE, errors="coerce")
    tr = tr[(tr.sh > 0) & (tr.px > 0)]
    own = own.drop_duplicates("ACCESSION_NUMBER")
    m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    m = m[m.DOCUMENT_TYPE.fillna("4").str.startswith("4")]
    m["filed"] = pd.to_datetime(m.FILING_DATE, format="mixed", dayfirst=True, errors="coerce")
    m["sym"] = m.ISSUERTRADINGSYMBOL.str.upper().str.strip()
    print(q, "عمليات شراء P:", len(m), flush=True)
    return m[["sym", "filed", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "sh", "px"]]


def score(buys, price, day):
    total = float((buys.sh * buys.px).sum())
    shares = float(buys.sh.sum())
    avg = total / shares if shares else None
    n = buys.RPTOWNERNAME.nunique()
    roles = " ".join((buys.RPTOWNER_TITLE.fillna("") + " " + buys.RPTOWNER_RELATIONSHIP.fillna("")).tolist())
    top = bool(TOP.search(roles))
    director = "Director" in roles
    days = (day - buys.filed.max()).days
    above = (price / avg - 1) * 100 if avg else None
    pts = (40 if n >= 3 else 30 if n == 2 else 15) + (20 if top else 12 if director else 8)
    pts += 20 if total >= 250_000 else 15 if total >= 100_000 else 10
    pts += 10 if days <= 7 else 7 if days <= 14 else 4
    pts += 10 if above is not None and above <= 10 else 5 if above is not None and above <= 25 else 0
    status = "ready" if pts >= 65 and (above is None or above <= 20) else "near" if pts >= 45 else "radar"
    if above is not None and above > 30:
        status = "radar"
    # المقترح: المنصب أكبر وزن (40)، العدد 20، المبلغ 10، الحداثة 10، القرب 5 — و«قوي» لازم CEO/CFO/الرئيس
    alt = (40 if top else 15 if director else 5) + (20 if n >= 3 else 15 if n == 2 else 8)
    alt += 10 if total >= 100_000 else 6
    alt += 10 if days <= 7 else 6 if days <= 14 else 3
    alt += 5 if above is not None and above <= 25 else 0
    alt_status = "ready" if alt >= 65 and top else "near" if alt >= 45 else "radar"
    return {"score": pts, "status": status, "total": total, "n": n, "top": top, "above": above,
            "alt": alt, "alt_status": alt_status}


def outcome(fut, entry):
    for _, r in fut.iterrows():
        if r["Low"] < entry * 0.9:
            return 0
        if r["High"] >= entry * 1.2:
            return 1
    return 0


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    print("tickers", len(tickers), flush=True)
    frames = []
    for q in quarters():
        try:
            f = load_q(q)
            if f is not None:
                frames.append(f)
        except Exception as e:
            print(q, "خطأ", repr(e)[:120], flush=True)
        time.sleep(1)
    tx = pd.concat(frames)
    tx = tx[tx.sym.isin(set(tickers))]
    print("عمليات شراء لأسهمنا:", len(tx), "· أسهم:", tx.sym.nunique(), flush=True)
    daily = {}
    syms = sorted(set(tickers))
    for i in range(0, len(syms), 100):
        try:
            daily.update(A.daily(syms[i:i + 100], days=760))
        except Exception as e:
            print("alpaca", repr(e)[:100], flush=True)
    rows, rnd = [], []
    rng = np.random.default_rng(3)
    for s, d in daily.items():
        if len(d) < 60:
            continue
        idx = d.index
        # عشوائي من نفس الأسهم (آخر 18 شهر)
        cand = [i for i in range(len(d) - 21) if idx[i] >= idx[-1] - pd.Timedelta(days=550)]
        for i in rng.choice(cand, size=min(12, len(cand)), replace=False) if cand else []:
            px = float(d["Close"].iloc[i])
            if 1 <= px <= 5:
                rnd.append({"u10": outcome(d.iloc[i + 1:i + 11], px),
                            "up30": int(float(d["High"].iloc[i + 1:i + 21].max()) / px - 1 >= 0.3),
                            "ret10": float(d["Close"].iloc[i + 10]) / px - 1, "ret20": float(d["Close"].iloc[i + 20]) / px - 1})
        b = tx[tx.sym == s]
        if b.empty:
            continue
        last_sig = None
        for fd in sorted(b.filed.dropna().unique()):
            fd = pd.Timestamp(fd)
            pos = idx.searchsorted(fd, side="right")   # أول يوم تداول بعد التقديم
            if pos >= len(d) - 21 or pos < 30:
                continue
            if last_sig is not None and pos - last_sig < 10:
                continue
            day = idx[pos]
            win = b[(b.filed <= fd) & (b.filed > fd - pd.Timedelta(days=30))]
            px = float(d["Close"].iloc[pos])
            total = float((win.sh * win.px).sum())
            if not (1 <= px <= 5) or total < 25_000:
                continue
            # فلتر معقولية (نفس الرادار): سعر الشراء ضمن 0.25-4 من السعر الحالي
            avgp = total / float(win.sh.sum())
            if not (0.25 * px <= avgp <= 4 * px):
                continue
            last_sig = pos
            sc = score(win, px, day)
            fut10, fut20 = d.iloc[pos + 1:pos + 11], d.iloc[pos + 1:pos + 21]
            rows.append({"t": s, "day": str(day.date()), **sc, "px": px,
                         "u10": outcome(fut10, px), "up30": int(float(fut20["High"].max()) / px - 1 >= 0.3),
                         "ret10": float(d["Close"].iloc[pos + 10]) / px - 1,
                         "ret20": float(d["Close"].iloc[pos + 20]) / px - 1})
    df, R = pd.DataFrame(rows), pd.DataFrame(rnd)
    print(f"\nالعشوائي (n={len(R)}): +20%قبل−10% {pct(R.u10.mean())} · طار30% {pct(R.up30.mean())} · "
          f"عائد10 وسيط {R.ret10.median() * 100:+.1f}% · عائد20 وسيط {R.ret20.median() * 100:+.1f}% · رابح بعد 20 {pct((R.ret20 > 0).mean())}")
    if df.empty:
        print("ما فيه إشارات"); return
    print("الفترة:", df.day.min(), "→", df.day.max())

    def line(sub, name):
        if len(sub) < 5:
            print(f"  {name}: n={len(sub)} (قليل)"); return
        print(f"  {name}: n={len(sub)} · +20%قبل−10% {pct(sub.u10.mean())} · طار30% {pct(sub.up30.mean())} · "
              f"عائد10 وسيط {sub.ret10.median() * 100:+.1f}% · عائد20 وسيط {sub.ret20.median() * 100:+.1f}% "
              f"(متوسط {sub.ret20.mean() * 100:+.1f}%) · رابح بعد 20 {pct((sub.ret20 > 0).mean())}")
    print("\n==== المطّلعين ====")
    line(df, "كل إشارات الشراء ≥25K")
    for st in ("ready", "near", "radar"):
        line(df[df.status == st], f"الحالة {st}")
    line(df[df.n >= 2], "مطّلعين 2+ (Cluster)")
    line(df[df.n == 1], "مطّلع واحد")
    line(df[df.top], "CEO/CFO/President")
    line(df[~df.top], "بدون المدير التنفيذي/المالي")
    line(df[df.total >= 100_000], "مجموع 100K+")
    line(df[df.total >= 250_000], "مجموع 250K+")
    line(df[df.total < 100_000], "مجموع 25-100K")
    line(df[df.above.notna() & (df.above <= 10)], "السعر ضمن 10% من متوسطهم")
    line(df[df.above.notna() & (df.above < 0)], "السعر تحت متوسطهم")
    line(df[df.above.notna() & (df.above > 20)], "السعر فوق متوسطهم 20%+")
    mid = df.day.sort_values().iloc[len(df) // 2]
    print("\n==== الأوزان الحالية مقابل المقترحة (لكل نص من الفترة) ====")
    for half, sub in (("النص الأول", df[df.day < mid]), ("النص الثاني", df[df.day >= mid])):
        print(f" {half}:")
        line(sub[sub.status == "ready"], "  الحالي: شراء قوي")
        line(sub[sub.status.isin(["ready", "near"])], "  الحالي: قوي+داخلي")
        line(sub[sub.alt_status == "ready"], "  المقترح: شراء قوي")
        line(sub[sub.alt_status.isin(["ready", "near"])], "  المقترح: قوي+داخلي")
    line(df[df.day < mid], "النص الأول من الفترة")
    line(df[df.day >= mid], "النص الثاني من الفترة")
    print("\nأمثلة ready:", ", ".join(f"{r.t}@{r.day}:{r.ret20 * 100:+.0f}%" for r in df[df.status == "ready"].tail(15).itertuples()))


if __name__ == "__main__":
    main()
