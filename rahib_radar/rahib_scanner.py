#!/usr/bin/env python3
"""🔥 رادار «فيصل الرهيب» — نظرية الارتكاز البحتة (من دليل فيصل). مستقل صامت. الكاشف في rahib.py.

فلتر فيصل الحرفي (ص6): سعر < 5$ · أقل من 300M · فلووت < 20M · متوسط فوليوم > 100K · أداء شهر/ربع هابط (قيعان) · RVOL أي.
الإعداد: دعم حقيقي + تجميع (قيعان أعلى + فوليوم بيع يقل + انكماش) + ثبات + مساحة صعود + مو ضخ.
خطة فيصل: منطقة طلب (Limit مقسّم) · دعم حرج (كسره = خروج) · جني ربح تدريجي عند المقاومة 1 ثم 2.

    python rahib_radar/rahib_scanner.py --out site/rahib/index.html
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "highrisk_radar"))

import scanner as data  # noqa: E402
import highrisk_scanner as HS  # noqa: E402
import highrisk as HR  # noqa: E402
import rahib as RB  # noqa: E402
import delist as DL  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import flags  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
N_ENRICH = 45


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def bleed_of(F):
    try:
        sc = HR.score(F.iloc[-1]); return sc.get("pb") if sc else None
    except Exception:
        return None


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    RC = cfg.get("rahib") or {}
    explicit = tickers is not None
    if tickers is None:
        tickers = HS.alpaca_universe()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    daily = HS.fetch_daily(tickers) if not explicit else data.download_daily(tickers, cfg)
    log(f"[الرهيب] اليومي: {len(daily)} سهم")

    # مرحلة 1: الكشف بدون معلومات (الفلتر الفني + السعر/الفوليوم/الهبوط)
    cands = []
    for t, d in daily.items():
        if d is None or len(d) < 70:
            continue
        try:
            r = RB.detect(d, None, cfg)
        except Exception as e:
            log(f"  [الرهيب] خطأ {t}:", repr(e)[:90]); continue
        if not r:
            continue
        cands.append((t, d, r))
    cands.sort(key=lambda x: -x[2]["score"])
    log(f"[الرهيب] مرشحون (فني): {len(cands)}")
    cands = cands[:RC.get("max_rows", 120)]

    # مرحلة 2: معلومات ياهو → فلتر الفلووت/القيمة (فلتر فيصل الكامل) + إثراء
    syms = [t for t, _, _ in cands]
    infos = data.get_info(syms, cfg) if syms else {}
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    rows = []
    for t, d, _ in cands:
        inf = infos.get(t) or {}
        r = RB.detect(d, inf, cfg)   # يطبّق فلتر الفلووت/القيمة
        if not r:
            continue
        sv, short_src = data.short_of(t, ib, inf)
        F = HR.features(d)
        try:
            dl = DL.assess(d)
        except Exception:
            dl = {"level": 0, "reasons": []}
        c = d["Close"]
        rows.append({
            "symbol": t, "name": inf.get("shortName") or inf.get("longName") or HS.NAMES.get(t) or t,
            "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
            "price": _r(c.iloc[-1]), "change_pct": _r((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
            "short": sv, "short_src": short_src, "vol20": int(d["Volume"].tail(20).mean()),
            "date": str(d.index[-1])[:10], "bleed": _r(bleed_of(F), 3), "delist": dl,
            "flags": [], "news": [], **r,
        })

    order = {"ready": 0, "near": 1, "watch": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -r["score"]))
    watch_hidden = sum(1 for r in rows if r["status"] == "watch")
    rows = [r for r in rows if r["status"] in ("ready", "near")]   # نعرض الجاهز والمتجمّع (المراقبة عدّاد)

    # إثراء الأعلى
    top = rows[:N_ENRICH]
    import yfinance as yf

    def news(t):
        try:
            nn = data.parse_news(yf.Ticker(t).news)
        except Exception:
            nn = []
        news_tools.enrich(nn, data.translate_ar)
        return t, nn
    if top:
        with ThreadPoolExecutor(6) as ex:
            nm = dict(ex.map(news, [r["symbol"] for r in top]))
        for r in top:
            r["news"] = nm.get(r["symbol"], [])
        news_tools.save_cache()
        try:
            sec_tools.annotate(top, log)
        except Exception as e:
            log("[الرهيب] SEC:", repr(e)[:120])
    try:
        flags.annotate(rows)
    except Exception as e:
        log("[الرهيب] أعلام:", repr(e)[:120])

    for r in rows:
        if (r.get("delist") or {}).get("level") == 2:
            r["flags"].append("⛔ خطر شطب عالي — " + "؛ ".join((r["delist"].get("reasons") or [])[:1]))
        if r.get("bleed") is not None and r["bleed"] >= 0.40:
            r["flags"].append(f"🩸 خطر نزيف عالي ({r['bleed']*100:.0f}% يلمس −30%)")
        if r.get("dilution") in ("high", "medium"):
            r["flags"].append("⚠ ملفات طرح/تخفيف SEC — تأكد قبل الدخول")

    cnt = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": len(daily),
               "seconds": round(time.time() - t0), "stocks": rows, "watch_hidden": watch_hidden}
    log(f"## فيصل الرهيب: جاهز {cnt['ready']} · يتجمّع {cnt['near']} · (مراقبة مخفي {watch_hidden}) · خلص في {payload['seconds']}ث")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status_ar']} ({r['score']}) · دعم {r['support']} (بعد {r['dist_support_pct']}%، لمسات {r['touches']}) · "
            f"تجميع {r['accum_signs']}/4 · شهر {r['perf_month_pct']}% · فلووت {r['float']} · شورت {r['short']} · هدف +{r['upside_pct']}%")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار فيصل الرهيب (الارتكاز)")
    ap.add_argument("--out", default=str(OUT / "rahib.html"))
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
    (OUT / "sig_rahib.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[الرهيب] التقرير: {out}")


if __name__ == "__main__":
    main()
