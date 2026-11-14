#!/usr/bin/env python3
"""🏹 رادار «الصيد» — طريقة فيصل الإنسان: جمّع في القاعدة قبل الانطلاق، والسهم يطير خلال أيام.

يمسح **كل** أسهم Alpaca النشطة (ما فيه فلتر يشيل سهم — زي قرارك بالمدمج)، يطبّق كاشف التجميع (hunt.py)،
ويرتّب الأقرب للانطلاق. كل فلتر «خطر» (شورت/HRT/طرح/شطب) = **تحذير بالبطاقة بس**، ما يستبعد.

الأساس المثبّت (diag/dna_backtest.py، 3 سنين مع المشطوبة): سهم بهالبصمة احتمال طيرانه ×2 خلال 20ج = 13.5%
مقابل 4.1% للعشوائي (×3.3، ثابت كل سنة). ⚠ الشراء الآلي يخسر؛ الأفضلية في الاختيار + تنفيذك ساعة القفزة.
البيع على المتداول. 46% من البصمة تلمس −30% ← حارس النزيف/الشطب أحمر.

    python hunt_radar/hunt_scanner.py --out site/hunt/index.html
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "highrisk_radar"))

import scanner as data  # noqa: E402
import highrisk_scanner as HS  # noqa: E402
import highrisk as HR  # noqa: E402
import hunt as HUNT  # noqa: E402
import delist as DL  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import flags  # noqa: E402
import holders  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
N_ENRICH = 45

# نتيجة الاختبار (diag/dna_backtest.py): طار ×2 خلال 20ج — البصمة 13.5% مقابل العشوائي 4.1% (×3.3) · 46% تلمس −30%
HIST = {"jump2": 13.5, "rand2": 4.1, "lift": 3.3, "bleed30": 46.6}


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def bleed_of(F) -> float | None:
    """احتمال النزيف 🩸 (يلمس −30% خلال 20ج) من نموذج عالي المخاطرة المعاير."""
    try:
        sc = HR.score(F.iloc[-1])
        return sc.get("pb") if sc else None
    except Exception:
        return None


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    HC = cfg.get("hunt") or {}
    pmin, pmax = HC.get("min_price", 0.3), HC.get("max_price", 15.0)
    min_vol20, min_dv20 = HC.get("min_vol20", 20_000), HC.get("min_dv20", 30_000)
    explicit = tickers is not None
    if tickers is None:
        tickers = HS.alpaca_universe()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    daily = HS.fetch_daily(tickers) if not explicit else data.download_daily(tickers, cfg)
    log(f"[الصيد] اليومي: {len(daily)} سهم")

    cands = []
    for t, d in daily.items():
        if d is None or len(d) < 80:
            continue
        px = float(d["Close"].iloc[-1])
        v20 = float(d["Volume"].tail(20).mean())
        dv20 = float((d["Close"].tail(20) * d["Volume"].tail(20)).mean())
        if not explicit and not (pmin <= px <= pmax and v20 >= min_vol20 and dv20 >= min_dv20):
            continue
        try:
            r = HUNT.detect(d, None)
        except Exception as e:
            log(f"  [الصيد] خطأ {t}:", repr(e)[:90]); continue
        if not r:
            continue
        cands.append((t, d, r))
    cands.sort(key=lambda x: -x[2]["score"])
    log(f"[الصيد] مرشحون بالبصمة: {len(cands)}")
    cands = cands[:HC.get("max_rows", 120)]

    syms = [t for t, _, _ in cands]
    infos = data.get_info(syms, cfg) if syms else {}
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}

    rows = []
    for t, d, _ in cands:
        inf = infos.get(t) or {}
        r = HUNT.detect(d, inf)          # إعادة مع الفلووت
        if not r:
            continue
        sv, short_src = data.short_of(t, ib, inf)
        F = HR.features(d)
        try:
            dl = DL.assess(d)
        except Exception:
            dl = {"level": 0, "reasons": []}
        c = d["Close"]
        tail = d.tail(160)
        ma20, ma50 = c.rolling(20).mean().tail(160), c.rolling(50).mean().tail(160)
        rows.append({
            "symbol": t, "name": inf.get("shortName") or inf.get("longName") or HS.NAMES.get(t) or t,
            "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
            "price": _r(c.iloc[-1]), "change_pct": _r((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
            "market_cap": inf.get("marketCap"), "short": sv, "short_src": short_src,
            "vol20": int(d["Volume"].tail(20).mean()), "date": str(d.index[-1])[:10],
            "bleed": _r(bleed_of(F), 3), "delist": dl, "flags": [], "news": [],
            **r,
            "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(x) for x in tail["Open"]],
                      "h": [_r(x) for x in tail["High"]], "l": [_r(x) for x in tail["Low"]],
                      "c": [_r(x) for x in tail["Close"]], "v": [int(x) for x in tail["Volume"]],
                      "ma20": [_r(x) for x in ma20], "ma50": [_r(x) for x in ma50]},
        })

    order = {"ready": 0, "near": 1, "watch": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -r["score"]))
    # تضييق القائمة (قرار المستخدم): الجاهز للصيد 🎯 + يتجمّع 🟡 بس — نشيل «مبكر» 👀 (نخليه عدّاد)
    watch_hidden = sum(1 for r in rows if r["status"] == "watch")
    rows = [r for r in rows if r["status"] in ("ready", "near")]

    # إثراء الأعلى: أخبار + SEC (تنبيه فقط) + أعلام + HRT (تنبيه فقط، ما نستبعد)
    top = rows[:N_ENRICH]
    import yfinance as yf

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n
    if top:
        with ThreadPoolExecutor(6) as ex:
            nm = dict(ex.map(news, [r["symbol"] for r in top]))
        for r in top:
            r["news"] = nm.get(r["symbol"], [])
        news_tools.save_cache()
        try:
            sec_tools.annotate(top, log)
        except Exception as e:
            log("[الصيد] SEC:", repr(e)[:120])
    try:
        flags.annotate(rows)
    except Exception as e:
        log("[الصيد] أعلام:", repr(e)[:120])
    if cfg["behavior"].get("exclude_hrt"):
        try:
            holders.annotate(rows, log)   # يحط r["hrt"] تنبيه — ما نستبعد
            for r in rows:
                if r.get("hrt"):
                    r["flags"].append("⚠ صندوق HRT من الملاك")
        except Exception as e:
            log("[الصيد] HRT:", repr(e)[:120])

    for r in rows:
        if (r.get("delist") or {}).get("level") == 2:
            r["flags"].append("⛔ خطر شطب عالي — " + "؛ ".join((r["delist"].get("reasons") or [])[:1]))
        if r.get("bleed") is not None and r["bleed"] >= 0.40:
            r["flags"].append(f"🩸 خطر نزيف عالي ({r['bleed']*100:.0f}% يلمس −30%)")

    cnt = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": len(daily),
               "seconds": round(time.time() - t0), "stocks": rows, "hist": HIST, "watch_hidden": watch_hidden,
               "price_range": [pmin, pmax], "liquidity": {"vol20": min_vol20, "dv20": min_dv20}}
    log(f"## رادار الصيد: جاهز {cnt['ready']} · يتجمّع {cnt['near']} · مبكر {cnt['watch']} · خلص في {payload['seconds']}ث")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status_ar']} ({r['score']}) · انهار {r['crash_pct']}% · طار {r['n_jumps']}× · "
            f"قاعدة {r['support']}–{r['ceiling']} ({r['age']}ج) · موقع {r['pos_in_base_pct']}% · فلووت {r['float']} · "
            f"نزيف {r['bleed']} · شطب {(r.get('delist') or {}).get('level')}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار الصيد")
    ap.add_argument("--out", default=str(OUT / "hunt.html"))
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
    (OUT / "sig_hunt.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[الصيد] التقرير: {out}")


if __name__ == "__main__":
    main()
