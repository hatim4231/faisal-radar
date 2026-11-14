#!/usr/bin/env python3
"""🔄 رادار «ما بعد التقسيم العكسي» — نمط أبوفرح (رادار مستقل صامت). الكاشف في revsplit.py.

يمسح كل أسهم Alpaca، يلقى اللي سوّى تقسيم عكسي خلال ~سنة ونزل لـ≤65% من سعره وبنى قاعدة ويجمّع.
بدون فلاتر تشيل — تحذيرات بس (خصوصاً خبر الطرح، 35% من المقسّمة ينزل −30% بيوم).

مثبت (diag/revsplit_study.py + revsplit_accum.py، 3 سنين مع المشطوبة):
  81% تنزل ≤60% (وسيط 13ج) · 99% تبني قاعدة · من القاعدة 26% تنفجر ≥+70% داخل اليوم خلال 40ج (×2.8 عن العشوائي 9%)،
  19% ≥+100%. الشراء الآلي يخسر (−3.8% وسيط) — المكسب من بيعك اللحظي بالقفزة. البيع على المتداول.

    python revsplit_radar/revsplit_scanner.py --out site/revsplit/index.html
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
import revsplit as RS  # noqa: E402
import delist as DL  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import flags  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
N_ENRICH = 45
HIST = {"drop60": 81, "drop_days": 13, "based": 99, "spike70": 26, "spike100": 19, "rand": 9, "lift": 2.8, "offer": 35}


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def bleed_of(F):
    try:
        sc = HR.score(F.iloc[-1]); return sc.get("pb") if sc else None
    except Exception:
        return None


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    RC = cfg.get("revsplit") or {}
    pmin, pmax = RC.get("min_price", 0.3), RC.get("max_price", 20.0)
    min_dv20 = RC.get("min_dv20", 30_000)
    explicit = tickers is not None
    if tickers is None:
        tickers = HS.alpaca_universe()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    daily = HS.fetch_daily(tickers) if not explicit else data.download_daily(tickers, cfg)
    log(f"[التقسيم] اليومي: {len(daily)} سهم")

    cands = []
    for t, d in daily.items():
        if d is None or len(d) < 40:
            continue
        px = float(d["Close"].iloc[-1])
        dv20 = float((d["Close"].tail(20) * d["Volume"].tail(20)).mean())
        if not explicit and not (pmin <= px <= pmax and dv20 >= min_dv20):
            continue
        try:
            r = RS.detect(d, None)
        except Exception as e:
            log(f"  [التقسيم] خطأ {t}:", repr(e)[:90]); continue
        if not r:
            continue
        cands.append((t, d, r))
    cands.sort(key=lambda x: -x[2]["score"])
    log(f"[التقسيم] مرشحون (تقسيم عكسي خلال سنة): {len(cands)}")
    cands = cands[:RC.get("max_rows", 120)]

    syms = [t for t, _, _ in cands]
    infos = data.get_info(syms, cfg) if syms else {}
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    rows = []
    for t, d, _ in cands:
        inf = infos.get(t) or {}
        r = RS.detect(d, inf)
        if not r:
            continue
        sv, short_src = data.short_of(t, ib, inf)   # المتاح للاقتراض من IB
        F = HR.features(d)
        try:
            dl = DL.assess(d)
        except Exception:
            dl = {"level": 0, "reasons": []}
        c = d["Close"]; tail = d.tail(160)
        ma20, ma50 = c.rolling(20).mean().tail(160), c.rolling(50).mean().tail(160)
        rows.append({
            "symbol": t, "name": inf.get("shortName") or inf.get("longName") or HS.NAMES.get(t) or t,
            "country": inf.get("country"), "price": _r(c.iloc[-1]), "change_pct": _r((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
            "market_cap": inf.get("marketCap"), "short": sv, "short_src": short_src,
            "vol20": int(d["Volume"].tail(20).mean()), "date": str(d.index[-1])[:10],
            "bleed": _r(bleed_of(F), 3), "delist": dl, "flags": [], "news": [], **r,
            "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(x) for x in tail["Open"]],
                      "h": [_r(x) for x in tail["High"]], "l": [_r(x) for x in tail["Low"]],
                      "c": [_r(x) for x in tail["Close"]], "v": [int(x) for x in tail["Volume"]],
                      "ma20": [_r(x) for x in ma20], "ma50": [_r(x) for x in ma50]},
        })

    order = {"ready": 0, "near": 1, "watch": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -r["score"]))
    watch_hidden = sum(1 for r in rows if r["status"] == "watch")
    rows = [r for r in rows if r["status"] in ("ready", "near")]   # نعرض اللي نزل وبنى قاعدة

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
            sec_tools.annotate(top, log)       # خبر الطرح مهم جداً لهالنمط
        except Exception as e:
            log("[التقسيم] SEC:", repr(e)[:120])
    try:
        flags.annotate(rows)
    except Exception as e:
        log("[التقسيم] أعلام:", repr(e)[:120])

    for r in rows:
        if r.get("dilution") in ("high", "medium") or r.get("had_offer_drop"):
            r["flags"].append("⚠ احذر خبر الطرح — تأكد من ملفات SEC (35% من المقسّمة ينزل −30% بيوم)")
        if (r.get("delist") or {}).get("level") == 2:
            r["flags"].append("⛔ خطر شطب عالي — " + "؛ ".join((r["delist"].get("reasons") or [])[:1]))
        if r.get("bleed") is not None and r["bleed"] >= 0.40:
            r["flags"].append(f"🩸 خطر نزيف عالي ({r['bleed']*100:.0f}% يلمس −30%)")

    cnt = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": len(daily),
               "seconds": round(time.time() - t0), "stocks": rows, "hist": HIST, "watch_hidden": watch_hidden,
               "price_range": [pmin, pmax]}
    log(f"## رادار التقسيم: جاهز {cnt['ready']} · يتجمّع {cnt['near']} · (مبكر مخفي {watch_hidden}) · خلص في {payload['seconds']}ث")
    for r in rows[:15]:
        rv = r["revsplit"]
        log(f"  {r['symbol']} {r['price']} {r['status_ar']} ({r['score']}) · {rv['ratio']} قبل {rv['days_ago']}ج · نزل {rv['drop_from_p0_pct']}% · "
            f"قاعدة {r['support']} ({r['age']}ج) · نزيف {r['bleed']} · شطب {(r.get('delist') or {}).get('level')} · طرح؟ {r.get('had_offer_drop')}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار ما بعد التقسيم العكسي")
    ap.add_argument("--out", default=str(OUT / "revsplit.html"))
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
    (OUT / "sig_revsplit.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[التقسيم] التقرير: {out}")


if __name__ == "__main__":
    main()
