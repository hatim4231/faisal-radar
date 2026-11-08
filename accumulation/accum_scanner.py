#!/usr/bin/env python3
"""
🔍 رادار التجميع — يكشف الأسهم اللي فيها بصمات تجميع مضارب قبل الانطلاق.

نفس الفلاتر العامة: 0.50-5$ (مع طبقة امتثال ناسداك)، الفوليوم، القيمة السوقية، الفلووت، بدون Warrants،
الشورت 100 ألف وأقل، استبعاد التخفيف المتكرر، والأخبار بالعربي.
تحذير خاص: ملف إعادة بيع (424B3 / F-1 / S-1) = أصحاب الأسهم جاهزين يبيعون — كثير من مضخات الأسهم الصينية يسبقها.

    python accumulation/accum_scanner.py --out site/accumulation/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import scanner as data  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import compliance_rule  # noqa: E402
import accum_rules as R  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
SHORT_HIST = data.CACHE / "short_history.json"


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    explicit = tickers is not None
    if tickers is None:
        tickers = data.get_universe(cfg)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not explicit:
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = data.download_daily(tickers, cfg)
    U = cfg["universe"]
    pre = {t: d for t, d in daily.items()
           if explicit or (U["min_price"] <= float(d["Close"].iloc[-1]) <= U["max_price"]
                           and float(d["Volume"].tail(30).mean()) >= U["min_avg_volume"] * 0.7)}
    infos = data.get_info(list(pre), cfg)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    try:
        hist = json.loads(SHORT_HIST.read_text(encoding="utf-8"))
    except Exception:
        hist = {}
    max_sh = cfg["behavior"].get("max_short_shares")  # None = الشورت مفتوح
    rows, moved = [], 0
    for t, d in pre.items():
        inf = infos.get(t) or {}
        mcap, flt = inf.get("marketCap"), inf.get("floatShares")
        if flt is not None and flt < 50_000:
            flt = None
        if not explicit and ((mcap and mcap > U["max_market_cap"]) or (flt and flt > U["max_float"])):
            continue
        sa = (ib.get(t) or {}).get("available")
        sv, src = (sa, "المتاح للاقتراض") if sa is not None else (inf.get("sharesShort"), "Short Interest")
        if sv is not None and max_sh and sv > max_sh and not explicit:
            continue
        try:
            a = R.analyze(d, inf, cfg, hist.get(t))
        except Exception as e:
            log(f"  خطأ في {t}:", repr(e)[:100])
            continue
        if not a:
            continue
        if a.get("status") == "moved":
            moved += 1
            continue
        prev = float(d["Close"].iloc[-2])
        a.update({"symbol": t, "name": inf.get("shortName") or inf.get("longName") or t,
                  "sector": inf.get("sector") or inf.get("industry"), "market_cap": mcap, "float": flt,
                  "change_pct": round((a["price"] / prev - 1) * 100, 2),
                  "short_used": {"value": sv, "source": src, "max": max_sh}, "flags": [], "tags": []})
        if a["chinese"]:
            a["tags"].append("🇨🇳 " + a["country"])
        if sv is None and max_sh:
            a["flags"].append("الشورت غير معروف — ما يترشح لين يتأكد")
            if a["status"] in ("ready", "near"):
                a["status"] = "radar"
        compliance_rule.apply_abdullah(a, d)  # امتثال ناسداك للأسهم تحت 1$ (نفس طبقة عبدالله)
        rows.append(a)
    log(f"[التجميع] {len(rows)} سهم فيها قاعدة · انطلقت (مستبعدة) {moved}")

    rows = [r for r in rows if explicit or r["score"] >= cfg["accumulation"]["report_min_score"]]
    import yfinance as yf

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n
    top = sorted(rows, key=lambda r: -r["score"])[:cfg["accumulation"]["deep_max"]]
    with ThreadPoolExecutor(6) as ex:
        nm = dict(ex.map(news, [r["symbol"] for r in top]))
    for r in rows:
        r["news"] = nm.get(r["symbol"], [])
    news_tools.save_cache()
    sec_tools.annotate(top, log)
    for r in top:
        hits = (r.get("dilution") or {}).get("hits", [])
        resale = [h for h in hits if h["form"].upper().startswith(("424B3", "S-1", "F-1"))]
        if resale:
            h = resale[0]
            r["flags"].insert(0, f"⚠ ملف إعادة بيع/تسجيل ({h['form']} · {h['date']}) — أصحاب أسهم جاهزين يبيعون، "
                                 f"وكثير من المضخات يسبقها. بع بدري.")
    rows, excluded = sec_tools.exclude_repeat_diluters(rows, log)
    import flags
    import holders
    flags.annotate(rows)
    excluded_hrt = []
    if cfg["behavior"].get("exclude_hrt"):
        try:
            holders.annotate(rows, log)
            rows, excluded_hrt = holders.exclude(rows, log)
        except Exception as e:
            log("فحص HRT تعطل:", repr(e)[:150])
    order = {"ready": 0, "near": 1, "radar": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -r["score"]))
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + Interactive Brokers + SEC", "scanned": len(pre), "moved": moved,
               "excluded_hrt": excluded_hrt,
               "seconds": round(time.time() - t0), "stocks": rows, "excluded_dilution": excluded}
    log(f"## رادار التجميع: مؤكد {c['ready']} · محتمل {c['near']} · مراقبة {c['radar']} · خلص في {payload['seconds']} ثانية")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status']} {r['score']} · قاعدة {r['base_len']}ج · دعم {r['support']} · "
            f"OBV {r['flow']} · صعود/هبوط {r['updown']} · دوران {r['float_turnover']} · {r['country']}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار التجميع")
    ap.add_argument("--out", default=str(OUT / "accumulation.html"))
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
    (OUT / "sig_accumulation.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[التجميع] التقرير: {out}")


if __name__ == "__main__":
    main()
