#!/usr/bin/env python3
"""
0️⃣ رادار المصفّر — طلب أبوفرح (2026-11-24): «من رادار فيصل الأصلي أو المطوّر أو رادار عبدالله أي سهم شورته صفر حطه في رادار ثاني وسمه رادار المصفّر».

الشورت صفر = المتاح للاقتراض عند Interactive Brokers = 0 (السهم مو بقائمة IB، أو موجود ورقمه 0) — نفس تعريف `scanner.short_of`
اللي تستخدمه كل الرادارات. ما نحسب شي جديد: نقرا نتائج نفس المسح من output/sig_faisal_dev.json و sig_faisal_orig.json و sig_abdullah.json
ونجمع الأسهم اللي `short_used.value == 0`، مع حالة السهم بكل رادار. التفاصيل = «تحليل فيصل الكامل» (stock_detail).
صامت (بدون تيليجرام ولا سجل). ⚠ الشورت صفر يعني ما أحد يقدر يبيع على المكشوف عند IB — مو ضمان صعود. ليس توصية.

    python zero_radar/zero_scanner.py --out site/zero/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"

SOURCES = [("faisal_dev", "📡 فيصل المطوّر", "sig_faisal_dev.json", {"ready": "✅ مطابق", "near": "🟡 قريب", "radar": "👀 تحت الرادار", "no": "غير مطابق"}),
           ("faisal_orig", "📜 فيصل الأصلي", "sig_faisal_orig.json", {"ready": "✅ مطابق", "near": "🟡 قريب", "radar": "👀 تحت الرادار", "no": "غير مطابق"}),
           ("abdullah", "🌊 عبدالله", "sig_abdullah.json", {"ready": "🎯 جاهز", "near": "🟡 قريب", "radar": "👀 مراقبة", "no": "غير مطابق"})]
RANK = {"ready": 0, "near": 1, "radar": 2, "no": 3}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _status(key, r):
    if key == "abdullah":
        return r.get("status"), r.get("score")
    I = r.get("irtikaz") or {}
    return I.get("status"), I.get("score")


def _is_zero(r) -> bool:
    v = (r.get("short_used") or {}).get("value")
    return v is not None and float(v) == 0


def collect() -> tuple[list[dict], dict]:
    by, meta = {}, {}
    for key, label, fn, ST in SOURCES:
        p = OUT / fn
        try:
            P = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"  {label}: ما انقرا ({repr(e)[:80]})")
            meta[key] = None
            continue
        rows = P.get("stocks") or []
        z = [r for r in rows if _is_zero(r)]
        meta[key] = {"label": label, "total": len(rows), "zero": len(z), "generated_at": P.get("generated_at")}
        log(f"  {label}: {len(z)} سهم شورته صفر من {len(rows)}")
        for r in z:
            st, sc = _status(key, r)
            e = by.setdefault(r["symbol"], {"symbol": r["symbol"], "sources": [], "rows": {}})
            e["sources"].append({"key": key, "radar": label, "status": st, "status_ar": ST.get(st, st or "-"),
                                 "score": round(float(sc), 1) if sc is not None else None})
            e["rows"][key] = r
    return list(by.values()), meta


def detail(e: dict, cfg: dict) -> dict | None:
    """تحليل فيصل الكامل للبطاقة: صف المطوّر نفسه، أو تحليل عبدالله المرفق، أو نبنيه من الشارت (الأصلي بس)."""
    R = e["rows"]
    if "faisal_dev" in R:
        return R["faisal_dev"]
    if R.get("abdullah", {}).get("faisal"):
        return R["abdullah"]["faisal"]
    r = R.get("faisal_orig") or R.get("abdullah")
    try:
        import scanner as data
        import stock_detail
        d = data.download_daily([e["symbol"]], cfg).get(e["symbol"])
        if d is None:
            return None
        info = {"floatShares": r.get("float"), "marketCap": r.get("market_cap"), "sharesOutstanding": r.get("shares_out"),
                "shortName": r.get("name"), "sector": r.get("sector"), "country": r.get("country")}
        return stock_detail.build(e["symbol"], d, info, cfg, news=r.get("news"), dilution=r.get("dilution"),
                                  short_available=r.get("short_available"), flag=r.get("flag"), country_ar=r.get("country_ar"))
    except Exception as ex:
        log(f"  {e['symbol']}: التحليل ما انبنى ({repr(ex)[:90]})")
        return None


def main():
    ap = argparse.ArgumentParser(description="رادار المصفّر")
    ap.add_argument("--out", default=str(OUT / "zero.html"))
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    items, meta = collect()
    stocks = []
    for e in items:
        best = min(e["sources"], key=lambda s: RANK.get(s["status"], 9))
        any_row = next(iter(e["rows"].values()))
        fa = detail(e, cfg)
        stocks.append({"symbol": e["symbol"], "name": any_row.get("name") or e["symbol"], "price": (fa or any_row).get("price"),
                       "change_pct": (fa or any_row).get("change_pct"), "flag": any_row.get("flag"), "country_ar": any_row.get("country_ar"),
                       "float": any_row.get("float"), "short_src": (any_row.get("short_used") or {}).get("source"),
                       "best": best["status"], "sources": sorted(e["sources"], key=lambda s: RANK.get(s["status"], 9)),
                       "tags": sorted({t for r in e["rows"].values() for t in (r.get("tags") or [])}),
                       "faisal": fa})
    stocks.sort(key=lambda s: (RANK.get(s["best"], 9), -len(s["sources"]), -max((x["score"] or 0) for x in s["sources"])))
    from collections import Counter
    c = Counter(s["best"] for s in stocks)
    log(f"## رادار المصفّر: {len(stocks)} سهم شورته صفر (أفضل حالة: ✅/🎯 {c['ready']} · 🟡 {c['near']} · 👀 {c['radar']} · غير مطابق {c['no']}) · {round(time.time() - t0)}ث")
    for s in stocks[:20]:
        log(f"  {s['symbol']} {s['price']} · " + " · ".join(f"{x['radar']} {x['status_ar']} {x['score']}" for x in s["sources"]))
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": stocks, "meta": meta, "seconds": round(time.time() - t0)}
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    (OUT / "sig_zero.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    import stock_detail
    html = stock_detail.inject((HERE / "template.html").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")), encoding="utf-8")
    log(f"[المصفّر] التقرير: {out}")


if __name__ == "__main__":
    main()
