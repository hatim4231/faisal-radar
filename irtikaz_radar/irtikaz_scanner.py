#!/usr/bin/env python3
"""
⚓ رادار الارتكاز — منشور نايف (@Naifo9x). طلب أبوفرح (2026-11-26): «ابنه» بعد الدراسة. صامت (بدون تيليجرام ولا سجل).
الكاشف irtikaz_naif.py. الحالات:
  ⚓ ارتكاز كامل = الأساسي (قاع سنوي ثابت + RSI<35 + تشبّع) + تقسيم عكسي خلال سنة + القيمة 2-50M + الشورت <20K
  🟡 ارتكاز      = الأساسي + القيمة 2-50M + الشورت <20K (بدون تقسيم)
  👀 مراقبة     = الأساسي بس (القيمة أو الشورت خارج شرط نايف أو غير معروف)
كل بطاقة فيها 🩸 خطر النزيف (highrisk) + خطر الشطب + SEC — لأن الدراسة بيّنت إن الشروط ترفع النزيف بنفس قدر الطيران.

    python irtikaz_radar/irtikaz_scanner.py --out site/irtikaz/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "highrisk_radar"))
import scanner as data  # noqa: E402
import alpaca_data as A  # noqa: E402
import irtikaz_naif as IN  # noqa: E402

log = data.log
OUT = ROOT / "output"
ORDER = {"full": 0, "ready": 1, "watch": 2}
LABEL = {"full": "⚓ ارتكاز كامل (مع تقسيم عكسي)", "ready": "🟡 ارتكاز", "watch": "👀 مراقبة"}
# نتيجة الدراسة (diag/irtikaz_naif.py، مرجع مطابق، 20 جلسة): [الإشارة%, المرجع%]
HIST = {"core": {"n": 2308, "x50": [14.8, 12.3], "bleed": [20.6, 16.8], "ret20": -0.1},
        "rs": {"n": 465, "x50": [24.0, 17.8], "bleed": [34.8, 26.7], "ret20": -2.1}}


def chart(d):
    t = d.tail(160)
    c = t["Close"]
    return {"d": [str(i)[:10] for i in t.index], "o": [round(float(x), 4) for x in t["Open"]], "h": [round(float(x), 4) for x in t["High"]],
            "l": [round(float(x), 4) for x in t["Low"]], "c": [round(float(x), 4) for x in c], "v": [int(x) for x in t["Volume"]],
            "ma20": [round(float(x), 4) if x == x else None for x in c.rolling(20).mean()]}


def run(cfg) -> dict:
    t0 = time.time()
    tickers = [t for t in data.get_universe(cfg) if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = data.download_daily(tickers, cfg)
    cand = {}
    for t, d in daily.items():
        try:
            if float(d["Volume"].tail(20).mean()) < cfg["universe"].get("min_avg_volume", 60_000):
                continue
            r = IN.detect(d)
        except Exception:
            continue
        if r and r["core"]:
            cand[t] = r
    log(f"[الارتكاز] الأساسي (قاع سنوي ثابت + RSI<35 + تشبّع): {len(cand)} من {len(daily)}")
    spl = {}
    if cand and A.available():
        try:
            syms = sorted(cand)
            for i in range(0, len(syms), 100):
                spl.update(A.splits(syms[i:i + 100], datetime.now(timezone.utc) - timedelta(days=400)))
        except Exception as e:
            log("  [الارتكاز] التقسيمات:", repr(e)[:100])
    infos = data.get_info(list(cand), cfg) if cand else {}
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    rows = []
    for t, r in cand.items():
        r = IN.detect(daily[t], spl.get(t))
        inf = infos.get(t) or {}
        mcap = inf.get("marketCap")
        sv, src = data.short_of(t, ib, inf)
        mcap_ok = mcap is not None and IN.MCAP[0] <= mcap <= IN.MCAP[1]
        short_ok = sv is not None and sv < IN.MAX_SHORT
        st = ("full" if r["rs"] else "ready") if (mcap_ok and short_ok) else "watch"
        why_not = []
        if not mcap_ok:
            why_not.append("القيمة السوقية " + (f"{mcap / 1e6:.1f}M" if mcap else "غير معروفة") + " (شرط نايف 2-50M)")
        if not short_ok:
            why_not.append("الشورت " + (f"{sv:,}" if sv is not None else "غير معروف") + " (شرط نايف أقل من 20 ألف)")
        d = daily[t]
        rows.append({"symbol": t, "name": inf.get("shortName") or inf.get("longName") or t, "country": inf.get("country"),
                     "market_cap": mcap, "float": inf.get("floatShares"), "short": sv, "short_src": src,
                     "mcap_ok": mcap_ok, "short_ok": short_ok, "status": st, "label": LABEL[st], "why_not": why_not,
                     "change_pct": round((float(d["Close"].iloc[-1]) / float(d["Close"].iloc[-2]) - 1) * 100, 2),
                     "vol20": int(d["Volume"].tail(20).mean()), "date": str(d.index[-1])[:10], **r, "chart": chart(d), "flags": []})
    # إثراء: 🩸 النزيف، خطر الشطب، SEC، الأعلام، تحليل فيصل
    try:
        import highrisk as HR
        import delist as DL
        for x in rows:
            try:
                sc = HR.score(HR.features(daily[x["symbol"]]).iloc[-1])
                x["bleed"] = round(sc["pb"], 3) if sc and sc.get("pb") is not None else None
            except Exception:
                x["bleed"] = None
            try:
                x["delist"] = DL.assess(daily[x["symbol"]])
            except Exception:
                x["delist"] = None
    except Exception as e:
        log("  [الارتكاز] نزيف/شطب:", repr(e)[:100])
    rows.sort(key=lambda x: (ORDER[x["status"]], x.get("bleed") if x.get("bleed") is not None else 1, x["above_low_pct"]))
    try:
        import flags
        import sec_tools
        flags.annotate(rows)
        live = [x for x in rows if x["status"] != "watch"] + [x for x in rows if x["status"] == "watch"][:20]
        sec_tools.annotate(live, log)
    except Exception as e:
        log("  [الارتكاز] إثراء:", repr(e)[:100])
    try:
        import stock_detail
        for x in rows:
            if x["status"] == "watch" and rows.index(x) > 40:
                continue
            fa = stock_detail.build(x["symbol"], daily[x["symbol"]], infos.get(x["symbol"]) or {}, cfg, dilution=x.get("dilution"),
                                    flag=x.get("flag"), country_ar=x.get("country_ar"))
            if fa:
                x["faisal"] = fa
    except Exception as e:
        log("  [الارتكاز] فيصل:", repr(e)[:100])
    c = Counter(x["status"] for x in rows)
    log(f"## رادار الارتكاز: ⚓ {c['full']} · 🟡 {c['ready']} · 👀 {c['watch']} · {round(time.time() - t0)}ث")
    for x in rows[:15]:
        log(f"  {x['symbol']} {x['price']} {x['label']} · RSI {x['rsi']} · Stoch {x['stoch']} · قاع سنوي {x['annual_low']} (+{x['above_low_pct']}%، قبل {x['low_days_ago']}ج)"
            f" · قيمة {x['market_cap']} · شورت {x['short']} · تقسيم {x['rs']} · نزيف {x.get('bleed')}")
    return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "scanned": len(daily), "stocks": rows, "hist": HIST,
            "rules": {"mcap": IN.MCAP, "max_short": IN.MAX_SHORT}, "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser(description="رادار الارتكاز")
    ap.add_argument("--out", default=str(OUT / "irtikaz.html"))
    a = ap.parse_args()
    cfg = data.load_cfg()
    payload = run(cfg)
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    OUT.mkdir(exist_ok=True)
    (OUT / "sig_irtikaz.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    import stock_detail
    html = stock_detail.inject((HERE / "template.html").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")), encoding="utf-8")
    log(f"[الارتكاز] التقرير: {out}")


if __name__ == "__main__":
    main()
