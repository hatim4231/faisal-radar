#!/usr/bin/env python3
"""
رادار عبدالله — يبحث عن: موجة تدبيل فعلية ← تصحيح إلى دعم ← ثبات ← عودة طلب لموجة ثانية.

    python abdullah/abdullah_scanner.py                 # مسح كامل
    python abdullah/abdullah_scanner.py --tickers ABC,XYZ
    python abdullah/abdullah_scanner.py --out site/abdullah/index.html

يستخدم نفس أدوات جلب البيانات من رادار فيصل (ياهو، ملف الشورت من IB، الأخبار والترجمة)،
لكن الشروط والتقرير مستقلة تماماً.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import scanner as data  # noqa: E402  (أدوات جلب البيانات فقط)
import abdullah_rules as ar  # noqa: E402
import news_tools  # noqa: E402
import sec_tools  # noqa: E402

log = data.log
CACHE = data.CACHE
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
STATE = CACHE / "abdullah_state_v3.json"
SEEN = CACHE / "abdullah_seen.json"
SHORT_HIST = CACHE / "short_history.json"

DILUTION = re.compile(r"offering|private placement|registered direct|at-the-market|\bATM\b|warrant|"
                      r"\bS-1\b|\bS-3\b|\bF-1\b|\bF-3\b|dilut|convertible|\bPIPE\b|shelf|equity line|"
                      r"purchase agreement|securities purchase", re.I)


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def update_short_history(ib: dict, syms: list[str], keep: int) -> dict:
    """يحفظ قراءة الشورت المتاح لكل مسح (مع تاريخ القراءة) عشان نعرض عدة أيام."""
    hist = load_json(SHORT_HIST, {})
    for s in syms:
        x = ib.get(s)
        if not x or x.get("available") is None:
            continue
        rows = hist.setdefault(s, [])
        if rows and rows[-1]["date"] == x["date"]:
            continue
        rows.append({"date": x["date"], "available": x["available"], "more_than": x.get("more_than", False)})
        hist[s] = rows[-keep:]
    SHORT_HIST.write_text(json.dumps(hist), encoding="utf-8")
    return hist


def run(cfg, tickers=None, deep=True) -> dict:
    t0 = time.time()
    A = cfg["abdullah"]
    explicit = tickers is not None
    if tickers is None:
        tickers = data.get_universe(cfg)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not explicit:
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = data.download_daily(tickers, cfg)
    log(f"[عبدالله] وصلت شارتات {len(daily)} سهم")

    U = cfg["universe"]
    pre = {t: d for t, d in daily.items()
           if explicit or (U["min_price"] <= float(d["Close"].iloc[-1]) <= U["max_price"]
                           and float(d["Volume"].tail(30).mean()) >= U["min_avg_volume"] * 0.7)}
    infos = data.get_info(list(pre), cfg)
    try:
        ib = data.load_ib_short()
    except Exception as e:
        ib = {}
        log("[عبدالله] الشورت المتاح ما اشتغل:", repr(e)[:120])
    hist = update_short_history(ib, list(pre), A["short_history_keep"])
    state = load_json(STATE, {})
    seen = load_json(SEEN, {})  # أول يوم ظهر فيه السهم في القائمة (فايدة القوائم: SSM دبّل بعد ~20 يوم)

    results = {}
    for t, d in pre.items():
        inf = infos.get(t) or {}
        try:
            a = ar.analyze(d, inf, cfg, state.get(t))
        except Exception:
            log(f"  خطأ في {t}:", traceback.format_exc(limit=1).splitlines()[-1])
            continue
        if not a:
            continue
        mcap, flt = inf.get("marketCap"), inf.get("floatShares")
        if flt is not None and flt < 50_000:
            flt = None
        if not explicit and ((mcap and mcap > U["max_market_cap"]) or (flt and flt > U["max_float"])):
            continue
        # الشورت: المتاح للاقتراض (عدة قراءات بتواريخها)، وإلا Short Interest
        readings = hist.get(t, [])
        sa = ib.get(t, {}).get("available")
        short_val, short_src = (sa, "المتاح للاقتراض") if sa is not None else (inf.get("sharesShort"), "Short Interest")
        if short_val is not None and short_val > A["max_short_shares"] and not explicit:
            continue
        a.update({
            "symbol": t, "name": inf.get("shortName") or inf.get("longName") or t,
            "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
            "market_cap": mcap, "float": flt, "shares_out": inf.get("sharesOutstanding"),
            "short_interest": inf.get("sharesShort"),
            "short_available": sa, "short_readings": readings[-8:],
            "short_used": {"value": short_val, "source": short_src, "max": A["max_short_shares"]},
            "flags": [],
        })
        if short_val is None:
            a["flags"].append("الشورت غير معروف — ما يترشح لين يتأكد")
            if a["status"] in ("ready", "near"):
                a["status"] = "radar"
        if a["dollar_volume"] < A["min_dollar_volume"]:
            a["flags"].append(f"سيولة ضعيفة: متوسط التداول ${a['dollar_volume'] / 1e3:,.0f}K يومياً")
        if flt and a.get("shares_out") and a["shares_out"] > flt * 3:
            a["flags"].append("عدد الأسهم أكبر بكثير من الفلووت — انتبه لأسهم مقيدة قد تنفك")
        if a["status"] in ("ready", "near", "radar"):
            first = seen.setdefault(t, a["date"])
            a["list_since"] = first
            a["list_days"] = int((pd.Timestamp(a["date"]) - pd.Timestamp(first)).days)
        else:
            seen.pop(t, None)
        results[t] = a

        # حفظ الدعم والإبطال أول مرة يصير السهم متماسك/قريب (ما يتغير بعدين لنفس الموجة)
        st = state.get(t)
        if a["status"] in ("ready", "near") and (not st or st.get("peak_date") != a["wave"]["peak_date"]):
            state[t] = {"peak_date": a["wave"]["peak_date"], "invalidation": a["invalidation"],
                        "support": a["support"], "since": a["date"], "broken": False}
        elif st and st.get("peak_date") == a["wave"]["peak_date"] and a["invalidated"]:
            st["broken"] = True
    STATE.write_text(json.dumps(state), encoding="utf-8")
    SEEN.write_text(json.dumps(seen), encoding="utf-8")
    log(f"[عبدالله] أسهم فيها موجة تدبيل: {len(results)}")

    if deep and results:
        deep_scan(results, cfg)

    # المطلوب أسهم ما طلعت: نشيل اللي طلعت أو كسرت الدعم
    from collections import Counter
    counts = Counter(r["status"] for r in results.values())
    rows = [r for r in results.values() if explicit or
            (r["status"] in ("ready", "near", "radar") and r["score"] >= cfg["scan"]["report_min_score"])]
    order = {"ready": 0, "near": 1, "radar": 2, "moved": 3, "invalid": 4}
    rows.sort(key=lambda r: (order[r["status"]], -r["score"]))
    if deep:
        sec_tools.annotate(rows, log)  # خطر التخفيف من ملفات SEC
        for r in rows:
            dl = r.get("dilution") or {}
            if dl.get("level") in ("high", "medium"):
                h = dl["hits"][0]
                r["flags"].insert(0, f"{'⛔' if dl['level'] == 'high' else '⚠'} SEC: {h['label']} ({h['form']} · {h['date']})")
    rows, excluded_dil = sec_tools.exclude_repeat_diluters(rows, log)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "Yahoo Finance + Interactive Brokers (الشورت المتاح)",
               "scanned": len(daily), "with_wave": len(results),
               "moved": counts.get("moved", 0), "invalid": counts.get("invalid", 0), "seconds": round(time.time() - t0),
               "config": cfg, "stocks": rows, "excluded_dilution": excluded_dil}
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    summary(rows)
    log(f"[عبدالله] خلص في {payload['seconds']} ثانية — {len(rows)} سهم في التقرير")
    return payload


def deep_scan(results: dict, cfg):
    import yfinance as yf
    cands = sorted(results.values(), key=lambda r: ({"ready": 0, "near": 1, "radar": 2, "moved": 3, "invalid": 4}[r["status"]], -r["score"]))
    syms = [r["symbol"] for r in cands if r["status"] in ("ready", "near", "radar")][:cfg["scan"]["deep_scan_max"]]
    if not syms:
        return
    log(f"[عبدالله] أخبار + 4 ساعات + ساعة لـ {len(syms)} سهم")

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n

    with ThreadPoolExecutor(6) as ex:
        for t, n in ex.map(news, syms):
            r = results[t]
            r["news"] = n
            dil = [x for x in n if DILUTION.search((x.get("title") or "") + " " + (x.get("text") or x.get("summary") or ""))]
            if dil:
                r["flags"].append(f"تمويل/تخفيف محتمل: {len(dil)} خبر عن طرح أو Warrants أو تمويل")
                r["dilution_news"] = [x.get("title_ar") or x["title"] for x in dil[:3]]
    news_tools.save_cache()
    try:
        h = yf.download(syms, period="60d", interval="1h", prepost=True, group_by="ticker",
                        auto_adjust=False, threads=True, progress=False)
        for t in syms:
            try:
                d = h[t] if isinstance(h.columns, pd.MultiIndex) else h
                d = d.dropna(subset=["Close"])
                d.index = pd.to_datetime(d.index)
                r = results[t]
                r["intraday"] = ar.intraday_check(d, r["support"], r["consolidation"]["top"])
                if r["intraday"].get("h4_sweep") and r["status"] in ("ready", "near"):
                    # 4H Sweep: نزل تحت قاع سابق ورجع فوقه = سحب سيولة (إشارة إيجابية عند عبدالله)
                    r["score"] = round(min(100, r["score"] + 5), 1)
                    r["flags_pos"] = r.get("flags_pos", []) + [
                        f"سحب سيولة على 4 ساعات إلى {r['intraday']['h4_sweep_low']} ({r['intraday']['h4_sweep_time']}) ثم رجع فوق القاع"]
            except Exception:
                pass
    except Exception as e:
        log("  الفريمات الصغيرة ما اشتغلت:", repr(e)[:120])


def summary(rows):
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    lines = [f"## رادار عبدالله: متماسك {c['ready']} · يتماسك {c['near']} · يصحح {c['radar']}", "",
             "| السهم | السعر | الحالة | النقاط | الموجة | التصحيح | الدعم | اختبارات | جلسات تماسك | الإبطال | الشورت |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows[:25]:
        w, k = r["wave"], r["consolidation"]
        lines.append(f"| {r['symbol']} | {r['price']} | {r['status']} | {r['score']:.0f} | {w['low']}→{w['peak']} x{w['mult']} ({w['len']}ج) | "
                     f"{w['retrace_pct']}% | {r['support']} | {k['touches']} | {k['len']} | {r['invalidation']} | {r['short_used']['value']} |")
    txt = "\n".join(lines)
    print(txt, flush=True)
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if sp:
        with open(sp, "a", encoding="utf-8") as f:
            f.write(txt + "\n")


def write_html(payload, out: Path):
    html = (HERE / "template.html").read_text(encoding="utf-8")
    js = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", js), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="رادار عبدالله")
    ap.add_argument("--tickers")
    ap.add_argument("--out")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    payload = run(cfg, a.tickers.split(",") if a.tickers else None, deep=not a.no_deep)
    out = Path(a.out) if a.out else OUT / "abdullah_report.html"
    write_html(payload, out)
    (OUT / "sig_abdullah.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[عبدالله] التقرير: {out}")
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
