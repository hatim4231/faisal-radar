#!/usr/bin/env python3
"""
🚀 رادار الطيارات — من أمثلة المستخدم (OMH و QMMM). تجريبي وصامت (بالسجل بدون تيليجرام). الكاشف في runners.py.

الفلاتر: قرارات المستخدم (القيمة السوقية/الفلووت، بدون Warrants، امتثال ناسداك، استبعاد التخفيف المتكرر، استبعاد HRT،
علم الدولة، الأخبار بالعربي) **مع استثناءات لهالرادار بس** (config.json → runners، قرار المستخدم):
  السعر مفتوح (باقي الرادارات 0.7-10$) · الشورت حتى 100 ألف (باقي الرادارات 50 ألف — OMH كان شورته ~55 ألف)
  · السيولة أخف: متوسط فوليوم 20 جلسة ≥ 20 ألف وسيولة يومية ≥ 50 ألف$ (OMH فوليومه ~67 ألف وكان ينشال بحد الـ 100 ألف).

المقارنة الصادقة: rand_p = كم مرة دخول عشوائي بنفس الهدف (+100%) والوقف ونفس التذبذب يوصل الهدف خلال 60 جلسة
(runners_baseline.json)، + التوأم العشوائي (3 أسهم من نفس المسح ونفس التذبذب) يتابعه السجل.

    python runners_radar/runners_scanner.py --out site/runners/index.html
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
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

import scanner as data  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import compliance_rule  # noqa: E402
import runners as RN  # noqa: E402
import flags  # noqa: E402
import holders  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
BASE_PATH = ROOT / "runners_baseline.json"
N_TWINS = 3

# نتايج الاختبار التاريخي (diag/backtest_runners2.py، 6,395 سهم مع المشطوبة، 3 سنين): يدبّل خلال 60 جلسة
# مقابل عشوائي بنفس التذبذب والسعر — V5 21.5% (15.3%) · V6 21.7% (17.5%) · V4 20.0% (16.7%)
HIST = {"V5": [21.5, 15.3], "V6": [21.7, 17.5], "V4": [20.0, 16.7]}


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def load_baseline() -> dict | None:
    try:
        return json.loads(BASE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def atr20(d) -> float:
    c, h, l = (d[k].to_numpy(float) for k in ("Close", "High", "Low"))
    return float(((h - l) / c)[-20:].mean() * 100)


def rand_p(B: dict | None, atr: float, stop_pct: float | None, target: float = 2.0) -> float | None:
    """كم مرة دخول عشوائي (نفس التذبذب) يوصل الهدف قبل الإغلاق تحت الوقف خلال 60 جلسة — من runners_baseline.json."""
    if not B or not stop_pct or target not in B.get("targets", []):
        return None
    T = np.asarray(B["hit_permille"][int(np.searchsorted(B["atr20_q"], atr))][B["targets"].index(target)], float) / 1000
    x = min(max(stop_pct - 1, 0.0), len(T) - 1.0001)
    a = int(x)
    return round(float(T[a] * (1 - (x - a)) + T[a + 1] * (x - a)), 3)


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    explicit = tickers is not None
    RC = cfg.get("runners") or {}
    MIN_VOL20, MIN_DV20 = RC.get("min_vol20", 20_000), RC.get("min_dv20", 50_000)
    pmin, pmax = RC.get("min_price"), RC.get("max_price")          # None = السعر مفتوح (قرار المستخدم)
    if tickers is None:
        cfg_u = copy.deepcopy(cfg)
        cfg_u["universe"]["min_avg_volume"] = 2 * MIN_VOL20 * 0.75   # فلتر ياهو: متوسط 3 شهور > 15 ألف
        cfg_u["universe"]["min_price"] = pmin if pmin is not None else 0.0001
        cfg_u["universe"]["max_price"] = pmax if pmax is not None else 1_000_000
        tickers = data.get_universe(cfg_u)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not explicit:
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = data.download_daily(tickers, cfg)
    U = cfg["universe"]

    def liquid(d):
        v20 = float(d["Volume"].tail(20).mean())
        return v20 >= MIN_VOL20 and float((d["Close"].tail(20) * d["Volume"].tail(20)).mean()) >= MIN_DV20

    def price_ok(px):
        return (pmin is None or px >= pmin) and (pmax is None or px <= pmax)

    pre = {t: d for t, d in daily.items() if explicit or (price_ok(float(d["Close"].iloc[-1])) and liquid(d))}
    infos = data.get_info(list(pre), cfg)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    B = load_baseline()
    max_sh = RC.get("max_short_shares", cfg["behavior"].get("max_short_shares"))  # لهالرادار: 100 ألف (None = مفتوح)
    rows, eligible = [], set()
    for t, d in pre.items():
        inf = infos.get(t) or {}
        mcap, flt = inf.get("marketCap"), inf.get("floatShares")
        if flt is not None and flt < 50_000:
            flt = None
        if not explicit and ((mcap and mcap > U["max_market_cap"]) or (flt and flt > U["max_float"])):
            continue
        sa = (ib.get(t) or {}).get("available")
        sv = sa if sa is not None else inf.get("sharesShort")   # المتاح للاقتراض (IB)، وإلا Short Interest
        if not explicit and max_sh and sv is not None and sv > max_sh:
            continue
        if not max_sh or sv is not None:
            eligible.add(t)
        try:
            r = RN.detect(d)
        except Exception as e:
            log(f"  [الطيارات] خطأ {t}:", repr(e)[:100])
            continue
        if not r:
            continue
        c = d["Close"]
        tail = d.tail(160)
        ma20, ma50 = c.rolling(20).mean().tail(160), c.rolling(50).mean().tail(160)
        row = {"symbol": t, "name": inf.get("shortName") or inf.get("longName") or t,
               "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
               "price": _r(c.iloc[-1]), "change_pct": _r((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
               "market_cap": mcap, "float": flt, "short": sv, "short_src": "IB" if sa is not None else "Short Interest",
               "vol20": int(d["Volume"].tail(20).mean()), "date": str(d.index[-1])[:10], "flags": [], "tags": [],
               **r,
               "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(x) for x in tail["Open"]],
                         "h": [_r(x) for x in tail["High"]], "l": [_r(x) for x in tail["Low"]],
                         "c": [_r(x) for x in tail["Close"]], "v": [int(x) for x in tail["Volume"]],
                         "ma20": [_r(x) for x in ma20], "ma50": [_r(x) for x in ma50]}}
        row["atr20"] = round(atr20(d), 2)
        row["rand_p"] = rand_p(B, row["atr20"], r["plan"]["stop_pct"])
        # امتثال ناسداك للأسهم تحت 1$ (نفس طبقة عبدالله — نحول المستويات مؤقتاً)
        row["status"] = {"ready": "ready", "near": "near", "watch": "radar"}[row["status"]]
        compliance_rule.apply_abdullah(row, d)
        row["status"] = {"ready": "ready", "near": "near"}.get(row["status"], "watch")
        if max_sh and sv is None and not explicit:
            row["flags"].append("الشورت غير معروف — ما يترشح لين يتأكد")
            row["status"] = "watch"
        rows.append(row)
    log(f"[الطيارات] {len(rows)} سهم فيه إشارة من {len(pre)} (السيولة الأخف: فوليوم ≥ {MIN_VOL20:,} وسيولة ≥ ${MIN_DV20:,})")
    order = {"ready": 0, "near": 1, "watch": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -(r.get("n_jumps") or 0), -(r.get("hl_pct") or 0)))
    import yfinance as yf

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n
    top = rows[:40]
    with ThreadPoolExecutor(6) as ex:
        nm = dict(ex.map(news, [r["symbol"] for r in top]))
    for r in rows:
        r["news"] = nm.get(r["symbol"], [])
    news_tools.save_cache()
    sec_tools.annotate(top, log)
    rows, excluded = sec_tools.exclude_repeat_diluters(rows, log)
    flags.annotate(rows)
    excluded_hrt = []
    if cfg["behavior"].get("exclude_hrt"):
        try:
            holders.annotate(rows, log)
            rows, excluded_hrt = holders.exclude(rows, log)
        except Exception as e:
            log("فحص HRT تعطل:", repr(e)[:150])
    # التوأم العشوائي: لكل 🚀/🟡 ثلاث أسهم عشوائية من نفس المسح ونفس شريحة التذبذب (السجل يتابعها بنفس الهدف والوقف)
    qs = (B or {}).get("atr20_q")
    atrs = {}
    for t in eligible:
        try:
            atrs[t] = atr20(pre[t])
        except Exception:
            pass
    atrs = {t: a for t, a in atrs.items() if math.isfinite(a)}
    if cfg["behavior"].get("exclude_hrt") and atrs:
        try:
            pool_rows = [{"symbol": t, "name": (infos.get(t) or {}).get("shortName") or (infos.get(t) or {}).get("longName")}
                         for t in atrs]
            holders.annotate(pool_rows, lambda *a: None)
            atrs = {x["symbol"]: atrs[x["symbol"]] for x in pool_rows if not x.get("hrt")}
        except Exception as e:
            log("التوأم العشوائي: فحص HRT تعطل:", repr(e)[:120])
    if not qs and len(atrs) >= 50:
        qs = [float(x) for x in np.quantile(list(atrs.values()), [0.2, 0.4, 0.6, 0.8])]
    if qs:
        bucket = {t: int(np.searchsorted(qs, a)) for t, a in atrs.items()}
        for r in rows:
            if r["status"] not in ("ready", "near") or r["symbol"] not in bucket:
                continue
            pool = sorted(t for t, q in bucket.items() if q == bucket[r["symbol"]] and t != r["symbol"])
            pick = random.Random(f"{r['symbol']}:{r['date']}").sample(pool, min(N_TWINS, len(pool)))
            r["twins"] = [[t, _r(pre[t]["Close"].iloc[-1])] for t in pick]
    cnt = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": len(pre), "seconds": round(time.time() - t0),
               "stocks": rows, "excluded_dilution": excluded, "excluded_hrt": excluded_hrt, "hist": HIST,
               "liquidity": {"vol20": MIN_VOL20, "dv20": MIN_DV20}, "max_short": max_sh, "price_range": [pmin, pmax],
               "baseline": {k: B.get(k) for k in ("version", "horizon", "n")} if B else None}
    log(f"## رادار الطيارات: جاهز {cnt['ready']} · قريب {cnt['near']} · مراقبة {cnt['watch']} · خلص في {payload['seconds']} ثانية")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status']} ({r['level']}) · انهار {r['crash_pct']}% · قاع أعلى +{r['hl_pct']}% · "
            f"RSI {r['rsi_bottom']}→{r['rsi_last_low']} · طار {r['n_jumps']} مرة · لمسات {r['touches']} · الوقف −{r['plan']['stop_pct']}%"
            + (f" · العشوائي بنفس المسافات {r['rand_p'] * 100:.0f}%" if r.get("rand_p") is not None else ""))
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار الطيارات")
    ap.add_argument("--out", default=str(OUT / "runners.html"))
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
    (OUT / "sig_runners.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[الطيارات] التقرير: {out}")


if __name__ == "__main__":
    main()
