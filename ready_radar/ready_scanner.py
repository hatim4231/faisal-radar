#!/usr/bin/env python3
"""
🎯 رادار «الجاهزة» — يرتب إشارات فيصل المطوّر + الأصلي + عبدالله ويطلع أقصر قائمة تكسب.
يشتغل بعد الرادارات الثلاثة (يقرا output/sig_faisal_dev.json و sig_faisal_orig.json و sig_abdullah.json).

    python ready_radar/ready_scanner.py --out site/ready/index.html

يطلع: قائمتين (🎯 الثبات top-3 · 🚀 الطيران top-3) + output/sig_ready.json + output/ready_picks.json
(اللي يقراها tracker.py عشان تيليجرام يرسل بس الجاهزة — قرار المستخدم 2026-11-13).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import scanner as data  # noqa: E402
import tracker  # noqa: E402
import ready as R  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
SRC = {"faisal_dev": ("📡 فيصل المطوّر", "sig_faisal_dev.json"),
       "faisal_orig": ("📜 فيصل الأصلي", "sig_faisal_orig.json"),
       "abdullah": ("🌊 عبدالله", "sig_abdullah.json")}
N_PICK = 3


def _r(x, n=4):
    try:
        x = float(x)
        return None if (np.isnan(x) or np.isinf(x)) else round(x, n)
    except (TypeError, ValueError):
        return None


def market_state() -> dict:
    """وضع سوق الصغيرة اليوم: IWM فوق متوسط 50 وأداء 20 يوم. الاتساع ثابت (وزنه صغير بالنموذج)."""
    try:
        import alpaca_data as A
        iw = A.bars(["IWM"], "1Day", datetime.now(timezone.utc) - timedelta(days=200)).get("IWM")
        s = iw["Close"].astype(float)
        return {"iwm_up": 1.0 if s.iloc[-1] > s.tail(50).mean() else 0.0,
                "iwm_ret20": float(s.iloc[-1] / s.iloc[-21] - 1) * 100, "breadth": 0.42}
    except Exception as e:
        log("[الجاهزة] IWM تعطل:", repr(e)[:100])
        return {"iwm_up": 0.5, "iwm_ret20": 0.0, "breadth": 0.42}


def fetch_daily(syms: list[str]) -> dict:
    import alpaca_data as A
    out = {}
    for i in range(0, len(syms), 100):
        for attempt in range(3):
            try:
                out.update({t: d for t, d in A.daily(syms[i:i + 100], days=420).items() if len(d) >= 120})
                break
            except Exception as e:
                log(f"  [الجاهزة] Alpaca (محاولة {attempt + 1}):", repr(e)[:100])
                time.sleep(4 * (attempt + 1))
    return out


def collect_signals() -> list[dict]:
    """كل إشارات ✅/🟡 من الرادارات الثلاثة، موحّدة (عبر tracker.signals)."""
    sigs = []
    for key, (name, fn) in SRC.items():
        payload = tracker.load(OUT / fn, None)
        if not payload:
            log(f"[الجاهزة] ما فيه نتائج لـ {name}")
            continue
        for s in tracker.signals(key, payload):
            if s.get("status") in ("ready", "near") and s.get("entry") and s.get("stop") and s.get("t1"):
                s["radar"], s["radar_name"] = key, name
                sigs.append(s)
    return sigs


def run() -> dict:
    t0 = time.time()
    if not R.model():
        log("[الجاهزة] ⚠ ما فيه ready_model.json — الرادار ما يشتغل")
        return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "steady": [], "runner": [], "all": []}
    sigs = collect_signals()
    log(f"[الجاهزة] إشارات ✅/🟡 من الرادارات الثلاثة: {len(sigs)}")
    if not sigs:
        return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "steady": [], "runner": [], "all": []}
    mkt = market_state()
    daily = fetch_daily(sorted({s["symbol"] for s in sigs}))
    log(f"[الجاهزة] شارتات: {len(daily)} سهم · السوق: IWM فوق متوسطه {mkt['iwm_up']} · 20 يوم {mkt['iwm_ret20']:+.1f}%")
    scored = []
    for s in sigs:
        d = daily.get(s["symbol"])
        f = R.feature_row(s, d, mkt) if d is not None else None
        if not f:
            continue
        sc = R.score_row(f)
        s = dict(s)
        s.update({"steady": sc["steady"]["score"], "steady_tier": sc["steady"]["tier"],
                  "runner": sc["runner"]["score"], "runner_tier": sc["runner"]["tier"],
                  "p": _r(f["_p"], 4), "pb": _r(f["_pb"], 3), "t1_pct": _r((s["t1"] / s["entry"] - 1) * 100, 1),
                  "stop_pct": _r((1 - s["stop"] / s["entry"]) * 100, 1), "last": _r(d["Close"].iloc[-1])})
        scored.append(s)
    # dedup بنفس السهم عبر الرادارات: نخلي الأعلى بمجموع الدرجتين
    best = {}
    for s in scored:
        k = s["symbol"]
        if k not in best or (s["steady"] + s["runner"]) > (best[k]["steady"] + best[k]["runner"]):
            best[k] = s
    uniq = list(best.values())
    steady = sorted(uniq, key=lambda s: -s["steady"])[:N_PICK]
    runner = sorted(uniq, key=lambda s: -s["runner"])[:N_PICK]
    pick_syms = {s["symbol"] for s in steady + runner}
    for s in uniq:
        s["in_steady"] = any(s["symbol"] == x["symbol"] for x in steady)
        s["in_runner"] = any(s["symbol"] == x["symbol"] for x in runner)
    uniq.sort(key=lambda s: -max(s["steady_tier"], s["runner_tier"]))
    # ملف للـ tracker: أي (رادار، سهم) جاهزة (عشان تيليجرام يرسلها بس) — نحفظ أصل الإشارة لكل سهم جاهز
    picks = []
    for lst, flavor in ((steady, "steady"), (runner, "runner")):
        for s in lst:
            picks.append({"radar": s["radar"], "symbol": s["symbol"], "flavor": flavor,
                          "tier": s["steady_tier"] if flavor == "steady" else s["runner_tier"], "date": s.get("date")})
    M = R.model()
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0),
               "source": "ترتيب فوق فيصل المطوّر + الأصلي + عبدالله · الأسعار Alpaca",
               "n_signals": len(sigs), "n_scored": len(uniq), "market": mkt,
               "steady": [_slim(s) for s in steady], "runner": [_slim(s) for s in runner],
               "all": [_slim(s) for s in uniq], "q": M["q"], "model_version": M.get("version"),
               "period": M.get("period"), "n_train": M.get("n")}
    (OUT / "ready_picks.json").write_text(json.dumps({"picks": picks, "generated_at": payload["generated_at"]}, ensure_ascii=False), encoding="utf-8")
    log(f"🎯 الثبات: {', '.join(s['symbol'] for s in steady) or '-'} · 🚀 الطيران: {', '.join(s['symbol'] for s in runner) or '-'}"
        f" · من {len(uniq)} مرشح · خلص في {payload['seconds']}ث")
    return payload


KEEP = ("symbol", "name", "flag", "country_ar", "radar", "radar_name", "status", "label", "entry", "stop", "t1", "t2",
        "last", "t1_pct", "stop_pct", "p", "pb", "steady", "steady_tier", "runner", "runner_tier",
        "in_steady", "in_runner", "why", "score", "date", "dilution")


def _slim(s: dict) -> dict:
    return {k: s.get(k) for k in KEEP}


def main():
    ap = argparse.ArgumentParser(description="رادار الجاهزة")
    ap.add_argument("--out", default=str(OUT / "ready.html"))
    a = ap.parse_args()
    payload = run()
    import os
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    html = (HERE / "template.html").read_text(encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")),
                   encoding="utf-8")
    (OUT / "sig_ready.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[الجاهزة] التقرير: {out}")


if __name__ == "__main__":
    main()
