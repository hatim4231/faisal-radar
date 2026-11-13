#!/usr/bin/env python3
"""
🎯 رادار «الجاهزة» — فيصل المدمج (الأصلي + المطوّر) بدون فلاتر، مع حارس الشطب (طلب المستخدم 2026-11-13):
«ماعاد أبغى كل الفلاتر ... تدمج بين كل الأفكار حقت فيصل الأصلي والمطور ... أهم شي أنه السهم ماينشطب ... مدة 5 أو 10 جلسات».

- يشغّل محرّكي فيصل (`faisal_rules` المطوّر + `faisal_rules_original`) على المجال **بدون** استبعاد الشورت/HRT/الطرح المتكرر/الفلووت/القيمة
  (يبقى بس المجال السعري 0.7-10$ وسيولة أساسية). السهم يترشح لو ✅/🟡 في **أي** محرّك.
- يرتب بعقلين (ready_model: 🎯 ثبات / 🚀 طيران) ويطلع أعلى 3 لكل عقل.
- **حارس الشطب** (delist_model): تحذير أحمر لو خطر الشطب عالي — **ما يستبعد**، بس يحذّر (قرار المستخدم).
- تفاصيل غنية لكل سهم (التقسيمات العكسية، الشورت، الفلووت، القيمة، SEC، الأخبار، الشارت، 🧨/🩸).
- المدة 10 جلسات. تيليجرام يرسل الجاهزة بس (عبر `tracker.py`، المفتاح `merged`).

    python ready_radar/ready_scanner.py --out site/ready/index.html
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "faisal_original"))

import scanner as data  # noqa: E402
import faisal_rules as fr  # noqa: E402
import faisal_rules_original as fo  # noqa: E402
import split_rule  # noqa: E402
import new_low_rule  # noqa: E402
import compliance_rule  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import flags  # noqa: E402
import holders  # noqa: E402
import highrisk as HR  # noqa: E402
import ready as R  # noqa: E402
import delist as DL  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
N_PICK = 3
HORIZON = 10


def _r(x, n=4):
    try:
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else round(x, n)
    except (TypeError, ValueError):
        return None


def relaxed(cfg: dict) -> dict:
    """نسخة إعدادات بدون سقف القيمة/الفلووت (يبقى المجال السعري والفوليوم) — عشان محرّك فيصل ما يستبعد بهم."""
    c = copy.deepcopy(cfg)
    c["universe"]["max_market_cap"] = 1e15
    c["universe"]["max_float"] = 1e15
    return c


def market_state() -> dict:
    try:
        import alpaca_data as A
        from datetime import timedelta, timezone
        iw = A.bars(["IWM"], "1Day", datetime.now(timezone.utc) - timedelta(days=200)).get("IWM")
        s = iw["Close"].astype(float)
        return {"iwm_up": 1.0 if s.iloc[-1] > s.tail(50).mean() else 0.0,
                "iwm_ret20": float(s.iloc[-1] / s.iloc[-21] - 1) * 100, "breadth": 0.42}
    except Exception:
        return {"iwm_up": 0.5, "iwm_ret20": 0.0, "breadth": 0.42}


def best_status(*sts):
    order = {"ready": 2, "near": 1}
    return max((s for s in sts if s in order), key=lambda s: order[s], default=None)


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    rcfg = relaxed(cfg)
    if tickers is None:
        tickers = data.get_universe(cfg)
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = data.download_daily(tickers, cfg)
    U = cfg["universe"]
    pre = {t: d for t, d in daily.items()
           if U["min_price"] <= float(d["Close"].iloc[-1]) <= U["max_price"] and float(d["Volume"].tail(30).mean()) >= U["min_avg_volume"] * 0.7}
    log(f"[الجاهزة] المجال بعد السعر/الفوليوم (بدون فلاتر الشورت/HRT/الطرح): {len(pre)} سهم")
    infos = data.get_info(list(pre), cfg)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    mkt = market_state()
    cand = []
    for t, d in pre.items():
        inf = infos.get(t) or {}
        try:
            a = fr.analyze(d, inf, rcfg)          # المطوّر
            if a:
                split_rule.apply(a, d)
                new_low_rule.apply(a, d)
                compliance_rule.apply_faisal(a, d)
            b = fo.analyze(d, inf, rcfg)           # الأصلي
            if b:
                new_low_rule.apply(b, d)
                compliance_rule.apply_faisal(b, d)
        except Exception as e:
            log("  [الجاهزة] خطأ", t, repr(e)[:90])
            continue
        sd = a["irtikaz"]["status"] if a and not a.get("gate_fail") else None
        so = b["irtikaz"]["status"] if b and not b.get("gate_fail") else None
        st = best_status(sd, so)
        if st not in ("ready", "near"):
            continue
        # نستخدم خطة المحرّك اللي حالته أقوى (تعادل = المطوّر)
        use_dev = (sd == st) if (sd in ("ready", "near")) else False
        if sd == so:
            use_dev = True
        A_, src = (a, "faisal_dev") if (use_dev and a) else (b, "faisal_orig")
        cand.append({"t": t, "d": d, "a": A_, "dev": a, "orig": b, "inf": inf, "status": st,
                     "sd": sd, "so": so, "src": src})
    log(f"[الجاهزة] مرشحين ✅/🟡 (أي محرّك): {len(cand)} · المطوّر {sum(1 for x in cand if x['sd'] in ('ready','near'))}"
        f" · الأصلي {sum(1 for x in cand if x['so'] in ('ready','near'))} · الاثنين {sum(1 for x in cand if x['sd'] in ('ready','near') and x['so'] in ('ready','near'))}")
    # الترتيب + حارس الشطب
    scored = []
    for x in cand:
        A_, d = x["a"], x["d"]
        p = A_["plan"]
        entry, stop, t1 = A_["price"], p.get("stop"), p.get("t1")
        if not stop or not t1 or stop >= entry or t1 <= entry:
            continue
        sig = {"radar": x["src"], "status": x["status"], "entry": entry, "stop": stop, "t1": t1, "score": A_["irtikaz"]["score"]}
        f = R.feature_row(sig, d, mkt)
        if not f:
            continue
        sc = R.score_row(f)
        dl = DL.assess(d)
        x.update({"entry": entry, "stop": stop, "t1": t1, "t2": (p.get("four_targets") or [None, None, None])[1] if p.get("four_targets") else None,
                  "steady": sc["steady"]["score"], "steady_tier": sc["steady"]["tier"],
                  "runner": sc["runner"]["score"], "runner_tier": sc["runner"]["tier"],
                  "p": _r(f["_p"], 4), "pb": _r(f["_pb"], 3), "delist": dl,
                  "t1_pct": _r((t1 / entry - 1) * 100, 1), "stop_pct": _r((1 - stop / entry) * 100, 1)})
        scored.append(x)
    steady = sorted(scored, key=lambda x: -x["steady"])[:N_PICK]
    runner = sorted(scored, key=lambda x: -x["runner"])[:N_PICK]
    picks_ids = {id(x) for x in steady + runner}
    for x in scored:
        x["in_steady"], x["in_runner"] = x in steady, x in runner
    scored.sort(key=lambda x: -max(x["steady_tier"], x["runner_tier"]))
    # تفاصيل غنية + أخبار + SEC + علم + HRT (للعرض بس — ما يستبعد)
    shown = list({id(x): x for x in steady + runner + scored}.values())[:60]
    rows = [build_row(x, ib) for x in shown]
    enrich(rows)
    by_sym = {r["symbol"]: r for r in rows}
    out_steady = [by_sym[x["t"]] for x in steady if x["t"] in by_sym]
    out_runner = [by_sym[x["t"]] for x in runner if x["t"] in by_sym]
    # ملف tracker (تيليجرام يرسل الجاهزة): نجمع أعلى 3 ثبات + 3 طيران (dedup)
    picks = []
    for lst, flavor in ((out_steady, "steady"), (out_runner, "runner")):
        for r in lst:
            picks.append({"radar": "merged", "symbol": r["symbol"], "flavor": flavor,
                          "tier": r["steady_tier"] if flavor == "steady" else r["runner_tier"]})
    (OUT / "ready_picks.json").write_text(json.dumps({"picks": picks, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False), encoding="utf-8")
    M = R.model()
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0),
               "source": "فيصل مدمج (المطوّر + الأصلي) بدون فلاتر · حارس الشطب · الأسعار Alpaca",
               "n_cand": len(cand), "n_scored": len(scored), "market": mkt, "horizon": HORIZON,
               "steady": out_steady, "runner": out_runner, "all": rows, "q": M["q"] if M else {},
               "model_version": M.get("version") if M else None, "n_train": M.get("n") if M else None}
    log(f"🎯 الثبات: {', '.join(r['symbol'] for r in out_steady) or '-'} · 🚀 الطيران: {', '.join(r['symbol'] for r in out_runner) or '-'}"
        f" · من {len(scored)} مرشح · تحذير شطب عالي {sum(1 for r in rows if (r.get('delist') or {}).get('level') == 2)} · خلص في {payload['seconds']}ث")
    return payload


def build_row(x: dict, ib: dict) -> dict:
    A_, d, inf, t = x["a"], x["d"], x["inf"], x["t"]
    px = A_["price"]
    sv, short_src = data.short_of(t, ib, inf)
    try:
        F = HR.features(d)
        hist = HR.history(d, F)
    except Exception:
        hist = {"episodes": [], "reverse_splits": [], "n_year": 0, "days_since": None, "cycle_days": None}
    tail = d.tail(120)
    epdates = set(e["date"] for e in hist["episodes"])
    p = A_["plan"]
    dl = x["delist"] or {}
    row = {"symbol": t, "name": inf.get("shortName") or inf.get("longName") or t, "sector": inf.get("sector") or inf.get("industry"),
           "country": inf.get("country"), "price": _r(px), "change_pct": _r(A_.get("change_pct"), 2),
           "status": x["status"], "label_dev": _status_ar(x["sd"]), "label_orig": _status_ar(x["so"]), "src": x["src"],
           "both": x["sd"] in ("ready", "near") and x["so"] in ("ready", "near"),
           "entry": _r(px), "stop": _r(x["stop"]), "t1": _r(x["t1"]), "t2": _r(x["t2"]),
           "t1_pct": x["t1_pct"], "stop_pct": x["stop_pct"], "zone": [_r(z) for z in (p.get("entry_zone") or [])],
           "market_cap": inf.get("marketCap"), "float": inf.get("floatShares"), "short": sv, "short_src": short_src,
           "steady_tier": x["steady_tier"], "runner_tier": x["runner_tier"], "in_steady": x["in_steady"], "in_runner": x["in_runner"],
           "p": x["p"], "pb": x["pb"], "rsi": A_.get("rsi"), "drawdown_pct": A_.get("drawdown_pct"),
           "n_year": hist["n_year"], "days_since": hist["days_since"],
           "episodes": hist["episodes"][-6:], "reverse_splits": (A_.get("reverse_splits") or hist["reverse_splits"])[-4:],
           "delist": {"level": dl.get("level", 0), "reasons": dl.get("reasons", []),
                      "u1_streak": dl.get("u1_streak"), "dd_pct": _r((dl.get("dd252") or 0) * 100, 0)},
           "why_dev": f"RSI {A_.get('rsi')} · دعم {data.fp(p.get('support')) if hasattr(data,'fp') else p.get('support')} (عنه {p.get('dist_support_pct')}%)",
           "why": _why(A_), "tags": A_.get("tags", []), "flags": [], "split_zone": A_.get("split"),
           "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(v) for v in tail["Open"]], "h": [_r(v) for v in tail["High"]],
                     "l": [_r(v) for v in tail["Low"]], "c": [_r(v) for v in tail["Close"]], "v": [int(v) for v in tail["Volume"]],
                     "ex": [str(i)[:10] in epdates for i in tail.index]}}
    return row


def _status_ar(s):
    return {"ready": "✅ مطابق", "near": "🟡 قريب"}.get(s)


def _why(a):
    p = a["plan"]
    sup = p.get("support")
    return (f"RSI {a.get('rsi')} · دعم {sup} (عنه {p.get('dist_support_pct')}%) · منطقة الطلب "
            f"{(p.get('entry_zone') or ['', ''])[0]}–{(p.get('entry_zone') or ['', ''])[1]}"
            + (" · " + "، ".join(a.get("tags", [])) if a.get("tags") else ""))


def enrich(rows):
    import yfinance as yf

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n
    top = [r for r in rows if r["in_steady"] or r["in_runner"]] + [r for r in rows if not (r["in_steady"] or r["in_runner"])][:30]
    try:
        with ThreadPoolExecutor(6) as ex:
            nm = dict(ex.map(news, [r["symbol"] for r in top]))
    except Exception:
        nm = {}
    for r in rows:
        r["news"] = nm.get(r["symbol"], [])
    news_tools.save_cache()
    try:
        sec_tools.annotate(rows[:40], log)        # تحذير فقط — ما يستبعد
    except Exception as e:
        log("[الجاهزة] SEC:", repr(e)[:100])
    try:
        flags.annotate(rows)
    except Exception:
        pass
    try:
        holders.annotate(rows, log)               # HRT: وسم فقط
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="رادار الجاهزة — فيصل المدمج")
    ap.add_argument("--out", default=str(OUT / "ready.html"))
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
    (OUT / "sig_ready.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[الجاهزة] التقرير: {out}")


if __name__ == "__main__":
    main()
