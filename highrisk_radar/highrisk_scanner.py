#!/usr/bin/env python3
"""
🧨 رادار «عالي مخاطرة» — الأسهم اللي تنفجر بشكل متكرر (طلب المستخدم 2026-11-11). تجريبي **صامت** (بالسجل بدون تيليجرام).

بدون أي قيود (قرار المستخدم: «في الرادار ذا لاتتقيد بأي قيود كل شي مفتوح»): كل أسهم Alpaca النشطة (NASDAQ/NYSE/AMEX)
بدون حد سعر أو شورت أو HRT أو طرح أو امتثال أو فلووت أو قيمة سوقية أو فوليوم — هذي تنعرض **معلومة وتحذير بس**.
(مو أسهم أصلاً فما تدخل: صناديق ETF والـ Warrants/Rights/Units.)

النموذج: highrisk.py + highrisk_model.json (من diag/backtest_highrisk.py على فرع diag-flye) — احتمال انفجار
(أعلى سعر ≥ ضعف إغلاق اليوم اللي قبله) خلال 10 جلسات، معاير على فترات ما شافها.

    python highrisk_radar/highrisk_scanner.py --out site/highrisk/index.html
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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
import flags  # noqa: E402
import holders  # noqa: E402
import highrisk as HR  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
N_TWINS = 3
SHOW = 120           # كم سهم ينعرض (الأعلى احتمال) + اللي انطلقت اليوم
ETF_RX = re.compile(r"\b(ETF|ETN|Fund|Trust|ProShares|Direxion|iShares|Leveraged|Daily|2X|3X|Bull|Bear|Ultra|Notes|"
                    r"Index|Shares|SPDR|Invesco|VanEck|WisdomTree|Global X|Vanguard|Schwab|Portfolio)\b", re.I)
# حالات الرادار (من الدراسة — highrisk_model.json → thresholds): الأعلى احتمال = 🧨، بعده 🟠، والباقي من المتكرر 👀
ST_AR = {"fire": "🔥 انطلق اليوم", "ready": "🧨 مرشح قوي", "near": "🟠 مرشح", "watch": "👀 متكرر"}
# الخطة الأقل سوءاً بالاختبار (7 طرق، خارج العينة): شراء Stop فوق إغلاق اليوم +15% باليوم التالي (ما تدخل إلا إذا بدأ يطير)،
# بيع Limit +30% من سعر الدخول، وقف إغلاق −15%، أقصى 3 جلسات — 🧨: +0.3% بالمتوسط مقابل −0.6% للعشوائي (تقريباً تعادل).
# الدخول على الإغلاق خسران بكل المدد (−2% إلى −3% مقابل ~0% للعشوائي). الحجم: خسارة الوقف ≤ 0.5% من المحفظة.
PLAN = {"trig": 0.15, "t1": 0.30, "stop": 0.15, "days": 3, "risk": 0.5}


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), n)


def alpaca_universe() -> list[str]:
    """كل الأسهم النشطة من Alpaca (بدون فلاتر) — احتياط: قائمة ياهو/ناسداك إذا Alpaca تعطل."""
    import requests
    import alpaca_data as A
    syms = []
    for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            r = requests.get(f"{base}/v2/assets", headers=A._headers(),
                             params={"status": "active", "asset_class": "us_equity"}, timeout=90)
            r.raise_for_status()
            for a in r.json():
                s, ex, nm = a.get("symbol", ""), a.get("exchange", ""), a.get("name") or ""
                if ex not in ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"):
                    continue
                if not re.fullmatch(r"[A-Z]{1,5}", s) or (len(s) == 5 and s[-1] in "WRU") or ETF_RX.search(nm):
                    continue
                syms.append(s)
            if syms:
                break
        except Exception as e:
            log("[عالي مخاطرة] قائمة Alpaca:", repr(e)[:120])
    return sorted(set(syms))


def fetch_daily(tickers: list[str]) -> dict:
    """يومي من Alpaca لكل المجال (~5 آلاف سهم): 420 يوم تكفي (الميزات تحتاج 252 جلسة) — نص حمل download_daily (730 يوم)."""
    import alpaca_data as A
    out = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        for attempt in range(3):
            try:
                out.update({t: d for t, d in A.daily(batch, days=420).items() if len(d) >= 120})
                break
            except Exception as e:
                log(f"  [عالي مخاطرة] Alpaca خطأ (محاولة {attempt + 1}):", repr(e)[:120])
                time.sleep(5 * (attempt + 1))
    log(f"[عالي مخاطرة] اليومي من Alpaca: {len(out)} من {len(tickers)}")
    return out


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    M = HR.model()
    if not M:
        log("[عالي مخاطرة] ⚠ ما فيه highrisk_model.json — الرادار ما يشتغل")
    if tickers is None:
        tickers = alpaca_universe()
        log(f"[عالي مخاطرة] المجال: {len(tickers)} سهم من Alpaca (بدون فلاتر)")
        if len(tickers) < 500:
            tickers = data.get_universe(dict(cfg, universe=dict(cfg["universe"], min_price=0.0001, max_price=1_000_000,
                                                                min_avg_volume=0)))
    daily = fetch_daily(tickers)
    TH = (M or {}).get("thresholds") or {}
    # الحالة: 🔥 انطلق اليوم (أعلى ≥ +50% فوق إغلاق أمس) · 🧨/🟠 حسب حدود الاحتمال من الدراسة · 👀 متكرر (انفجر مرتين+ بسنة)
    p_ready, p_near = TH.get("ready", 1.0), TH.get("near", 1.0)
    rows_all = []
    for t in list(daily):
        d = daily.pop(t)
        if len(d) < 120:
            continue
        try:
            F = HR.features(d)
            last = F.iloc[-1]
            sc = HR.score(last)
        except Exception as e:
            log("  [عالي مخاطرة] خطأ", t, repr(e)[:100])
            continue
        if not sc:
            continue
        jump = float(last["jump"]) if np.isfinite(last["jump"]) else 1.0
        p = sc["p_raw"]
        st = ("fire" if jump >= 1.5 else "ready" if p >= p_ready else "near" if p >= p_near
              else "watch" if last["n_ep252"] >= 2 else None)
        # نخلي الشارت والميزات كاملة بس للي ممكن ينعرض (توفير ذاكرة: المجال ~5 آلاف سهم)
        rows_all.append({"t": t, "d": d if st else None, "F": F if st else None, "last": last, "sc": sc, "jump": jump, "status": st})
    n_scanned = len(rows_all)
    if not rows_all:
        return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": [], "scanned": n_scanned}
    order = {"fire": 0, "ready": 1, "near": 2, "watch": 3}
    shown = [x for x in rows_all if x["status"]]
    shown.sort(key=lambda x: (order[x["status"]], -x["sc"]["p_raw"]))
    shown = [x for x in shown if x["status"] == "fire"] + [x for x in shown if x["status"] != "fire"][:SHOW]
    infos = data.get_info([x["t"] for x in shown], cfg)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    rows = []
    for x in shown:
        t, d, F, last, sc = x["t"], x["d"], x["F"], x["last"], x["sc"]
        inf = infos.get(t) or {}
        c = d["Close"]
        px = float(c.iloc[-1])
        sv, short_src = data.short_of(t, ib, inf)
        hist = HR.history(d, F)
        tail = d.tail(160)
        epdates = set(e["date"] for e in hist["episodes"])
        row = {"symbol": t, "name": inf.get("shortName") or inf.get("longName") or t,
               "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
               "price": _r(px), "change_pct": _r((px / float(c.iloc[-2]) - 1) * 100, 2),
               "jump_today_pct": _r((x["jump"] - 1) * 100, 1), "status": x["status"], "status_ar": ST_AR[x["status"]],
               "market_cap": inf.get("marketCap"), "float": inf.get("floatShares"), "short": sv, "short_src": short_src,
               "dv20": _r(last["dv20"], 0), "vol20": int(d["Volume"].tail(20).mean()), "date": str(d.index[-1])[:10],
               "p_raw": _r(sc["p_raw"], 4), "decile": sc["decile"], "e10": _r(sc["e10"], 3), "e20": _r(sc["e20"], 3),
               "base_e10": _r(sc["base"], 3), "tr50_hist": _r(sc.get("tr50"), 3),
               "why": [[f, _r(v, 3)] for f, v in sc["top"]],
               "fp": {k: _r(last[k], 3) for k in ("vol5_60", "vol1_20", "atr5_60", "rng20", "pos60", "dd252", "ret5", "ret20",
                                                     "rsi", "closepos10", "upv10", "obv20", "hl_up", "base_ratio", "from_peak")},
               "n_year": hist["n_year"], "days_since": hist["days_since"], "cycle_days": hist["cycle_days"],
               "episodes": hist["episodes"][-8:], "reverse_splits": hist["reverse_splits"][-4:],
               "short_hist": len(d) < 260,
               "plan": {"trigger": _r(px * (1 + PLAN["trig"])), "t1": _r(px * (1 + PLAN["trig"]) * (1 + PLAN["t1"])),
                        "stop": _r(px * (1 + PLAN["trig"]) * (1 - PLAN["stop"])), "days": PLAN["days"],
                        "base_low": _r(float(last["lo20"])), "size_pct": round(PLAN["risk"] / PLAN["stop"], 1)},
               "flags": [], "tags": [],
               "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(v) for v in tail["Open"]],
                         "h": [_r(v) for v in tail["High"]], "l": [_r(v) for v in tail["Low"]],
                         "c": [_r(v) for v in tail["Close"]], "v": [int(v) for v in tail["Volume"]],
                         "ex": [str(i)[:10] in epdates for i in tail.index]}}
        if row["short_hist"]:
            row["flags"].append("سهم جديد (أقل من سنة تداول) — تاريخ الانفجارات ناقص والاحتمال أقل دقة")
        rsd = int(last["days_rs_raw"])
        if rsd <= 60:
            row["flags"].append(f"✂️ تقسيم عكسي قبل {rsd} جلسة")
        if px < 1:
            row["flags"].append("🏛 تحت 1$ — خطر إنذار ناسداك/تقسيم عكسي")
        rows.append(row)
    # معلومات وتحذيرات (ما تشيل شي — قرار المستخدم: كل شي مفتوح)
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
    try:
        sec_tools.annotate(rows[:80], log)
        for r in rows[:80]:
            ev = sec_tools.dilution_events(r)
            if len(ev) >= 2:
                r["dilution_events"] = [{"date": e["date"], "what": " + ".join(e["what"][:3])} for e in ev]
    except Exception as e:
        log("[عالي مخاطرة] SEC تعطل:", repr(e)[:120])
    flags.annotate(rows)
    try:
        holders.annotate(rows, log)
    except Exception as e:
        log("[عالي مخاطرة] HRT تعطل:", repr(e)[:120])
    # التوأم العشوائي: لكل 🧨/🟠 ثلاث أسهم عشوائية من نفس المسح بنفس شريحة التذبذب والسعر (السجل يتابعها بنفس الهدف والوقف)
    atrp = {x["t"]: float(x["last"]["atr20p"]) for x in rows_all if np.isfinite(x["last"]["atr20p"])}
    if len(atrp) >= 50:
        qs = np.quantile(list(atrp.values()), [0.2, 0.4, 0.6, 0.8])
        tier = lambda p: 0 if p < 1 else 1 if p < 3 else 2 if p < 10 else 3
        px_of = {x["t"]: float(x["last"]["close"]) for x in rows_all}
        key = {t: (int(np.searchsorted(qs, a)), tier(px_of[t])) for t, a in atrp.items()}
        for r in rows:
            if r["status"] not in ("ready", "near") or r["symbol"] not in key:
                continue
            pool = sorted(t for t, k in key.items() if k == key[r["symbol"]] and t != r["symbol"])
            pick = random.Random(f"{r['symbol']}:{r['date']}").sample(pool, min(N_TWINS, len(pool)))
            r["twins"] = [[t, _r(px_of[t])] for t in pick]
    cnt = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": n_scanned, "scored": len(rows_all),
               "seconds": round(time.time() - t0), "stocks": rows, "plan": PLAN,
               "model": {k: (M or {}).get(k) for k in ("version", "base_e10", "base_e20", "n_rows", "n_eps", "period", "summary")},
               "counts": dict(cnt), "serial_total": int(sum(1 for x in rows_all if x["last"]["n_ep252"] >= 2))}
    log(f"## رادار عالي مخاطرة: انطلق اليوم {cnt['fire']} · مرشح قوي {cnt['ready']} · مرشح {cnt['near']} · متكرر {cnt['watch']}"
        f" · من {len(rows_all)} سهم · خلص في {payload['seconds']} ثانية")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status']} · احتمال ×2 خلال 10 جلسات {r['e10']} (الأساس {r['base_e10']}) · "
            f"انفجر {r['n_year']} مرة/سنة · آخر انفجار قبل {r['days_since']} جلسة")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار عالي مخاطرة")
    ap.add_argument("--out", default=str(OUT / "highrisk.html"))
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
    (OUT / "sig_highrisk.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[عالي مخاطرة] التقرير: {out}")


if __name__ == "__main__":
    main()
