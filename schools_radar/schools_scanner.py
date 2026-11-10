#!/usr/bin/env python3
"""
🧭 رادار المدارس — كلاسيكي + وايكوف + إليوت (الكواشف في schools.py). تجريبي وصامت (بدون تيليجرام).

نفس الفلاتر العامة (0.7-10$، الفوليوم، القيمة السوقية، الفلووت، الشورت 50 ألف وأقل، بدون Warrants، امتثال ناسداك تحت 1$،
استبعاد التخفيف المتكرر، استبعاد الأسهم اللي HRT من ملاكها، علم الدولة، الأخبار بالعربي).

المقارنة الصادقة (درس الاختبار التاريخي: نسبة نجاح النمط تجي غالباً من بعد الهدف/الوقف):
- rand_p: كم مرة يوصل الهدف الأول قبل الإغلاق تحت الوقف خلال 10 جلسات لو دخلت عشوائي بنفس الهدف والوقف
  ونفس شريحة التذبذب (جدول schools_baseline.json من الاختبار، 17 ألف صفقة عشوائية).
- twins: لكل فرصة 🎯/🟡 ثلاث أسهم عشوائية من نفس المسح ونفس شريحة التذبذب ← السجل يتابعها بنفس الهدف والوقف
  (نفس اليوم ونفس السوق) عشان نقارن نتيجة الرادار الحقيقية بالعشوائي.

    python schools_radar/schools_scanner.py --out site/schools/index.html
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
import schools as SC  # noqa: E402
import flags  # noqa: E402
import holders  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
MODEL_PATH = ROOT / "schools_model.json"
BASE_PATH = ROOT / "schools_baseline.json"
N_TWINS = 3


def load_baseline() -> dict | None:
    try:
        return json.loads(BASE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def atr20(d) -> float:
    c, h, l = (d[k].to_numpy(float) for k in ("Close", "High", "Low"))
    return float(((h - l) / c)[-20:].mean() * 100)


def rand_p(B: dict | None, atr: float, entry, stop, t1) -> float | None:
    """نسبة وصول الهدف الأول قبل الإغلاق تحت الوقف خلال 10 جلسات لدخول عشوائي بنفس الهدف والوقف ونفس شريحة التذبذب."""
    if not B or not entry or not stop or not t1:
        return None
    u, s = (t1 / entry - 1) * 100, (1 - stop / entry) * 100
    if u <= 0 or s <= 0:
        return None
    T = np.asarray(B["hit_permille"][int(np.searchsorted(B["atr20_q"], atr))], float) / 1000
    iu, is_ = min(max(u - 1, 0.0), T.shape[0] - 1.0001), min(max(s - 1, 0.0), T.shape[1] - 1.0001)
    a, b = int(iu), int(is_)
    fa, fb = iu - a, is_ - b
    v = T[a, b] * (1 - fa) * (1 - fb) + T[a + 1, b] * fa * (1 - fb) + T[a, b + 1] * (1 - fa) * fb + T[a + 1, b + 1] * fa * fb
    return round(float(v), 3)


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def status_of(r: dict) -> str:
    """مبدئياً (لين يجهز نموذج الثقة): توافق مدرستين+ وجودة = قوي."""
    q = r["best"]["quality"]
    if r["confluence"] >= 2 and q >= 60:
        return "strong"
    if r["confluence"] >= 2 or q >= 70:
        return "good"
    return "watch"


def trust(row: dict, d, mkt: dict) -> dict | None:
    """ثقة رادار المدارس (لو النموذج المختبر موجود) — نفس عوامل الاختبار التاريخي."""
    try:
        m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    c, h, l, v = (d[k].to_numpy(float) for k in ("Close", "High", "Low", "Volume"))
    rs = [ix for ix, x in d["Stock Splits"].items() if x and 0 < float(x) < 1] if "Stock Splits" in d else []
    day = d.index[-1]
    try:
        import confidence
        dil, earn = confidence.sec_dates_cached(row["symbol"])
    except Exception:
        dil, earn = [], []
    import pandas as pd
    dp = [pd.Timestamp(x) for x in dil if pd.Timestamp(x) <= day]
    ep = [pd.Timestamp(x) for x in earn if pd.Timestamp(x) <= day]
    de = (day - max(ep)).days if ep else 200
    ma50 = c[-50:].mean()
    p = row["plan"]
    f = {"quality": row["quality"], "confluence": row["confluence"], "n_setups": len(row["setups"]),
         "stop_pct": (p["entry"] - p["stop"]) / p["entry"] * 100, "t1_pct": (p["t1"] / p["entry"] - 1) * 100,
         "price_log": math.log10(c[-1]), "dollar_vol": math.log10(max(1, (c[-20:] * v[-20:]).mean())),
         "atr20": float(((h - l) / c)[-20:].mean() * 100), "ret20": float(c[-1] / c[-21] - 1) * 100,
         "ret60": float(c[-1] / c[-61] - 1) * 100 if len(c) > 61 else 0.0, "drawdown": float(c[-1] / h[-252:].max() - 1) * 100,
         "vol_ratio": float(v[-5:].mean() / max(1, v[-50:].mean())), "above_ma50": float(c[-1] > ma50),
         "ma50_up": float(ma50 > c[-60:-10].mean()), "rs365": float(sum(1 for x in rs if (day - x).days <= 365)),
         "days_rs": float(min(400, (day - max(rs)).days)) if rs else 400.0,
         "dil90": float(sum(1 for x in dp if (day - x).days <= 90)), "days_earn": float(min(200, de)),
         "earn_soon": float(75 <= de <= 100), "iwm_up": mkt.get("iwm_up", 0.5), "iwm_ret20": mkt.get("iwm_ret20", 0.0),
         "breadth": mkt.get("breadth", 0.4)}
    for code in SC.SCHOOLS:
        f[f"has_{code}"] = float(any(s["setup"] == code for s in row["setups"]))
    z = m["intercept"]
    for k, w in m["coef"].items():
        x = (f.get(k, m["mean"][k]) - m["mean"][k]) / (m["std"][k] or 1)
        z += w * max(-5.0, min(5.0, x))
    pr = 1 / (1 + math.exp(-z))
    cal = m.get("calibration")
    if cal:
        pr = float(np.interp(pr, [a[0] for a in cal], [a[1] for a in cal]))
    tiers = m["tiers"]
    label = "عالية" if pr >= tiers["high"] else "متوسطة" if pr >= tiers["mid"] else "منخفضة"
    ts = (m.get("tier_stats") or {}).get(label)
    return {"p": round(pr, 3), "label": label, "similar": ts, "horizon": m.get("horizon", 10), "version": m.get("version")}


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
        import alpaca_data
        import confidence
        iwm = alpaca_data.daily(["IWM"], days=200).get("IWM")
        mkt = confidence.market_state(iwm, {t: d["Close"] for t, d in daily.items()})
    except Exception:
        mkt = {}
    B = load_baseline()
    max_sh = cfg["behavior"].get("max_short_shares")  # قرار المستخدم: فوق 50 ألف ما يترشح (None = مفتوح)
    rows, eligible = [], set()   # eligible = نفس فلاتر الرادار قبل النمط (مجال التوأم العشوائي)
    for t, d in pre.items():
        inf = infos.get(t) or {}
        mcap, flt = inf.get("marketCap"), inf.get("floatShares")
        if flt is not None and flt < 50_000:
            flt = None
        if not explicit and ((mcap and mcap > U["max_market_cap"]) or (flt and flt > U["max_float"])):
            continue
        sv, _src = data.short_of(t, ib, inf)   # المتاح للاقتراض (IB)؛ مو بالقائمة = 0؛ IB تعطل = Short Interest
        if not explicit and max_sh and sv is not None and sv > max_sh:
            continue
        if not max_sh or sv is not None:
            eligible.add(t)   # التوأم بس من الأسهم اللي ممكن تترشح (الشورت معروف وتحت الحد)
        try:
            r = SC.detect(d)
        except Exception as e:
            log(f"  [المدارس] خطأ {t}:", repr(e)[:100])
            continue
        if not r:
            continue
        c = d["Close"]
        tail = d.tail(120)
        ma20, ma50 = c.rolling(20).mean().tail(120), c.rolling(50).mean().tail(120)
        row = {"symbol": t, "name": inf.get("shortName") or inf.get("longName") or t,
               "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
               "price": _r(c.iloc[-1]), "change_pct": _r((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
               "market_cap": mcap, "float": flt, "short": sv,
               "setups": r["setups"], "schools": r["schools"], "confluence": r["confluence"], "types": r["types"],
               "plan": {k: _r(v) for k, v in r["plan"].items()}, "best": r["best"], "quality": r["best"]["quality"],
               "context": r["context"], "date": str(d.index[-1])[:10], "flags": [], "tags": [],
               "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(x) for x in tail["Open"]],
                         "h": [_r(x) for x in tail["High"]], "l": [_r(x) for x in tail["Low"]],
                         "c": [_r(x) for x in tail["Close"]], "v": [int(x) for x in tail["Volume"]],
                         "ma20": [_r(x) for x in ma20], "ma50": [_r(x) for x in ma50]}}
        row["status"] = status_of(r)
        row["trust"] = trust(row, d, mkt)
        row["atr20"] = round(atr20(d), 2)
        row["rand_p"] = rand_p(B, row["atr20"], row["plan"]["entry"], row["plan"]["stop"], row["plan"]["t1"])
        # امتثال ناسداك للأسهم تحت 1$ (نفس طبقة عبدالله — نحول المستويات مؤقتاً)
        row["status"] = {"strong": "ready", "good": "near", "watch": "radar"}[row["status"]]
        compliance_rule.apply_abdullah(row, d)
        row["status"] = {"ready": "strong", "near": "good"}.get(row["status"], "watch")
        if max_sh and sv is None and not explicit:
            row["flags"].append("الشورت غير معروف — ما يترشح لين يتأكد")
            row["status"] = "watch"
        rows.append(row)
    log(f"[المدارس] {len(rows)} سهم فيه فرصة من {len(pre)}")
    # الأخبار + SEC + التخفيف المتكرر + HRT + الأعلام
    order = {"strong": 0, "good": 1, "watch": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -r["confluence"], -r["quality"]))
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
    # التوأم العشوائي: لكل فرصة 🎯/🟡 أسهم عشوائية من نفس المسح ونفس شريحة التذبذب (السجل يتابعها بنفس الهدف والوقف)
    import random
    qs = (B or {}).get("atr20_q")
    atrs = {}
    for t in eligible:
        try:
            atrs[t] = atr20(pre[t])
        except Exception:
            pass
    atrs = {t: a for t, a in atrs.items() if math.isfinite(a)}
    if cfg["behavior"].get("exclude_hrt") and atrs:   # نفس استبعاد HRT للتوأم (التخفيف المتكرر ما ينفحص له — مكلف)
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
            if r["status"] not in ("strong", "good") or r["symbol"] not in bucket:
                continue
            pool = sorted(t for t, q in bucket.items() if q == bucket[r["symbol"]] and t != r["symbol"])
            pick = random.Random(f"{r['symbol']}:{r['date']}").sample(pool, min(N_TWINS, len(pool)))
            r["twins"] = [[t, _r(pre[t]["Close"].iloc[-1])] for t in pick]
    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": len(pre), "seconds": round(time.time() - t0),
               "stocks": rows, "excluded_dilution": excluded, "excluded_hrt": excluded_hrt,
               "model": None,
               "baseline": {k: B.get(k) for k in ("version", "period", "horizon", "n_random")} if B else None}
    try:
        mm = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        payload["model"] = {k: mm.get(k) for k in ("version", "tier_stats", "base_rate", "oos_auc", "period", "horizon")}
    except Exception:
        pass
    log(f"## رادار المدارس: قوي {cnt['strong']} · جيد {cnt['good']} · مراقبة {cnt['watch']} · خلص في {payload['seconds']} ثانية")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status']} · توافق {r['confluence']} ({'+'.join(r['schools'])}) · "
            f"{', '.join(s['setup'] for s in r['setups'])} · جودة {r['quality']}"
            + (f" · العشوائي بنفس الهدف/الوقف {r['rand_p'] * 100:.0f}%" if r.get("rand_p") is not None else "")
            + (f" · ثقة {r['trust']['p'] * 100:.0f}% ({r['trust']['label']})" if r.get("trust") else ""))
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار المدارس")
    ap.add_argument("--out", default=str(OUT / "schools.html"))
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
    (OUT / "sig_schools.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[المدارس] التقرير: {out}")


if __name__ == "__main__":
    main()
