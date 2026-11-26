#!/usr/bin/env python3
"""
🕘 رادار الماركت — نظرية «سهم الماركت» لعبدالإله (@z_v11v) بمثال فيصل على APUS. طلب أبوفرح (2026-11-26): «شوف البوستات وسوي رادار اسمه رادار الماركت».
القواعد كلها في market.py (مشتركة مع الدراسة diag/market_study.py). صامت: بدون تيليجرام ولا سجل التوصيات العام.

المراحل (بتوقيت السعودية = نيويورك + 8):
  الأفتر أمس ← قفزة ≥ 15% · قاعدة 1 (جنى وهبط) تلغي.
  البري ← قاعدة 2 (كسر دعم الأفتر) تلغي · 3 عصراً (7:00 نيويورك) حافظ على أدنى شمعة ← 🟡 سهم الماركت.
  الماركت 5:30–6:00 م (9:30–10:00) ← اختبار دعم البري: ✅ ثبت = دخول بسعر 6:00 م، هدف +20%، وقف إغلاق تحت دعم الأفتر · ❌ فشل.
سجل الصدق: كل ✅ ينحفظ في trackdata/market_log.json ونقيّمه بعد يومين (الهدف قبل الوقف) مقابل كل قفزات الأفتر بنفس اليوم.

    python market_radar/market_scanner.py --out site/market/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import scanner as data  # noqa: E402
import alpaca_data as A  # noqa: E402
import market as M  # noqa: E402

log = data.log
OUT = ROOT / "output"
LOG = Path(os.environ.get("TRACK_DIR", ROOT / "trackdata")) / "market_log.json"
ORDER = {"held": 0, "testing": 1, "market": 2, "wait_pm": 3, "fail": 4, "cancel_pm": 5, "cancel_ah": 6}
LABEL = {"held": "✅ ثبت دعم البري — دخول", "testing": "⏳ يختبر دعم البري الحين", "market": "🟡 سهم الماركت — ينتظر الافتتاح",
         "wait_pm": "⌛ ينتظر البري / 3 عصراً", "fail": "❌ فشل بالماركت", "cancel_pm": "❌ مُلغى — البري", "cancel_ah": "❌ مُلغى — جنى وهبط بالأفتر"}


def now_ny():
    return pd.Timestamp.now(tz=M.NY)


def session_days():
    """D = آخر جلسة قبل اليوم N. N = اليوم لو يوم تداول وبدأ البري (4:00 نيويورك)، وإلا None (ننتظر الجلسة الجاية)."""
    ref = A.bars(["SPY"], "5Min", (now_ny() - pd.Timedelta(days=7)).tz_convert("UTC").to_pydatetime()).get("SPY")
    ref.index = ref.index.tz_convert(M.NY)
    m = ref.index.hour * 60 + ref.index.minute
    rth_days = sorted(set(ref.index[(m >= 570) & (m < 960)].date))
    today, n = now_ny().date(), now_ny()
    has_today = any(d == today for d in set(ref.index.date))
    if has_today and n.hour >= 4:
        prev = [d for d in rth_days if d < today]
        return (prev[-1] if prev else None), today
    return (rth_days[-1] if rth_days else None), None


def mini_chart(b: pd.DataFrame) -> dict:
    x = b.tail(400)
    return {"t": [i.strftime("%m-%d %H:%M") for i in x.index], "c": [round(float(v), 4) for v in x["Close"]],
            "h": [round(float(v), 4) for v in x["High"]], "l": [round(float(v), 4) for v in x["Low"]]}


def evaluate_log(entries: list[dict]) -> None:
    """سجل الصدق: لكل ✅ — الهدف +20% قبل الإغلاق تحت دعم الأفتر خلال يوم الدخول واليوم اللي بعده."""
    todo = [e for e in entries if not e.get("res")]
    for e in todo:
        N = pd.Timestamp(e["N"]).tz_localize(M.NY)
        if now_ny() < N + pd.Timedelta(days=1, hours=16, minutes=20):
            continue
        try:
            b = A.bars([e["symbol"]], "5Min", (N + pd.Timedelta(hours=10)).tz_convert("UTC").to_pydatetime(),
                       min(N + pd.Timedelta(days=5), now_ny() - pd.Timedelta(minutes=20)).tz_convert("UTC").to_pydatetime()).get(e["symbol"])
        except Exception:
            continue
        if b is None or b.empty:
            continue
        b.index = b.index.tz_convert(M.NY)
        days = sorted(set(b.index.date))[:2]
        if len(days) < 2 or (days[1] == now_ny().date() and now_ny() < pd.Timestamp(days[1]).tz_localize(M.NY) + pd.Timedelta(hours=16, minutes=20)):
            continue
        rows = []
        for d in days:
            _, rth, _ = M.sessions(b[b.index.date == d])
            rows.append(rth)
        seq = pd.concat(rows)
        seq = seq[seq.index >= N + pd.Timedelta(hours=10)]
        r = M.result_after({"ah_support": e["stop"]}, {"entry": e["entry"]}, seq)
        e["res"], e["ret_pct"] = r["res"], round(r["ret"] * 100, 1)
        e["max_pct"] = round((float(seq["High"].max()) / e["entry"] - 1) * 100, 1) if len(seq) else None


def run(cfg) -> dict:
    t0 = time.time()
    if not A.available():
        return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "error": "مفاتيح Alpaca مو موجودة", "stocks": []}
    D, N = session_days()
    now = now_ny() - pd.Timedelta(minutes=16)          # الخطة المجانية متأخرة 15 دقيقة
    tickers = [t for t in data.get_universe(cfg) if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    start = (pd.Timestamp(D).tz_localize(M.NY) + pd.Timedelta(hours=9)).tz_convert("UTC").to_pydatetime()
    found = []
    for i in range(0, len(tickers), 50):
        try:
            got = A.bars(tickers[i:i + 50], "5Min", start)
        except Exception as e:
            log("  [الماركت] Alpaca:", repr(e)[:120])
            continue
        for sym, b in got.items():
            if b is None or b.empty:
                continue
            b = b.copy()
            b.index = b.index.tz_convert(M.NY)
            ev = M.ah_event(b[b.index.date == D])
            if not ev:
                continue
            if N is not None:
                cls = M.classify(ev, b[b.index.date == N], now)
            else:
                cls = M.classify(ev, b.iloc[0:0]) if ev["faded"] else {"status": "wait_pm", "why": ["ننتظر بري الجلسة الجاية"],
                                                                         "pm_support": None, "entry": None, "target": None, "stop": ev["ah_support"]}
            px = float(b["Close"].iloc[-1])
            found.append({"symbol": sym, "D": str(D), "N": str(N) if N else None, "price": round(px, 4), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()},
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cls.items()},
                          "chart": mini_chart(b)})
    log(f"[الماركت] أفتر {D} ← جلسة {N or 'الجاية'} · قفزات أفتر ≥{int(M.MIN_JUMP*100)}%: {len(found)}")
    # إثراء خفيف: الاسم، العلم، الشورت، SEC (ما يشيل — تنبيه بس)
    if found:
        infos = data.get_info([f["symbol"] for f in found], cfg)
        try:
            ib = data.load_ib_short()
        except Exception:
            ib = {}
        for f in found:
            inf = infos.get(f["symbol"]) or {}
            f["name"] = inf.get("shortName") or inf.get("longName") or f["symbol"]
            f["float"] = inf.get("floatShares")
            f["country"] = inf.get("country")
            f["short"], f["short_src"] = data.short_of(f["symbol"], ib, inf)
        try:
            import flags
            import sec_tools
            flags.annotate(found)
            live = [f for f in found if f["status"] in ("held", "testing", "market", "wait_pm")]
            sec_tools.annotate(live, log)
            for f in live:
                if (f.get("dilution") or {}).get("level") == "high":
                    f.setdefault("warn", []).append("⛔ ملف طرح/تخفيف SEC خلال 30 يوم")
        except Exception as e:
            log("  [الماركت] إثراء:", repr(e)[:120])
        try:   # تحليل فيصل الكامل للأسهم الحيّة (نفس باقي الرادارات)
            import stock_detail
            live = [f for f in found if f["status"] in ("held", "testing", "market")]
            daily = data.download_daily([f["symbol"] for f in live], cfg) if live else {}
            for f in live:
                d = daily.get(f["symbol"])
                if d is not None:
                    fa = stock_detail.build(f["symbol"], d, infos.get(f["symbol"]) or {}, cfg, dilution=f.get("dilution"),
                                            flag=f.get("flag"), country_ar=f.get("country_ar"))
                    if fa:
                        f["faisal"] = fa
        except Exception as e:
            log("  [الماركت] فيصل:", repr(e)[:120])
    for f in found:
        f["label"] = LABEL.get(f["status"], f["status"])
    found.sort(key=lambda f: (ORDER.get(f["status"], 9), -f["jump_pct"]))
    # سجل الصدق
    entries = []
    try:
        entries = json.loads(LOG.read_text(encoding="utf-8")).get("entries", [])
    except Exception:
        pass
    have = {(e["symbol"], e["N"]) for e in entries}
    for f in found:
        if f["status"] == "held" and (f["symbol"], f["N"]) not in have:
            entries.append({"symbol": f["symbol"], "D": f["D"], "N": f["N"], "entry": f["entry"], "target": f["target"],
                            "stop": f["stop"], "tested": f.get("tested"), "jump_pct": f["jump_pct"]})
    try:
        evaluate_log(entries)
    except Exception as e:
        log("  [الماركت] تقييم السجل:", repr(e)[:120])
    if LOG.parent.exists():
        LOG.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=1), encoding="utf-8")
    done = [e for e in entries if e.get("res")]
    stats = {"n": len(done), "target": sum(e["res"] == "target" for e in done), "stop": sum(e["res"] == "stop" for e in done),
             "avg_ret": round(sum(e["ret_pct"] for e in done) / len(done), 1) if done else None}
    from collections import Counter
    c = Counter(f["status"] for f in found)
    log(f"## رادار الماركت: ✅ {c['held']} · ⏳ {c['testing']} · 🟡 {c['market']} · ⌛ {c['wait_pm']} · ❌ {c['fail'] + c['cancel_pm'] + c['cancel_ah']} · "
        f"السجل {stats['n']} (هدف {stats['target']} · وقف {stats['stop']}) · {round(time.time() - t0)}ث")
    for f in found[:15]:
        log(f"  {f['symbol']} {f['label']} · قفزة +{f['jump_pct']}% · دعم الأفتر {f['ah_support']} · دعم البري {f.get('pm_support')} · دخول {f.get('entry')}")
    return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "ny_time": now_ny().strftime("%Y-%m-%d %H:%M"),
            "D": str(D), "N": str(N) if N else None, "stocks": found, "log": sorted(entries, key=lambda e: e["N"], reverse=True)[:200],
            "stats": stats, "rules": {"jump": M.MIN_JUMP * 100, "giveback": M.MAX_GIVEBACK * 100, "target": M.TARGET * 100},
            "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser(description="رادار الماركت")
    ap.add_argument("--out", default=str(OUT / "market.html"))
    a = ap.parse_args()
    cfg = data.load_cfg()
    payload = run(cfg)
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    OUT.mkdir(exist_ok=True)
    (OUT / "sig_market.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    import stock_detail
    html = stock_detail.inject((HERE / "template.html").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")), encoding="utf-8")
    log(f"[الماركت] التقرير: {out}")


if __name__ == "__main__":
    main()
