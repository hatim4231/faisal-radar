#!/usr/bin/env python3
"""
📅 قائمة الأسبوع — طلب أبوفرح (2026-11-28): «ابنه وارسله» بعد دراسة diag/movers_weekly.py.
كل نهاية أسبوع (بعد إغلاق آخر جلسة بالأسبوع) نرتب كل الأسهم بنموذج الانفجار (highrisk) ونطلع أعلى 10 ← تيليجرام مرة وحدة بالأسبوع.
الدراسة (55 أسبوع): أعلى 10 انفجرت ×2 خلال الأسبوع الجاي 12.4% ±3.5 مقابل 0.5% للسوق (×25) · لمست +50% 20% ·
لكن الشراء الآلي (الاثنين ← الجمعة، أو Limit +50%/+100%) سالب بعد التكاليف، والسهم الوسطي ينزل −20% خلال الأسبوع ← قائمة مراقبة:
حجم صغير جداً + البيع نفس يوم القفزة (بكرة يفتح −8% وبعد 5 جلسات −21% بالوسيط).
يقرا output/hr_all.json (يكتبه رادار عالي مخاطرة بنفس التشغيل). السجل: trackdata/weekly_log.json (القائمة + وش صار بعد 5 جلسات).

    python weekly_radar/weekly_scanner.py --out site/weekly/index.html
"""
from __future__ import annotations

import argparse
import html as H
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import scanner as data  # noqa: E402

log = data.log
OUT = ROOT / "output"
LOG = Path(os.environ.get("TRACK_DIR", ROOT / "trackdata")) / "weekly_log.json"
SITE = "https://hatim4231.github.io/faisal-radar/weekly/"
TOP = 10
PRICE = (0.2, 20.0)
MIN_DV = 30_000
STUDY = {"weeks": 55, "top10_ex": 12.4, "base_ex": 0.5, "top10_hit50": 20.0, "median_low": -20, "next_open": -8, "after5": -21}


def now_ny():
    return pd.Timestamp.now(tz="America/New_York")


def week_due(sig_date: str) -> bool:
    """القائمة تنبني بعد ما تخلص آخر جلسة بالأسبوع: الجمعة بعد 4:15 العصر نيويورك، أو السبت/الأحد."""
    n = now_ny()
    if n.weekday() >= 5:
        return True
    return n.weekday() == 4 and n.hour * 60 + n.minute >= 16 * 60 + 15 and sig_date == str(n.date())


def build(cfg, hr: dict) -> dict:
    rows = [r for r in hr["rows"] if not r["stale"] and r.get("p") is not None and r.get("praw") is not None
            and PRICE[0] <= r["praw"] <= PRICE[1] and (r.get("dv20") or 0) >= MIN_DV and not r.get("short_hist")]
    rows.sort(key=lambda r: -r["p"])
    picks = rows[:TOP]
    syms = [r["t"] for r in picks]
    infos = data.get_info(syms, cfg) if syms else {}
    import delist as DL
    import alpaca_data as A
    daily = {}
    try:
        daily = A.daily(syms, days=420)
    except Exception as e:
        log("  [الأسبوع] اليومي:", repr(e)[:100])
    out = []
    for i, r in enumerate(picks, 1):
        inf = infos.get(r["t"]) or {}
        d = daily.get(r["t"])
        dl = None
        if d is not None:
            try:
                dl = DL.assess(d)
            except Exception:
                dl = None
        out.append({"rank": i, "symbol": r["t"], "name": inf.get("shortName") or inf.get("longName") or r.get("name") or r["t"],
                    "country": inf.get("country"), "price": r["close"], "praw": r["praw"], "p": r["p"], "bleed": r.get("pb"),
                    "n_ep252": int(r.get("n_ep252") or 0), "days_ep": r.get("days_ep"), "days_rs": r.get("days_rs"),
                    "dd252": r.get("dd252"), "dv20": r.get("dv20"), "float": inf.get("floatShares"), "delist": dl})
    try:
        import flags
        flags.annotate(out)
    except Exception:
        pass
    try:
        import sec_tools
        sec_tools.annotate(out, log)
        for x in out:
            x["dil"] = (x.pop("dilution", None) or {}).get("level")
    except Exception as e:
        log("  [الأسبوع] SEC:", repr(e)[:100])
    try:   # تحليل فيصل + شارت (للعرض بالصفحة بس)
        import stock_detail
        for x in out:
            d = daily.get(x["symbol"])
            if d is not None:
                fa = stock_detail.build(x["symbol"], d, infos.get(x["symbol"]) or {}, cfg, flag=x.get("flag"), country_ar=x.get("country_ar"))
                if fa:
                    x["faisal"] = fa
    except Exception as e:
        log("  [الأسبوع] فيصل:", repr(e)[:100])
    return {"key": hr["sig_date"], "made_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_eligible": len(rows), "picks": out}


def evaluate(weeks: list[dict]) -> None:
    """بعد 5 جلسات من القائمة: انفجر ×2؟ أعلى سعر من افتتاح الاثنين؟ لمس +50%/+100%؟ إغلاق الجمعة؟"""
    import alpaca_data as A
    import highrisk as HR
    for w in weeks:
        if w.get("result"):
            continue
        key = pd.Timestamp(w["key"])
        syms = [p["symbol"] for p in w["picks"]]
        try:
            raw = A.daily(syms, days=max(30, (pd.Timestamp.now() - key).days + 15))
        except Exception:
            continue
        res, done = {}, True
        for s in syms:
            d = raw.get(s)
            if d is None:
                continue
            idx = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index, "tz", None) else pd.to_datetime(d.index)
            after = np.flatnonzero(idx.normalize() > key)
            if len(after) < 5:
                done = False
                break
            a, b = after[0], after[0] + 5
            o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
            prev = c[a - 1]
            dvd = v * (h + l + c) / 3
            ex = any(h[k] >= 2 * c[k - 1] and dvd[k] >= HR.EX_MIN_DV for k in range(a, b))
            mo = o[a]
            res[s] = {"ex": bool(ex), "hi_o": round((h[a:b].max() / mo - 1) * 100, 1), "lo_o": round((l[a:b].min() / mo - 1) * 100, 1),
                      "cl_o": round((c[b - 1] / mo - 1) * 100, 1), "hit50": bool(h[a:b].max() >= mo * 1.5), "hit100": bool(h[a:b].max() >= mo * 2),
                      "hi_prev": round((h[a:b].max() / prev - 1) * 100, 1)}
        if done and res:
            w["result"] = res
            w["summary"] = {"n": len(res), "ex": sum(r["ex"] for r in res.values()), "hit50": sum(r["hit50"] for r in res.values()),
                            "hit100": sum(r["hit100"] for r in res.values())}


def tg_list(w: dict) -> str:
    L = [f"📅 <b>قائمة الأسبوع</b> — أعلى {len(w['picks'])} احتمال طيران ×2 (بعد إغلاق {w['key']})"]
    for x in w["picks"]:
        warn = []
        if (x.get("bleed") or 0) >= 0.4:
            warn.append(f"🩸{round(x['bleed'] * 100)}%")
        if (x.get("delist") or {}).get("level") == 2:
            warn.append("⛔شطب")
        if x.get("dil") == "high":
            warn.append("⛔طرح")
        fl = x.get("flag") or ""
        L.append(f"{x['rank']}) {fl} <b>{H.escape(x['symbol'])}</b> ${x['price']} · طار {x['n_ep252']}× بسنة" + (" · " + " ".join(warn) if warn else ""))
    L.append(f"\n📊 تاريخياً (55 أسبوع): أعلى 10 انفجر منها ×2 بالأسبوع <b>{STUDY['top10_ex']}%</b> مقابل {STUDY['base_ex']}% للسوق، ولمس +50% {STUDY['top10_hit50']}%.")
    L.append("⚠ الشراء الآلي سالب: السهم الوسطي ينزل ~−20% خلال الأسبوع. <b>حجم صغير جداً + البيع نفس يوم القفزة</b> (بكرة يفتح −8% بالوسيط).")
    L.append(f"{SITE}\nليس توصية شراء أو بيع.")
    return "\n".join(L)


def tg_result(w: dict) -> str:
    S, R = w["summary"], w["result"]
    best = sorted(R.items(), key=lambda kv: -kv[1]["hi_o"])[:3]
    return (f"📒 <b>نتيجة قائمة أسبوع {w['key']}</b>: انفجر ×2 {S['ex']} من {S['n']} · لمس +50% من افتتاح الاثنين {S['hit50']} · +100% {S['hit100']}\n"
            + "الأعلى: " + " · ".join(f"{s} +{r['hi_o']}% (إغلاق الجمعة {r['cl_o']:+}%)" for s, r in best) + f"\n{SITE}")


def main():
    ap = argparse.ArgumentParser(description="قائمة الأسبوع")
    ap.add_argument("--out", default=str(OUT / "weekly.html"))
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--force", action="store_true", help="ابنِ القائمة الحين حتى لو مو نهاية الأسبوع")
    a = ap.parse_args()
    t0 = time.time()
    cfg = data.load_cfg()
    try:
        store = json.loads(LOG.read_text(encoding="utf-8"))
    except Exception:
        store = {"weeks": []}
    weeks = store.setdefault("weeks", [])
    msgs = []
    try:
        hr = json.loads((OUT / "hr_all.json").read_text(encoding="utf-8"))
    except Exception as e:
        hr = None
        log("[الأسبوع] ما فيه hr_all.json (رادار عالي مخاطرة ما اشتغل؟):", repr(e)[:100])
    if hr and (a.force or week_due(hr["sig_date"])) and not any(w["key"] == hr["sig_date"] for w in weeks):
        w = build(cfg, hr)
        weeks.append(w)
        log(f"[الأسبوع] قائمة جديدة {w['key']}: " + ", ".join(f"{x['symbol']}({x['p']:.3f})" for x in w["picks"]))
        msgs.append(tg_list(w))
    try:
        before = {w["key"] for w in weeks if w.get("result")}
        evaluate(weeks)
        for w in weeks:
            if w.get("result") and w["key"] not in before:
                msgs.append(tg_result(w))
                log(f"[الأسبوع] نتيجة {w['key']}: {w['summary']}")
    except Exception as e:
        log("[الأسبوع] التقييم:", repr(e)[:120])
    weeks.sort(key=lambda w: w["key"])
    if LOG.parent.exists():
        slim = [{k: v for k, v in w.items()} for w in weeks]
        for w in slim[:-1]:            # تحليل فيصل يبقى لآخر قائمة بس (عشان الصفحة) — الحجم
            w["picks"] = [{k: v for k, v in x.items() if k != "faisal"} for x in w["picks"]]
        LOG.write_text(json.dumps({"weeks": slim[-104:]}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if msgs and not a.no_telegram:
        import tracker
        tracker.telegram(msgs)
    done = [w for w in weeks if w.get("result")]
    tot = {"weeks": len(done), "n": sum(w["summary"]["n"] for w in done), "ex": sum(w["summary"]["ex"] for w in done),
           "hit50": sum(w["summary"]["hit50"] for w in done)}
    cur = weeks[-1] if weeks else None
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "current": cur, "history": list(reversed(weeks[:-1]))[:52] if weeks else [],
               "totals": tot, "study": STUDY, "seconds": round(time.time() - t0)}
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    for w in payload["history"]:
        w["picks"] = [{k: v for k, v in x.items() if k != "faisal"} for x in w["picks"]]
    import stock_detail
    html = stock_detail.inject((HERE / "template.html").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")), encoding="utf-8")
    log(f"[الأسبوع] التقرير: {out} · رسائل تيليجرام {len(msgs)}")


if __name__ == "__main__":
    main()
