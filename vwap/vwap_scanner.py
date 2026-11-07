"""
📈 رادار الارتداد من VWAP (تجريبي صامت: بدون تيليجرام لين نشوف نتايجه — قرار المستخدم: رادار منفصل).

نفس الاختبار التاريخي (diag/backtest_vwap.py على فرع diag-flye):
الأسهم: 0.50-5$ · فجوة افتتاح 10%+ عن إغلاق أمس · فوليوم كافي.
الإعداد: بعد الافتتاح السهم يطلع فوق VWAP بـ 3%+ ، ثم يرجع يلمس VWAP بين 9:35 و 10:30 (نيويورك).
الدخول: Limit عند VWAP لحظة اللمسة · الهدف/الوقف من الإعدادات تحت · الصفقة تقفل نفس اليوم.
الأسعار من Alpaca (شموع 5 دقائق، بتأخير ~15 دقيقة في الخطة المجانية).
النتايج تنحفظ في trackdata/vwap.json.

    python vwap/vwap_scanner.py --out site/vwap/index.html
"""
from __future__ import annotations

import argparse
import json
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
import alpaca_data as A  # noqa: E402
import sec_tools  # noqa: E402

log = data.log
NY = "America/New_York"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
STORE = ROOT / "trackdata" / "vwap.json"

MIN_GAP, MIN_RUN, MIN_VOL = 0.10, 0.03, 200_000      # الفوليوم من الافتتاح لين الحين
WIN_START, WIN_END = 5, 60                           # دقائق بعد الافتتاح (9:35 – 10:30)
TARGET, STOP = 10.0, 5.0                             # % — تتعدل بعد نتيجة الاختبار التاريخي


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def analyze(sym: str, b: pd.DataFrame, pc: float) -> dict | None:
    """b: شموع 5 دقائق لجلسة اليوم (9:30 وبعد). pc: إغلاق أمس."""
    if b.empty or not pc:
        return None
    o = float(b["Open"].iloc[0])
    gap = o / pc - 1
    vol = float(b["Volume"].sum())
    if gap < MIN_GAP or not (0.5 <= o <= 5) or vol < MIN_VOL:
        return None
    b = b.copy()
    px = b["VWAP"] if "VWAP" in b else (b["High"] + b["Low"] + b["Close"]) / 3
    b["vwap"] = (px * b["Volume"]).cumsum() / b["Volume"].cumsum().replace(0, np.nan)
    b["vprev"] = b["vwap"].shift(1)
    mins = ((b.index.hour * 60 + b.index.minute) - 570).values
    last = float(b["Close"].iloc[-1])
    vw = float(b["vwap"].iloc[-1])
    r = {"symbol": sym, "pc": round(pc, 4), "open": round(o, 4), "gap_pct": round(gap * 100, 1),
         "vol": int(vol), "last": round(last, 4), "vwap": round(vw, 4),
         "vs_vwap_pct": round((last / vw - 1) * 100, 1), "high": round(float(b["High"].max()), 4),
         "asof": b.index[-1].strftime("%H:%M")}
    ran, touch = False, None
    for i in range(1, len(b)):
        if mins[i] > WIN_END:
            break
        vp = b["vprev"].iloc[i]
        if b["High"].iloc[:i].max() >= vp * (1 + MIN_RUN):
            ran = True
        if ran and mins[i] >= WIN_START and b["Low"].iloc[i] <= vp:
            touch = i
            break
    r["ran"] = ran
    if touch is None:
        late = mins[-1] > WIN_END
        r["state"] = "missed" if late else ("waiting" if ran else "not_ran")
        return r
    entry = float(b["vprev"].iloc[touch])
    tg, st = entry * (1 + TARGET / 100), entry * (1 - STOP / 100)
    r.update({"state": "touched", "entry": round(entry, 4), "target": round(tg, 4), "stop": round(st, 4),
              "touch_at": b.index[touch].strftime("%H:%M"),
              "run_pct": round((float(b["High"].iloc[:touch].max()) / entry - 1) * 100, 1)})
    # استرجاع (للمعلومة): أول إغلاق فوق VWAP خلال 30 دقيقة بعد اللمسة
    for j in range(touch, min(touch + 7, len(b))):
        if b["Close"].iloc[j] > b["vwap"].iloc[j]:
            r["reclaim_at"] = b.index[j].strftime("%H:%M")
            break
    # النتيجة لين آخر شمعة (شمعة اللمسة: نفحص الوقف بس)
    res, pnl = None, None
    for k in range(touch, len(b)):
        if b["Low"].iloc[k] <= st:
            res, pnl = "stop", -STOP
            break
        if k > touch and b["High"].iloc[k] >= tg:
            res, pnl = "target", TARGET
            break
    if res is None:
        pnl = round((last / entry - 1) * 100, 1)
    r["result_live"], r["pnl"] = res, pnl
    return r


def stats(sigs: list[dict]) -> dict:
    def one(sub):
        done = [s for s in sub if s.get("final")]
        n = len(done)
        if not n:
            return {"n": 0}
        return {"n": n, "target": round(sum(s["result"] == "target" for s in done) / n * 100),
                "stop": round(sum(s["result"] == "stop" for s in done) / n * 100),
                "avg": round(sum(s["pnl"] for s in done) / n, 2)}
    return {"eligible": one([s for s in sigs if s.get("eligible")]), "all": one(sigs)}


def run(cfg) -> dict:
    t0 = time.time()
    store = {"signals": []}
    try:
        store = json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        pass
    sigs = store.setdefault("signals", [])
    base = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "rules": {"gap": MIN_GAP * 100, "run": MIN_RUN * 100, "target": TARGET, "stop": STOP,
                      "window": "9:35 – 10:30"}}
    if not A.available():
        log("[VWAP] مفاتيح Alpaca مو موجودة")
        return {**base, "error": "مفاتيح Alpaca مو موجودة", "today": [], "history": sigs, "stats": stats(sigs)}

    # آخر يوم فيه جلسة
    ny = now_ny()
    ref = A.bars(["SPY"], "5Min", (ny - pd.Timedelta(days=6)).tz_convert("UTC").to_pydatetime()).get("SPY")
    ref = ref[A.session_of(ref.index).values == "rth"] if ref is not None else None
    if ref is None or ref.empty:
        return {**base, "error": "ما فيه بيانات جلسة", "today": [], "history": sigs, "stats": stats(sigs)}
    D = ref.index[-1].date()
    day0 = pd.Timestamp(D).tz_localize(NY)
    closed = ny >= day0 + pd.Timedelta(hours=16, minutes=20)

    tickers = [t for t in data.get_universe(cfg) if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    rows = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            dd = A.bars(batch, "1Day", (day0 - pd.Timedelta(days=10)).tz_convert("UTC").to_pydatetime(),
                        (day0 - pd.Timedelta(hours=1)).tz_convert("UTC").to_pydatetime())
            m5 = A.bars(batch, "5Min", (day0 + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC").to_pydatetime(),
                        min(ny - pd.Timedelta(minutes=16), day0 + pd.Timedelta(hours=16)).tz_convert("UTC").to_pydatetime())
        except Exception as e:
            log("  [VWAP] Alpaca:", repr(e)[:120])
            continue
        for s, b in m5.items():
            d = dd.get(s)
            if d is None or d.empty:
                continue
            b = b[(b.index.date == D) & (A.session_of(b.index).values == "rth")]
            try:
                r = analyze(s, b, float(d["Close"].iloc[-1]))
            except Exception as e:
                log("  [VWAP]", s, repr(e)[:80])
                r = None
            if r:
                rows.append(r)
    log(f"[VWAP] يوم {D} · {'الجلسة خلصت' if closed else 'الجلسة شغالة'} · فجوات 10%+: {len(rows)} · "
        f"لمست VWAP: {sum(r['state'] == 'touched' for r in rows)}")

    # الفلاتر العامة: الشورت 100 ألف + التخفيف المتكرر (نسجل الكل ونفصل المؤهل)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    infos = data.get_info([r["symbol"] for r in rows], cfg) if rows else {}
    max_sh = cfg["behavior"].get("max_short_shares", 100000)
    for r in rows:
        inf = infos.get(r["symbol"]) or {}
        sa = (ib.get(r["symbol"]) or {}).get("available")
        sv = sa if sa is not None else inf.get("sharesShort")
        r.update({"name": inf.get("shortName") or inf.get("longName") or r["symbol"], "price": r["last"],
                  "float": inf.get("floatShares"), "short": sv, "flags": [], "news": []})
        if sv is None:
            r["flags"].append("الشورت غير معروف")
        elif sv > max_sh:
            r["flags"].append(f"الشورت {sv:,} فوق {max_sh:,}")
    touched = [r for r in rows if r["state"] in ("touched", "waiting")]
    if touched:
        sec_tools.annotate(touched, log)
        _, excl = sec_tools.exclude_repeat_diluters(list(touched), log)
        bad = {e["symbol"] for e in excl}
        for r in touched:
            if r["symbol"] in bad:
                r["flags"].append("⛔ تخفيف/طرح متكرر")
            if (r.get("dilution") or {}).get("level"):
                r["dilution_level"] = r["dilution"]["level"]
    for r in rows:
        r.pop("dilution", None)
        r["eligible"] = not r["flags"]

    # السجل: كل لمسة تنحفظ، والنتيجة تثبت بعد الإغلاق
    idx = {(s["symbol"], s["D"]): s for s in sigs}
    for r in rows:
        if r["state"] != "touched":
            continue
        k = (r["symbol"], str(D))
        s = idx.get(k)
        if s is None:
            s = {"symbol": r["symbol"], "D": str(D)}
            sigs.append(s)
            idx[k] = s
        if s.get("final"):
            continue
        s.update({f: r.get(f) for f in ("name", "gap_pct", "open", "entry", "target", "stop", "touch_at",
                                         "reclaim_at", "run_pct", "eligible", "flags", "vol")})
        s["result"] = r["result_live"] or "close"
        s["pnl"] = r["pnl"]
        s["final"] = closed or r["result_live"] is not None
    if STORE.parent.exists():
        STORE.write_text(json.dumps(store, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    order = {"touched": 0, "waiting": 1, "not_ran": 2, "missed": 3}
    rows.sort(key=lambda r: (order[r["state"]], not r["eligible"], -r["gap_pct"]))
    sigs.sort(key=lambda s: (s["D"], s.get("touch_at") or ""), reverse=True)
    st = stats(sigs)
    log(f"[VWAP] النتايج (المؤهلة): {st['eligible']} · الكل: {st['all']} · {round(time.time() - t0)}ث")
    return {**base, "day": str(D), "closed": closed, "today": rows, "history": sigs[:300], "stats": st}


def main():
    ap = argparse.ArgumentParser(description="رادار VWAP")
    ap.add_argument("--out", default=str(OUT / "vwap.html"))
    a = ap.parse_args()
    payload = run(data.load_cfg())
    html = (HERE / "template.html").read_text(encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")),
                   encoding="utf-8")
    (OUT / "sig_vwap.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[VWAP] التقرير: {out}")


if __name__ == "__main__":
    main()
