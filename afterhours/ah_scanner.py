"""
🌙 رادار الأفتر (صامت: بدون تيليجرام لين نشوف نتايجه — قرار المستخدم).

نفس الاختبار التاريخي (diag/backtest_ah_alpaca.py على فرع diag-flye):
الإشارة يوم D: إغلاق 0.50-5$ · فوليوم الجلسة 100K+ · أعلى سعر في الأفتر ≥ الإغلاق +15%
· محافظ على 70%+ من القفزة آخر الأفتر · فوليوم الأفتر 50K+.
الخطة: طلب Limit عند فواب (متوسط سعر) الأفتر في البري ← الهدف +10% / +15% · الوقف −10% (نفس الشمعة = وقف).
النتيجة تنحسب يوم التداول التالي من شموع 5 دقائق (Alpaca) وتنحفظ في trackdata/afterhours.json.

    python afterhours/ah_scanner.py --out site/afterhours/index.html
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
import sec_tools  # noqa: E402

log = data.log
NY = "America/New_York"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
STORE = ROOT / "trackdata" / "afterhours.json"

MIN_JUMP, MIN_HOLD, MIN_AH_VOL, MIN_RTH_VOL = 0.15, 0.70, 50_000, 100_000
PRICE = (0.7, 10.0)   # قرار المستخدم (نوفمبر 2026): كل الرادارات 0.7$ إلى 10$ — يتحدث من config.json تحت
T1, T2, STOP = 0.10, 0.15, 0.10
PM_VOL_RATIO = 0.07   # فوليوم البري لين التنفيذ ÷ فوليوم الأفتر — فوقه: متوسط الصفقة +2.2% بدل +1.4%


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def outcome(b: pd.DataFrame, entry: float, tg: float) -> str:
    for _, r in b.iterrows():
        if r["Low"] <= entry * (1 - STOP):
            return "fail"
        if r["High"] >= entry * (1 + tg):
            return "hit"
    return "none"


def find_today(sym: str, h: pd.DataFrame, D) -> dict | None:
    """إشارة الأفتر ليوم D من شموع 5 دقائق."""
    h = h[h.index.date == D]
    if h.empty:
        return None
    sess = A.session_of(h.index).values
    rth, ah = h[sess == "rth"], h[sess == "ah"]
    if rth.empty or ah.empty:
        return None
    close = float(rth["Close"].iloc[-1])
    rth_vol = float(rth["Volume"].sum())
    ah_hi, ah_last, ah_vol = float(ah["High"].max()), float(ah["Close"].iloc[-1]), float(ah["Volume"].sum())
    if not (PRICE[0] <= close <= PRICE[1]) or rth_vol < MIN_RTH_VOL or ah_vol < MIN_AH_VOL:
        return None
    jump = ah_hi / close - 1
    if jump < MIN_JUMP:
        return None
    hold = (ah_last - close) / (ah_hi - close)
    if hold < MIN_HOLD:
        return None
    vwap = float((ah["Close"] * ah["Volume"]).sum() / ah_vol)
    return {"symbol": sym, "D": str(D), "close": round(close, 4), "ah_high": round(ah_hi, 4),
            "ah_last": round(ah_last, 4), "ah_vwap": round(vwap, 4), "jump_pct": round(jump * 100, 1),
            "hold_pct": round(hold * 100), "ah_vol": int(ah_vol), "rth_vol": int(rth_vol),
            # الخطة: Limit عند فواب الأفتر (إذا الفواب أعلى من آخر سعر = ما فيه نزول نستناه ← الدخول = آخر سعر)
            "limit": round(min(vwap, ah_last), 4)}


def evaluate(sig: dict) -> None:
    """النتيجة يوم التداول التالي (بعد ما يخلص)."""
    D = pd.Timestamp(sig["D"]).date()
    s = pd.Timestamp(sig["D"]).tz_localize(NY) + pd.Timedelta(days=1)
    e = min(now_ny() - pd.Timedelta(minutes=16), s + pd.Timedelta(days=5))
    if e <= s:
        return  # اليوم التالي ما بدا (كان يطلب فترة بالمستقبل ← Alpaca يرد 403)
    b = A.bars([sig["symbol"]], "5Min", s.tz_convert("UTC").to_pydatetime(), e.tz_convert("UTC").to_pydatetime()).get(sig["symbol"])
    if b is None or b.empty:
        return
    b = b[b.index.date > D]
    if b.empty:
        return
    N = b.index[0].date()
    if now_ny() < pd.Timestamp(N).tz_localize(NY) + pd.Timedelta(hours=16, minutes=20):
        return  # اليوم التالي ما خلص
    x = b[b.index.date == N]
    sess = A.session_of(x.index).values
    pm, rth = x[sess == "pm"], x[sess == "rth"]
    seq = pd.concat([pm, rth])
    if seq.empty:
        return
    lim = sig["limit"]
    sig["N"] = str(N)
    sig["pm_high_pct"] = round((float(pm["High"].max()) / sig["ah_last"] - 1) * 100, 1) if not pm.empty else None
    sig["close_next_pct"] = round((float(rth["Close"].iloc[-1]) / sig["ah_last"] - 1) * 100, 1) if not rth.empty else None
    if sig["ah_vwap"] < sig["ah_last"] * 0.995:
        dip = pm[pm["Low"] <= lim]
        if dip.empty:
            sig["result"] = "nofill"
            return
        after = seq.loc[dip.index[0]:]
    else:
        after = seq
    r1, r2 = outcome(after, lim, T1), outcome(after, lim, T2)
    sig["r10"], sig["r15"] = r1, r2
    sig["result"] = "t2" if r2 == "hit" else "t1" if r1 == "hit" else "stop" if r1 == "fail" else "flat"
    if sig["result"] == "flat" and not rth.empty:
        sig["flat_pct"] = round((float(rth["Close"].iloc[-1]) / lim - 1) * 100, 1)


def stats(sigs: list[dict]) -> dict:
    def one(sub):
        done = [s for s in sub if s.get("result") in ("t1", "t2", "stop", "flat")]
        n = len(done)
        if not n:
            return {"n": 0}
        return {"n": n, "nofill": sum(s.get("result") == "nofill" for s in sub),
                "t1": round(sum(s["result"] in ("t1", "t2") for s in done) / n * 100),
                "t2": round(sum(s["result"] == "t2" for s in done) / n * 100),
                "stop": round(sum(s["result"] == "stop" for s in done) / n * 100)}
    ok = [s for s in sigs if s.get("eligible")]
    return {"eligible": one(ok), "all": one(sigs)}


def last_session_day(start):
    """آخر يوم فيه جلسة عادية (من SPY)، وهل الأفتر حقه خلص (بعد 8:15 مساء نيويورك)."""
    ref = A.bars(["SPY"], "5Min", start).get("SPY")
    if ref is None or ref.empty:
        return None, False
    ref = ref[A.session_of(ref.index).values == "rth"]
    if ref.empty:
        return None, False
    D = ref.index[-1].date()
    done = now_ny() >= pd.Timestamp(D).tz_localize(NY) + pd.Timedelta(hours=20, minutes=15)
    return D, done


def run(cfg) -> dict:
    global PRICE
    PRICE = (cfg["universe"]["min_price"], cfg["universe"]["max_price"])
    t0 = time.time()
    store = {"signals": []}
    try:
        store = json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        pass
    sigs = store.setdefault("signals", [])
    if not A.available():
        log("[الأفتر] مفاتيح Alpaca مو موجودة — ما نقدر نشغل الرادار")
        return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "error": "مفاتيح Alpaca مو موجودة",
                "today": [], "history": sigs, "stats": stats(sigs)}

    # 1) نتايج الإشارات السابقة
    for s in sigs:
        if not s.get("result"):
            try:
                evaluate(s)
            except Exception as e:
                log("  [الأفتر] تقييم", s["symbol"], repr(e)[:80])

    # 2) إشارات آخر أفتر
    tickers = [t for t in data.get_universe(cfg) if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    start = (now_ny() - pd.Timedelta(days=5)).tz_convert("UTC").to_pydatetime()
    found = []
    D, done = last_session_day(start)
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        try:
            b = A.bars(batch, "5Min", start)
        except Exception as e:
            log("  [الأفتر] Alpaca:", repr(e)[:120])
            continue
        for sym, h in b.items():
            try:
                x = find_today(sym, h, D) if D else None
            except Exception:
                x = None
            if x:
                found.append(x)
    log(f"[الأفتر] يوم {D} · الأفتر {'خلص' if done else 'ما خلص (مبدئي)'} · قفزات محافظة: {len(found)}")

    # 3) الفلاتر العامة: الشورت (حد config، قرار المستخدم 50 ألف) + التخفيف المتكرر (نسجل الكل ونفصل المؤهل)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    infos = data.get_info([f["symbol"] for f in found], cfg) if found else {}
    max_sh = cfg["behavior"].get("max_short_shares")  # 50 ألف (None = مفتوح)
    for f in found:
        inf = infos.get(f["symbol"]) or {}
        sv, src = data.short_of(f["symbol"], ib, inf)   # مو بقائمة IB = 0 متاح (تعريف فيصل)
        f.update({"name": inf.get("shortName") or inf.get("longName") or f["symbol"], "price": f["ah_last"],
                  "float": inf.get("floatShares"), "short": sv, "short_src": src, "flags": [], "news": [],
                  "country": inf.get("country")})
        if max_sh and sv is None:
            f["flags"].append("الشورت غير معروف")
        elif max_sh and sv > max_sh:
            f["flags"].append(f"الشورت {sv:,} فوق {max_sh:,}")
    if found:
        import flags
        import holders
        flags.annotate(found)
        if cfg["behavior"].get("exclude_hrt"):
            try:
                holders.annotate(found, log)
                for f in found:
                    if f.get("hrt"):
                        f["flags"].append(f"⛔ HRT من الملاك ({f['hrt']['period']})")
            except Exception as e:
                log("فحص HRT تعطل:", repr(e)[:150])
        sec_tools.annotate(found, log)
        kept, excl = sec_tools.exclude_repeat_diluters(found, log)
        bad = {e["symbol"] for e in excl}
        for f in found:
            if f["symbol"] in bad:
                f["flags"].append("⛔ تخفيف/طرح متكرر")
            dl = (f.get("dilution") or {}).get("level")
            if dl:
                f["dilution_level"] = dl
            f.pop("dilution", None)
            f["eligible"] = not f["flags"]
            f["t1"], f["t2"], f["stop"] = (round(f["limit"] * (1 + T1), 4), round(f["limit"] * (1 + T2), 4),
                                           round(f["limit"] * (1 - STOP), 4))
            # شرط فوليوم البري (اختبار سنة): ادخل بس إذا فوليوم البري لين لحظة الشراء ≥ 7% من فوليوم الأفتر
            f["pm_min_vol"] = int(f["ah_vol"] * PM_VOL_RATIO)

    # 4) نسجل بس لما الأفتر يخلص (ما نكرر نفس السهم لنفس اليوم)
    if done and found:
        have = {(s["symbol"], s["D"]) for s in sigs}
        new = [f for f in found if (f["symbol"], f["D"]) not in have]
        sigs.extend(new)
        log(f"[الأفتر] انسجلت {len(new)} إشارة جديدة (صامت، بدون تيليجرام)")
    if STORE.parent.exists():
        STORE.write_text(json.dumps(store, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    found.sort(key=lambda f: (not f["eligible"], -f["hold_pct"], -f["jump_pct"]))
    sigs.sort(key=lambda s: s["D"], reverse=True)
    st = stats(sigs)
    log(f"[الأفتر] النتايج (المؤهلة): {st['eligible']} · الكل: {st['all']} · {round(time.time() - t0)}ث")
    last_final = max((s["D"] for s in sigs), default=None)
    final_list = sorted([s for s in sigs if s["D"] == last_final],
                        key=lambda f: (not f.get("eligible"), -f["hold_pct"], -f["jump_pct"]))
    return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "day": str(D) if D else None,
            "final": done, "today": found if (found and not done) else [],
            "last_final_day": last_final, "last_final": final_list, "history": sigs[:300], "stats": st,
            "rules": {"jump": MIN_JUMP * 100, "hold": MIN_HOLD * 100, "ah_vol": MIN_AH_VOL,
                      "t1": T1 * 100, "t2": T2 * 100, "stop": STOP * 100}}


def main():
    ap = argparse.ArgumentParser(description="رادار الأفتر")
    ap.add_argument("--out", default=str(OUT / "afterhours.html"))
    a = ap.parse_args()
    payload = run(data.load_cfg())
    html = (HERE / "template.html").read_text(encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")),
                   encoding="utf-8")
    (OUT / "sig_afterhours.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[الأفتر] التقرير: {out}")


if __name__ == "__main__":
    main()
