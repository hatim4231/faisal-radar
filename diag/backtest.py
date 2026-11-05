"""
اختبار تاريخي (تشخيص فقط — ما يغير الرادار):
يشغّل رادار فيصل على آخر ~6 شهور كل 3 جلسات، ويقارن:
  BASE_READY  = ✅ مطابق (طريقة فيصل الحالية)
  BASE_NEAR   = 🟡 قريب
  BOX_READY   = مطابق بعد اقتراح (ب): سقف التماسك ما يُعتبر مقاومة
  CONSOL      = اقتراح (أ): مسار "تماسك جاهز" (3+ اختبارات، 8+ جلسات، مدى ضيق، RSI <= حد)
  RANDOM      = أي سهم في أي يوم (المرجع)
النتيجة خلال 10 جلسات: هل لمس +20% قبل ما يكسر الوقف؟
"""
import sys, time, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "abdullah"))
import numpy as np, pandas as pd
import scanner, faisal_rules as fr, abdullah_rules as ar

STEP, LOOKBACK_DAYS, HORIZON, TARGET = 3, 126, 10, 0.20
RSI_CAPS = [40, 45, 50]


def consol_metrics(df):
    n = len(df)
    c = ar.consolidation(df, max(0, n - 60), 20)
    if not c:
        return None
    avg10 = float(df["Volume"].tail(10).mean())
    avg50 = float(df["Volume"].tail(60).head(50).mean()) or 1
    return {"len": c["len"], "touches": c["touches"], "range": c["range_pct"], "support": c["support"],
            "top": c["top"], "dry": avg10 < avg50 * 0.7}


def outcome(df_future, entry, stop):
    for _, r in df_future.iterrows():
        if r["Low"] < stop:
            return "fail"
        if r["High"] >= entry * (1 + TARGET):
            return "hit"
    return "none"


def run_one(args):
    t, d, info, cfg = args
    out = []
    n = len(d)
    last_sig = {}
    for end in range(n - LOOKBACK_DAYS - HORIZON, n - HORIZON, STEP):
        if end < 260:
            continue
        hist = d.iloc[:end]
        price = float(hist["Close"].iloc[-1])
        if not (1 <= price <= 5) or float(hist["Volume"].tail(30).mean()) < 100000:
            continue
        fut = d.iloc[end:end + HORIZON]
        try:
            a = fr.analyze(hist, info, cfg)
        except Exception:
            continue
        if not a:
            continue
        date = str(hist.index[-1].date())
        p = a["plan"]
        stop_base = p["stop"] or price * 0.9
        res_rand = outcome(fut, price, price * 0.9)
        out.append(("RANDOM", t, date, res_rand))
        st = a["irtikaz"]["status"]
        cm = consol_metrics(hist)
        # (ب) سقف التماسك ليس مقاومة
        box_score = a["irtikaz"]["score"]
        up = next(c for c in a["irtikaz"]["checks"] if c["key"] == "upside")
        if cm and p["t1"] and p["t1"] <= cm["top"] * 1.03 and not up["ok"]:
            nxt = [r for r in p["resistances"] if r > cm["top"] * 1.03]
            if nxt and (nxt[0] / price - 1) * 100 >= 20:
                box_score = min(100, box_score + (10 - up["got"]))
        core_ok = all(c["ok"] for c in a["irtikaz"]["checks"] if c["key"] in ("support", "hold", "rsi"))
        sigs = []
        if st == "ready":
            sigs.append("BASE_READY")
        if st == "near":
            sigs.append("BASE_NEAR")
        if box_score >= 75 and core_ok:
            sigs.append("BOX_READY")
        if cm and cm["touches"] >= 3 and cm["len"] >= 8 and cm["range"] <= 20 and price <= cm["support"] * 1.10:
            for cap in RSI_CAPS:
                if a["rsi"] is not None and a["rsi"] <= cap:
                    sigs.append(f"CONSOL_RSI{cap}" + ("_DRY" if cm["dry"] else ""))
                    if cm["dry"]:
                        sigs.append(f"CONSOL_RSI{cap}")
        for s in set(sigs):
            if last_sig.get(s, -99) >= end - HORIZON:
                continue  # نفس السهم ما يتكرر خلال 10 جلسات
            last_sig[s] = end
            stop = (cm["support"] * 0.97) if s.startswith("CONSOL") and cm else stop_base
            out.append((s, t, date, outcome(fut, price, stop)))
    return out


def main():
    cfg = scanner.load_cfg()
    cfg["scan"]["history_period"] = "2y"
    tickers = scanner.get_universe(cfg)
    tickers = [x for x in tickers if not (len(x) == 5 and x[-1] in "WRU")]
    daily = scanner.download_daily(tickers, cfg)
    infos = scanner.get_info(list(daily), cfg)
    jobs = [(t, d, infos.get(t) or {}, cfg) for t, d in daily.items() if len(d) > 300]
    print("tickers", len(jobs), flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(i, round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows, columns=["sig", "t", "date", "res"])
    print("\n==== RESULTS (خلال 10 جلسات: +20% قبل الوقف) ====")
    g = df.groupby("sig")["res"].value_counts().unstack(fill_value=0)
    for col in ("hit", "fail", "none"):
        if col not in g:
            g[col] = 0
    g["n"] = g.sum(axis=1)
    g["hit%"] = (g["hit"] / g["n"] * 100).round(1)
    g["fail%"] = (g["fail"] / g["n"] * 100).round(1)
    print(g[["n", "hit", "fail", "none", "hit%", "fail%"]].sort_index().to_string())
    for s in sorted(set(df["sig"]) - {"RANDOM"}):
        sub = df[df.sig == s]
        print(f"\n-- {s} أمثلة:", ", ".join(f"{r.t}@{r.date}:{r.res}" for r in sub.tail(25).itertuples()))
    print("\n-- FLYE:", df[df.t == "FLYE"][["sig", "date", "res"]].to_string())


if __name__ == "__main__":
    main()
