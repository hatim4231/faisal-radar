"""
اختبار تاريخي: هل تأكيد التجميع يحسّن إشارات فيصل المطوّر وعبدالله؟ (تشخيص فقط)
آخر ~6 شهور، كل 3 جلسات. لكل إشارة ready/near: مع تأكيد تجميع (ready/near) أو بدونه.
النتيجة: +20% قبل −10% خلال 10 جلسات، العائد بعد 10، وطار +30%/+50% خلال 20 جلسة. والمرجع: سهم عشوائي.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "abdullah", ROOT / "accumulation"):
    sys.path.insert(0, str(p))
import pandas as pd
import scanner
import faisal_rules as fr
import abdullah_rules as ar
import accum_confirm

STEP, LOOKBACK, H10, H20 = 3, 126, 10, 20
ACFG = json.loads((ROOT / "abdullah" / "config.json").read_text(encoding="utf-8"))


def first_hit(fut, entry):
    for _, r in fut.iterrows():
        if r["High"] >= entry * 1.2:
            return "hit"
        if r["Low"] < entry * 0.9:
            return "fail"
    return "none"


def run_one(args):
    t, d, info, cfg = args
    out, last = [], {}
    n = len(d)
    for end in range(n - LOOKBACK - H20, n - H20, STEP):
        if end < 200:
            continue
        hist = d.iloc[:end]
        px = float(hist["Close"].iloc[-1])
        if not (0.5 <= px <= 5) or float(hist["Volume"].tail(30).mean()) < 100000:
            continue
        f10, f20 = d.iloc[end:end + H10], d.iloc[end:end + H20]
        base = {"t": t, "date": str(hist.index[-1].date()), "u10": first_hit(f10, px),
                "ret10": float(f10["Close"].iloc[-1]) / px - 1, "maxup20": float(f20["High"].max()) / px - 1}
        out.append({**base, "sig": "RANDOM"})
        sigs = []
        try:
            a = fr.analyze(hist, info, cfg)
            if a and a["irtikaz"]["status"] in ("ready", "near"):
                sigs.append("FAISAL")
        except Exception:
            pass
        try:
            b = ar.analyze(hist, info, ACFG)
            if b and b["status"] in ("ready", "near"):
                sigs.append("ABDULLAH")
        except Exception:
            pass
        if not sigs:
            continue
        acc = accum_confirm.check(hist, info, cfg["accumulation"])
        tag = "+ACC" if acc else "-noACC"
        for s in sigs:
            key = s + tag
            if last.get(key, -99) >= end - H10:
                continue
            last[key] = end
            out.append({**base, "sig": key})
            out.append({**base, "sig": s + "_ALL"})
    return out


def main():
    cfg = scanner.load_cfg()
    cfg["scan"]["history_period"] = "2y"
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU")]
    daily = scanner.download_daily(tickers, cfg)
    infos = scanner.get_info(list(daily), cfg)
    jobs = [(t, d, infos.get(t) or {}, cfg) for t, d in daily.items() if len(d) > 330]
    print("tickers", len(jobs), flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(i, round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby("sig").agg(
        n=("t", "size"),
        hit20_before_stop10=("u10", lambda x: (x == "hit").mean() * 100),
        stop10_first=("u10", lambda x: (x == "fail").mean() * 100),
        median_ret10=("ret10", lambda x: x.median() * 100),
        win10=("ret10", lambda x: (x > 0).mean() * 100),
        up30_in20=("maxup20", lambda x: (x >= 0.30).mean() * 100),
        up50_in20=("maxup20", lambda x: (x >= 0.50).mean() * 100))
    print("\n==== النتائج ====")
    print(g.round(1).to_string())
    for s in ("FAISAL+ACC", "ABDULLAH+ACC"):
        sub = df[df.sig == s].sort_values("maxup20", ascending=False)
        print(f"\n-- {s} أمثلة:", ", ".join(f"{r.t}@{r.date}:max+{r.maxup20 * 100:.0f}%/ret{r.ret10 * 100:+.0f}%" for r in sub.head(15).itertuples()))


if __name__ == "__main__":
    main()
