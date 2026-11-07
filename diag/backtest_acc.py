"""
اختبار تاريخي لرادار التجميع (تشخيص فقط):
آخر ~6 شهور، كل 3 جلسات، لكل سهم في المجال (0.50-5$):
  ACC_READY / ACC_NEAR  = إشارات الرادار (نفس السهم ما يتكرر خلال 10 جلسات)
  RANDOM                = أي سهم في أي يوم (المرجع)
النتائج:
  - +20% قبل وقف موحد −10% خلال 10 جلسات
  - خطة الرادار: سقف القاعدة (الهدف) قبل الوقف (أقل سعر −10%، بالإغلاق) خلال 20 جلسة
  - أقصى صعود خلال 20 جلسة، ونسبة اللي طاروا +30% و +50%
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accumulation"))
import pandas as pd
import scanner
import accum_rules as R

STEP, LOOKBACK, H10, H20 = 3, 126, 10, 20


def first_hit(fut, entry, target, stop, by_close=False):
    for _, r in fut.iterrows():
        if r["High"] >= target:
            return "hit"
        if (r["Close"] if by_close else r["Low"]) < stop:
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
        base = {"t": t, "date": str(hist.index[-1].date()),
                "u10": first_hit(f10, px, px * 1.2, px * 0.9),
                "maxup20": float(f20["High"].max()) / px - 1, "ret10": float(f10["Close"].iloc[-1]) / px - 1,
                "chinese": (info.get("country") or "") in ("China", "Hong Kong", "Macau", "Taiwan", "Singapore", "Malaysia")}
        out.append({**base, "sig": "RANDOM", "plan": None})
        try:
            a = R.analyze(hist, info, cfg)
        except Exception:
            continue
        if not a or a.get("status") not in ("ready", "near"):
            continue
        sig = "ACC_READY" if a["status"] == "ready" else "ACC_NEAR"
        if last.get(sig, -99) >= end - H10:
            continue
        last[sig] = end
        plan = first_hit(f20, px, a["targets"][0], a["stop"], by_close=True)
        out.append({**base, "sig": sig, "plan": plan, "score": a["score"]})
    return out


def main():
    cfg = scanner.load_cfg()
    cfg["scan"]["history_period"] = "2y"
    tickers = scanner.get_universe(cfg)
    tickers = [x for x in tickers if not (len(x) == 5 and x[-1] in "WRU")]
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
    for title, sub in (("الكل", df), ("الأسهم الصينية/الآسيوية", df[df.chinese])):
        print(f"\n==== {title} ====")
        g = sub.groupby("sig").agg(
            n=("t", "size"),
            hit20_before_stop10=("u10", lambda x: (x == "hit").mean() * 100),
            stop10_first=("u10", lambda x: (x == "fail").mean() * 100),
            median_ret10=("ret10", lambda x: x.median() * 100),
            win10=("ret10", lambda x: (x > 0).mean() * 100),
            median_maxup20=("maxup20", lambda x: x.median() * 100),
            up30_in20=("maxup20", lambda x: (x >= 0.30).mean() * 100),
            up50_in20=("maxup20", lambda x: (x >= 0.50).mean() * 100))
        print(g.round(1).to_string())
        for s in ("ACC_READY", "ACC_NEAR"):
            p = sub[sub.sig == s]["plan"].value_counts(normalize=True).mul(100).round(1)
            if len(p):
                print(f"  خطة الرادار ({s}): وصل سقف القاعدة قبل الوقف {p.get('hit', 0)}% · وقف {p.get('fail', 0)}% · لا هذا ولا هذا {p.get('none', 0)}%")
    for s in ("ACC_READY", "ACC_NEAR"):
        sub = df[df.sig == s].sort_values("maxup20", ascending=False)
        print(f"\n-- {s} أمثلة (أعلى صعود):", ", ".join(f"{r.t}@{r.date}:+{r.maxup20 * 100:.0f}%" for r in sub.head(10).itertuples()))
        print(f"-- {s} أمثلة (أسوأ):", ", ".join(f"{r.t}@{r.date}:{r.ret10 * 100:+.0f}%" for r in sub.tail(8).itertuples()))


if __name__ == "__main__":
    main()
