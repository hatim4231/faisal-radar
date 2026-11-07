"""
تحليل الشروط (تشخيص فقط): أي شرط في فيصل المطوّر وعبدالله فعلاً يرفع نسبة النجاح؟
بيانات Alpaca اليومية (سنتين) · آخر ~12 شهر · كل 3 جلسات · نفس كود main (الخطوة في الـ workflow تنسخه).
لكل سهم/يوم: نشغل القاعدتين ونسجل كل شرط (تحقق؟ ونقاطه) + النقاط والحالة، والنتيجة:
  u10  = +20% قبل −10% خلال 10 جلسات (نفس الشمعة = وقف)
  up30 = طار +30% خلال 20 جلسة · ret10 = العائد بعد 10 جلسات
تنبيه: معلومات الشركة (فلووت/شورت) حالية مو تاريخية ← الشروط اللي تعتمد عليها أقل موثوقية.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "abdullah"):
    sys.path.insert(0, str(p))
import numpy as np
import pandas as pd
import scanner
import faisal_rules as fr
import abdullah_rules as ar

STEP, LOOKBACK, H10, H20 = 3, 250, 10, 20
ACFG = json.loads((ROOT / "abdullah" / "config.json").read_text(encoding="utf-8"))


def u10(fut, entry):
    for _, r in fut.iterrows():
        if r["Low"] < entry * 0.9:
            return 0
        if r["High"] >= entry * 1.2:
            return 1
    return 0


def run_one(args):
    t, d, info, cfg = args
    out = []
    n = len(d)
    for end in range(max(200, n - LOOKBACK - H20), n - H20, STEP):
        hist = d.iloc[:end]
        px = float(hist["Close"].iloc[-1])
        if not (0.5 <= px <= 5) or float(hist["Volume"].tail(30).mean()) < 100000:
            continue
        f10, f20 = d.iloc[end:end + H10], d.iloc[end:end + H20]
        base = {"t": t, "date": str(hist.index[-1].date()), "u10": u10(f10, px),
                "up30": int(float(f20["High"].max()) / px - 1 >= 0.30),
                "ret10": float(f10["Close"].iloc[-1]) / px - 1, "px": px,
                "dd10": int(float(f10["Low"].min()) < px * 0.9),
                "atr": float(((hist["High"] - hist["Low"]) / hist["Close"]).tail(20).mean())}
        out.append({**base, "radar": "RANDOM"})
        try:
            a = fr.analyze(hist, info, cfg)
        except Exception:
            a = None
        if a:
            I = a["irtikaz"]
            row = {**base, "radar": "FAISAL", "score": I["score"], "base": I.get("base"), "status": I["status"],
                   "rsi": a.get("rsi")}
            for c in I["checks"]:
                row["ok_" + c["key"]] = int(bool(c["ok"]))
                row["pc_" + c["key"]] = (c["got"] / c["weight"]) if c["weight"] else 0
            out.append(row)
        try:
            b = ar.analyze(hist, info, ACFG)
        except Exception:
            b = None
        if b:
            row = {**base, "radar": "ABDULLAH", "score": b["score"], "status": b["status"]}
            for c in b["checks"]:
                row["ok_" + c["key"]] = int(bool(c["ok"]))
                row["pc_" + c["key"]] = (c["got"] / c["weight"]) if c["weight"] else 0
            out.append(row)
    return out


def pct(x):
    return f"{x * 100:.1f}%"


def report(df, name, rnd):
    print(f"\n\n######## {name} ########")
    print(f"العشوائي: +20% قبل −10% = {pct(rnd.u10.mean())} · طار 30% = {pct(rnd.up30.mean())} · نزل−10% {pct(rnd.dd10.mean())} "
          f"· عائد10 وسيط {rnd.ret10.median() * 100:+.1f}% (n={len(rnd)})")
    # 1) هل النقاط الأعلى = نجاح أكثر؟
    print("\n-- حسب النقاط --")
    bins = [0, 30, 40, 50, 60, 70, 80, 101]
    df = df.copy()
    df["b"] = pd.cut(df["score"], bins, right=False)
    g = df.groupby("b", observed=True).agg(n=("u10", "size"), u10=("u10", "mean"), up30=("up30", "mean"),
                                            ret10=("ret10", "median"))
    for k, r in g.iterrows():
        print(f"  {k}: n={r.n:<6} +20%قبل−10%={pct(r.u10):>6} طار30%={pct(r.up30):>6} عائد10(وسيط)={r.ret10 * 100:+.1f}%")
    print("\n-- حسب الحالة --")
    for st, sub in df.groupby("status"):
        print(f"  {st}: n={len(sub)} +20%قبل−10%={pct(sub.u10.mean())} طار30%={pct(sub.up30.mean())} "
              f"نزل−10%={pct(sub.dd10.mean())} عائد10(وسيط)={sub.ret10.median() * 100:+.1f}% ATR={sub.atr.median() * 100:.1f}%")
    # 1ب) الحالة مقابل عشوائي بنفس التذبذب
    edges = rnd["atr"].quantile([0, .2, .4, .6, .8, 1]).to_numpy().copy()
    edges[0], edges[-1] = -1, 99
    rq = pd.cut(rnd["atr"], edges, labels=False)
    base_u = rnd.groupby(rq)["u10"].mean()
    base_r = rnd.groupby(rq)["ret10"].median()
    print("\n-- الحالة مقابل عشوائي بنفس التذبذب --")
    for st, sub in df.groupby("status"):
        q = pd.cut(sub["atr"], edges, labels=False)
        exp_u = q.map(base_u).mean()
        exp_r = q.map(base_r).mean()
        print(f"  {st}: نجاح {pct(sub.u10.mean())} مقابل عشوائي مماثل {pct(exp_u)} ({(sub.u10.mean() - exp_u) * 100:+.1f}) · "
              f"عائد10 وسيط {sub.ret10.median() * 100:+.1f}% مقابل {exp_r * 100:+.1f}%")
    # 2) كل شرط: متحقق مقابل مو متحقق — في كل الأسهم، وفي المرشحين (نقاط 50+)
    for label, pop in (("كل الأسهم", df), ("المرشحين (نقاط 50+)", df[df.score >= 50])):
        print(f"\n-- الشروط: {label} (n={len(pop)}) — الفرق = نجاح لما يتحقق − لما ما يتحقق --")
        res = []
        for col in [c for c in pop.columns if c.startswith("ok_")]:
            on, off = pop[pop[col] == 1], pop[pop[col] == 0]
            if len(on) < 30 or len(off) < 30:
                continue
            # ثبات: النصف الأول من الفترة مقابل الثاني
            mid = pop["date"].sort_values().iloc[len(pop) // 2]
            h1 = pop[pop.date < mid]
            h2 = pop[pop.date >= mid]
            d1 = h1[h1[col] == 1].u10.mean() - h1[h1[col] == 0].u10.mean()
            d2 = h2[h2[col] == 1].u10.mean() - h2[h2[col] == 0].u10.mean()
            res.append((col[3:], len(on), on.u10.mean(), off.u10.mean(), on.u10.mean() - off.u10.mean(),
                        on.up30.mean() - off.up30.mean(), d1, d2))
        # نفس التذبذب: نقارن داخل كل خُمس من ATR (عشان الشرط ما ينحسب له أو عليه بس لأنه يختار أسهم أهدى/أعنف)
        pop = pop.assign(aq=pd.qcut(pop["atr"], 5, labels=False, duplicates="drop"))
        vm, rk = {}, {}
        for col in [c for c in pop.columns if c.startswith("ok_")]:
            diffs, ws, dds, rets = [], [], [], []
            for _, g in pop.groupby("aq"):
                on, off = g[g[col] == 1], g[g[col] == 0]
                if len(on) >= 15 and len(off) >= 15:
                    w = len(on)
                    diffs.append((on.u10.mean() - off.u10.mean()) * w)
                    dds.append((on.dd10.mean() - off.dd10.mean()) * w)
                    rets.append((on.ret10.median() - off.ret10.median()) * w)
                    ws.append(w)
            if ws:
                vm[col[3:]] = (sum(diffs) / sum(ws), sum(dds) / sum(ws), sum(rets) / sum(ws))
        for k, non, a, b, d, d30, d1, d2 in sorted(res, key=lambda x: -x[4]):
            v = vm.get(k)
            if v:
                print(f"  {'':<14} ↳ بنفس التذبذب: نجاح {v[0] * 100:+.1f} · نزل −10% {v[1] * 100:+.1f} · عائد10 وسيط {v[2] * 100:+.1f}")
            stable = "✅ ثابت" if (d1 > 0) == (d2 > 0) else "⚠ متقلب"
            print(f"  {k:<14} متحقق n={non:<6} نجاح {pct(a):>6} | ما تحقق {pct(b):>6} | الفرق {d * 100:+.1f} نقطة "
                  f"· طار30% {d30 * 100:+.1f} · نصف1 {d1 * 100:+.1f} نصف2 {d2 * 100:+.1f} {stable}")
    # 3) RSI (فيصل)
    if "rsi" in df and df["rsi"].notna().any():
        print("\n-- RSI --")
        for lo, hi in ((0, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 101)):
            s = df[(df.rsi >= lo) & (df.rsi < hi)]
            if len(s):
                print(f"  RSI {lo}-{hi}: n={len(s)} +20%قبل−10%={pct(s.u10.mean())} طار30%={pct(s.up30.mean())}")
    # 4) السعر
    print("\n-- السعر (المرشحين 50+) --")
    c = df[df.score >= 50]
    for lo, hi in ((0.5, 1), (1, 2), (2, 3.5), (3.5, 5.01)):
        s = c[(c.px >= lo) & (c.px < hi)]
        if len(s):
            print(f"  {lo}-{hi}$: n={len(s)} +20%قبل−10%={pct(s.u10.mean())} طار30%={pct(s.up30.mean())}")


def main():
    cfg = scanner.load_cfg()
    tickers = [x for x in scanner.get_universe(cfg) if not (len(x) == 5 and x[-1] in "WRU") and "-" not in x]
    daily = scanner.download_daily(tickers, cfg)
    infos = scanner.get_info(list(daily), cfg)
    jobs = [(t, d, infos.get(t) or {}, cfg) for t, d in daily.items() if len(d) > 260]
    print("tickers", len(jobs), flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(i, round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("diag_out", exist_ok=True)
    df.to_csv("diag_out/features.csv.gz", index=False, compression="gzip")
    rnd = df[df.radar == "RANDOM"]
    print("الفترة:", rnd.date.min(), "→", rnd.date.max(), "· أيام-أسهم:", len(rnd))
    report(df[df.radar == "FAISAL"].dropna(axis=1, how="all"), "فيصل المطوّر", rnd)
    report(df[df.radar == "ABDULLAH"].dropna(axis=1, how="all"), "عبدالله", rnd)


if __name__ == "__main__":
    main()
