"""
🎯 تحدي المستخدم (2026-11-11): «القائمة كثيرة ... عطني اللي جاهز إنه بينطلق قريب» — اختيار قليل وحاسم (تشخيص فقط).

1) اختبار قواعد الاختيار على سنة ما شافها النموذج (تدريب قبل 2025-10، اختبار 2025-11 ← 2026-10):
   A = 🧨 أعلى 1% من النموذج · B = بصمة اليوم اللي قبل الانفجار (فوليوم آخر جلسة ≥ 2× وسيط 20 جلسة، بدون انفجار)
   · B3 = نفسها ≥ 3× · C = متكرر (انفجر مرتين+ بسنة) · «اختيار اليوم» = أعلى 3 كل يوم من A∧B بالاحتمال.
   لكل قاعدة: كم انفجر (×2 بيوم) خلال 5 و10 جلسات، ووصل +50% خلال 10، ونتيجة الشراء المعلّق (+15% ← +30%/−15%/3)
   ودخول الإغلاق (+30%/−15%/5) — مقابل عشوائي بنفس التذبذب والسعر.
2) اختيار اليوم من آخر جلسة مكتملة (نموذج الإنتاج highrisk_model.json) + نشاط البري اليوم + الشركة/الشورت/SEC/الأخبار.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
sys.path.insert(0, str(ROOT / "highrisk_radar"))
import numpy as np
import pandas as pd
import alpaca_data as A
import backtest_highrisk as BH
from backtest_confidence import fit_logit, prep, auc


def part1():
    t0 = time.time()
    syms = BH.universe()
    start = datetime.now(timezone.utc) - timedelta(days=BH.DAYS)
    parts = []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            batch = syms[i:i + 100]
            try:
                raw = A.bars(batch, "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:100], flush=True)
                continue
            try:
                spl = A.splits(batch, start)
            except Exception:
                spl = {}
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")], spl.get(s, [])))
            for rows, _ in ex.map(BH.feat, jobs, chunksize=4):
                if rows is not None and len(rows):
                    parts.append(rows)
            if (i // 100) % 15 == 0:
                print(f"  {i + 100}/{len(syms)} · {round(time.time() - t0)}ث", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    for f_ in BH.FEATS:
        x = df[f_].replace([np.inf, -np.inf], np.nan)
        df[f_] = x.fillna(x.median()).astype(np.float32)
    df["vq"] = pd.qcut(df["atr20p"].rank(method="first"), 5, labels=False)
    df["pq"] = pd.cut(df["close"], [0, 1, 3, 10, 1e9], labels=False)
    tr = df[df.D < 20251001]
    te = df[(df.D >= 20251101)].copy()
    Z, mu, sd = prep(tr, BH.FEATS)
    w = fit_logit(Z, tr.E10.astype(float).values, lam=5.0)
    Zt, _, _ = prep(te, BH.FEATS, mu, sd)
    te["p"] = 1 / (1 + np.exp(-(w[0] + Zt @ w[1:])))
    print(f"\nاختبار خارج العينة 2025-11 ← 2026-10: n={len(te):,} · AUC {auc(te.E10.values, te.p.values):.3f} · الأساس خلال 5 {BH.pr(te.E5.mean())} · خلال 10 {BH.pr(te.E10.mean())}", flush=True)
    q99 = te.p.quantile(0.99)
    Am = te.p >= q99
    Bm = te.vol1_20 >= 2
    B3 = te.vol1_20 >= 3
    Cm = te.n_ep252 >= 2
    AB = te[Am & Bm]
    picks3 = AB.sort_values("p", ascending=False).groupby("D").head(3)
    picksA = te[Am].sort_values("p", ascending=False).groupby("D").head(3)
    groups = [("A 🧨 أعلى 1%", te[Am]), ("B فوليوم ≥ 2× (بدون انفجار)", te[Bm]), ("C متكرر", te[Cm]), ("A∧B", AB),
              ("A∧B3 (فوليوم ≥ 3×)", te[Am & B3]), ("A∧C", te[Am & Cm]), ("A∧B∧C", te[Am & Bm & Cm]), ("B∧C", te[Bm & Cm]),
              ("اختيار اليوم: أعلى 3 كل يوم من A∧B", picks3), ("مقارنة: أعلى 3 كل يوم من A", picksA)]
    print("\nالقاعدة: n · أيام فيها اختيار · انفجر خلال 5 · خلال 10 (عشوائي بنفس التذبذب والسعر) · +50% خلال 10 · شراء معلّق (عشوائي) · دخول إغلاق +30%/−15%/5 (عشوائي)")
    for nm, g in groups:
        if len(g) < 30:
            print(f"  {nm}: عدد قليل ({len(g)})")
            continue
        gs = g.dropna(subset=["v_s15b"])
        print(f"  {nm}: n={len(g):,} · أيام {g.D.nunique()} · خلال 5 {BH.pr(g.E5.mean())} · خلال 10 {BH.pr(g.E10.mean())} "
              f"({BH.pr(BH.matched_expect(g, te, 'E10'))}) · +50% {BH.pr((g.mx10 >= 0.5).mean())} · "
              f"معلّق {BH.pr(gs.v_s15b.mean())} n={len(gs)} ({BH.pr(BH.matched_expect(gs, te.dropna(subset=['v_s15b']), 'v_s15b'))}) · "
              f"إغلاق {BH.pr(g.v_c5.mean())} ({BH.pr(BH.matched_expect(g, te, 'v_c5'))})", flush=True)
    # كم مرة على الأقل واحد من الثلاث انفجر خلال 10 جلسات (من الأيام اللي فيها اختيار)
    day_hit = picks3.groupby("D").E10.max()
    print(f"\nأيام فيها اختيار A∧B: {len(day_hit)} · واحد على الأقل من الثلاث انفجر خلال 10 جلسات: {BH.pr(day_hit.mean())}", flush=True)
    return float(q99)


def part2():
    import highrisk as HR
    import highrisk_scanner as S
    import scanner
    import sec_tools
    M = HR.model()
    th = M["thresholds"]
    tickers = S.alpaca_universe()
    daily = S.fetch_daily(tickers)
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    rows = []
    for t, d in daily.items():
        if len(d) and d.index[-1].normalize() >= today:
            d = d.iloc[:-1]
        if len(d) < 120:
            continue
        try:
            F = HR.features(d)
            last = F.iloc[-1]
            sc = HR.score(last)
        except Exception:
            continue
        rows.append({"t": t, "p": sc["p_raw"], "e10": sc["e10"], "date": str(d.index[-1].date()), "close": float(last["close"]),
                     "vol1_20": float(last["vol1_20"]), "vol5_60": float(last["vol5_60"]), "n_year": int(last["n_ep252"]),
                     "days_ep": int(last["days_ep_raw"]), "days_rs": int(last["days_rs_raw"]), "dd252": float(last["dd252"]),
                     "pos60": float(last["pos60"]), "ret5": float(last["ret5"]), "ret20": float(last["ret20"]),
                     "jump": float(last["jump"]) if np.isfinite(last["jump"]) else 1.0, "dv20": float(last["dv20"]),
                     "vol_last": float(d["Volume"].iloc[-1])})
    R = pd.DataFrame(rows)
    print(f"\n==== اختيار اليوم (آخر جلسة مكتملة {R.date.max()}) — من {len(R):,} سهم ====", flush=True)
    R["A"] = R.p >= th["ready"]
    R["B"] = (R.vol1_20 >= 2) & (R.jump < 2)
    cand = R[R.A & R.B].sort_values("p", ascending=False)
    topA = R[R.A].sort_values("p", ascending=False)
    print(f"🧨 (A): {int(R.A.sum())} · A∧B (فوليوم آخر جلسة ≥ 2×): {len(cand)}")
    show = pd.concat([cand.head(12), topA.head(12)]).drop_duplicates("t")
    syms = list(show.t)
    infos = scanner.get_info(syms, scanner.load_cfg())
    try:
        ib = scanner.load_ib_short()
    except Exception:
        ib = {}
    # نشاط البري اليوم (4:00 ← الحين بتوقيت نيويورك)
    pm = {}
    try:
        st = (today + pd.Timedelta(hours=4)).tz_localize("America/New_York").tz_convert("UTC").to_pydatetime()
        bars = A.bars(syms, "5Min", st)
        for s_, b in bars.items():
            b = b[b.index.hour < 10]
            if len(b):
                pm[s_] = {"vol": float(b.Volume.sum()), "hi": float(b.High.max()), "last": float(b.Close.iloc[-1])}
    except Exception as e:
        print("البري:", repr(e)[:100])
    sec_rows = [{"symbol": s_, "name": (infos.get(s_) or {}).get("shortName"), "news": []} for s_ in syms]
    try:
        sec_tools.annotate(sec_rows, lambda *a: None)
    except Exception:
        pass
    dil = {r["symbol"]: ((r.get("dilution") or {}).get("level"), [(h["form"], h["date"]) for h in (r.get("dilution") or {}).get("hits", [])][:3],
                         len(sec_tools.dilution_events(r))) for r in sec_rows}
    import yfinance as yf
    print("\nالسهم · A∧B؟ · الاحتمال الخام · السعر · فوليوم آخر جلسة ÷ الوسيط · انفجر/سنة · آخر انفجار قبل · تقسيم قبل · من قمة السنة · 20 جلسة · "
          "سيولة$ · فلووت · القيمة · الدولة · شورت IB · SEC · البري اليوم")
    for _, r in show.iterrows():
        inf = infos.get(r.t) or {}
        sv, _ = scanner.short_of(r.t, ib, inf)
        p_ = pm.get(r.t)
        pmtxt = f"فوليوم {p_['vol']:,.0f} · أعلى {p_['hi']:.3f} ({(p_['hi'] / r.close - 1) * 100:+.0f}%) · آخر {p_['last']:.3f}" if p_ else "ما فيه تداول بري"
        lv, hits, nev = dil.get(r.t, (None, [], 0))
        print(f"  {r.t:6s} · {'✅' if r.A and r.B else '—'} · {r.p:.3f} · {r.close:.3f} · {r.vol1_20:.1f}× · {r.n_year} · "
              f"{r.days_ep if r.days_ep < 9999 else '-'} · {r.days_rs if r.days_rs < 9999 else '-'} · {r.dd252 * 100:.0f}% · {r.ret20 * 100:+.0f}% · "
              f"{r.dv20:,.0f} · {inf.get('floatShares')} · {inf.get('marketCap')} · {inf.get('country')} · {sv} · {lv} {hits} أحداث {nev} · {pmtxt}", flush=True)
        try:
            n = scanner.parse_news(yf.Ticker(r.t).news)[:2]
            for x in n:
                print(f"        📰 {str(x.get('time'))[:16]} {x.get('title')}")
        except Exception:
            pass


if __name__ == "__main__":
    t0 = time.time()
    for fn in (part2, part1):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{fn.__name__} تعطل:", repr(e)[:200], flush=True)
    print(f"\nخلص في {round(time.time() - t0)} ثانية")
