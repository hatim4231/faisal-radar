"""
🚀 الطيارات — هل الأخبار والتجميع يحسّنون اختيار الطيران؟ (طلب أبوفرح: «الأخبار والتجميع»)

السؤال: داخل إشارات الطيارات (أي حالة)، هل وجود خبر قبل الإشارة أو بصمة تجميع (رادار التجميع accum_rules) يرفع
احتمال الطيران (+50%/+100% خلال 20 جلسة) أو يقلل النزيف (≤ −30%) — **بشكل يصمد بنصي الفترة**؟
- نفس مجال ومنهج runners_features: أيام مؤهلة (0.5-15$، سيولة ≥ 50 ألف$، خطوة 3 جلسات، مع المشطوبة)، المرجع = نفس السنة + فئة السعر + شريحة التذبذب،
  أول ظهور كل 20 جلسة، هامش 95% مجمّع بالسهم.
- التجميع: accum_rules.analyze على نافذة الإشارة (score + status).
- الأخبار: Alpaca News (Benzinga) لكل سهم إشارة: عدد الأخبار قبل الإشارة بـ 3 و10 أيام + كلمات طرح/محفّز بالعناوين.
- الثبات: كل ميزة تنقاس بالنص الأول والثاني بالتاريخ — اللي ما يثبت صدفة. تشخيص فقط.
"""
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accumulation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import alpaca_data as A  # noqa: E402
import runners as RN  # noqa: E402
import accum_rules as AR  # noqa: E402
from backtest_runners2 import universe, first_only  # noqa: E402

DAYS, STEP, WIN = 1180, 3, 420
CODE = {None: 0, "watch": 1, "near": 2, "ready": 3}
CFG = json.load(open(ROOT / "config.json", encoding="utf-8"))
OFFER = ("offering", "registered direct", "private placement", "warrant", "priced", "public offering", "at-the-market", "atm ", "shelf")
CATAL = ("fda", "approval", "contract", "partnership", "agreement", "acquisition", "acquire", "merger", "patent", "order", "launch", "collaboration", "trial", "award")


def process(job):
    sym, d = job
    if d is None or len(d) < 300:
        return None
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    rng = (h - l) / np.maximum(c, 1e-9)
    atr20 = pd.Series(rng).rolling(20).mean().values * 100
    dv20 = pd.Series(c * v).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    rows = []
    for t in range(270, n - 1, STEP):
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000) or not np.isfinite(atr20[t]):
            continue
        w = d.iloc[max(0, t - WIN + 1):t + 1]
        try:
            r = RN.detect(w)
        except Exception:
            r = None
        rec = {"t": sym, "date": str(d.index[t].date()), "px": px, "atr20": atr20[t], "R": CODE[r["status"]] if r else 0,
               "acc_score": np.nan, "acc_status": ""}
        if r:
            try:
                a = AR.analyze(w, {}, CFG)
                if a:
                    rec["acc_status"] = a.get("status", "")
                    rec["acc_score"] = float(a.get("score", np.nan)) if "score" in a else np.nan
            except Exception:
                pass
        fh, fl = h[t + 1:], l[t + 1:]
        for k in (20,):
            rec[f"mfe{k}"] = float(fh[:k].max() / px - 1) if len(fh) >= k else np.nan
        rec["mae20"] = float(fl[:20].min() / px - 1) if len(fl) >= 20 else np.nan
        rows.append(rec)
    return rows


def collect():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i in range(0, len(syms), 100):
            try:
                raw = A.bars(syms[i:i + 100], "1Day", start)
            except Exception as e:
                print("bars err", repr(e)[:100], flush=True)
                continue
            jobs = []
            for s, dd in raw.items():
                dd = dd[["Open", "High", "Low", "Close", "Volume"]].copy()
                dd.index = pd.to_datetime(dd.index).tz_localize(None).normalize()
                jobs.append((s, dd[~dd.index.duplicated(keep="last")]))
            for rows in ex.map(process, jobs, chunksize=4):
                if rows:
                    out += rows
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · أيام مؤهلة {len(out):,} · {round(time.time() - t0)}ث", flush=True)
    return pd.DataFrame(out)


def fetch_news(sym, start, end):
    import requests
    items, tok = [], None
    for _ in range(15):
        p = {"symbols": sym, "start": start, "end": end, "limit": 50, "sort": "asc", "include_content": "false"}
        if tok:
            p["page_token"] = tok
        for att in range(5):
            try:
                r = requests.get("https://data.alpaca.markets/v1beta1/news", headers=A._headers(), params=p, timeout=60)
            except Exception:
                time.sleep(2)
                continue
            if r.status_code == 429:
                time.sleep(5 + 5 * att)
                continue
            break
        else:
            return sym, None
        if r.status_code != 200:
            return sym, None if not items else items
        j = r.json()
        items += [(x["created_at"][:19], (x.get("headline") or "").lower()) for x in j.get("news", [])]
        tok = j.get("next_page_token")
        if not tok:
            break
    return sym, items


def news_feats(sig):
    syms = sorted(sig.t.unique())
    start = (pd.to_datetime(sig.date.min()) - timedelta(days=15)).strftime("%Y-%m-%dT00:00:00Z")
    end = (pd.to_datetime(sig.date.max()) + timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    news, t0 = {}, time.time()
    with ThreadPoolExecutor(3) as ex:
        for i, (s, it) in enumerate(ex.map(lambda s: fetch_news(s, start, end), syms)):
            news[s] = it
            if i % 200 == 0:
                print(f"  أخبار {i}/{len(syms)} · {round(time.time() - t0)}ث", flush=True)
    ok = sum(v is not None for v in news.values())
    print(f"أخبار جلبناها لـ {ok}/{len(syms)} سهم", flush=True)
    n3, n10, off, cat, have = [], [], [], [], []
    for s, dt in zip(sig.t.values, pd.to_datetime(sig.date).values):
        it = news.get(s)
        if it is None:
            n3.append(np.nan); n10.append(np.nan); off.append(np.nan); cat.append(np.nan); have.append(0)
            continue
        tt = pd.to_datetime([x[0] for x in it]).values if it else np.array([], dtype="datetime64[ns]")
        m3 = (tt > dt - np.timedelta64(3, "D")) & (tt <= dt + np.timedelta64(1, "D"))
        m10 = (tt > dt - np.timedelta64(10, "D")) & (tt <= dt + np.timedelta64(1, "D"))
        hl = [it[i][1] for i in np.where(m10)[0]]
        n3.append(int(m3.sum())); n10.append(int(m10.sum()))
        off.append(float(any(any(k in x for k in OFFER) for x in hl)))
        cat.append(float(any(any(k in x for k in CATAL) for x in hl)))
        have.append(1)
    sig = sig.copy()
    sig["n3"], sig["n10"], sig["off10"], sig["cat10"], sig["news_ok"] = n3, n10, off, cat, have
    return sig


def analyze(df):
    df = df.copy()
    df["year"] = df.date.str[:4]
    df["sub1"] = df.px < 1
    qa = np.nanquantile(df.atr20, [0.2, 0.4, 0.6, 0.8])
    df["q"] = np.searchsorted(qa, df.atr20.values)
    df["cell"] = df.year + "|" + df.sub1.astype(str) + "|" + df.q.astype(str)
    df["h50"] = np.where(df.mfe20.isna(), np.nan, (df.mfe20 >= 0.5).astype(float))
    df["h100"] = np.where(df.mfe20.isna(), np.nan, (df.mfe20 >= 1.0).astype(float))
    df["risk"] = np.where(df.mae20.isna(), np.nan, (df.mae20 <= -0.30).astype(float))
    for c in ("h50", "h100", "risk"):
        df["b_" + c] = df.groupby("cell")[c].transform("mean")
    print(f"\nأيام مؤهلة {len(df):,} · إشارات (أي حالة) {int((df.R >= 1).sum()):,}", flush=True)
    sig = first_only(df[df.R >= 1].assign(i=lambda x: x.index))
    print(f"إشارات أول ظهور: {len(sig):,} · {sig.t.nunique():,} سهم", flush=True)
    sig = news_feats(sig)
    mid = sig.date.sort_values().iloc[len(sig) // 2]
    sig["half"] = np.where(sig.date < mid, 1, 2)

    def stat(g, c):
        ok = g[c].notna() & g["b_" + c].notna()
        if ok.sum() < 20:
            return None
        x = g[ok]
        d = (x[c] - x["b_" + c]).groupby(x.t).mean()
        return x[c].mean() * 100, x["b_" + c].mean() * 100, (x[c] - x["b_" + c]).mean() * 100, 1.96 * d.std() / max(1, len(d)) ** .5 * 100

    def fmt(s):
        return "      -      " if s is None else f"{s[0]:4.1f}|{s[1]:4.1f}|{s[2]:+5.1f}±{s[3]:.1f}"

    def line(g, title):
        if len(g) < 25:
            print(f"  {title:<44} n={len(g)} (قليل)")
            return
        print(f"  {title:<44} n={len(g):>5}  +50%: {fmt(stat(g, 'h50'))}  +100%: {fmt(stat(g, 'h100'))}  نزيف: {fmt(stat(g, 'risk'))}")

    def block(name, masks):
        print(f"\n══ {name} ══   (إشارة | مرجع مطابق | الفرق ± هامش)")
        for title, m in masks:
            line(sig[m], title)
            for hlf in (1, 2):
                line(sig[m & (sig.half == hlf)], f"    └ نص {hlf}")

    print(f"\nأخبار متوفرة لـ {int(sig.news_ok.sum())} من {len(sig)} إشارة")
    block("كل إشارات الطيارات", [("الكل", sig.R >= 1)])
    block("🗞 الأخبار قبل الإشارة", [
        ("بدون أي خبر خلال 10 أيام", (sig.n10 == 0)),
        ("خبر 1-2 خلال 10 أيام", (sig.n10 >= 1) & (sig.n10 <= 2)),
        ("3+ أخبار خلال 10 أيام", (sig.n10 >= 3)),
        ("خبر خلال 3 أيام", (sig.n3 >= 1)),
        ("عنوان طرح/Warrants خلال 10 أيام", (sig.off10 == 1)),
        ("عنوان محفّز (FDA/عقد/اندماج…) خلال 10 أيام", (sig.cat10 == 1)),
    ])
    block("🔍 التجميع (رادار التجميع على نفس الإشارة)", [
        ("تجميع مؤكد (≥70)", sig.acc_score >= 70),
        ("تجميع محتمل (55-70)", (sig.acc_score >= 55) & (sig.acc_score < 70)),
        ("تجميع ضعيف (40-55)", (sig.acc_score >= 40) & (sig.acc_score < 55)),
        ("تجميع أقل من 40", sig.acc_score < 40),
        ("ما فيه قاعدة تجميع أصلاً", sig.acc_score.isna()),
    ])
    block("🤝 التجميع المؤكد/المحتمل + خبر", [
        ("تجميع ≥55 + خبر خلال 10 أيام", (sig.acc_score >= 55) & (sig.n10 >= 1)),
        ("تجميع ≥55 بدون خبر", (sig.acc_score >= 55) & (sig.n10 == 0)),
        ("جاهز 🚀 + تجميع ≥55", (sig.R == 3) & (sig.acc_score >= 55)),
        ("جاهز 🚀 بدون تجميع ≥55", (sig.R == 3) & ~(sig.acc_score >= 55)),
    ])
    sig.to_csv("diag_out/runners_news_accum_signals.csv.gz", index=False, compression="gzip")
    print("\n⚠ تشخيص فقط — ليس توصية. المرجع = كل الأيام المؤهلة بنفس السنة وفئة السعر وشريحة التذبذب. عناوين الأخبار تُصنّف بكلمات مفتاحية (تقريبي).")


def main():
    os.makedirs("diag_out", exist_ok=True)
    df = collect()
    analyze(df)


if __name__ == "__main__":
    main()
