"""
🚀 الطيارات — دراسة «وين الدخول الصح؟» (طلب أبوفرح بعد RAYA: «دخول صحيح»)

المشكلة: الاختبار القديم (backtest_runners2.py) دخل بإغلاق يوم الإشارة (e = c[t])، بينما الصفحة تنصح «Limit داخل المنطقة
قرب آخر قاع أعلى» — يعني نصيحة المنطقة ما انختبرت أصلاً. هنا نقارن 7 طرق دخول على نفس الإشارات:

  E0 إغلاق يوم الإشارة (اللي انختبر قبل — ما تقدر تنفذه فعلياً)   E1 افتتاح اليوم التالي (سوق)
  E2 Limit أعلى المنطقة (آخر قاع أعلى R × 1.08)   E3 Limit وسط المنطقة (R × 1.04)   E4 Limit عند آخر قاع أعلى R
  E5 Limit −10% من الإغلاق   E6 أمر شراء توقف +10% فوق الإغلاق (مطاردة قوة)
  الأوامر صالحة 10 جلسات، والإدارة بعد الدخول: وقف = إغلاق تحت القاع الكبير −10%، هدف +100% من سعر الدخول (لمس)، أقصى 60 جلسة.

المرجع العادل: صفقات عشوائية بنفس شريحة التذبذب **ونفس آلية الأمر بالضبط** (نفس مسافة الـ Limit ونفس بعد الوقف) —
لأن أمر الـ Limit ينملئ غالباً لما السهم ينزل، فالمقارنة بدخول عشوائي بإغلاق ظالمة. النتيجة بالعائد لكل إشارة
(اللي ما انملأ = 0)، وفترة الثقة بتجميع الأسهم (إشارات نفس السهم مو مستقلة).
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import alpaca_data as A  # noqa: E402
from backtest_runners2 import universe, rsi_np, first_only, pct  # noqa: E402

DAYS, STEP, RAND_P = 1180, 2, 0.03
W, H = 10, 60                  # صلاحية الأمر / مدة الاحتفاظ بعد الدخول
NC = W + H + 2                 # أعمدة المسار: 0 = يوم الإشارة، 1.. = الجلسات بعده
ENTRIES = {"E0": "إغلاق يوم الإشارة (القديم — مو قابل للتنفيذ)", "E1": "افتتاح بكرة (سوق)",
           "E2": "Limit أعلى المنطقة (قاع أعلى ×1.08)", "E3": "Limit وسط المنطقة (×1.04)",
           "E4": "Limit عند آخر قاع أعلى", "E5": "Limit −10% من الإغلاق", "E6": "شراء توقف +10% (مطاردة قوة)"}


def process(job):
    sym, d = job
    if d is None or len(d) < 320:
        return [], []
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    dates, n = d.index, len(c)
    r = rsi_np(c)
    jump = np.r_[0.0, h[1:] / np.maximum(c[:-1], 1e-9)]
    fr = np.cumsum(jump >= 1.5)
    rng = (h - l) / np.maximum(c, 1e-9)
    atr20 = pd.Series(rng).rolling(20).mean().values * 100
    dv20 = pd.Series(c * v).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().values
    rnd = np.random.default_rng(int.from_bytes(sym.encode(), "little") % 10_000_019)
    sig, rands = [], []

    def path(t):
        sl = slice(t, t + NC)
        P = {k: (a[sl] / c[t]).astype(np.float32) for k, a in (("ph", h), ("pl", l), ("po", o), ("pc", c))}
        for k in P:
            P[k][0] = 1.0                      # عمود 0 = يوم الإشارة (إغلاقه = 1)
        return P

    for t in range(100, n - NC, STEP):
        px = c[t]
        if not (0.5 <= px <= 15) or not (dv20[t] >= 50_000) or not (v20[t] >= 20_000):
            continue
        if rnd.random() < RAND_P:
            rands.append({"atr20": atr20[t], "date": str(dates[t].date()), "px": px, "P": path(t)})
        w0 = max(0, t - 119)
        b = t - int(np.argmin(l[w0:t + 1][::-1]))
        if t - b < 15:
            continue
        pre0 = max(0, b - 250)
        if b - pre0 < 20:
            continue
        B = l[b]
        if h[pre0:b].max() / B < 3 or not (1.05 <= px / B <= 2.0):
            continue
        j = t - 9 + int(np.argmin(l[t - 9:t + 1]))
        R = l[j]
        if not (R >= 1.08 * B and j - b >= 10 and r[j] >= r[b] + 5):
            continue
        if t - 29 <= b:
            continue
        m1, m2, m3 = l[t - 29:t - 19].min(), l[t - 19:t - 9].min(), l[t - 9:t + 1].min()
        if not (m2 >= m1 * 0.99 and m3 >= m2 * 0.99 and m3 >= m1 * 1.03):
            continue
        nfr = fr[t] - fr[max(0, t - 250)]
        if nfr < 1:
            continue                                            # V4 فما فوق
        top = h[b:t + 1].max()
        near = np.where(h[b:t + 1] >= 0.93 * top)[0]
        touches = 0 if len(near) == 0 else 1 + int((np.diff(near) >= 5).sum())
        sig.append({"t": sym, "date": str(dates[t].date()), "px": px, "atr20": atr20[t], "dv20": dv20[t],
                    "R": R / px, "stop": 0.9 * B / px, "V5": touches >= 2, "V6": nfr >= 2,
                    "ready": touches >= 2 or nfr >= 2, "P": path(t)})
    return sig, rands


def simulate(P, lr, s, kind):
    """P: dict (n, NC). lr: مستوى الأمر (نسبة من إغلاق الإشارة). s: بعد الوقف كنسبة من سعر الدخول.
    يرجع (filled, hit, stop, ret) — ret = 0 لو ما انملأ. الهدف يُفحص قبل الوقف بنفس اليوم."""
    ph, pl, po, pc = P["ph"], P["pl"], P["po"], P["pc"]
    n = len(ph)
    lr = np.broadcast_to(np.asarray(lr, float), (n,))
    s = np.broadcast_to(np.asarray(s, float), (n,))
    ar = np.arange(n)
    if kind == "close":
        has, fi, pr = np.ones(n, bool), np.zeros(n, int), np.ones(n)
    elif kind == "open":
        has, fi, pr = np.ones(n, bool), np.ones(n, int), po[:, 1].astype(float)
    elif kind == "limit":
        cond = pl[:, 1:W + 1] <= lr[:, None]
        has, fi = cond.any(1), cond.argmax(1) + 1
        pr = np.minimum(lr, po[ar, fi])
    else:                                                          # stop-buy
        cond = ph[:, 1:W + 1] >= lr[:, None]
        has, fi = cond.any(1), cond.argmax(1) + 1
        pr = np.maximum(lr, po[ar, fi])
    hit, stp, ret = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)
    for f in np.unique(fi[has]):
        idx = np.where(has & (fi == f))[0]
        p0 = pr[idx][:, None]
        hs = ph[idx, f + 1:f + 1 + H] / p0
        cs = pc[idx, f:f + H] / p0
        big = H + 5
        tj = np.where((hs >= 2).any(1), (hs >= 2).argmax(1), big)
        sk = np.where((cs < (1 - s[idx])[:, None]).any(1), (cs < (1 - s[idx])[:, None]).argmax(1), big)
        win = (tj < big) & (tj + 1 <= sk)
        los = ~win & (sk < big)
        rr = np.where(win, 1.0, np.where(los, cs[np.arange(len(idx)), np.minimum(sk, H - 1)] - 1, cs[:, H - 1] - 1))
        hit[idx], stp[idx], ret[idx] = win, los, rr
    return has, hit, stp, ret


def levels(S):
    """مستوى الأمر وبعد الوقف النسبي لكل طريقة (نسبة من إغلاق يوم الإشارة)."""
    R, sr = S["R"], S["stop"]
    spec = {"E0": ("close", np.ones_like(R)), "E1": ("open", np.ones_like(R)),
            "E2": ("limit", R * 1.08), "E3": ("limit", R * 1.04), "E4": ("limit", R.copy()),
            "E5": ("limit", np.full_like(R, 0.90)), "E6": ("stop", np.full_like(R, 1.10))}
    out = {}
    for k, (kind, lv) in spec.items():
        enom = np.minimum(lv, 1.0) if kind == "limit" else np.maximum(lv, 1.0)
        out[k] = (kind, lv, np.clip(1 - sr / enom, 0.01, 0.9))
    return out


def run():
    syms = universe()
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    SG, RD = [], []
    t0 = time.time()
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
            for sg, rd in ex.map(process, jobs, chunksize=4):
                SG += sg
                RD += rd
            if (i // 100) % 10 == 0:
                print(f"  {i + 100}/{len(syms)} · إشارات {len(SG)} · عشوائي {len(RD)} · {round(time.time() - t0)}ث", flush=True)
    return SG, RD


def stack(items):
    return {k: np.vstack([x["P"][k] for x in items]) for k in ("ph", "pl", "po", "pc")}


def main():
    SG, RD = run()
    print(f"\nإشارات V4+: {len(SG)} · أسهم {len({x['t'] for x in SG})} · عشوائي {len(RD)}", flush=True)
    if len(SG) < 60:
        print("إشارات قليلة"); return
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "P"} for x in SG])
    df["i"] = np.arange(len(df))
    qa = np.nanquantile([x["atr20"] for x in RD], [0.2, 0.4, 0.6, 0.8])
    rb = np.searchsorted(qa, np.nan_to_num([x["atr20"] for x in RD]))
    df["q"] = np.searchsorted(qa, df.atr20.fillna(0).values)
    rng = np.random.default_rng(7)
    pools = {}
    for q in range(5):
        ids = np.where(rb == q)[0]
        ids = rng.choice(ids, min(len(ids), 2500), replace=False) if len(ids) else ids
        pools[q] = stack([RD[k] for k in ids]) if len(ids) else None
    print("حجم العشوائي لكل شريحة تذبذب:", {q: (0 if p is None else len(p['ph'])) for q, p in pools.items()}, flush=True)

    # لكل إشارة × طريقة: نتيجة الإشارة + متوسط العشوائي بنفس الآلية
    Sall = stack(SG)
    LV = levels(df)
    res = {}
    for k, (kind, lv, s) in LV.items():
        has, hit, stp, ret = simulate(Sall, lv, s, kind)
        er, eh, ef = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        for i in range(len(df)):
            pool = pools.get(int(df.q.iat[i]))
            if pool is None:
                er[i] = eh[i] = ef[i] = np.nan; continue
            hh, hi_, _, rr = simulate(pool, lv[i], s[i], kind)
            er[i], eh[i], ef[i] = rr.mean(), hi_.mean(), hh.mean()
        res[k] = {"fill": has, "hit": hit, "stop": stp, "ret": ret, "er": er, "eh": eh, "ef": ef}
        print(f"  {k} محسوب · {round(time.time() - 0)}", flush=True)

    def report(mask, title):
        g = first_only(df[mask])
        if len(g) < 30:
            print(f"\n── {title}: n={len(g)} (قليل)"); return
        ix = g.i.values
        print(f"\n══ {title} — n={len(g)} إشارة · {g.t.nunique()} سهم ══")
        print(f"{'الطريقة':<42}{'انملأ':>7}{'وصل+100%':>10}{'(من الكل)':>10}{'وقف':>7}{'عائد/إشارة':>12}{'عشوائي':>9}{'الفرق':>14}")
        for k in ENTRIES:
            r_ = res[k]
            ret, er = r_["ret"][ix], r_["er"][ix]
            ok = np.isfinite(er)
            fill = r_["fill"][ix].mean()
            hit_all = r_["hit"][ix].mean()
            stop_f = r_["stop"][ix].sum() / max(1, r_["fill"][ix].sum())
            diff = pd.Series(ret[ok] - er[ok]).groupby(g.t.values[ok]).mean()
            ci = 1.96 * diff.std() / max(1, len(diff)) ** 0.5
            print(f"{k} {ENTRIES[k]:<39}{fill*100:6.0f}%{hit_all*100:9.1f}%{(r_['hit'][ix].sum()/max(1,r_['fill'][ix].sum()))*100:9.1f}%"
                  f"{stop_f*100:6.0f}%{ret.mean()*100:+11.1f}%{np.nanmean(er)*100:+8.1f}%{(ret[ok]-er[ok]).mean()*100:+8.1f}±{ci*100:.1f}")

    report(df.V5 | df.V6, "🚀 جاهز (V5 أو V6)")
    report(df.V5 & ~df.V6, "V5 فقط (اختبر المقاومة مرتين)")
    report(df.V6, "V6 (طار مرتين+)")
    report(np.ones(len(df), bool), "V4 كلها (🟡 قريب فما فوق)")
    # ثبات بالسنين + السعر
    for nm, m in (("سعر < 1$", df.px < 1), ("سعر ≥ 1$", df.px >= 1)):
        report((df.V5 | df.V6) & m, f"جاهز · {nm}")
    for y in sorted(df.date.str[:4].unique()):
        report((df.V5 | df.V6) & (df.date.str[:4] == y), f"جاهز · سنة {y}")
    # كم تبعد منطقة الدخول عن السعر وقت الإشارة؟
    g = first_only(df[df.V5 | df.V6])
    print("\nبعد أعلى المنطقة (R×1.08) عن إغلاق الإشارة (نسبة):", np.round(np.percentile(g.R * 1.08, [10, 25, 50, 75, 90]), 3))
    print("وقف القاع الكبير −10% كنسبة من الإغلاق:", np.round(np.percentile(g.stop, [10, 25, 50, 75, 90]), 3))
    print("\n⚠ تشخيص فقط — ليس توصية. «الفرق» = عائد الإشارة − عشوائي بنفس آلية الأمر والتذبذب (±95%، مجمّع بالسهم).")


if __name__ == "__main__":
    main()
