#!/usr/bin/env python3
"""
رادار طريقة فيصل — ماسح الأسهم الأمريكية الصغيرة (Penny / Small caps)

الاستخدام:
    python scanner.py                 # مسح كامل لسوق ناسداك وفتح التقرير
    python scanner.py --serve         # يشغل موقع محلي http://127.0.0.1:8765 فيه زر تحديث
    python scanner.py --tickers BJDX,GDHG,LFS     # فحص أسهم محددة فقط
    python scanner.py --watchlist my_list.txt     # فحص قائمتك (سهم في كل سطر)
    python scanner.py --demo          # تقرير تجريبي ببيانات وهمية (بدون إنترنت)

المصدر: الأسعار من Alpaca (قرار المستخدم، ياهو للطوارئ بس) · معلومات الشركة والأخبار من Yahoo Finance. تأخير ~15 دقيقة.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import os
import sys
import threading
import time
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import faisal_rules as fr
import news_tools
import split_rule
import new_low_rule
import compliance_rule
import sec_tools
import report

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "output"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def load_cfg(path: Path | None = None) -> dict:
    p = path or ROOT / "config.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# 1) قائمة الأسهم (Universe)
# ----------------------------------------------------------------------------

YAHOO_EXCH = {"NASDAQ": ["NMS", "NCM", "NGM"], "NYSE_AMERICAN": ["ASE"], "NYSE": ["NYQ"]}


def universe_from_yahoo_screener(cfg) -> list[str]:
    """أسرع طريقة: فلتر ياهو نفسه (سعر/فوليوم/قيمة سوقية) قبل ما نحمل الشارتات."""
    import yfinance as yf
    from yfinance import EquityQuery as Q

    U = cfg["universe"]
    exch = sum((YAHOO_EXCH.get(e, []) for e in U["exchanges"]), [])
    q = Q("and", [
        Q("is-in", ["exchange", *exch]),
        Q("lt", ["intradayprice", U["max_price"]]),
        Q("gt", ["intradayprice", U["min_price"]]),
        Q("gt", ["avgdailyvol3m", U["min_avg_volume"] * 0.5]),
        Q("lt", ["intradaymarketcap", U["max_market_cap"] * 1.5]),
    ])
    out, offset = [], 0
    while True:
        res = yf.screen(q, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=True)
        quotes = res.get("quotes", []) if isinstance(res, dict) else []
        out += [x["symbol"] for x in quotes if x.get("symbol")]
        total = res.get("total", 0) if isinstance(res, dict) else 0
        offset += 250
        if not quotes or offset >= total or offset > 5000:
            break
    return sorted(set(out))


def universe_from_nasdaqtrader(cfg) -> list[str]:
    """البديل: قائمة كل أسهم ناسداك الرسمية ثم نفلتر السعر بأنفسنا."""
    import requests
    syms = []
    bad_words = ("Warrant", "Unit", " Right", "Preferred", "Notes", "Debenture")
    want = cfg["universe"]["exchanges"]
    if "NASDAQ" in want:
        txt = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|")
        df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
        df = df[~df["Security Name"].astype(str).str.contains("|".join(bad_words))]
        syms += df["Symbol"].dropna().astype(str).tolist()
    if any(e in want for e in ("NYSE", "NYSE_AMERICAN")):
        txt = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|")
        codes = (["A"] if "NYSE_AMERICAN" in want else []) + (["N"] if "NYSE" in want else [])
        df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N") & (df["Exchange"].isin(codes))]
        df = df[~df["Security Name"].astype(str).str.contains("|".join(bad_words))]
        syms += df["ACT Symbol"].dropna().astype(str).tolist()
    syms = [s.replace(".", "-") for s in syms if "$" not in s and s.isascii()]
    return sorted(set(syms))


def get_universe(cfg) -> list[str]:
    try:
        u = universe_from_yahoo_screener(cfg)
        if len(u) >= 20:
            log(f"قائمة الأسهم من فلتر ياهو: {len(u)} سهم")
            return u
        log("فلتر ياهو رجع نتائج قليلة، أجرب قائمة ناسداك الرسمية...")
    except Exception as e:
        log("فلتر ياهو ما اشتغل:", repr(e)[:200])
    u = universe_from_nasdaqtrader(cfg)
    log(f"قائمة الأسهم من nasdaqtrader: {len(u)} سهم (بيتم فلترة السعر بعد التحميل)")
    return u


# ----------------------------------------------------------------------------
# 2) تحميل الشارتات اليومية + بيانات الشركة
# ----------------------------------------------------------------------------

def download_daily(tickers: list[str], cfg) -> dict[str, pd.DataFrame]:
    """الشارت اليومي: من Alpaca دائماً (قرار المستخدم). ياهو للطوارئ بس: إذا Alpaca تعطل أو المفاتيح مو موجودة.
    السهم اللي Alpaca ما عنده بيانات كافية (أقل من 60 جلسة) ما ينجاب من ياهو — ينكتب اسمه في السجل."""
    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    if cfg["scan"].get("data_source", "alpaca") != "alpaca":
        return _download_daily_yahoo(tickers, cfg)
    try:
        import alpaca_data
        if not alpaca_data.available():
            log("⚠ مفاتيح Alpaca مو موجودة — طوارئ: الأسعار من ياهو")
            return _download_daily_yahoo(tickers, cfg)
        for i in range(0, len(tickers), 100):
            batch = tickers[i:i + 100]
            got = None
            for attempt in range(3):
                try:
                    got = alpaca_data.daily(batch, days=730)
                    break
                except Exception as e:
                    log(f"  Alpaca خطأ (محاولة {attempt + 1}):", repr(e)[:120])
                    time.sleep(5 * (attempt + 1))
            if got is None:
                failed += batch
                continue
            out.update({t: d for t, d in got.items() if len(d) >= 60})
    except Exception as e:
        log("⚠ Alpaca ما اشتغل — طوارئ: الأسعار من ياهو:", repr(e)[:120])
        return _download_daily_yahoo(tickers, cfg)
    missing = [t for t in tickers if t not in out and t not in failed]
    log(f"الشارت اليومي من Alpaca: {len(out)} من {len(tickers)}"
        + (f" · ما عنده بيانات كافية ({len(missing)}): {', '.join(missing[:20])}" if missing else ""))
    if failed:
        log(f"⚠ طوارئ: {len(failed)} سهم Alpaca تعطل معها ← من ياهو")
        out.update(_download_daily_yahoo(failed, cfg))
    return out


def _download_daily_yahoo(tickers: list[str], cfg) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    S = cfg["scan"]
    out: dict[str, pd.DataFrame] = {}
    bs = S["batch_size"]
    if tickers:
        log(f"الشارت اليومي من ياهو: {len(tickers)} سهم")
    for i in range(0, len(tickers), bs):
        batch = tickers[i:i + bs]
        log(f"تحميل الشارت اليومي {i + 1}-{i + len(batch)} من {len(tickers)}")
        for attempt in range(3):
            try:
                data = yf.download(batch, period=S["history_period"], interval="1d",
                                   group_by="ticker", actions=True, auto_adjust=False,
                                   threads=True, progress=False)
                break
            except Exception as e:
                log("  خطأ تحميل، إعادة المحاولة:", repr(e)[:120])
                time.sleep(3 * (attempt + 1))
        else:
            continue
        if data is None or data.empty:
            continue
        for t in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.get_level_values(0):
                        continue
                    d = data[t].copy()
                else:
                    d = data.copy()
                d = d.dropna(subset=["Close"])
                if len(d) >= 60:
                    d.index = pd.to_datetime(d.index).tz_localize(None)
                    out[t] = d
            except Exception:
                pass
    return out


def intraday(syms: list[str], interval: str, days: int) -> pd.DataFrame:
    """شموع داخل اليوم فيها البري والأفتر، بشكل yf.download(group_by='ticker'). Alpaca أولاً ثم ياهو."""
    try:
        import alpaca_data
        if alpaca_data.available():
            tf = {"1h": "1Hour", "5m": "5Min", "15m": "15Min"}[interval]
            f = alpaca_data.intraday_frame(syms, tf, days)
            if not f.empty:
                return f
    except Exception as e:
        log("  Alpaca (داخل اليوم) ما اشتغل، نكمل بياهو:", repr(e)[:100])
    import yfinance as yf
    return yf.download(syms, period=f"{days}d" if interval != "5m" else "1d", interval=interval, prepost=True,
                       group_by="ticker", auto_adjust=False, threads=True, progress=False)


INFO_KEYS = ["marketCap", "floatShares", "sharesOutstanding", "sharesShort",
             "shortPercentOfFloat", "shortRatio", "exchange", "longName", "shortName",
             "sector", "industry", "country", "dateShortInterest"]


def get_info(tickers: list[str], cfg) -> dict[str, dict]:
    import yfinance as yf
    day = datetime.now().strftime("%Y%m%d")
    path = CACHE / f"info_{day}.json"
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    need = [t for t in tickers if t not in cache]
    if need:
        log(f"تحميل بيانات الشركات (فلووت/شورت/قيمة سوقية) لـ {len(need)} سهم...")

        def one(t):
            for attempt in range(2):
                try:
                    inf = yf.Ticker(t).info or {}
                    return t, {k: inf.get(k) for k in INFO_KEYS}
                except Exception:
                    time.sleep(1.5)
            return t, {}

        done = 0
        with ThreadPoolExecutor(cfg["scan"]["info_workers"]) as ex:
            for fut in as_completed([ex.submit(one, t) for t in need]):
                t, inf = fut.result()
                cache[t] = inf
                done += 1
                if done % 50 == 0:
                    log(f"  {done}/{len(need)}")
                    path.write_text(json.dumps(cache), encoding="utf-8")
        path.write_text(json.dumps(cache), encoding="utf-8")
    return {t: cache.get(t, {}) for t in tickers}


# ----------------------------------------------------------------------------
# 3) التحليل العميق للمرشحين: الأخبار، فريم 4 ساعات، الفواب
# ----------------------------------------------------------------------------

def parse_news(items) -> list[dict]:
    out = []
    for it in items or []:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
        prov = (c.get("provider") or {}).get("displayName") or c.get("publisher")
        ts = c.get("pubDate") or c.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        summary = c.get("summary") or c.get("description") or ""
        summary = re.sub(r"<[^>]+>", " ", summary)
        summary = re.sub(r"\s+", " ", summary).strip()[:600]
        out.append({"title": title, "summary": summary, "url": url, "source": prov, "time": ts})
    return out[:5]


def translate_ar(texts: list[str]) -> list[str]:
    """ترجمة للعربي (خدمة قوقل المجانية). إذا فشلت يرجع النص الأصلي."""
    import requests
    out = []
    for t in texts:
        if not t:
            out.append(t)
            continue
        try:
            r = requests.get("https://translate.googleapis.com/translate_a/single",
                             params={"client": "gtx", "sl": "auto", "tl": "ar", "dt": "t", "q": t},
                             timeout=15)
            out.append("".join(seg[0] for seg in r.json()[0] if seg and seg[0]) or t)
        except Exception:
            out.append(t)
    return out


def load_ib_short() -> dict[str, dict]:
    """
    الشورت المتاح للاقتراض لكل الأسهم الأمريكية دفعة وحدة، من ملف Interactive Brokers العام
    (نفس المصدر اللي يستخدمه iBorrowDesk). يتحدث كل ~15 دقيقة.
    """
    from ftplib import FTP
    buf = io.BytesIO()
    ftp = FTP("ftp2.interactivebrokers.com", timeout=60)
    ftp.login("shortstock", "")
    ftp.retrbinary("RETR usa.txt", buf.write)
    ftp.quit()
    out, stamp = {}, ""
    for line in buf.getvalue().decode("utf-8", "ignore").splitlines():
        if line.startswith("#BOF"):
            stamp = " ".join(line.split("|")[1:3])
            continue
        if line.startswith("#"):
            continue
        p = line.split("|")
        if len(p) < 8:
            continue
        sym, fee, avail = p[0].strip(), p[6].strip(), p[7].strip()
        try:
            a = int(avail.lstrip(">").replace(",", "")) if avail else None
        except ValueError:
            a = None
        try:
            f = float(fee) if fee and fee.upper() != "NA" else None
        except ValueError:
            f = None
        out[sym.replace(".", "-").replace(" ", "-")] = {"available": a, "fee": f, "date": stamp,
                                                       "more_than": avail.startswith(">")}
    return out


IB_MIN_ROWS = 1000   # أقل من كذا = ملف IB ما نزل كامل (عطل)


def ib_ok(ib: dict | None) -> bool:
    return bool(ib) and len(ib) >= IB_MIN_ROWS


def short_of(sym: str, ib: dict | None, info: dict | None) -> tuple[int | None, str]:
    """الشورت بتعريف فيصل = المتاح للاقتراض عند IB (تصحيح نوفمبر 2026 بعد OMH، بموافقة المستخدم):
    - السهم موجود بقائمة IB ← رقمه.
    - مو موجود بالقائمة (والقائمة نزلت كاملة) ← ما فيه أسهم متاحة للشورت عند IB = 0.
      (كان الكود ياخذ بداله Short Interest من ياهو — رقم قديم ومختلف — وشال OMH بـ 55,556 وهو غالباً 0 متاح)
    - القائمة ما نزلت (عطل) ← Short Interest من ياهو كاحتياط."""
    if ib_ok(ib):
        x = ib.get(sym)
        if x is None:
            return 0, "غير متاح للشورت عند IB"
        if x.get("available") is not None:
            return x["available"], "المتاح للاقتراض (IB)"
    return (info or {}).get("sharesShort"), "Short Interest (ياهو — احتياط)"


def short_available(t: str) -> dict | None:
    """الشورت المتاح للاقتراض (اللي يقصده فيصل بـ "شورت 10 آلاف") من iBorrowDesk."""
    import requests
    try:
        r = requests.get(f"https://iborrowdesk.com/api/ticker/{t}", timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        js = r.json()
        rows = js.get("daily") or js.get("real_time") or []
        if not rows:
            return None
        last = rows[-1]
        return {"available": last.get("available"), "fee": last.get("fee"),
                "date": str(last.get("date") or last.get("time") or "")[:16]}
    except Exception:
        return None


def deep_scan(results: dict, cfg):
    import yfinance as yf
    S = cfg["scan"]
    cands = sorted(results.values(),
                   key=lambda r: -max(r["irtikaz"]["score"], r["momentum"]["score"]))
    cands = [r for r in cands if max(r["irtikaz"]["score"], r["momentum"]["score"]) >= S["deep_scan_min_score"]]
    cands = cands[:S["deep_scan_max"]]
    if not cands:
        return
    syms = [r["symbol"] for r in cands]
    log(f"تحليل عميق (أخبار + فريم 4 ساعات) لأفضل {len(syms)} سهم...")

    # الأخبار
    def news(t):
        try:
            n = parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, translate_ar)
        return t, n

    with ThreadPoolExecutor(6) as ex:
        for t, n in ex.map(news, syms):
            results[t]["news"] = n
            fresh = False
            for x in n:
                try:
                    ts = pd.Timestamp(x["time"])
                    ts = ts.tz_convert(None) if ts.tzinfo else ts
                    if (pd.Timestamp.utcnow().tz_localize(None) - ts).total_seconds() < 36 * 3600:
                        fresh = True
                except Exception:
                    pass
            m = results[t]["momentum"]
            for c in m["checks"]:
                if c["key"] == "news":
                    c["ok"] = fresh
                    c["got"] = 10 if fresh else 0
                    c["progress"] = 1 if fresh else 0
                    c["detail"] = "يوجد خبر خلال آخر 36 ساعة" if fresh else ("آخر خبر قديم" if n else "لا توجد أخبار")
            m["score"] = round(min(100, m["score"] + (10 if fresh else 0)), 1)

    n_news = sum(len(results[t].get("news", [])) for t in syms)
    n_tr = sum(1 for t in syms for x in results[t].get("news", []) if x.get("title_ar") and x["title_ar"] != x["title"])
    news_tools.save_cache()
    allnews = [x for t in syms for x in results[t].get("news", [])]
    log(f"  أخبار: {n_news} (مترجم {n_tr}) · كاملة {sum(1 for x in allnews if x.get('full'))} · "
        f"إيجابي {sum(1 for x in allnews if x.get('sentiment') == 'positive')} · "
        f"سلبي {sum(1 for x in allnews if x.get('sentiment') == 'negative')}")
    for x in allnews[:40]:
        print(f"    [{x.get('sentiment')}] {x['title'][:110]} | +{x.get('why_pos')} -{x.get('why_neg')}", flush=True)

    # فريم 4 ساعات (يشمل البري والأفتر)
    try:
        h = intraday(syms, "1h", 60)
        for t in syms:
            try:
                d = h[t] if isinstance(h.columns, pd.MultiIndex) else h
                d = d.dropna(subset=["Close"])
                d.index = pd.to_datetime(d.index)
                d4 = d.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                           "Close": "last", "Volume": "sum"}).dropna()
                if len(d4) < 20:
                    continue
                r4 = fr.rsi(d4["Close"]).iloc[-1]
                last30 = d4.tail(30)  # ~5 جلسات كاملة
                lowest = float(last30["Low"].min())
                broken = float(d4["Low"].tail(3).min()) < lowest * 1.0001 and \
                    int(last30["Low"].values.argmin()) >= len(last30) - 3
                # تذبذب 4 ساعات: هل فيه شموع طويلة بدون جسم تصل 20% (قروبات)؟
                rng = ((d4["High"] - d4["Low"]) / d4["Close"] * 100).tail(60)
                pumps4 = int((rng >= 20).sum())
                results[t]["h4"] = {"rsi": round(float(r4), 1), "low5": round(lowest, 4),
                                    "new_low_recent": bool(broken), "big_candles": pumps4}
            except Exception:
                pass
    except Exception as e:
        log("  فريم 4 ساعات ما اشتغل:", repr(e)[:120])

    # الفواب لمرشحين الزخم (5 دقائق اليوم)
    mom = [r["symbol"] for r in cands if r["momentum"]["status"] != "no"][:25]
    if mom:
        try:
            m5 = intraday(mom, "5m", 1)
            for t in mom:
                try:
                    d = m5[t] if isinstance(m5.columns, pd.MultiIndex) else m5
                    d = d.dropna(subset=["Close"])
                    if d.empty:
                        continue
                    tp = (d["High"] + d["Low"] + d["Close"]) / 3
                    vwap = float((tp * d["Volume"]).cumsum().iloc[-1] / max(1, d["Volume"].sum()))
                    last = float(d["Close"].iloc[-1])
                    first5 = d.iloc[0]
                    results[t]["vwap"] = {"vwap": round(vwap, 4), "above": last > vwap,
                                          "first5_high": round(float(first5["High"]), 4)}
                    m = results[t]["momentum"]
                    m["checks"].append({"key": "vwap", "title": "السعر فوق الفواب (VWAP)", "weight": 10,
                                        "got": 10 if last > vwap else 0, "ok": last > vwap,
                                        "detail": f"VWAP {fr.fmt_price(vwap)} / السعر {fr.fmt_price(last)}",
                                        "remaining": "" if last > vwap else f"باقي {fr.pct(vwap, last):.1f}% ليرجع فوق الفواب",
                                        "progress": 1 if last > vwap else 0})
                    if last <= vwap:
                        m["score"] = round(max(0, m["score"] - 10), 1)
                except Exception:
                    pass
        except Exception as e:
            log("  الفواب ما اشتغل:", repr(e)[:120])


# ----------------------------------------------------------------------------
# 4) التشغيل الكامل
# ----------------------------------------------------------------------------

def run_scan(cfg, tickers: list[str] | None = None, deep: bool = True) -> dict:
    t0 = time.time()
    explicit = tickers is not None
    if tickers is None:
        tickers = get_universe(cfg)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not explicit:
        # ناسداك: الرمز الخماسي اللي ينتهي بـ W/R/U = Warrant / Rights / Unit (مو سهم عادي)
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in "WRU") and "-" not in t]
    daily = download_daily(tickers, cfg)
    log(f"وصلت شارتات {len(daily)} سهم")

    U = cfg["universe"]
    pre = {}
    for t, d in daily.items():
        last = float(d["Close"].iloc[-1])
        avgv = float(d["Volume"].tail(30).mean())
        if explicit or (U["min_price"] <= last <= U["max_price"] and avgv >= U["min_avg_volume"] * 0.7):
            pre[t] = d
    log(f"بعد فلتر السعر والفوليوم: {len(pre)} سهم")

    infos = get_info(list(pre), cfg)
    try:
        ib_short = load_ib_short()
        log(f"الشورت المتاح (Interactive Brokers): {len(ib_short)} سهم")
    except Exception as e:
        ib_short = {}
        log("الشورت المتاح ما اشتغل:", repr(e)[:150])
    results = {}
    for t, d in pre.items():
        try:
            inf_t = dict(infos.get(t) or {})
            if ib_ok(ib_short) and (t not in ib_short or ib_short[t].get("available") is not None):
                inf_t["shortAvailable"] = short_of(t, ib_short, inf_t)[0]   # مو بقائمة IB = 0 متاح
            a = fr.analyze(d, inf_t, cfg)
        except Exception:
            log(f"  خطأ في تحليل {t}:", traceback.format_exc(limit=1).splitlines()[-1])
            continue
        if not a:
            continue
        split_rule.apply(a, d)  # منطقة التقسيم (عبدالإله): وسم ومعلومات فقط
        new_low_rule.apply(a, d)  # قاع جديد: ما يترشح لين يتماسك عند الدعم الثاني ويختبره
        compliance_rule.apply_faisal(a, d)  # امتثال ناسداك للأسهم تحت 1$
        if a["gate_fail"] and not explicit:
            continue
        inf = infos.get(t) or {}
        a["symbol"] = t
        a["name"] = inf.get("shortName") or inf.get("longName") or t
        a["sector"] = inf.get("sector") or inf.get("industry")
        a["country"] = inf.get("country")
        a["exchange"] = inf.get("exchange")
        if t in ib_short:
            a["short_available"] = ib_short[t]
        a["_short"] = short_of(t, ib_short, inf)
        results[t] = a

    if deep and results:
        deep_scan(results, cfg)

    # قرار نهائي على فريم 4 ساعات
    for r in results.values():
        h4 = r.get("h4")
        if h4 and h4.get("new_low_recent"):
            r["irtikaz"]["adjustments"].append({"name": "قاع جديد على 4 ساعات", "pts": -5,
                                                "detail": "كسر أدنى قاع آخر 5 جلسات على فريم 4 ساعات"})
            r["irtikaz"]["score"] = max(0, r["irtikaz"]["score"] - 5)

    # شرط الشورت (قرار المستخدم، نوفمبر 2026): فوق 50 ألف ما يترشح (config: behavior.max_short_shares، None = مفتوح)
    max_sh = cfg["behavior"].get("max_short_shares")
    if True:
        for t in list(results):
            r = results[t]
            val, src = r.pop("_short", (None, "غير معروف"))
            r["short_used"] = {"value": val, "source": src, "max": max_sh}
            if not max_sh:
                continue
            if val is None:
                # الشورت غير معروف: ما يترشح (حده الأعلى رادار)
                r["irtikaz"]["adjustments"].append({"name": "الشورت غير معروف", "pts": 0,
                                                    "detail": f"ما يترشح لين يتأكد إن الشورت {max_sh:,} أو أقل"})
                if r["irtikaz"]["status"] in ("ready", "near"):
                    r["irtikaz"]["status"] = "radar"
                if r["momentum"]["status"] != "no":
                    r["momentum"]["status"] = "no"
            elif val > max_sh and not explicit:
                del results[t]

    min_s = cfg["scan"]["report_min_score"]
    rows = [r for r in results.values()
            if explicit or max(r["irtikaz"]["score"], r["momentum"]["score"]) >= min_s]
    rows.sort(key=lambda r: -r["irtikaz"]["score"])
    if deep:
        sec_tools.annotate(rows, log)  # خطر التخفيف من ملفات SEC
    # قرار المستخدم: شركات الطرح/التخفيف المتكرر ما تترشح
    rows, excluded_dil = sec_tools.exclude_repeat_diluters(rows, log)
    # قرار المستخدم: علم الدولة + استبعاد الأسهم اللي صندوق HRT من ملاكها
    import flags
    import holders
    flags.annotate(rows)
    excluded_hrt, hrt_info = [], {}
    if cfg["behavior"].get("exclude_hrt"):
        try:
            hrt_info = holders.annotate(rows, log)
            rows, excluded_hrt = holders.exclude(rows, log)
        except Exception as e:
            log("فحص HRT تعطل:", repr(e)[:150])
    # 🎯 أداة الثقة: احتمال نجاح التوصية (فيصل المطوّر فقط — النسخة الأصلية ما تتغير)
    if getattr(fr, "__name__", "") == "faisal_rules":
        try:
            import confidence
            n_conf = confidence.apply(rows, daily, log)
            if n_conf:
                log(f"أداة الثقة: انحسبت لـ {n_conf} سهم")
                for r in rows:
                    c = r.get("confidence")
                    if c and r["irtikaz"]["status"] in ("ready", "near"):
                        log(f"  🎯 {r['symbol']} {r['irtikaz']['status']} · ثقة {c['p'] * 100:.0f}% ({c['label']}) · "
                            f"معها: {'، '.join(c['pos']) or '-'} · ضدها: {'، '.join(c['neg']) or '-'}")
        except Exception as e:
            log("أداة الثقة تعطلت:", repr(e)[:150])
    # صفحة فيصل المطوّر: الحالات اللي تنعرض من config (scan.dev_page_statuses). قرار المستخدم (نوفمبر 2026):
    # المطابق ✅ ثم القريب 🟡 ثم تحت الرادار 👀 بالترتيب، و«غير مطابق» ما ينعرض. لو «تحت الرادار» مو من الحالات المعروضة
    # يبقى ملخص خفيف مطوي (ليش مو جاهز). النسخة الأصلية تعرض الكل زي ما هي.
    from collections import Counter
    all_rows, hidden, status_first = rows, None, False
    show = cfg["scan"].get("dev_page_statuses")
    if show and getattr(fr, "__name__", "") == "faisal_rules" and not explicit:
        rows = [r for r in all_rows if r["irtikaz"]["status"] in show]
        status_first = True
        if "radar" not in show:
            hidden = [radar_brief(r) for r in all_rows if r["irtikaz"]["status"] == "radar"]
        n = Counter(r["irtikaz"]["status"] for r in rows)
        log(f"صفحة فيصل المطوّر: {len(rows)} سهم (✅ {n['ready']} · 🟡 {n['near']} · 👀 {n['radar']})"
            + (f" · مخفي تحت الرادار {len(hidden)}" if hidden is not None else ""))
    # التوصية المفتوحة (من السجل) لكل سهم: الصفحة تفصل بين «خطة صفقتك» و«التقييم الحالي لشراء جديد»
    # (طلب المستخدم بعد FBGL: التقييم نزل بعد ما اشترى — لأن السعر ابتعد عن الدعم، مو لأن الصفقة خربت)
    try:
        import tracker
        ot = tracker.open_trades("faisal_dev" if getattr(fr, "__name__", "") == "faisal_rules" else "faisal_orig")
        for r in rows:
            if r["symbol"] in ot:
                r["open_trade"] = ot[r["symbol"]]
        if ot:
            log(f"توصيات مفتوحة بالسجل: {len(ot)} · منها بالصفحة {sum(r['symbol'] in ot for r in rows)}")
    except Exception as e:
        log("التوصيات المفتوحة ما انقرت:", repr(e)[:120])
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": "الأسعار: Alpaca · معلومات الشركة والأخبار: Yahoo — تأخير ~15 دقيقة",
        "scanned": len(daily), "passed_filter": len(results), "seconds": round(time.time() - t0),
        "config": cfg, "stocks": rows, "excluded_dilution": excluded_dil, "excluded_hrt": excluded_hrt,
        "hrt_report": {k: hrt_info.get(k) for k in ("name", "period", "filed", "n")} if hrt_info else None,
        "hidden_radar": hidden, "status_first": status_first,
        "status_counts": dict(Counter(r["irtikaz"]["status"] for r in all_rows)),
    }
    (OUT / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    write_csv(all_rows)
    log(f"خلص المسح في {payload['seconds']} ثانية — {len(rows)} سهم في التقرير")
    write_summary(all_rows)
    return payload


def radar_brief(r: dict) -> dict:
    """ملخص خفيف لسهم «تحت الرادار» (بدون شارت ولا أخبار): وين وصل وليش مو جاهز."""
    I, p = r["irtikaz"], r.get("plan") or {}
    why = []
    if I.get("status_before_new_low"):
        why.append("قاع جديد — ينتظر تماسك الدعم الثاني")
    if I.get("status_before_compliance"):
        why.append("امتثال ناسداك (تحت 1$)")
    if any(x.get("name") == "الشورت غير معروف" for x in I.get("adjustments") or []):
        why.append("الشورت غير معروف — ما يترشح لين يتأكد")
    miss = [c.get("remaining") for c in sorted(I.get("checks") or [], key=lambda c: -(c.get("weight") or 0))
            if not c.get("ok") and c.get("remaining")]
    return {"symbol": r["symbol"], "flag": r.get("flag"), "country_ar": r.get("country_ar"), "price": r.get("price"),
            "change_pct": r.get("change_pct"), "score": round(I["score"]), "rsi": r.get("rsi"),
            "dist_support_pct": p.get("dist_support_pct"), "why": why + miss[:2]}


def write_summary(rows):
    """ملخص أفضل الأسهم في سجل التشغيل (ويظهر في صفحة GitHub Actions)."""
    from collections import Counter
    c = Counter(r["irtikaz"]["status"] for r in rows)
    lines = [f"## رادار فيصل: مطابق {c['ready']} · قريب {c['near']} · رادار {c['radar']}", "",
             "| السهم | السعر | الجاهزية | الحالة | RSI | عن الدعم | فلووت | السلوك |", "|---|---|---|---|---|---|---|---|"]
    for r in rows[:20]:
        flt = f"{r['float']/1e6:.1f}M" if r.get("float") else "-"
        lines.append(f"| {r['symbol']} | {r['price']} | {r['irtikaz']['score']:.0f}% | {r['irtikaz']['status']} | "
                     f"{r['rsi']} | {r['plan']['dist_support_pct']}% | {flt} | {'، '.join(r['tags'])} |")
    txt = "\n".join(lines)
    print(txt, flush=True)
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if sp:
        with open(sp, "a", encoding="utf-8") as f:
            f.write(txt + "\n")


def write_csv(rows):
    recs = []
    for r in rows:
        p = r["plan"]
        recs.append({
            "السهم": r["symbol"], "السعر": r["price"], "التغير%": r["change_pct"],
            "جاهزية الارتكاز": r["irtikaz"]["score"], "حالة الارتكاز": r["irtikaz"]["status"],
            "جاهزية الزخم": r["momentum"]["score"], "RSI": r["rsi"],
            "الدعم": p["support"], "البعد عن الدعم%": p["dist_support_pct"],
            "منطقة الطلب": f"{p['entry_zone'][0]} - {p['entry_zone'][1]}", "الوقف": p["stop"],
            "الهدف الأول": p["t1"], "الأهداف الأربعة": " / ".join(str(x) for x in p["four_targets"]),
            "فلووت(M)": round(r["float"] / 1e6, 2) if r.get("float") else None,
            "RVOL": r["rel_volume"], "السلوك": "، ".join(r["tags"]),
        })
    pd.DataFrame(recs).to_csv(OUT / "latest.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------
# 5) موقع محلي فيه زر تحديث
# ----------------------------------------------------------------------------

def serve(cfg, port: int, tickers, interval_min: int):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    state = {"running": False, "last_error": None}
    lock = threading.Lock()

    def scan_bg():
        with lock:
            if state["running"]:
                return
            state["running"] = True
        try:
            p = run_scan(load_cfg(), tickers)
            report.write(p, OUT / "faisal_report.html", live=True)
            state["last_error"] = None
        except Exception as e:
            state["last_error"] = repr(e)
            log("خطأ في المسح:", traceback.format_exc())
        finally:
            state["running"] = False

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/status"):
                latest = OUT / "latest.json"
                gen = None
                if latest.exists():
                    try:
                        gen = json.loads(latest.read_text(encoding="utf-8"))["generated_at"]
                    except Exception:
                        pass
                self._send(200, json.dumps({"running": state["running"], "generated_at": gen,
                                            "error": state["last_error"]}).encode(), "application/json")
            elif self.path.startswith("/api/rescan"):
                threading.Thread(target=scan_bg, daemon=True).start()
                self._send(200, b'{"ok":true}', "application/json")
            else:
                f = OUT / "faisal_report.html"
                if not f.exists():
                    self._send(200, report.waiting_page().encode())
                else:
                    self._send(200, f.read_bytes())

    def loop():
        while True:
            scan_bg()
            if not interval_min:
                return
            time.sleep(interval_min * 60)

    threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}"
    log(f"الموقع شغال: {url}   (Ctrl+C للإيقاف)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="رادار طريقة فيصل")
    ap.add_argument("--tickers", help="أسهم محددة مفصولة بفاصلة")
    ap.add_argument("--watchlist", help="ملف فيه سهم بكل سطر")
    ap.add_argument("--serve", action="store_true", help="تشغيل موقع محلي")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--every", type=int, default=0, help="مع --serve: إعادة المسح كل X دقيقة")
    ap.add_argument("--no-deep", action="store_true", help="بدون أخبار و4 ساعات (أسرع)")
    ap.add_argument("--no-open", action="store_true", help="لا تفتح المتصفح")
    ap.add_argument("--demo", action="store_true", help="تقرير تجريبي ببيانات وهمية")
    ap.add_argument("--config", help="مسار ملف إعدادات بديل")
    ap.add_argument("--out", help="مسار ملف التقرير (يستخدم في GitHub Pages)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config) if args.config else None)
    tickers = None
    if args.tickers:
        tickers = args.tickers.split(",")
    if args.watchlist:
        tickers = (tickers or []) + [l.strip() for l in Path(args.watchlist).read_text(encoding="utf-8").splitlines()
                                     if l.strip() and not l.startswith("#")]

    if args.demo:
        import demo_data
        payload = demo_data.build(cfg)
        out = OUT / "faisal_report_demo.html"
        report.write(payload, out)
        log(f"التقرير التجريبي: {out}")
        if not args.no_open:
            webbrowser.open(out.as_uri())
        return

    if args.serve:
        serve(cfg, args.port, tickers, args.every)
        return

    payload = run_scan(cfg, tickers, deep=not args.no_deep)
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:  # يشتغل على GitHub Actions: زر "تحديث الآن" يفتح صفحة تشغيل المسح
        payload["actions_url"] = f"https://github.com/{repo}/actions/workflows/scan.yml"
    out = Path(args.out) if args.out else OUT / "faisal_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write(payload, out)
    log(f"التقرير: {out}")
    log(f"ملف Excel/CSV: {OUT / 'latest.csv'}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
