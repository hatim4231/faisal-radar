#!/usr/bin/env python3
"""
🧨 رادار «عالي مخاطرة» — الأسهم اللي تنفجر بشكل متكرر (طلب المستخدم 2026-11-11). **صامت** (بدون تيليجرام).

النسخة 2 (2026-11-12، طلب المستخدم: «طوره وخله ثقة والتوصية تكون مفيدة وقوية وتبني محفظتي ... فاجئني»).
ثلاث دراسات على 7 سنين (diag/backtest_highrisk2.py و backtest_highrisk3.py، فرع diag-flye) قالت:
  - توقع الانفجار صحيح (أعلى 3 كل يوم: 10% تنفجر خلال 10 جلسات مقابل 2% لأسهم مثلها) —
  - **لكن ولا خطة شراء قبل الانفجار ربحت** بعد التكاليف، بكل السنين (20 خطة + نموذج مباشر للربح + أمر بيع بالبري والأفتر).
  - اللي يتوقع بقوة وثبات هو **النزيف**: احتمال السهم يلمس −30% خلال 20 جلسة (معاير: المتوقع ≈ الفعلي).
فالرادار صار أداة «حقيقة وحماية» بدل قائمة شراء:
  🔎 افحص أي سهم (كل الأسهم ~5 آلاف) · 💼 محفظتي (بالمتصفح) · 🩸 فخاخ من توصيات الرادارات الثانية ·
  🎯 أقوى 3 للمراقبة + «التذكرة» (أقل الخطط خسارة، مكتوب إنها يانصيب) · 📒 سجل الصدق (يتحقق من كلام الرادار بنفسه).

بدون أي قيود (قرار المستخدم): كل أسهم Alpaca النشطة — السعر/الشورت/HRT/الطرح/الامتثال **تحذير بس**.
(مو أسهم فما تدخل: ETF والـ Warrants/Rights/Units. وأقل من 5 آلاف$ تداول يومي أو 0.05$ ما ينشرى أصلاً فما يترشح.)

    python highrisk_radar/highrisk_scanner.py --out site/highrisk/index.html
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import scanner as data  # noqa: E402
import sec_tools  # noqa: E402
import news_tools  # noqa: E402
import flags  # noqa: E402
import holders  # noqa: E402
import highrisk as HR  # noqa: E402

log = data.log
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
STORE = ROOT / "trackdata" / "highrisk_log.json"       # 📒 سجل الصدق (فرع track-data، مثل afterhours.json)
ETF_RX = re.compile(r"\b(ETF|ETN|Fund|Trust|ProShares|Direxion|iShares|Leveraged|Daily|2X|3X|Bull|Bear|Ultra|Notes|"
                    r"Warrant|Rights?|Units?|Acquisition|SPDR|Invesco|Vanguard|Global X|MicroSectors|GraniteShares|"
                    r"Tradr|Defiance|T-Rex|YieldMax|Roundhill|Volatility|VIX)\b", re.I)   # نفس الدراسة
# مو أسهم عادية (Warrants بأي حرف مثل OPENZ/AMPGZ — أول تشغيل دخلت وقلبت أعلى القائمة، حقوق، وحدات، ممتازة، سندات)
NOT_COMMON_RX = re.compile(r"\b(Warrants?|Wts?|Rights?|Units?|Preferred|Pref|Depositary Shares? Representing|Notes?|Debentures?|"
                           r"Subordinated|Perpetual|Contingent Value)\b", re.I)
NAMES: dict[str, str] = {}
ST_AR = {"fire": "🔥 انطلق للتو", "ready": "🧨 مرشح قوي", "near": "🟠 مرشح", "watch": "👀 متكرر"}   # 🔥 = اليوم وقت السوق، وإلا آخر جلسة
CAP = {"fire": 20, "ready": 30, "near": 20, "watch": 20}     # المستخدم: «القائمة كثيرة» — والباقي يطلع بـ «افحص أي سهم»
# 🎟 التذكرة (أقل الخطط خسارة بالدراسة — مو أفضلية): دخول افتتاح الجلسة الجاية · أمر بيع Limit +50% شغّال بالبري والأفتر ·
# بدون وقف · اطلع بإغلاق الجلسة 20. أعلى 3: −1.2% للتذكرة (بالجلسة العادية بس −5.4%) · وصلت الهدف 35% (29%).
TICKET = {"limit": 0.50, "days": 20, "margin": 0.02}
E_DAYS, B_DAYS = 10, 20                                    # نافذة الانفجار / النزيف (نفس الدراسة)
RADAR_LO, RADAR_HI = 0.7, 10.0                             # مجال رادارات المستخدم (لفخاخ 🩸)
OTHERS = {  # الرادارات الثانية: نقرا نتايج نفس المسح (output/sig_*.json) عشان «🩸 فخاخ من توصياتك»
    "faisal_dev": ("📡 فيصل المطوّر", "sig_faisal_dev.json", {"ready": "✅ مطابق", "near": "🟡 قريب"}),
    "faisal_orig": ("📜 فيصل الأصلي", "sig_faisal_orig.json", {"ready": "✅ مطابق", "near": "🟡 قريب"}),
    "abdullah": ("🌊 عبدالله", "sig_abdullah.json", {"ready": "🎯 متماسك", "near": "🟡 بدأ يتماسك"}),
    "insiders": ("🟢 المطّلعين", "sig_insiders.json", {"ready": "🟢 قوي", "near": "🟡"}),
    "accumulation": ("🔍 التجميع", "sig_accumulation.json", {"ready": "مؤكد", "near": "محتمل"}),
    "schools": ("🧭 المدارس", "sig_schools.json", {"ready": "🎯 قوي", "near": "🟡 جيد"}),
    "runners": ("🚀 الطيارات", "sig_runners.json", {"ready": "🚀 جاهز", "near": "🟡 قريب"}),
}


def _r(x, n=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, n)


def cost_of(px: float) -> float:
    """تكلفة ذهاب وعودة (فرق السعر والانزلاق) — نفس الدراسة: 4% تحت 1$ · 2% تحت 5$ · 1% فوقها."""
    return 0.04 if px < 1 else 0.02 if px < 5 else 0.01


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York")


def alpaca_universe() -> list[str]:
    """كل الأسهم النشطة من Alpaca (بدون فلاتر) — احتياط: قائمة ياهو/ناسداك إذا Alpaca تعطل. يعبّي NAMES."""
    import requests
    import alpaca_data as A
    syms = []
    for base in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            r = requests.get(f"{base}/v2/assets", headers=A._headers(),
                             params={"status": "active", "asset_class": "us_equity"}, timeout=90)
            r.raise_for_status()
            for a in r.json():
                s, ex, nm = a.get("symbol", ""), a.get("exchange", ""), a.get("name") or ""
                if ex not in ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"):
                    continue
                if not re.fullmatch(r"[A-Z]{1,5}", s) or (len(s) == 5 and s[-1] in "WRUZ") or ETF_RX.search(nm) or NOT_COMMON_RX.search(nm):
                    continue
                syms.append(s)
                NAMES[s] = nm
            if syms:
                break
        except Exception as e:
            log("[عالي مخاطرة] قائمة Alpaca:", repr(e)[:120])
    return sorted(set(syms))


def fetch_daily(tickers: list[str]) -> dict:
    """يومي من Alpaca لكل المجال (~5 آلاف سهم): 420 يوم تكفي (الميزات تحتاج 252 جلسة) — نص حمل download_daily (730 يوم)."""
    import alpaca_data as A
    out = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        for attempt in range(3):
            try:
                out.update({t: d for t, d in A.daily(batch, days=420).items() if len(d) >= 120})
                break
            except Exception as e:
                log(f"  [عالي مخاطرة] Alpaca خطأ (محاولة {attempt + 1}):", repr(e)[:120])
                time.sleep(5 * (attempt + 1))
    log(f"[عالي مخاطرة] اليومي من Alpaca: {len(out)} من {len(tickers)}")
    return out


def ext_highs(syms: list[str], start: pd.Timestamp) -> dict:
    """قمم البري (قبل 9 الصبح) والأفتر (من 4 العصر) لكل (سهم، يوم) من شموع الساعة — نفس قواعد الدراسة:
    الشمعة الخفيفة (تداول أقل من 10 آلاف$ أو أقل من 10 صفقات) ما تنحسب."""
    import alpaca_data as A
    XH = {}
    st = pd.Timestamp(start).tz_localize("America/New_York").tz_convert("UTC").to_pydatetime()
    for j in range(0, len(syms), 30):
        try:
            bars = A.bars(syms[j:j + 30], "1Hour", st)
        except Exception as e:
            log("  [عالي مخاطرة] شموع الساعة:", repr(e)[:100])
            continue
        for s, b in bars.items():
            ix = b.index
            hr = ix.hour.to_numpy()
            hi = b["High"].to_numpy(float)
            px = b["VWAP"].to_numpy(float) if "VWAP" in b else b["Close"].to_numpy(float)
            ok = b["Volume"].to_numpy(float) * px >= 10_000
            if "Trades" in b:
                ok &= b["Trades"].to_numpy(float) >= 10
            g = pd.DataFrame({"D": ix.strftime("%Y-%m-%d"), "pre": np.where((hr <= 8) & ok, hi, np.nan),
                              "post": np.where((hr >= 16) & ok, hi, np.nan)}).groupby("D").max()
            for D_, a_, p_ in zip(g.index, g["pre"].to_numpy(), g["post"].to_numpy()):
                XH[(s, D_)] = (float(a_), float(p_))
    return XH


class Scoreboard:
    """📒 سجل الصدق: كل مسح خارج وقت السوق نسجل وش قال الرادار (أقوى 3، 🧨، وأسهم 🩸 بمجال راداراتنا)، وبعد 10/20 جلسة
    نشوف وش صار فعلاً مقابل كل الأسهم بنفس الأيام — عشان كلام الرادار يتحقق منه بالأرقام مو بالكلام.
    التقييم من نفس بيانات المسح (اليومي لكل الأسهم) + شموع الساعة لتذاكر أقوى 3 بس."""

    def __init__(self, path: Path = STORE, liq: float = 30_000):
        self.path, self.liq = path, liq
        try:
            self.log = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.log = {"days": []}
        self.pend = [x for x in self.log["days"] if not x.get("done")]
        self.dt = pd.DatetimeIndex([pd.Timestamp(x["date"]) for x in self.pend])
        self.acc = [dict(n10=0, e10=0, n20=0, b20=0, nr=0, br=0) for _ in self.pend]
        self.out = [dict() for _ in self.pend]
        self.want = [set(x.get("picks_s", [])) | set(x.get("ready", [])) | set(x.get("b25", [])) | set(x.get("b40", []))
                     for x in self.pend]
        self.cal, self.cal_key = pd.DatetimeIndex([]), None
        self.err = 0

    def observe(self, sym: str, d: pd.DataFrame, F: pd.DataFrame):
        if len(d) and (self.cal_key is None or (d.index[-1], len(d)) > self.cal_key):
            self.cal, self.cal_key = d.index, (d.index[-1], len(d))
        if not self.pend:
            return
        try:
            pos = d.index.get_indexer(self.dt)
        except Exception:
            self.err += 1
            return
        if (pos < 0).all():
            return
        o, h, l, c = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        ex, dv20 = F["ex"].to_numpy(bool), F["dv20"].to_numpy(float)
        n = len(c)
        for j, i in enumerate(pos):
            if i < 0:
                continue
            after = n - 1 - i
            act = bool(c[i] >= HR.ACT_MIN_PX and dv20[i] >= HR.ACT_MIN_DV20)
            a = self.acc[j]
            e = b = None
            if after >= E_DAYS:
                e = bool(ex[i + 1:i + 1 + E_DAYS].any())
                if act:
                    a["n10"] += 1
                    a["e10"] += e
            if after >= B_DAYS:
                b = bool(l[i + 1:i + 1 + B_DAYS].min() / c[i] - 1 <= -0.3)
                if act:
                    a["n20"] += 1
                    a["b20"] += b
                    if RADAR_LO <= c[i] <= RADAR_HI and dv20[i] >= self.liq:
                        a["nr"] += 1
                        a["br"] += b
            if sym in self.want[j]:
                k = min(after, B_DAYS)
                self.out[j][sym] = {"after": after, "e": e, "b": b, "c0": float(c[i]),
                                    "o1": float(o[i + 1]) if after >= 1 else None,
                                    "h": [float(x) for x in h[i + 1:i + 1 + k]], "c": [float(x) for x in c[i + 1:i + 1 + k]],
                                    "dates": [str(x.date()) for x in d.index[i + 1:i + 1 + k]]}

    def _ticket(self, o: dict, XH: dict, sym: str) -> dict | None:
        """التذكرة: دخول افتتاح أول جلسة · Limit +50% (البري من ثاني جلسة، الأفتر ما ينحسب بآخر جلسة) · بدون وقف · 20 جلسة."""
        if not o or not o.get("o1") or o["o1"] <= 0:
            return None
        ent, tgt = o["o1"], o["o1"] * (1 + TICKET["limit"])
        hit = src = None
        for k, (hk, dk) in enumerate(zip(o["h"], o["dates"])):
            pre, post = XH.get((sym, dk), (float("nan"), float("nan")))
            if k > 0 and pre == pre and pre >= tgt * (1 + TICKET["margin"]):
                hit, src = dk, "ext"
            elif hk >= tgt:
                hit, src = dk, "reg"
            elif k < B_DAYS - 1 and post == post and post >= tgt * (1 + TICKET["margin"]):
                hit, src = dk, "ext"
            if hit:
                break
        t = {"entry": round(ent, 4), "limit": round(tgt, 4), "hit": hit, "src": src, "n": len(o["h"])}
        if hit or len(o["c"]) >= B_DAYS:
            t["ret"] = round((TICKET["limit"] if hit else o["c"][B_DAYS - 1] / ent - 1) - cost_of(o["c0"]), 4)
        return t

    def finalize(self, off_hours: bool):
        """يكمّل نتايج الأيام اللي نضجت (10 جلسات للانفجار، 20 للنزيف والتذاكر)."""
        if not self.pend or not len(self.cal):
            return
        psyms = sorted({s for j, x in enumerate(self.pend) for s in x.get("picks_s", []) if self.out[j].get(s, {}).get("o1")})
        XH = {}
        if off_hours and psyms:
            try:
                XH = ext_highs(psyms, self.dt.min() + pd.Timedelta(days=1))
            except Exception as e:
                log("[عالي مخاطرة] شموع الساعة للتذاكر تعطلت:", repr(e)[:100])
        for j, x in enumerate(self.pend):
            k, a, o = int((self.cal > self.dt[j]).sum()), self.acc[j], self.out[j]
            cnt = lambda key, f: [o[s][f] for s in x.get(key, []) if s in o and o[s][f] is not None]
            if not off_hours:                        # وقت السوق: لا شموع ساعة ولا نتايج نهائية — نخلي آخر حالة
                continue
            new = {s: self._ticket(o.get(s), XH, s) for s in x.get("picks_s", [])}
            x["tickets"] = {**(x.get("tickets") or {}), **{s: t for s, t in new.items() if t}}   # السهم اللي اختفت بياناته يبقى بآخر حالة
            if k >= E_DAYS and "res10" not in x and a["n10"]:
                r = cnt("ready", "e")
                x["res10"] = {"base_n": a["n10"], "base_e": a["e10"], "ready_n": len(r), "ready_e": int(sum(r)),
                              "picks": {s: o[s]["e"] for s in x.get("picks_s", []) if s in o and o[s]["e"] is not None}}
            if k >= B_DAYS and a["n20"]:
                b25, b40 = cnt("b25", "b"), cnt("b40", "b")
                x["res20"] = {"all_n": a["n20"], "all_b": a["b20"], "radar_n": a["nr"], "radar_b": a["br"],
                              "b25_n": len(b25), "b25_b": int(sum(b25)), "b40_n": len(b40), "b40_b": int(sum(b40))}
                x["done"] = True
                for key in ("ready", "b25", "b40"):
                    x.pop(key, None)

    def add_day(self, date: str, picks: list[dict], ready: list[str], b25: list[str], b40: list[str]):
        if any(x["date"] == date for x in self.log["days"]):
            return False
        self.log["days"].append({"date": date, "picks": picks, "picks_s": [p["s"] for p in picks],
                                 "ready": ready, "b25": b25, "b40": b40})
        self.log["days"] = self.log["days"][-400:]
        return True

    def save(self):
        if self.path.parent.exists():
            self.path.write_text(json.dumps(self.log, ensure_ascii=False), encoding="utf-8")

    def summary(self) -> dict:
        D = self.log["days"]
        S = {"days": len(D), "start": D[0]["date"] if D else None,
             "picks": {"n": 0, "e": 0, "wait": 0}, "ready": {"n": 0, "e": 0}, "base10": {"n": 0, "e": 0},
             "b40": {"n": 0, "b": 0}, "b25": {"n": 0, "b": 0}, "radar": {"n": 0, "b": 0}, "all": {"n": 0, "b": 0},
             "tickets": {"n": 0, "hit": 0, "ext": 0, "sum": 0.0, "open": 0}, "recent": []}
        for x in D:
            r10, r20 = x.get("res10"), x.get("res20")
            if r10:
                S["picks"]["n"] += len(r10["picks"])
                S["picks"]["e"] += sum(bool(v) for v in r10["picks"].values())
                S["ready"]["n"] += r10["ready_n"]
                S["ready"]["e"] += r10["ready_e"]
                S["base10"]["n"] += r10["base_n"]
                S["base10"]["e"] += r10["base_e"]
            else:
                S["picks"]["wait"] += len(x.get("picks_s", []))
            if r20:
                for k in ("b40", "b25", "radar", "all"):
                    S[k]["n"] += r20[k + "_n"]
                    S[k]["b"] += r20[k + "_b"]
            for s, t in (x.get("tickets") or {}).items():
                if "ret" in t:
                    S["tickets"]["n"] += 1
                    S["tickets"]["hit"] += bool(t["hit"])
                    S["tickets"]["ext"] += t.get("src") == "ext"
                    S["tickets"]["sum"] += t["ret"]
                else:
                    S["tickets"]["open"] += 1
        for x in D[-12:][::-1]:
            for p in x.get("picks", []):
                t = (x.get("tickets") or {}).get(p["s"]) or {}
                e = ((x.get("res10") or {}).get("picks") or {}).get(p["s"])
                S["recent"].append({"date": x["date"], "s": p["s"], "c": p.get("c"), "e": e, "entry": t.get("entry"),
                                    "hit": t.get("hit"), "src": t.get("src"), "ret": t.get("ret"), "n": t.get("n")})
        S["recent"] = S["recent"][:24]
        return S


def others_map() -> dict[str, list[str]]:
    """أسهم الرادارات الثانية اليوم (✅/🟡) + التوصيات المفتوحة بالسجل — عشان نقول أي منها عليه خطر نزيف عالي."""
    out: dict[str, list[str]] = {}
    for key, (name, fn, lab) in OTHERS.items():
        try:
            p = json.loads((OUT / fn).read_text(encoding="utf-8"))
        except Exception:
            continue
        for s in p.get("stocks", []):
            if s.get("status") in lab and s.get("symbol"):
                out.setdefault(s["symbol"], []).append(f"{name} {lab[s['status']]}")
    try:
        tr = json.loads((ROOT / "trackdata" / "track.json").read_text(encoding="utf-8"))
        for t in tr.get("trades", []):
            if t.get("result") in ("open", "t1") and t.get("radar") != "highrisk" and not t.get("pending"):
                lab = f"📌 توصية مفتوحة بالسجل — {t.get('radar_name') or t.get('radar')} ({t.get('date')})"
                if lab not in out.setdefault(t["symbol"], []):
                    out[t["symbol"]].append(lab)
    except Exception:
        pass
    return {k: v[:4] for k, v in out.items()}


def run(cfg, tickers=None) -> dict:
    t0 = time.time()
    M = HR.model()
    if not M:
        log("[عالي مخاطرة] ⚠ ما فيه highrisk_model.json — الرادار ما يشتغل")
    if tickers is None:
        tickers = alpaca_universe()
        log(f"[عالي مخاطرة] المجال: {len(tickers)} سهم من Alpaca (بدون فلاتر)")
        if len(tickers) < 500:
            tickers = data.get_universe(dict(cfg, universe=dict(cfg["universe"], min_price=0.0001, max_price=1_000_000,
                                                                min_avg_volume=0)))
    daily = fetch_daily(tickers)
    now = now_ny()
    today = now.normalize().tz_localize(None)
    rth = now.weekday() < 5 and (9 * 60 + 30) <= now.hour * 60 + now.minute < 16 * 60
    off_hours = not rth          # السجل يسجل بس من مسح خارج وقت السوق (بدون نظرة للمستقبل)
    TH = (M or {}).get("thresholds") or {}
    LIQ = float((M or {}).get("liq") or 30_000)
    p_ready, p_near = TH.get("ready", 1.0), TH.get("near", 1.0)
    sb = Scoreboard(liq=LIQ)
    rows_all = []
    for t in list(daily):
        d = daily.pop(t)
        part = None
        if rth and len(d) and d.index[-1].normalize() >= today:
            part, d = d.iloc[-1], d.iloc[:-1]      # شمعة اليوم الناقصة
        if len(d) < 120:
            continue
        try:
            F = HR.features(d)
            last = F.iloc[-1]
            sc = HR.score(last)
        except Exception as e:
            log("  [عالي مخاطرة] خطأ", t, repr(e)[:100])
            continue
        if not sc:
            continue
        sb.observe(t, d, F)
        nfull = len(d)
        lr = d.iloc[-1]
        jump_last = float(last["jump"]) if np.isfinite(last["jump"]) else 1.0
        if part is not None:
            jump = float(part["High"]) / float(lr["Close"])
            dvd = float(part["Volume"]) * float(part["High"] + part["Low"] + part["Close"]) / 3
            d = pd.concat([d, part.to_frame().T])  # للعرض (السعر الحالي والشارت)
        else:
            jump, dvd = jump_last, float(lr["Volume"]) * float(lr["High"] + lr["Low"] + lr["Close"]) / 3
        px, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
        dv20 = float(last["dv20"]) if np.isfinite(last["dv20"]) else 0.0
        # نشط = ينشرى فعلاً (نفس حد الدراسة). وآخر جلسة مكتملة لو فيها ×2 الدراسة ما قيّمتها (ما تترشح 🧨/🟠 ذاك اليوم)
        active = float(lr["Close"]) >= HR.ACT_MIN_PX and dv20 >= HR.ACT_MIN_DV20
        p = sc["p_raw"]
        st = None
        if jump >= 1.5 and dvd >= HR.EX_MIN_DV and px >= HR.ACT_MIN_PX:
            st = "fire"                                  # انطلق (اليوم وقت السوق، وإلا آخر جلسة) بتداول حقيقي
        elif active and jump_last < 2:
            st = "ready" if p >= p_ready else "near" if p >= p_near else None
        if st is None and active and last["n_ep252"] >= 2:
            st = "watch"
        rows_all.append({"t": t, "d": d if st else None, "F": F if st else None, "last": last, "sc": sc, "jump": jump,
                         "status": st, "active": active, "px": px, "chg": (px / prev - 1) * 100 if prev > 0 else None,
                         "dv20": dv20, "short_hist": nfull < 260, "date": str(d.index[-1])[:10],
                         "sig_date": str(lr.name)[:10], "sig_close": float(lr["Close"])})
    n_scanned = len(rows_all)
    if not rows_all:
        return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": [], "scanned": n_scanned}
    sig_date = max(x["sig_date"] for x in rows_all)      # آخر جلسة مكتملة
    for x in rows_all:
        x["stale"] = x["sig_date"] < sig_date            # ما تداول بآخر جلسة ← ما يترشح (الدراسة تختار من أسهم ذاك اليوم بس)
        if x["stale"] and x["status"]:
            x.update(status=None, d=None, F=None)
    liq_ok = lambda x: x["dv20"] >= LIQ
    key = {"fire": lambda x: -x["jump"], "ready": lambda x: (not liq_ok(x), -x["sc"]["p_raw"]),
           "near": lambda x: (not liq_ok(x), -x["sc"]["p_raw"]), "watch": lambda x: -x["sc"]["p_raw"]}
    # 🎯 أقوى 3 للمراقبة = نفس مجموعة «أعلى 3 كل يوم» بالدراسة: أعلى احتمال بسيولة ≥ 30 ألف$ (وتاريخ كامل، وما انفجر بآخر جلسة)
    cand = sorted((x for x in rows_all if x["status"] == "ready" and liq_ok(x) and not x["short_hist"]), key=lambda x: -x["sc"]["p_raw"])
    picks = [x["t"] for x in cand[:3]]
    shown = []
    for k in ("fire", "ready", "near", "watch"):
        grp = sorted((x for x in rows_all if x["status"] == k), key=key[k])
        shown += grp[:CAP[k]] + [x for x in grp[CAP[k]:] if x["t"] in picks]
    infos = data.get_info([x["t"] for x in shown], cfg)
    try:
        ib = data.load_ib_short()
    except Exception:
        ib = {}
    rows = []
    for x in shown:
        t, d, F, last, sc = x["t"], x["d"], x["F"], x["last"], x["sc"]
        inf = infos.get(t) or {}
        px = x["px"]
        sv, short_src = data.short_of(t, ib, inf)
        hist = HR.history(d, F)
        tail = d.tail(120)
        epdates = set(e["date"] for e in hist["episodes"])
        row = {"symbol": t, "name": inf.get("shortName") or inf.get("longName") or NAMES.get(t) or t,
               "sector": inf.get("sector") or inf.get("industry"), "country": inf.get("country"),
               "price": _r(px), "change_pct": _r(x["chg"], 2),
               "jump_today_pct": _r((x["jump"] - 1) * 100, 1), "status": x["status"], "status_ar": ST_AR[x["status"]],
               "market_cap": inf.get("marketCap"), "float": inf.get("floatShares"), "short": sv, "short_src": short_src,
               "dv20": _r(x["dv20"], 0), "vol20": int(d["Volume"].tail(20).mean()), "date": x["date"],
               "p_raw": _r(sc["p_raw"], 4), "pb": _r(sc["pb"], 3), "liq": bool(liq_ok(x)),
               "why": [[f, _r(v, 3)] for f, v in sc["top"]], "why_b": [[f, _r(v, 3)] for f, v in sc["bleed_top"]],
               "fp": {k: _r(last[k], 3) for k in ("vol5_60", "vol1_20", "atr5_60", "rng20", "pos60", "dd252", "ret5", "ret20",
                                                     "rsi", "closepos10", "upv10", "obv20", "hl_up", "base_ratio", "from_peak")},
               "n_year": hist["n_year"], "days_since": hist["days_since"], "cycle_days": hist["cycle_days"],
               "episodes": hist["episodes"][-8:], "reverse_splits": hist["reverse_splits"][-4:],
               "short_hist": x["short_hist"], "pick": (picks.index(t) + 1) if t in picks else None,
               "plan": {"limit_ref": _r(px * (1 + TICKET["limit"])), "days": TICKET["days"], "base_low": _r(float(last["lo20"]))},
               "flags": [], "tags": [],
               "chart": {"d": [str(i)[:10] for i in tail.index], "o": [_r(v) for v in tail["Open"]],
                         "h": [_r(v) for v in tail["High"]], "l": [_r(v) for v in tail["Low"]],
                         "c": [_r(v) for v in tail["Close"]], "v": [int(v) for v in tail["Volume"]],
                         "ex": [str(i)[:10] in epdates for i in tail.index]}}
        if row["short_hist"]:
            row["flags"].append("سهم جديد (أقل من سنة تداول) — تاريخ الانفجارات ناقص والاحتمال أقل دقة")
        if not row["liq"]:
            row["flags"].append(f"💧 سيولة ضعيفة (وسيط التداول اليومي أقل من {int(LIQ / 1000)} ألف$) — صعب تدخل وتطلع")
        rsd = int(last["days_rs_raw"])
        if rsd <= 60:
            row["flags"].append(f"✂️ تقسيم عكسي قبل {rsd} جلسة")
        if px < 1:
            row["flags"].append("🏛 تحت 1$ — خطر إنذار ناسداك/تقسيم عكسي")
        rows.append(row)
    # معلومات وتحذيرات (ما تشيل شي — قرار المستخدم: كل شي مفتوح)
    import yfinance as yf

    def news(t):
        try:
            n = data.parse_news(yf.Ticker(t).news)
        except Exception:
            n = []
        news_tools.enrich(n, data.translate_ar)
        return t, n
    top = [r for r in rows if r["pick"]] + [r for r in rows if not r["pick"]][:30]
    with ThreadPoolExecutor(6) as ex:
        nm = dict(ex.map(news, [r["symbol"] for r in top]))
    for r in rows:
        r["news"] = nm.get(r["symbol"], [])
    news_tools.save_cache()
    try:
        sec_tools.annotate(rows[:60], log)
        for r in rows[:60]:
            ev = sec_tools.dilution_events(r)
            if len(ev) >= 2:
                r["dilution_events"] = [{"date": e["date"], "what": " + ".join(e["what"][:3])} for e in ev]
    except Exception as e:
        log("[عالي مخاطرة] SEC تعطل:", repr(e)[:120])
    flags.annotate(rows)
    try:
        holders.annotate(rows, log)
    except Exception as e:
        log("[عالي مخاطرة] HRT تعطل:", repr(e)[:120])
    # 🔎 كل الأسهم بشكل مضغوط — للفاحص ومحفظتي (أي سهم حتى لو مو بالقائمة)
    SC = {None: 0, "fire": 1, "ready": 2, "near": 3, "watch": 4}
    cols = ["s", "n", "c", "ch", "p", "pb", "ny", "ds", "dd", "dv", "st", "rs", "sh", "j", "r20", "ac"]
    allrows = []
    for x in rows_all:
        L = x["last"]
        ds, rs = int(L["days_ep_raw"]), int(L["days_rs_raw"])
        allrows.append([x["t"], (NAMES.get(x["t"]) or "")[:30], _r(x["px"]), _r(x["chg"], 1), _r(x["sc"]["p_raw"], 4), _r(x["sc"]["pb"], 3),
                        int(L["n_ep252"]), ds if ds < 9999 else None, _r(L["dd252"], 2), int(x["dv20"]), SC[x["status"]],
                        rs if rs <= 252 else None, int(x["short_hist"]), _r((x["jump"] - 1) * 100, 0), _r(L["ret20"], 2),
                        2 if x["stale"] else int(x["active"])])
    # 📒 سجل الصدق: كمّل اللي نضج، وسجل اليوم (مسح خارج وقت السوق بس — قرار قبل ما تبدأ الجلسة اللي بعدها)
    score = {}
    try:
        sb.finalize(off_hours)
        if off_hours and picks:
            by = {x["t"]: x for x in rows_all}
            rr = [x for x in rows_all if x["active"] and liq_ok(x) and RADAR_LO <= x["sig_close"] <= RADAR_HI and x["sc"]["pb"] is not None]
            added = sb.add_day(sig_date, [{"s": t, "c": _r(by[t]["sig_close"]), "p": _r(by[t]["sc"]["p_raw"], 4), "pb": _r(by[t]["sc"]["pb"], 3)}
                                          for t in picks],
                               [x["t"] for x in rows_all if x["status"] == "ready"],
                               [x["t"] for x in rr if 0.25 <= x["sc"]["pb"] < 0.40], [x["t"] for x in rr if x["sc"]["pb"] >= 0.40])
            if added:
                log(f"[عالي مخاطرة] 📒 سجل الصدق: انضاف يوم {sig_date}")
        sb.save()
        score = sb.summary()
    except Exception as e:
        log("[عالي مخاطرة] سجل الصدق تعطل:", repr(e)[:160])
    cnt = Counter(r["status"] for r in rows)
    tot = Counter(x["status"] for x in rows_all if x["status"])
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": "الأسعار: Alpaca + الشركة: Yahoo + SEC", "scanned": n_scanned, "scored": len(rows_all),
               "seconds": round(time.time() - t0), "stocks": rows, "ticket": TICKET,
               "model": {k: (M or {}).get(k) for k in ("version", "base_e10", "base_e5", "base_bleed", "n_rows", "period", "liq",
                                                        "thresholds", "tiers", "cal", "top3_year", "ext", "study")},
               "counts": dict(cnt), "totals": dict(tot), "serial_total": int(sum(1 for x in rows_all if x["last"]["n_ep252"] >= 2)),
               "rth": rth, "sig_date": sig_date, "picks": picks, "all": {"cols": cols, "rows": allrows}, "others": others_map(), "score": score,
               "radar": [RADAR_LO, RADAR_HI]}
    log(f"🎯 أقوى 3 للمراقبة: {', '.join(picks) or '-'}")
    log(f"## رادار عالي مخاطرة: انطلق {tot['fire']} · مرشح قوي {tot['ready']} · مرشح {tot['near']} · متكرر {tot['watch']}"
        f" · من {len(rows_all)} سهم · خلص في {payload['seconds']} ثانية")
    for r in rows[:15]:
        log(f"  {r['symbol']} {r['price']} {r['status']} · احتمال خام {r['p_raw']} · 🩸 خطر −30% خلال 20 جلسة {r['pb']} · "
            f"انفجر {r['n_year']} مرة/سنة · آخر انفجار قبل {r['days_since']} جلسة")
    return payload


def main():
    ap = argparse.ArgumentParser(description="رادار عالي مخاطرة")
    ap.add_argument("--out", default=str(OUT / "highrisk.html"))
    ap.add_argument("--tickers")
    a = ap.parse_args()
    cfg = data.load_cfg()
    payload = run(cfg, a.tickers.split(",") if a.tickers else None)
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    html = (HERE / "template.html").read_text(encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")),
                   encoding="utf-8")
    # للسجل العام (tracker.py): بدون جدول كل الأسهم — والتوصيات الجديدة ما عاد تنحفظ هناك (السجل صار «سجل الصدق» بصفحة الرادار)
    (OUT / "sig_highrisk.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "all"}, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    log(f"[عالي مخاطرة] التقرير: {out}")


if __name__ == "__main__":
    main()
