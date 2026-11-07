#!/usr/bin/env python3
"""
سجل التوصيات + تنبيهات تيليجرام — لرادارات فيصل (المطوّر والأصلي) وعبدالله.

كل مسح:
  1) يقرأ نتائج الرادارات (output/sig_*.json)
  2) أي سهم جديد صار ✅ مطابق / 🟡 قريب (فيصل) أو 🎯 متماسك / 🟡 بدأ يتماسك (عبدالله)
     ينحفظ كتوصية: تاريخ، سعر الدخول، الوقف، الهدف الأول والثاني.
  3) يتابع كل توصية مفتوحة يوم بيوم: هل وصل الهدف الأول قبل الوقف؟ أقصى ربح؟ أقصى نزول؟
     التوصية تنقفل إذا ضرب الوقف أو الهدف الثاني، أو بعد 20 جلسة.
  4) يرسل تيليجرام: توصية جديدة، وصل هدف، ضرب وقف.
  5) يبني صفحة "📈 سجل التوصيات" بنسب النجاح لكل رادار.

السجل محفوظ في فرع git منفصل (track-data) عشان ما يضيع.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACK_DIR", ROOT / "trackdata"))
DATA.mkdir(exist_ok=True)
TRACK = DATA / "track.json"
TG_STATE = DATA / "telegram.json"
SITE_URL = "https://hatim4231.github.io/faisal-radar"

MAX_SESSIONS = 20
# قرار المستخدم: الوقف الجديد وقواعد القياس الجديدة لفيصل المطوّر وعبدالله فقط
NEW_RULES = {"faisal_dev", "abdullah", "accumulation"}
MUTED = {"accumulation"}   # قرار المستخدم: رادار التجميع نراقب نتايجه بالسجل بدون رسائل تيليجرام
COOLDOWN_DAYS = 7          # ~5 جلسات قبل ما نفتح توصية جديدة لنفس السهم من نفس الرادار
STOP_RULE_VERSION = 2
RADARS = {
    "faisal_dev": ("📡 فيصل المطوّر", ROOT / "output" / "sig_faisal_dev.json"),
    "faisal_orig": ("📜 فيصل الأصلي", ROOT / "output" / "sig_faisal_orig.json"),
    "abdullah": ("🌊 عبدالله", ROOT / "output" / "sig_abdullah.json"),
    "insiders": ("🟢 المطّلعين", ROOT / "output" / "sig_insiders.json"),
    "accumulation": ("🔍 التجميع", ROOT / "output" / "sig_accumulation.json"),
}
STATUS_AR = {"ready": "✅ مطابق", "near": "🟡 قريب"}
STATUS_AR_ABD = {"ready": "🎯 متماسك", "near": "🟡 بدأ يتماسك"}
STATUS_AR_INS = {"ready": "🟢 شراء داخلي قوي", "near": "🟡 شراء داخلي"}
STATUS_AR_ACC = {"ready": "🔍 تجميع مؤكد", "near": "🟡 تجميع محتمل"}


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), "[سجل]", *a, flush=True)


def load(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def fp(x):
    return "-" if x is None else (f"{x:.3f}" if x < 1 else f"{x:.2f}")


# ----------------------------------------------------------------------------
# 1) استخراج التوصيات من نتائج كل رادار
# ----------------------------------------------------------------------------

def signals(key: str, payload: dict) -> list[dict]:
    out = []
    for s in payload.get("stocks", []):
        if key == "accumulation":
            st = s.get("status")
            if st not in ("ready", "near"):
                continue
            tg = s.get("targets") or [None, None]
            out.append({"symbol": s["symbol"], "status": st, "label": STATUS_AR_ACC[st],
                        "entry": s["price"], "stop": s.get("stop"), "t1": tg[0], "t2": tg[1],
                        "score": s.get("score"), "date": s.get("date"), "name": s.get("name"),
                        "why": f"قاعدة {s.get('base_len')} جلسة بين {fp(s.get('support'))} و {fp(s.get('base_top'))}",
                        "dilution": (s.get("dilution") or {}).get("level")})
            continue
        if key == "insiders":
            st = s.get("status")
            if st not in ("ready", "near"):
                continue
            p0 = (s.get("people") or [{}])[0]
            out.append({"symbol": s["symbol"], "status": st, "label": STATUS_AR_INS[st],
                        "entry": s["price"], "stop": s.get("stop"), "t1": s["targets"][0], "t2": s["targets"][1],
                        "score": s.get("score"), "date": s.get("date"), "name": s.get("name"),
                        "why": f"{s['insiders']} مطّلع اشترى بـ ${s['total']:,} (متوسط {fp(s.get('avg_price'))}) · "
                               f"{p0.get('name', '')} — {'، '.join(p0.get('roles', []))} · آخر شراء {s.get('last_buy')}",
                        "dilution": (s.get("dilution") or {}).get("level"),
                        "accum": (s.get("accum") or {}).get("status")})
            continue
        if key == "abdullah":
            st = s.get("status")
            if st not in ("ready", "near"):
                continue
            tg = s.get("targets") or []
            out.append({"symbol": s["symbol"], "status": st, "label": STATUS_AR_ABD[st],
                        "entry": s["price"], "stop": s.get("invalidation"), "zone": s.get("entry"),
                        "t1": tg[0] if tg else None, "t2": tg[-1] if len(tg) > 1 else None,
                        "score": s.get("score"), "date": s.get("date"), "name": s.get("name"),
                        "why": f"موجة {fp(s['wave']['low'])}→{fp(s['wave']['peak'])} · دعم {fp(s.get('support'))} · "
                               f"اختبر {s['consolidation']['touches']} مرات · تماسك {s['consolidation']['len']} جلسة",
                        "dilution": (s.get("dilution") or {}).get("level"),
                        "accum": (s.get("accum") or {}).get("status")})
        else:
            st = s["irtikaz"]["status"]
            if st not in ("ready", "near"):
                continue
            p = s["plan"]
            ft = p.get("four_targets") or []
            out.append({"symbol": s["symbol"], "status": st, "label": STATUS_AR[st],
                        "entry": s["price"], "stop": p.get("stop"), "t1": p.get("t1"), "zone": p.get("entry_zone"),
                        "t2": ft[1] if len(ft) > 1 else None, "score": s["irtikaz"]["score"],
                        "date": s.get("date"), "name": s.get("name"),
                        "why": f"RSI {s.get('rsi')} · دعم {fp(p.get('support'))} (عنه {p.get('dist_support_pct')}%) · "
                               f"منطقة الطلب {fp(p['entry_zone'][0])}–{fp(p['entry_zone'][1])}",
                        "dilution": (s.get("dilution") or {}).get("level"),
                        "accum": (s.get("accum") or {}).get("status")})
    return out


# ----------------------------------------------------------------------------
# 2) المتابعة
# ----------------------------------------------------------------------------

def evaluate(t: dict, d: pd.DataFrame) -> list[str]:
    """يحدّث التوصية من الشارت اليومي ويرجع الأحداث الجديدة (للتنبيه)."""
    events = []
    after = d[d.index > pd.Timestamp(t["date"])]
    t["sessions"] = len(after)
    if len(d):
        t["last"] = round(float(d["Close"].iloc[-1]), 4)
        t["last_date"] = str(d.index[-1].date())
    if after.empty:
        return events
    t["max_gain_pct"] = round((float(after["High"].max()) / t["entry"] - 1) * 100, 1)
    t["max_dd_pct"] = round((float(after["Low"].min()) / t["entry"] - 1) * 100, 1)
    t["ret_pct"] = round((t["last"] / t["entry"] - 1) * 100, 1)
    new_rules = t.get("radar") in NEW_RULES
    for ix, r in after.iterrows():
        day = str(ix.date())
        if new_rules:
            # القواعد الجديدة: الهدف يُفحص أول (لمسه يصير خلال اليوم، والوقف ما ينحسب إلا بإغلاق اليوم تحت الوقف)
            if t.get("t1") and r["High"] >= t["t1"] and t["result"] == "open":
                t["result"], t["t1_date"] = "t1", day
                events.append(f"✅ {t['symbol']} وصل الهدف الأول {fp(t['t1'])} (+{(t['t1'] / t['entry'] - 1) * 100:.0f}%) — "
                              f"{t['radar_name']} · خذ جزء وارفع وقفك")
            if t.get("t2") and r["High"] >= t["t2"] and t["result"] == "t1":
                t["result"], t["closed"] = "t2", day
                events.append(f"🏆 {t['symbol']} وصل الهدف الثاني {fp(t['t2'])} (+{(t['t2'] / t['entry'] - 1) * 100:.0f}%)")
                break
            if t.get("stop") and r["Close"] < t["stop"] and t["result"] in ("open", "t1"):
                if t["result"] == "t1":
                    t["result"], t["closed"] = "t1_then_stop", day
                    events.append(f"🔻 {t['symbol']} أغلق تحت الوقف {fp(t['stop'])} بعد ما وصل الهدف الأول")
                else:
                    t["result"], t["closed"] = "stop", day
                    events.append(f"❌ {t['symbol']} أغلق تحت الوقف {fp(t['stop'])} ({t['radar_name']})")
                break
            if t.get("stop") and r["Low"] < t["stop"] <= r["Close"]:
                t.setdefault("sweeps", []).append(day)  # نزل تحت الوقف ورجع فوقه = مسح سيولة
            continue
        # القواعد القديمة (فيصل الأصلي والمطّلعين): لمس الوقف = وقف، ونفس اليوم فيه الوقف والهدف = وقف (الأحوط)
        if t.get("stop") and r["Low"] <= t["stop"] and t["result"] in ("open", "t1"):
            if t["result"] == "t1":
                t["result"], t["closed"] = "t1_then_stop", day
                events.append(f"🔻 {t['symbol']} رجع وضرب الوقف {fp(t['stop'])} بعد ما وصل الهدف الأول")
            else:
                t["result"], t["closed"] = "stop", day
                events.append(f"❌ {t['symbol']} ضرب الوقف {fp(t['stop'])} ({t['radar_name']})")
            break
        if t.get("t1") and r["High"] >= t["t1"] and t["result"] == "open":
            t["result"], t["t1_date"] = "t1", day
            events.append(f"✅ {t['symbol']} وصل الهدف الأول {fp(t['t1'])} (+{(t['t1'] / t['entry'] - 1) * 100:.0f}%) — "
                          f"{t['radar_name']} · خذ جزء وارفع وقفك")
        if t.get("t2") and r["High"] >= t["t2"] and t["result"] == "t1":
            t["result"], t["closed"] = "t2", day
            events.append(f"🏆 {t['symbol']} وصل الهدف الثاني {fp(t['t2'])} (+{(t['t2'] / t['entry'] - 1) * 100:.0f}%)")
            break
    if t["result"] in ("open", "t1") and t["sessions"] >= MAX_SESSIONS:
        t["closed"] = t.get("last_date")
        t["result"] = "expired" if t["result"] == "open" else "t1_expired"
    return events


def price_alerts(t: dict) -> list[str]:
    """تنبيه لحظي (بتأخير ~15 دقيقة): السعر دخل منطقة الطلب، أو قرّب من الوقف. كل تنبيه مرة وحدة."""
    out = []
    px = t.get("last")
    if px is None or t["result"] not in ("open", "t1"):
        return out
    z = t.get("zone")
    if z and z[0] and z[1] and z[0] <= px <= z[1] and not t.get("zone_alerted"):
        t["zone_alerted"] = t.get("last_date")
        out.append(f"⚡ {t['symbol']} دخل منطقة الطلب {fp(z[0])} – {fp(z[1])} (السعر {fp(px)}) — {t['radar_name']}\n"
                   f"الوقف {fp(t.get('stop'))} · الهدف 1 {fp(t.get('t1'))} · ضع طلب Limit ولا تشتري من العرض")
    st = t.get("stop")
    if st and st < px <= st * 1.03 and not t.get("stop_warned"):
        t["stop_warned"] = t.get("last_date")
        out.append(f"⚠ {t['symbol']} قرّب من الوقف {fp(st)} (السعر {fp(px)}) — {t['radar_name']} · انتبه للإغلاق")
    return out


def daily_prices(syms: list[str]) -> dict[str, pd.DataFrame]:
    """أسعار المتابعة اليومية من Alpaca دائماً (قرار المستخدم). ياهو للطوارئ بس إذا Alpaca تعطل."""
    try:
        import alpaca_data
        if alpaca_data.available():
            out = {}
            for i in range(0, len(syms), 100):
                out.update(alpaca_data.daily(syms[i:i + 100], days=100))
            log(f"أسعار المتابعة من Alpaca: {len(out)} من {len(syms)}")
            return out
        log("⚠ مفاتيح Alpaca مو موجودة — طوارئ: أسعار المتابعة من ياهو")
    except Exception as e:
        log("⚠ Alpaca تعطل — طوارئ: أسعار المتابعة من ياهو:", repr(e)[:120])
    import yfinance as yf
    data = yf.download(syms, period="3mo", interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True, progress=False)
    out = {}
    for s in syms:
        try:
            d = data[s] if isinstance(data.columns, pd.MultiIndex) else data
            d = d.dropna(subset=["Close"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            out[s] = d
        except Exception:
            pass
    return out


def migrate_stops(track: dict) -> int:
    """مرة وحدة: الوقف الجديد للتوصيات القديمة (فيصل المطوّر وعبدالله) وإعادة تقييمها من البداية."""
    if track.get("stop_rule_version", 1) >= STOP_RULE_VERSION:
        return 0
    todo = [t for t in track["trades"] if t["radar"] in NEW_RULES and t.get("stop")]
    for t in todo:
        t["stop_old"] = t["stop"]
        base = t["stop"] / (0.97 if t["radar"] == "faisal_dev" else 0.99)  # الدعم/أدنى سعر من الوقف القديم
        t["stop"] = round(base * 0.90, 4)
        t.update({"result": "open", "closed": None, "t1_date": None, "sweeps": []})
    if todo:
        syms = sorted({t["symbol"] for t in todo})
        data = daily_prices(syms)
        for t in todo:
            try:
                d = data[t["symbol"]]
                evaluate(t, d)  # بدون تنبيهات تيليجرام لكل حدث قديم
            except Exception as e:
                log("خطأ إعادة حساب", t["symbol"], repr(e)[:100])
    track["stop_rule_version"] = STOP_RULE_VERSION
    log(f"إعادة حساب الوقف: {len(todo)} توصية")
    return len(todo)


def run():
    track = load(TRACK, {"trades": []})
    # توصيات انحفظت ببيانات غلط (تُحذف من السجل)
    PURGE = {"insiders:LONA:2026-11-06", "insiders:LONA:2026-11-05"}
    track["trades"] = [t for t in track["trades"] if t.get("id") not in PURGE]
    trades = track["trades"]
    new_events = []
    migrated = migrate_stops(track)
    if migrated:
        new_events.append(f"🔁 أعدت حساب {migrated} توصية (فيصل المطوّر وعبدالله) بالوقف الجديد: "
                          f"أقل سعر عند الدعم − 10%، والوقف ينحسب بإغلاق اليوم تحته.")

    # توصيات جديدة
    for key, (name, path) in RADARS.items():
        payload = load(path, None)
        if not payload:
            log(f"ما فيه نتائج لـ {name}")
            continue
        open_keys = {(t["radar"], t["symbol"]) for t in trades if t["result"] in ("open", "t1")}
        recent_closed = {}
        if key in NEW_RULES:
            for x in trades:
                if x["radar"] == key and x.get("closed"):
                    recent_closed[x["symbol"]] = max(recent_closed.get(x["symbol"], ""), x["closed"])
        for s in signals(key, payload):
            if (key, s["symbol"]) in open_keys or not s.get("entry"):
                continue
            rc = recent_closed.get(s["symbol"])
            if rc and (date.today() - datetime.strptime(rc, "%Y-%m-%d").date()).days < COOLDOWN_DAYS:
                continue  # انقفلت قريب — ننتظر ~5 جلسات
            t = {"id": f"{key}:{s['symbol']}:{s['date']}", "radar": key, "radar_name": name, "sub1": s["entry"] < 1,
                 "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "result": "open", **s}
            if any(x["id"] == t["id"] for x in trades):
                continue
            trades.append(t)
            warn = {"high": "\n⛔ عليه خطر تخفيف SEC عالي", "medium": "\n⚠ عليه ملفات طرح محتملة"}.get(s.get("dilution"), "")
            warn = {"ready": "\n🔍 مؤكد بالتجميع (بصمات مضارب يجمع)", "near": "\n🔍 فيه بصمات تجميع محتملة"}.get(s.get("accum"), "") + warn
            if key in MUTED:
                continue
            new_events.append(
                f"🔔 توصية جديدة — {name}\n{s['label']} <b>{s['symbol']}</b> ({s.get('name') or ''})\n"
                f"السعر {fp(s['entry'])} · الوقف {fp(s['stop'])} · الهدف 1 {fp(s['t1'])}"
                + (f" · الهدف 2 {fp(s['t2'])}" if s.get("t2") else "") + f"\n{s['why']}{warn}")

    # متابعة المفتوحة
    open_t = [t for t in trades if t["result"] in ("open", "t1")]
    if open_t:
        syms = sorted({t["symbol"] for t in open_t})
        log(f"متابعة {len(open_t)} توصية مفتوحة ({len(syms)} سهم)")
        data = daily_prices(syms)
        for t in open_t:
            try:
                d = data[t["symbol"]]
                ev = evaluate(t, d)
                if t["radar"] not in MUTED:
                    new_events += ev + price_alerts(t)
            except Exception as e:
                log("خطأ متابعة", t["symbol"], repr(e)[:100])

    TRACK.write_text(json.dumps(track, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"السجل: {len(trades)} توصية · أحداث جديدة {len(new_events)}")
    for e in new_events:
        log("  ", e.replace("\n", " | "))
    telegram(new_events)
    build_page(track)


# ----------------------------------------------------------------------------
# 3) تيليجرام
# ----------------------------------------------------------------------------

def telegram(events: list[str]):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log("تيليجرام: ما فيه TELEGRAM_BOT_TOKEN — تخطي")
        return
    import requests
    api = f"https://api.telegram.org/bot{token}"
    st = load(TG_STATE, {})
    chat = os.environ.get("TELEGRAM_CHAT_ID") or st.get("chat_id")
    if not chat:
        try:
            ups = requests.get(f"{api}/getUpdates", timeout=20).json().get("result", [])
            chats = [u.get("message", {}).get("chat", {}).get("id") for u in ups if u.get("message")]
            chat = next((c for c in reversed(chats) if c), None)
        except Exception as e:
            log("تيليجرام getUpdates فشل:", repr(e)[:120])
        if not chat:
            log("تيليجرام: ما لقيت محادثة — أرسل /start للبوت من جوالك")
            return
        st["chat_id"] = chat
        TG_STATE.write_text(json.dumps(st), encoding="utf-8")
        events = ["✅ تم ربط رادارات فيصل وعبدالله والمطّلعين بتيليجرام. بتوصلك هنا التوصيات الجديدة والأهداف والوقف."] + events
    sent = 0
    for e in events[:25]:
        try:
            r = requests.post(f"{api}/sendMessage", timeout=20, data={
                "chat_id": chat, "text": e + f"\n\n{SITE_URL}", "parse_mode": "HTML",
                "disable_web_page_preview": "true"})
            sent += r.ok
        except Exception:
            pass
    if len(events) > 25:
        requests.post(f"{api}/sendMessage", timeout=20,
                      data={"chat_id": chat, "text": f"… و {len(events) - 25} تنبيه ثاني — شوف سجل التوصيات\n{SITE_URL}/track/"})
    log(f"تيليجرام: أُرسل {sent} من {len(events)}")


# ----------------------------------------------------------------------------
# 4) صفحة السجل
# ----------------------------------------------------------------------------

def stats(trades):
    # المحسوبة = اللي تحدد مصيرها (وصلت هدف أو وقف أو انتهت). المفتوحة بدون نتيجة ما تنحسب
    closed_like = [t for t in trades if t["result"] != "open"]
    n = len(closed_like)
    hit = sum(t["result"] in ("t1", "t2", "t1_then_stop", "t1_expired") for t in closed_like)
    stop = sum(t["result"] == "stop" for t in closed_like)
    gains = [t.get("max_gain_pct", 0) for t in closed_like]
    rets = [t.get("ret_pct", 0) for t in closed_like]
    return {"n_total": len(trades), "n": n, "hit": hit, "stop": stop,
            "hit_pct": round(hit / n * 100, 1) if n else None,
            "stop_pct": round(stop / n * 100, 1) if n else None,
            "avg_max_gain": round(sum(gains) / n, 1) if n else None,
            "median_ret": round(sorted(rets)[n // 2], 1) if n else None,
            "open": sum(t["result"] in ("open", "t1") for t in trades)}


def build_page(track):
    trades = track["trades"]
    by = {k: stats([t for t in trades if t["radar"] == k]) for k in RADARS}
    by_status = {f"{k}:{s}": stats([t for t in trades if t["radar"] == k and t["status"] == s])
                 for k in RADARS for s in ("ready", "near")}
    for k in RADARS:  # مع تأكيد التجميع مقابل بدونه
        by_status[f"{k}:acc"] = stats([t for t in trades if t["radar"] == k and t.get("accum")])
        by_status[f"{k}:noacc"] = stats([t for t in trades if t["radar"] == k and not t.get("accum")])
    for k in RADARS:  # الأسهم تحت 1$ مقابل فوقه
        by_status[f"{k}:sub1"] = stats([t for t in trades if t["radar"] == k and t.get("entry", 9) < 1])
        by_status[f"{k}:over1"] = stats([t for t in trades if t["radar"] == k and t.get("entry", 0) >= 1])
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "radars": {k: v[0] for k, v in RADARS.items()}, "stats": by, "stats_status": by_status,
               "trades": sorted(trades, key=lambda t: t["opened_at"], reverse=True)}
    html = (ROOT / "template_track.html").read_text(encoding="utf-8")
    out = ROOT / "site" / "track" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")),
                   encoding="utf-8")
    log(f"صفحة السجل: {out}")


if __name__ == "__main__":
    run()
