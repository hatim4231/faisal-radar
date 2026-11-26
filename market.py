"""
🕘 نظرية «سهم الماركت» (عبدالإله @z_v11v، أعاد نشرها فيصل @faisal_kisar مع مثال APUS — طلب أبوفرح 2026-11-26: «سوي رادار اسمه رادار الماركت»).

نص النظرية (الأوقات بتوقيت السعودية = نيويورك + 8 ساعات):
  1) سهم صعد في الأفتر وجنى ربحه وهبط ← يُلغى.
  2) سهم صعد في الأفتر وجنى ربحه في البري وهبط ← يُلغى.
  3) سهم صعد في الأفتر، جنى ربح في الأفتر، هبط في البري وحافظ على أدنى شمعة عند الساعة 3 عصراً (7:00 نيويورك) ← «سهم الماركت».
  دعوم البري مهمة يختبرها بعد افتتاح الماركت بـ 15 إلى 30 دقيقة؛ ثبات الدعم بعد الاختبار ← احتمال صعود 20% وأعلى.
  جلسة لندن بعد 8 مساءً (12:00 نيويورك) يتضح هل يكمل للأفتر. المثال APUS: البري ثابت، الماركت حقق دعم 1.90 وما كسر دعم الأفتر 1.82
  («لو كسر يكون الفشل»)، وحقق هدف 2.48 أعلى من هدف الأفتر، وكمل بالأفتر والليلة اللي بعدها.

ترجمتها لقواعد قابلة للفحص (ثابتة ومشتركة بين الرادار الحي والدراسة diag/market_study.py):
  - يوم D: قفزة أفتر ≥ +15% (أعلى سعر الأفتر ÷ إغلاق الجلسة) بفوليوم أفتر ≥ 50 ألف وفوليوم الجلسة ≥ 100 ألف.
  - دعم الأفتر = أدنى سعر بعد قمة الأفتر (الجني). قاعدة 1: آخر سعر بالأفتر رجّع أكثر من 70% من القفزة ← مُلغى.
  - البري (اليوم التالي 4:00–9:30): قاعدة 2: كسر دعم الأفتر (أدنى من دعم الأفتر −2%) أو آخر سعر بالبري تحته ← مُلغى.
    قاعدة 3: دعم البري = أدنى سعر بالبري لين 7:00، وما انكسر من 7:00 لين الافتتاح ← 🟡 «سهم الماركت».
  - الماركت (9:30–10:00): ✅ «ثبت الدعم» = ما أغلقت شمعة 5 دقايق تحت دعم البري ولا انكسر دعم الأفتر لين 10:00 (الدخول = سعر 10:00)؛
    ❌ «فشل» = إغلاق تحت دعم الأفتر. الهدف +20% · الوقف إغلاق تحت دعم الأفتر.
"""
from __future__ import annotations

import pandas as pd

NY = "America/New_York"
MIN_JUMP, MIN_AH_VOL, MIN_RTH_VOL = 0.15, 50_000, 100_000
MAX_GIVEBACK = 0.70          # قاعدة 1: رجّع أكثر من 70% من قفزة الأفتر = جنى وهبط
AH_BREAK = 0.98              # كسر دعم الأفتر = أدنى من الدعم بـ 2%
PM_CUTOFF = (7, 0)           # 3 عصراً بالسعودية = 7:00 نيويورك
TEST_END = (10, 0)           # اختبار دعم البري بأول 15–30 دقيقة من الماركت
TARGET = 0.20


def _t(ix, hm):
    return (ix.hour, ix.minute) >= hm


def sessions(x: pd.DataFrame):
    """يفصل شموع يوم واحد (توقيت نيويورك) إلى بري/جلسة/أفتر."""
    m = x.index.hour * 60 + x.index.minute
    return x[(m >= 240) & (m < 570)], x[(m >= 570) & (m < 960)], x[(m >= 960) & (m < 1200)]


def ah_event(dayD: pd.DataFrame) -> dict | None:
    """قفزة الأفتر بيوم D (شموع 5 دقايق بتوقيت نيويورك)."""
    _, rth, ah = sessions(dayD)
    if rth.empty or ah.empty:
        return None
    close = float(rth["Close"].iloc[-1])
    hi = float(ah["High"].max())
    if close <= 0 or hi / close - 1 < MIN_JUMP or float(ah["Volume"].sum()) < MIN_AH_VOL or float(rth["Volume"].sum()) < MIN_RTH_VOL:
        return None
    t_hi = ah["High"].idxmax()
    after = ah.loc[t_hi:]
    sup = float(after["Low"].min())
    last = float(ah["Close"].iloc[-1])
    giveback = (hi - last) / (hi - close)
    return {"close": close, "ah_high": hi, "ah_support": sup, "ah_last": last, "jump_pct": round((hi / close - 1) * 100, 1),
            "giveback_pct": round(giveback * 100), "ah_vol": int(ah["Volume"].sum()), "rth_vol": int(rth["Volume"].sum()),
            "faded": giveback > MAX_GIVEBACK}


def classify(ev: dict, dayN: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """حالة السهم باليوم التالي N حسب النظرية. now=None ← اليوم كامل (للدراسة)."""
    out = {"status": None, "why": [], "pm_support": None, "entry": None, "target": None, "stop": ev["ah_support"]}
    if ev["faded"]:
        out.update(status="cancel_ah", why=[f"جنى ربحه بالأفتر وهبط (رجّع {ev['giveback_pct']}% من القفزة)"])
        return out
    if now is not None:
        dayN = dayN[dayN.index <= now]
    pm, rth, _ = sessions(dayN)
    if pm.empty:
        out.update(status="wait_pm", why=["البري ما بدا"])
        return out
    brk = ev["ah_support"] * AH_BREAK
    if float(pm["Low"].min()) < brk or float(pm["Close"].iloc[-1]) < ev["ah_support"]:
        out.update(status="cancel_pm", why=[f"البري كسر دعم الأفتر {ev['ah_support']:.4g}"])
        return out
    early = pm[[not _t(i, PM_CUTOFF) for i in pm.index]]
    late = pm[[_t(i, PM_CUTOFF) for i in pm.index]]
    if early.empty:
        out.update(status="wait_pm", why=["البري بدري — ننتظر 7:00 نيويورك (3 عصراً)"])
        return out
    sup = float(early["Low"].min())
    out["pm_support"] = sup
    if late.empty and (now is None or not _t(now, PM_CUTOFF)):
        out.update(status="wait_pm", why=["ننتظر 7:00 نيويورك (3 عصراً) عشان نعرف هل حافظ على أدنى شمعة"])
        return out
    if not late.empty and float(late["Low"].min()) < sup:
        out.update(status="cancel_pm", why=[f"البري كسر أدنى شمعته ({sup:.4g}) بعد 3 عصراً"])
        return out
    out["status"] = "market"
    out["why"] = [f"حافظ البري على أدنى شمعة {sup:.4g} بعد 3 عصراً", f"دعم الأفتر {ev['ah_support']:.4g} ما انكسر"]
    if rth.empty:
        return out
    test = rth[[not _t(i, TEST_END) for i in rth.index]]
    if test.empty:
        return out
    if (test["Close"] < ev["ah_support"]).any():
        out.update(status="fail", why=out["why"] + [f"❌ الماركت أغلق تحت دعم الأفتر {ev['ah_support']:.4g} بأول نص ساعة = فشل"])
        return out
    if (test["Close"] < sup).any():
        out.update(status="fail", why=out["why"] + [f"❌ أغلق تحت دعم البري {sup:.4g} وقت الاختبار"])
        return out
    if now is not None and not _t(now, TEST_END):
        out.update(status="testing", why=out["why"] + ["⏳ وقت اختبار دعم البري (أول نص ساعة)"])
        return out
    touched = float(test["Low"].min()) <= sup * 1.03
    entry = float(test["Close"].iloc[-1])
    out.update(status="held", entry=entry, target=round(entry * (1 + TARGET), 4), tested=touched,
               why=out["why"] + [("✅ اختبر دعم البري وثبت" if touched else "✅ ثبت فوق دعم البري (ما نزل يختبره)") + " لين 10:00"])
    after = rth[[_t(i, TEST_END) for i in rth.index]]
    if len(after):                                      # الرادار الحي: وش صار بعد الدخول لين الحين
        out["now_price"] = float(after["Close"].iloc[-1])
        out["day_high"] = float(after["High"].max())
        res = result_after(ev, out, after)["res"]       # بالترتيب الزمني: أول شي صار (الهدف أو الوقف)
        if res != "flat":
            out["after"] = res
    return out


def result_after(ev: dict, cls: dict, rest: pd.DataFrame) -> dict:
    """للدراسة: من الدخول (10:00) — هل وصل +20% قبل إغلاق تحت دعم الأفتر، باقي اليوم وبعده بجلسة."""
    e = cls["entry"]
    for _, r in rest.iterrows():
        if r["Close"] < ev["ah_support"]:
            return {"res": "stop", "ret": ev["ah_support"] / e - 1}
        if r["High"] >= e * (1 + TARGET):
            return {"res": "target", "ret": TARGET}
    return {"res": "flat", "ret": float(rest["Close"].iloc[-1]) / e - 1 if len(rest) else 0.0}
