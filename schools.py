"""
🧭 رادار المدارس الثلاث — كلاسيكي + وايكوف + إليوت (طلب المستخدم: «تجمع بين المدارس وتكون ناجحة في اقتناص الأسهم»).

كل كاشف يشتغل على الشارت اليومي لين يوم معيّن بس (بدون نظر للمستقبل) ويرجع «فرصة»:
  {school, setup, name_ar, entry, stop, t1, t2, quality (0-100), parts{...}, why[...]}

الكلاسيكي:  CBR اختراق ثم إعادة اختبار · CDB قاع مزدوج (W) · CBF علم صاعد
وايكوف:     WSP سبرنق + اختبار (هزة تحت الدعم ورجوع بفوليوم هادي) · WLPS قوة (SOS) ثم آخر نقطة دعم (LPS)
إليوت:      EW2 الموجة الثانية (تصحيح 50-78.6% بعد موجة اندفاع) · EABC نهاية تصحيح ABC

التوافق = كم مدرسة (من الثلاث) عندها فرصة فعالة بنفس اليوم.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHOOLS = {"CBR": "classic", "CDB": "classic", "CBF": "classic", "WSP": "wyckoff", "WLPS": "wyckoff",
           "EW2": "elliott", "EABC": "elliott"}
SCHOOL_AR = {"classic": "الكلاسيكية", "wyckoff": "وايكوف", "elliott": "إليوت"}
SETUP_AR = {
    "CBR": "اختراق ثم إعادة اختبار (Retest)", "CDB": "قاع مزدوج W", "CBF": "علم صاعد (Bull Flag)",
    "WSP": "سبرنق + اختبار (وايكوف)", "WLPS": "إشارة قوة ثم آخر نقطة دعم (SOS → LPS)",
    "EW2": "نهاية الموجة الثانية (إليوت)", "EABC": "نهاية تصحيح ABC (إليوت)",
}


# ----------------------------------------------------------------------------
# أدوات
# ----------------------------------------------------------------------------
def _arr(d):
    return (d["Open"].to_numpy(float), d["High"].to_numpy(float), d["Low"].to_numpy(float),
            d["Close"].to_numpy(float), d["Volume"].to_numpy(float))


def zigzag(h, l, thr):
    """قمم وقيعان مؤكدة (سببية): [(index, price, 'H'/'L', confirm_index)] + الضلع الحالي (غير مؤكد)."""
    n = len(h)
    piv = []
    if n < 5:
        return piv, None
    direc = 0
    hi_i, lo_i = 0, 0
    for i in range(1, n):
        if direc >= 0:
            if h[i] >= h[hi_i]:
                hi_i = i
        if direc <= 0:
            if l[i] <= l[lo_i]:
                lo_i = i
        if direc == 0:
            if h[hi_i] >= l[lo_i] * (1 + thr) and hi_i > lo_i:
                piv.append((lo_i, l[lo_i], "L", i))
                direc, hi_i = 1, i if h[i] >= h[hi_i] else hi_i
            elif l[lo_i] <= h[hi_i] * (1 - thr) and lo_i > hi_i:
                piv.append((hi_i, h[hi_i], "H", i))
                direc, lo_i = -1, i if l[i] <= l[lo_i] else lo_i
        elif direc == 1:
            if l[i] <= h[hi_i] * (1 - thr):
                piv.append((hi_i, h[hi_i], "H", i))
                direc, lo_i = -1, i
        else:
            if h[i] >= l[lo_i] * (1 + thr):
                piv.append((lo_i, l[lo_i], "L", i))
                direc, hi_i = 1, i
    cur = ("up", hi_i, h[hi_i]) if direc == 1 else ("down", lo_i, l[lo_i]) if direc == -1 else None
    return piv, cur


def _thr(h, l, c):
    rng = np.median(((h - l) / np.maximum(c, 1e-9))[-60:])
    return float(max(0.10, min(0.35, 2.5 * rng)))


def _vavg(v, a, b):
    a, b = max(0, a), max(a + 1, b)
    return float(np.mean(v[a:b])) if b > a else float("nan")


def _q(x, lo, hi):
    """0..1 خطي بين lo و hi."""
    if hi == lo:
        return 1.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _setup(code, entry, stop, t1, t2, parts, why, extra=None):
    q = int(round(100 * sum(parts.values()) / max(1, len(parts))))
    out = {"setup": code, "school": SCHOOLS[code], "name_ar": SETUP_AR[code], "entry": float(entry),
           "stop": float(stop), "t1": float(t1), "t2": float(t2) if t2 else None, "quality": q,
           "parts": {k: round(float(v), 3) for k, v in parts.items()}, "why": why}
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------
# إليوت
# ----------------------------------------------------------------------------
def elliott_w2(h, l, c, v, piv):
    """موجة 1 اندفاعية من قاع ← تصحيح الموجة 2 (38-88%) بفوليوم أقل ← فرصة قبل الموجة 3."""
    t = len(c) - 1
    conf = [p for p in piv if p[3] <= t]
    if len(conf) < 2 or conf[-1][2] != "H" or conf[-2][2] != "L":
        return None
    (i0, L0, _, _), (i1, H1, _, _) = conf[-2], conf[-1]
    g1 = H1 / L0 - 1
    dur1 = i1 - i0
    if g1 < 0.25 or not (3 <= dur1 <= 60):
        return None
    if i0 < 40 or L0 > np.min(l[max(0, i0 - 60):i0]) * 1.08:
        return None  # الموجة 1 لازم تبدأ من قاع حقيقي
    since = slice(i1 + 1, t + 1)
    if t - i1 < 3 or t - i1 > 45:
        return None
    lo2 = float(np.min(l[since]))
    if lo2 <= L0:
        return None  # كسر بداية الموجة 1 = إبطال
    r_low = (H1 - lo2) / (H1 - L0)
    r_now = (H1 - c[t]) / (H1 - L0)
    if not (0.382 <= r_low <= 0.886) or not (0.2 <= r_now <= 0.82):
        return None
    vol1 = _vavg(v, i0, i1 + 1)
    vbase = _vavg(v, i0 - 50, i0)
    vol2 = _vavg(v, i1 + 1, t + 1)
    vexp, vcon = vol1 / max(1, vbase), vol2 / max(1, vol1)
    turn = 1.0 if (c[t] > c[t - 1] and (c[t] - l[t]) >= 0.5 * (h[t] - l[t] + 1e-9)) else 0.4 if c[t] > lo2 * 1.04 else 0.0
    parts = {"impulse": _q(g1, 0.25, 1.0), "vol_expansion": _q(vexp, 1.2, 3.0), "fib_zone": 1.0 if 0.5 <= r_low <= 0.786 else 0.5,
             "vol_contraction": _q(1.2 - vcon, 0.3, 0.9), "turning": turn,
             "time_ratio": 1.0 if (t - i1) <= 2.0 * dur1 else 0.4}
    stop = min(lo2, H1 - 0.786 * (H1 - L0)) * 0.97
    t2 = lo2 + 1.618 * (H1 - L0)
    why = [f"موجة 1: من {L0:.4g} إلى {H1:.4g} (+{g1 * 100:.0f}%) بفوليوم {vexp:.1f}x",
           f"التصحيح وصل {r_low * 100:.0f}% من الموجة 1 (فيبوناتشي) بفوليوم {vcon:.2f}x من فوليوم الموجة 1",
           f"الإبطال: كسر {L0:.4g} (بداية الموجة 1) · هدف الموجة 3 ≈ {t2:.4g}"]
    return _setup("EW2", c[t], stop, H1, t2, parts, why, {"type": "continuation"})


def elliott_abc(h, l, c, v, piv, cur):
    """تصحيح ABC: A نزول، B ارتداد 38-79%، C نزول ≈ A (0.8-1.7 من A) تحت قاع A ← ارتداد من C."""
    t = len(c) - 1
    conf = [p for p in piv if p[3] <= t]
    if len(conf) < 3:
        return None
    # C إما قاع مؤكد قريب، أو الضلع النازل الحالي (قبل التأكيد) مع ارتداد واضح 4%+
    if len(conf) >= 4 and "".join(p[2] for p in conf[-4:]) == "HLHL":
        (iH0, H0, _, _), (iA, A, _, _), (iB, B, _, _), (iC, C, _, _) = conf[-4:]
    elif "".join(p[2] for p in conf[-3:]) == "HLH" and cur and cur[0] == "down":
        (iH0, H0, _, _), (iA, A, _, _), (iB, B, _, _) = conf[-3:]
        iC, C = cur[1], cur[2]
        if c[t] < C * 1.04 or t - iC > 5:
            return None
    else:
        return None
    lenA, lenC = H0 - A, B - C
    if lenA <= 0 or lenC <= 0 or C >= A:
        return None
    rb = (B - A) / lenA
    if not (0.382 <= rb <= 0.786) or not (0.8 <= lenC / lenA <= 1.75):
        return None
    if t - iC > 8 or l[iC:t + 1].min() < C * 0.99:
        return None
    if (H0 / C - 1) < 0.25:
        return None
    vA, vC = _vavg(v, iH0, iA + 1), _vavg(v, iB, iC + 1)
    rebound = c[t] / C - 1
    parts = {"decline": _q(H0 / C - 1, 0.25, 0.8), "b_retrace": 1.0 if 0.5 <= rb <= 0.618 else 0.6,
             "c_equals_a": 1.0 if 0.9 <= lenC / lenA <= 1.25 else 0.6 if lenC / lenA <= 1.618 else 0.3,
             "c_volume_dry": _q(1.3 - vC / max(1, vA), 0.2, 0.8), "rebound": _q(rebound, 0.03, 0.15)}
    t1, t2 = B, C + 0.618 * (H0 - C)
    why = [f"تصحيح ABC: A {H0:.4g}→{A:.4g} · B ارتد {rb * 100:.0f}% · C نزل لين {C:.4g} (C/A = {lenC / lenA:.2f})",
           f"فوليوم C {vC / max(1, vA):.2f}x من فوليوم A · الارتداد من C +{rebound * 100:.0f}%"]
    return _setup("EABC", c[t], C * 0.97, t1, t2, parts, why, {"type": "bottom"})


# ----------------------------------------------------------------------------
# وايكوف
# ----------------------------------------------------------------------------
def _trading_range(h, l, c, end, n):
    """نطاق تداول في الشموع [end-n, end): الدعم/المقاومة واحتواء الإغلاقات."""
    a = end - n
    if a < 60:
        return None
    hh, ll, cc = h[a:end], l[a:end], c[a:end]
    R, S = float(np.percentile(hh, 90)), float(np.percentile(ll, 10))
    if S <= 0:
        return None
    width = R / S - 1
    inside = float(np.mean((cc >= S * 0.97) & (cc <= R * 1.03)))
    prior_hi = float(np.max(h[max(0, a - 80):a]))
    return {"R": R, "S": S, "width": width, "inside": inside, "markdown": prior_hi / R - 1, "a": a, "n": n}


def wyckoff_spring(o, h, l, c, v):
    """نطاق تجميع بعد هبوط ← سبرنق (كسر كاذب تحت الدعم ورجوع) ← اختبار بفوليوم أقل."""
    t = len(c) - 1
    best = None
    for n in (30, 45, 60, 90):
        for back in range(0, 8):  # يوم السبرنق
            j = t - back
            tr = _trading_range(h, l, c, j, n)
            if not tr or not (0.12 <= tr["width"] <= 0.70) or tr["inside"] < 0.8 or tr["markdown"] < 0.25:
                continue
            S, R = tr["S"], tr["R"]
            if not (l[j] < S * 0.985 and (S - l[j]) / S <= 0.15):
                continue
            # رجع فوق الدعم خلال 3 جلسات، والحين فوقه
            back_in = any(c[k] >= S * 0.99 for k in range(j, min(t, j + 3) + 1))
            if not back_in or c[t] < S * 0.99 or c[t] > S + 0.5 * (R - S):
                continue
            spring_low = float(np.min(l[j:t + 1]))
            if spring_low < l[j] * 0.995:
                continue  # نزل تحت قاع السبرنق = فشل
            vavg = _vavg(v, tr["a"], j)
            vsp = v[j] / max(1, vavg)
            if back >= 1:
                vtest = _vavg(v, j + 1, t + 1) / max(1, v[j])
                tested = 1.0 if vtest < 0.7 else 0.5 if vtest < 1.0 else 0.0
            else:
                vtest, tested = None, 0.3
            a = tr["a"]
            dc = np.diff(c[a - 1:j])
            vv = v[a:j]
            upv, dnv = vv[dc > 0].sum(), vv[dc < 0].sum()
            effort = upv / max(1, dnv)
            parts = {"range_quality": _q(tr["inside"], 0.8, 0.95), "markdown_before": _q(tr["markdown"], 0.25, 1.0),
                     "spring_depth": 1.0 if (S - l[j]) / S <= 0.08 else 0.5, "test": tested,
                     "effort_vs_result": _q(effort, 0.9, 1.6), "close_strength": _q((c[t] - l[t]) / (h[t] - l[t] + 1e-9), 0.3, 0.8)}
            why = [f"نطاق تجميع {n} جلسة بين {S:.4g} و {R:.4g} (+{tr['width'] * 100:.0f}%) بعد هبوط {tr['markdown'] * 100:.0f}%",
                   f"سبرنق: نزل لين {l[j]:.4g} تحت الدعم ورجع (فوليوم {vsp:.1f}x)"
                   + (f" · اختبار بفوليوم {vtest:.2f}x من يوم السبرنق" if vtest is not None else " · ننتظر الاختبار"),
                   f"فوليوم الصعود ÷ النزول داخل النطاق = {effort:.2f}"]
            s = _setup("WSP", c[t], spring_low * 0.97, R, R + (R - S), parts, why,
                       {"type": "bottom", "range": [S, R], "range_len": n})
            if not best or s["quality"] > best["quality"]:
                best = s
        if best:
            break
    return best


def wyckoff_lps(o, h, l, c, v):
    """اختراق النطاق بفوليوم قوي (SOS) ← رجوع هادي لسقف النطاق (LPS/BU) يثبت فوقه."""
    t = len(c) - 1
    best = None
    for k in range(3, 21):  # يوم الـ SOS
        s_i = t - k
        for n in (30, 45, 60, 90):
            tr = _trading_range(h, l, c, s_i, n)
            if not tr or not (0.12 <= tr["width"] <= 0.80) or tr["inside"] < 0.8:
                continue
            S, R = tr["S"], tr["R"]
            vavg = _vavg(v, tr["a"], s_i)
            if not (c[s_i] > R * 1.02 and v[s_i] >= 1.8 * vavg):
                continue
            post_hi = float(np.max(h[s_i:t + 1]))
            low_since = float(np.min(l[s_i + 1:t + 1]))
            if low_since < R * 0.93 or not (R * 0.97 <= c[t] <= R * 1.12) or post_hi < R * 1.08:
                continue
            vpull = _vavg(v, s_i + 1, t + 1) / max(1, _vavg(v, s_i, s_i + 2))
            parts = {"sos_volume": _q(v[s_i] / vavg, 1.8, 5.0), "sos_thrust": _q(post_hi / R - 1, 0.08, 0.5),
                     "quiet_pullback": _q(1.2 - vpull, 0.3, 0.9), "hold_creek": _q(low_since / R, 0.93, 1.0),
                     "range_quality": _q(tr["inside"], 0.8, 0.95)}
            why = [f"نطاق {n} جلسة بين {S:.4g} و {R:.4g} ← اختراق قوي (SOS) بفوليوم {v[s_i] / vavg:.1f}x قبل {k} جلسة",
                   f"رجع لسقف النطاق (LPS) بفوليوم {vpull:.2f}x وثبت فوق {low_since:.4g}"]
            s = _setup("WLPS", c[t], min(low_since, R * 0.95) * 0.97, post_hi, R + (R - S), parts, why,
                       {"type": "continuation", "range": [S, R], "range_len": n})
            if not best or s["quality"] > best["quality"]:
                best = s
            break
        if best:
            break
    return best


# ----------------------------------------------------------------------------
# الكلاسيكي
# ----------------------------------------------------------------------------
def classic_breakout_retest(h, l, c, v, piv):
    """مقاومة أفقية (قمتين+) ← اختراق بفوليوم ← رجوع يختبر المستوى المكسور ويثبت (المقاومة صارت دعم)."""
    t = len(c) - 1
    highs = [p for p in piv if p[2] == "H" and p[3] <= t and t - p[0] <= 160]
    best = None
    for k in range(3, 21):
        b = t - k
        prior = [p for p in highs if p[0] < b - 2]
        if len(prior) < 2:
            continue
        lvls = sorted(p[1] for p in prior)
        for i in range(len(lvls)):
            grp = [x for x in lvls if abs(x / lvls[i] - 1) <= 0.04]
            if len(grp) < 2:
                continue
            L = float(np.mean(grp))
            v50 = _vavg(v, b - 50, b)
            if not (c[b] > L * 1.03 and v[b] >= 2.0 * v50 and c[b - 1] <= L * 1.01):
                continue
            post_hi = float(np.max(h[b:t + 1]))
            low_since = float(np.min(l[b + 1:t + 1])) if t > b else l[b]
            if low_since < L * 0.92 or not (L * 0.97 <= c[t] <= L * 1.10) or post_hi < L * 1.08:
                continue
            base_low = float(np.min(l[max(0, b - 60):b]))
            vpull = _vavg(v, b + 1, t + 1) / max(1, v[b])
            parts = {"touches": _q(len(grp), 2, 4), "breakout_volume": _q(v[b] / v50, 2.0, 6.0),
                     "thrust": _q(post_hi / L - 1, 0.08, 0.5), "quiet_retest": _q(1.0 - vpull, 0.2, 0.8),
                     "hold": _q(low_since / L, 0.92, 1.0)}
            why = [f"مقاومة {L:.4g} انلمست {len(grp)} مرات ← اختراق قبل {k} جلسة بفوليوم {v[b] / v50:.1f}x",
                   f"رجع يختبر المستوى المكسور ({c[t] / L * 100 - 100:+.0f}%) بفوليوم هادي وثبت فوق {low_since:.4g}"]
            s = _setup("CBR", c[t], min(low_since, L * 0.95) * 0.97, post_hi, L + (L - base_low), parts, why,
                       {"type": "continuation", "level": L})
            if not best or s["quality"] > best["quality"]:
                best = s
        if best:
            break
    return best


def classic_double_bottom(h, l, c, v, piv):
    """قاعين متقاربين (±6%) بينهم قمة 12%+ ← السعر يطلع من القاع الثاني نحو خط العنق أو اخترقه قريب."""
    t = len(c) - 1
    lows = [p for p in piv if p[2] == "L" and p[3] <= t and t - p[0] <= 120]
    if len(lows) < 2:
        return None
    (i1, L1, _, _), (i2, L2, _, _) = lows[-2], lows[-1]
    if not (8 <= i2 - i1 <= 80) or abs(L2 / L1 - 1) > 0.06:
        return None
    M = float(np.max(h[i1:i2 + 1]))
    if M / max(L1, L2) - 1 < 0.12:
        return None
    lo = min(L1, L2)
    if l[i2:t + 1].min() < lo * 0.97 or t - i2 > 30:
        return None
    broke = bool(np.any(c[i2:t + 1] > M * 1.01))
    if broke and c[t] < M * 0.95:
        return None
    if not broke and c[t] > M:
        return None
    v1, v2 = _vavg(v, i1 - 2, i1 + 3), _vavg(v, i2 - 2, i2 + 3)
    height = M - lo
    parts = {"symmetry": _q(1 - abs(L2 / L1 - 1) / 0.06, 0, 1), "higher_second": 1.0 if L2 >= L1 else 0.5,
             "second_volume_dry": _q(1.3 - v2 / max(1, v1), 0.2, 0.8), "depth": _q(height / lo, 0.12, 0.5),
             "neckline": 1.0 if broke else _q(c[t] / M, 0.85, 1.0)}
    t1 = M if not broke else M + 0.5 * height
    why = [f"قاع مزدوج W: {L1:.4g} و {L2:.4g} (فرق {abs(L2 / L1 - 1) * 100:.1f}%) · خط العنق {M:.4g}",
           ("اخترق خط العنق ← الهدف المقاس " if broke else "يطلع نحو خط العنق ← بعده الهدف المقاس ") + f"{M + height:.4g}",
           f"فوليوم القاع الثاني {v2 / max(1, v1):.2f}x من الأول"]
    return _setup("CDB", c[t], lo * 0.97, t1, M + height, parts, why, {"type": "bottom", "neckline": M})


def classic_bull_flag(h, l, c, v):
    """سارية (صعود 30%+ خلال ≤10 جلسات بفوليوم قوي) ← علم ضيق هادي بالنص العلوي ← فرصة قبل الكمال."""
    t = len(c) - 1
    best = None
    for k in range(3, 16):  # طول العلم
        top_i = t - k
        if top_i < 70:
            break
        pole_hi = h[top_i]
        if pole_hi < np.max(h[top_i:t + 1]) * 0.999:
            continue
        a = max(0, top_i - 10)
        pole_lo_i = a + int(np.argmin(l[a:top_i + 1]))
        pole_lo = l[pole_lo_i]
        pole = pole_hi / pole_lo - 1
        if pole < 0.30 or top_i - pole_lo_i < 2:
            continue
        v50 = _vavg(v, pole_lo_i - 50, pole_lo_i)
        vpole = _vavg(v, pole_lo_i, top_i + 1)
        if vpole < 2.0 * v50:
            continue
        fl, fh = float(np.min(l[top_i + 1:t + 1])), float(np.max(h[top_i + 1:t + 1]))
        retr = (pole_hi - fl) / (pole_hi - pole_lo)
        if retr > 0.5 or fh / fl - 1 > 0.25:
            continue
        vflag = _vavg(v, top_i + 1, t + 1) / max(1, vpole)
        parts = {"pole": _q(pole, 0.3, 1.0), "pole_volume": _q(vpole / v50, 2.0, 6.0), "shallow": _q(0.5 - retr, 0.0, 0.35),
                 "tight": _q(0.25 - (fh / fl - 1), 0.0, 0.17), "quiet_flag": _q(0.9 - vflag, 0.2, 0.7)}
        why = [f"سارية +{pole * 100:.0f}% بفوليوم {vpole / v50:.1f}x ثم علم {k} جلسات بين {fl:.4g} و {fh:.4g}",
               f"التصحيح {retr * 100:.0f}% من السارية بفوليوم {vflag:.2f}x"]
        s = _setup("CBF", c[t], fl * 0.97, pole_hi, fl + (pole_hi - pole_lo), parts, why, {"type": "continuation"})
        if not best or s["quality"] > best["quality"]:
            best = s
    return best


# ----------------------------------------------------------------------------
# التجميع
# ----------------------------------------------------------------------------
def detect(d: pd.DataFrame) -> dict | None:
    """كل الفرص لسهم في آخر يوم من d. يرجع {setups, schools, confluence, best, context} أو None."""
    if len(d) < 120:
        return None
    d = d.tail(260)
    o, h, l, c, v = _arr(d)
    if c[-1] <= 0 or np.isnan(c[-1]):
        return None
    piv, cur = zigzag(h, l, _thr(h, l, c))
    found = []
    for fn in (lambda: classic_breakout_retest(h, l, c, v, piv), lambda: classic_double_bottom(h, l, c, v, piv),
               lambda: classic_bull_flag(h, l, c, v), lambda: wyckoff_spring(o, h, l, c, v),
               lambda: wyckoff_lps(o, h, l, c, v), lambda: elliott_w2(h, l, c, v, piv),
               lambda: elliott_abc(h, l, c, v, piv, cur)):
        try:
            s = fn()
        except Exception:
            s = None
        if s and s["stop"] < s["entry"] < s["t1"]:
            found.append(s)
    if not found:
        return None
    schools = sorted({s["school"] for s in found})
    # الخطة المجمعة: الوقف = الأعلى بين الوقوف (الأقرب)، الهدف = الأقرب بين الأهداف
    stop = max(s["stop"] for s in found)
    t1 = min(s["t1"] for s in found)
    t2s = [s["t2"] for s in found if s.get("t2") and s["t2"] > t1]
    best = max(found, key=lambda s: s["quality"])
    ma50 = float(np.mean(c[-50:]))
    ctx = {"above_ma50": bool(c[-1] > ma50), "ma50_up": bool(ma50 > float(np.mean(c[-60:-10]))),
           "atr_pct": float(np.mean(((h - l) / c)[-20:]) * 100)}
    return {"setups": found, "schools": schools, "confluence": len(schools), "best": best,
            "plan": {"entry": float(c[-1]), "stop": float(stop), "t1": float(t1), "t2": float(min(t2s)) if t2s else None},
            "types": sorted({s["type"] for s in found}), "context": ctx}
