"""
🧭 المدارس — جدول «العشوائي بنفس الهدف والوقف» للرادار الحي (تشخيص فقط).

يقرا بيانات اختبار المدارس المحفوظة (DATA_RUN = رقم التشغيل اللي فيه diag_out/schools_data.csv.gz)،
ويبني من الصفقات العشوائية جدول: لكل (بعد الهدف %، بعد الوقف %، شريحة التذبذب) كم مرة وصل الهدف الأول
قبل الإغلاق تحت الوقف خلال 10 جلسات (نفس قواعد السجل). الجدول بخطوة 1% ويتأكد إنه يطابق المرجع الدقيق (0.5%).
يطبع JSON بين SCHOOLS_BASE_BEGIN و SCHOOLS_BASE_END ← ينحفظ في main باسم schools_baseline.json.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diag"))
import numpy as np
import pandas as pd
from backtest_schools import geometry_tables, matched_baseline, U_GRID, S_GRID

H = 10
UP = np.arange(1, 81)      # الهدف +1% .. +80%
SP = np.arange(1, 51)      # الوقف −1% .. −50%


def load():
    rid = os.environ.get("DATA_RUN")
    subprocess.run(["gh", "run", "download", rid, "-n", "diag-out", "-D", "diag_in"], check=True)
    return pd.read_csv("diag_in/schools_data.csv.gz")


def coarse(HIT):
    """من جدول 0.5% إلى 1%: U_GRID[i] = 1 + 0.005(i+1) ← +k% عند i = 2k−1 (نفس الشي للوقف)."""
    return HIT[np.ix_(2 * UP - 1, 2 * SP - 1)]


def lookup(T, u_pct, s_pct):
    iu = np.clip(u_pct - 1, 0, len(UP) - 1.0001)
    is_ = np.clip(s_pct - 1, 0, len(SP) - 1.0001)
    a, b = iu.astype(int), is_.astype(int)
    fa, fb = iu - a, is_ - b
    return T[a, b] * (1 - fa) * (1 - fb) + T[a + 1, b] * fa * (1 - fb) + T[a, b + 1] * (1 - fa) * fb + T[a + 1, b + 1] * fa * fb


def main():
    df = load()
    R = df[df.kind == "random"].copy()
    Sx = df[df.kind == "setup"].copy()
    R["ph"] = R.ph.map(json.loads)
    R["pc"] = R.pc.map(json.loads)
    print(f"الفترة {df.date.min()} → {df.date.max()} · عشوائي {len(R)} · فرص {len(Sx)}", flush=True)
    PH, PC = np.array(R.ph.tolist()), np.array(R.pc.tolist())
    qs = np.quantile(R.atr20, [0.2, 0.4, 0.6, 0.8])
    rq = np.searchsorted(qs, R.atr20.values)
    tables, ns = [], []
    for q in range(5):
        m = rq == q
        HIT, _ = geometry_tables(PH[m] if m.sum() >= 200 else PH, PC[m] if m.sum() >= 200 else PC, H)
        tables.append(coarse(HIT))
        ns.append(int(m.sum()))
        print(f"  شريحة التذبذب {q + 1}: {m.sum()} صفقة عشوائية", flush=True)
    # تحقق: الجدول المختصر (1%) مقابل المرجع الدقيق (0.5%) على الخطة المجمعة
    C = Sx[Sx.setup == "COMBO"].copy()
    exact, _ = matched_baseline(C, R, H)
    sq = np.searchsorted(qs, C.atr20.values)
    approx = np.array([lookup(tables[q], u, s) for q, u, s in zip(sq, C.t1_pct.values, C.stop_pct.values)])
    ok = ~np.isnan(exact)
    print(f"تحقق (الخطة المجمعة، {ok.sum()} فرصة): المرجع الدقيق {exact[ok].mean() * 100:.2f}% · الجدول {approx[ok].mean() * 100:.2f}% · "
          f"متوسط الفرق المطلق {np.abs(exact[ok] - approx[ok]).mean() * 100:.2f} نقطة · أقصى فرق {np.abs(exact[ok] - approx[ok]).max() * 100:.2f}", flush=True)
    print(f"نجاح الخطة الفعلي {C.hit10.mean() * 100:.2f}% (للمقارنة)", flush=True)
    out = {"version": datetime.now().strftime("%Y-%m-%d"), "horizon": H,
           "rule": "T1 touched before close<stop within 10 sessions (random entries, same target/stop, same volatility quintile)",
           "period": [str(df.date.min()), str(df.date.max())], "n_random": int(len(R)), "n_by_quintile": ns,
           "atr20_q": [round(float(x), 4) for x in qs], "u_pct": UP.tolist(), "s_pct": SP.tolist(),
           "hit_permille": [np.rint(T * 1000).astype(int).tolist() for T in tables]}
    print("\nSCHOOLS_BASE_BEGIN")
    print(json.dumps(out, separators=(",", ":")))
    print("SCHOOLS_BASE_END")


if __name__ == "__main__":
    main()
