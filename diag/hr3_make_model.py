"""يبني highrisk_model.json من سجلات تشغيل diag/backtest_highrisk3.py (فرع diag-flye) — أداة مرة وحدة لكل تحديث.

    python diag/hr3_make_model.py <سجل التشغيل الكامل> [<سجل التشغيل السريع>] > highrisk_model.json

التشغيل الكامل (بدون DATA_RUN) فيه قسم التداول الممتد `ext`؛ السريع (DATA_RUN=fast) فيه جداول المعايرة `cal` بس.
لو التشغيل الكامل فيه `cal` (السكربت الحالي) يكفي لحاله. بلوك `study` أرقام مكتوبة باليد من نفس السجلات (المصدر جنب كل رقم).
"""
import json
import sys


def grab(fn):
    s = open(fn, encoding="utf-8").read()
    a, b = s.index("HR3_BEGIN"), s.index("HR3_END")
    return json.loads("".join(ln.split("H3|", 1)[1] for ln in s[a:b].splitlines() if "H3|" in ln))


full = grab(sys.argv[1])
fast = grab(sys.argv[2]) if len(sys.argv) > 2 else full
M = {k: fast[k] for k in ("version", "feats", "w", "feats_b", "wb", "mean", "std", "thresholds", "liq",
                          "base_e10", "base_e5", "base_bleed", "n_rows", "period", "cal", "top3_year")}
M["tiers"] = {k: {a: b for a, b in v.items() if a != "by_year_o5"} for k, v in fast["tiers"].items()}
KEEP = ("0.5:None:20", "0.3:None:20", "1.0:None:20", "0.5:0.3:20", "0.5:None:10", "lad")
M["ext"] = {g: {k: v for k, v in e.items() if k in KEEP} for g, e in (full.get("ext") or {}).items()}
# أرقام الدراسة للنصوص (بعد التكاليف بالسعر الحقيقي، خارج العينة 2021 ← 2026) — من سجلات التشغيل:
M["study"] = {
    "stocks": 5460, "rows": 3577143, "from": "2019-09", "to": "2026-10", "oos": "2021 ← 2026",   # run 43175273449
    "auc_e": [0.906, 0.943], "auc_b": [0.725, 0.854],
    # «أعلى 3 كل يوم» — كل خطط الشراء (القسم 3 بـ run 43175273449): المتوسط بعد التكاليف
    "plans_top3": {"o5": -0.042, "o20": -0.055, "olot": -0.053, "lad": -0.053, "on": -0.017, "h5": -0.026, "s15b": -0.051},
    "plans_n": 20, "direct_best": -0.004,       # النموذج المباشر للربح: أفضل نتيجة (إغلاق ← إغلاق بكرة، أعلى 3)
    # 🎟 التذكرة = دخول افتتاح بكرة · بيع Limit +50% (شغّال بالبري والأفتر) · بدون وقف · 20 جلسة (القسم 5)
    "ticket": {"limit": 0.5, "days": 20, "by_price": {"lt1": 0.014, "1to5": -0.018, "gt5": -0.067}},
    # «اليوم الثاني» بعد انفجار حقيقي (run 43172444632 القسم 7، n=4,166): الشراء بإغلاق يوم الانفجار
    "day2": {"n": 4166, "open": -0.043, "close": -0.064, "d5": -0.124, "win_open": 0.26},
    # الانفجارات الحقيقية بالسنة (run 43172444632 القسم 0)
    "ex_per_year": {"2021": 292, "2025": 950, "2026": 923},
}
print(json.dumps(M, ensure_ascii=False))
