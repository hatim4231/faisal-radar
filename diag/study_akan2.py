"""متابعة AKAN: هل رادار المدمج (المرخّى) كان بيمسكه لو وُجد أيام كان سعره فوق 0.70؟ + حارس الشطب."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "faisal_original"))
import alpaca_data as A
import faisal_rules as fr
import faisal_rules_original as fo
import delist as DL

SYM = "AKAN"
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
rcfg = copy.deepcopy(cfg); rcfg["universe"]["max_market_cap"] = 1e15; rcfg["universe"]["max_float"] = 1e15

info = {"shortName": "Akanda Corp.", "country": "Canada", "floatShares": 709911,
        "marketCap": 3589675, "sharesOutstanding": 2388212, "sharesShort": 193393, "shortPercentOfFloat": 0.081}

d = A.daily([SYM], days=730)[SYM].sort_index()
dates = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"]
print("التاريخ      سعر   | المطوّر(عادي) | الأصلي(عادي) | المدمج(مرخّى) يترشح؟ | فلتر القائمة | شطب")
for ds in dates:
    sub = d[d.index <= ds]
    if len(sub) < 60:
        continue
    px = float(sub["Close"].iloc[-1])
    ad = fr.analyze(sub.copy(), info, cfg)
    ao = fo.analyze(sub.copy(), info, cfg)
    adr = fr.analyze(sub.copy(), info, rcfg)
    aor = fo.analyze(sub.copy(), info, rcfg)
    # المدمج: يترشح لو ✅/🟡 بأي محرك (مرخّى) وما فيه فلتر قائمة يرفضه
    def st(a): return a["irtikaz"]["status"] if a else "—"
    merged_gate = (adr and not adr["gate_fail"]) or (aor and not aor["gate_fail"])
    merged_sig = any(st(a) in ("ready", "near") for a in (adr, aor))
    qual = "✅ نعم" if (merged_sig and merged_gate) else "❌ لا"
    gate = (ad or {}).get("gate_fail") or []
    dl = DL.assess(sub.copy())
    lvl = {0: "آمن", 1: "متوسط", 2: "🔴 عالي"}[dl["level"]]
    print(f"{ds} {px:5.3f} | {st(ad):5}({ad['irtikaz']['score'] if ad else 0:.0f}) | {st(ao):5}({ao['irtikaz']['score'] if ao else 0:.0f}) | {qual:8} | {('؛'.join(gate))[:34] or 'يمر':34} | {lvl}")

# تفصيل حارس الشطب عند 11-12
dl = DL.assess(d[d.index <= "2026-11-12"].copy())
print("\nحارس الشطب يوم 2026-11-12:", {k: dl[k] for k in ("level", "u1_streak", "dd252", "price")})
for r in dl["reasons"]:
    print("  -", r)
