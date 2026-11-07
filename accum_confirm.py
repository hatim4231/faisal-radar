"""
طبقة تأكيد التجميع (قرار المستخدم): لرادارات فيصل (النسختين) وعبدالله.

إذا السهم مرشح في الرادار وفوقه بصمة تجميع من رادار التجميع (accumulation/accum_rules.py)
← وسم «🔍 مؤكد بالتجميع» + بطاقة + سطر في تيليجرام، ويُسجّل في السجل عشان نقيس الفرق.
ما يغير النقاط ولا الحالة.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "accumulation"))
import accum_rules  # noqa: E402

LABEL = {"ready": "🔍 مؤكد بالتجميع", "near": "🔍 تجميع محتمل"}


def check(df, info: dict | None, cfg_acc: dict, short_readings=None) -> dict | None:
    try:
        r = accum_rules.analyze(df, info or {}, {"accumulation": cfg_acc}, short_readings)
    except Exception:
        return None
    if not r or r.get("status") not in ("ready", "near"):
        return None
    good = [c["title"] for c in r["checks"] if c["ok"]]
    return {"status": r["status"], "label": LABEL[r["status"]], "score": r["score"], "base_len": r["base_len"],
            "support": r["support"], "base_top": r["base_top"], "spring": r["spring"], "flow": r["flow"],
            "signs": good[:5]}


def apply_faisal(a: dict, df, info, cfg: dict, short_readings=None) -> None:
    c = check(df, info, cfg["accumulation"], short_readings)
    if c:
        a["accum"] = c
        a["tags"].append(c["label"])


def apply_abdullah(a: dict, df, info, cfg_acc: dict, short_readings=None) -> None:
    c = check(df, info, cfg_acc, short_readings)
    if c:
        a["accum"] = c
        a.setdefault("flags_pos", []).append(f"{c['label']} ({c['score']:.0f}) — " + "، ".join(c["signs"][:3]))
