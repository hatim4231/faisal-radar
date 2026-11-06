#!/usr/bin/env python3
"""
رادار فيصل — النسخة الأصلية (قواعد التحليل كما كانت في أول نسخة، commit 1c5f0a0).
نفس الفلاتر اللي اختارها المستخدم (1-5$، الشورت 50 ألف، الأخبار، بدون Warrants) لأنها في scanner.py و config.json،
لكن الحساب والتصنيف من faisal_rules_original.py بدون أي تعديل.

    python faisal_original/run_original.py --out site/original/index.html
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import scanner  # noqa: E402
import report  # noqa: E402
import faisal_rules_original  # noqa: E402

# نفس خط الجلب والفلاتر، بس بمحرك القواعد الأصلي
scanner.fr = faisal_rules_original


def main():
    ap = argparse.ArgumentParser(description="رادار فيصل — النسخة الأصلية")
    ap.add_argument("--out", default=str(HERE.parent / "output" / "faisal_original.html"))
    ap.add_argument("--tickers")
    a = ap.parse_args()
    cfg = scanner.load_cfg()
    payload = scanner.run_scan(cfg, a.tickers.split(",") if a.tickers else None)
    payload["variant"] = "original"
    import os
    if os.environ.get("GITHUB_REPOSITORY"):
        payload["actions_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/scan.yml"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write(payload, out)
    scanner.log(f"[الأصلي] التقرير: {out}")


if __name__ == "__main__":
    main()
