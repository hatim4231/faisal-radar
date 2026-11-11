"""يبني صفحة الويب (ملف HTML واحد، يشتغل بدون إنترنت) من نتائج المسح."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "template.html"


def write(payload: dict, out: Path, live: bool = False):
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", data).replace("/*__LIVE__*/false", "true" if live else "false")
    Path(out).write_text(html, encoding="utf-8")


def waiting_page() -> str:
    return """<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>رادار فيصل</title>
<body style="font-family:system-ui;background:#0f1115;color:#e8e8e8;display:grid;place-items:center;height:100vh;margin:0">
<div style="text-align:center"><h2>⏳ جاري أول مسح للسوق...</h2>
<p>أول مرة ياخذ 5-15 دقيقة (تحميل بيانات كل الأسهم). الصفحة تتحدث تلقائياً.</p>
<p style="opacity:.6">تابع التقدم في نافذة الأوامر</p></div></body></html>"""
