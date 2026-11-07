"""
رسالة مني (Claude) للمستخدم على تيليجرام عبر بوت الرادار.
النص يجي من متغير MESSAGE (مدخل الـ workflow). المحادثة: TELEGRAM_CHAT_ID أو telegram.json في فرع track-data.
"""
import json
import os
import subprocess
import sys

import requests

token = os.environ.get("TELEGRAM_BOT_TOKEN")
msg = (os.environ.get("MESSAGE") or "").replace("\\n", "\n").strip()
if not token or not msg:
    print("ما فيه توكن أو رسالة — تخطي")
    sys.exit(0)
chat = os.environ.get("TELEGRAM_CHAT_ID")
if not chat:
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "track-data"], check=True)
        st = json.loads(subprocess.run(["git", "show", "FETCH_HEAD:telegram.json"], check=True,
                                       capture_output=True, text=True).stdout)
        chat = st.get("chat_id")
    except Exception as e:
        print("ما قدرت أقرا رقم المحادثة:", repr(e)[:120])
if not chat:
    print("ما فيه رقم محادثة")
    sys.exit(1)
parts, cur = [], ""
for line in msg.split("\n"):
    if len(cur) + len(line) + 1 > 3800:
        parts.append(cur)
        cur = ""
    cur += line + "\n"
if cur.strip():
    parts.append(cur)
ok = 0
for p in parts:
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                      data={"chat_id": chat, "text": p, "disable_web_page_preview": "true"})
    ok += r.ok
    if not r.ok:
        print("فشل:", r.status_code, r.text[:200])
print(f"أُرسل {ok} من {len(parts)}")
