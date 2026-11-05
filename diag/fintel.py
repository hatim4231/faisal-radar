import re, requests
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
def get(u):
    try:
        r = requests.get(u, headers=H, timeout=20)
        return r.status_code, r.text
    except Exception as e:
        return None, repr(e)
for u in ["https://fintel.io/robots.txt"]:
    c, t = get(u); print("==", u, c); print(t[:1500])
for u in ["https://fintel.io/terms", "https://fintel.io/legal/terms", "https://fintel.io/tos"]:
    c, t = get(u); print("==", u, c)
    txt = re.sub(r"<[^>]+>", " ", t); txt = re.sub(r"\s+", " ", txt)
    for kw in ("scrap", "robot", "spider", "automated", "crawl"):
        for m in re.finditer(kw, txt, re.I):
            print("   ...", txt[max(0, m.start()-250): m.start()+250]); break
for u in ["https://fintel.io/ss/us/flye", "https://fintel.io/sbr/us/flye"]:
    c, t = get(u); print("==", u, c, len(t))
    txt = re.sub(r"<script.*?</script>", " ", t, flags=re.S); txt = re.sub(r"<[^>]+>", " ", txt); txt = re.sub(r"\s+", " ", txt)
    for kw in ("Short Shares Available", "Shares Available", "Borrow", "Short Interest", "Log In", "Subscribe", "premium", "Cloudflare", "Just a moment"):
        i = txt.find(kw)
        if i >= 0: print(f"   [{kw}] ...{txt[max(0,i-150):i+300]}")
