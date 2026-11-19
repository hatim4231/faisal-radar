"""مكوّن عرض مشترك: «تحليل فيصل الكامل» لأي سهم في أي رادار (ماعدا الأفتر).

طلب أبوفرح (2026-11-18): «نفس المعلومات وطريقة عرض المعلومات والبيانات للأسهم اللي في رادار
فيصل الأصلي والمطوّر تسويها في كل الرادارات ماعدا رادار الأفتر».

الفكرة: نشغّل محرك فيصل `faisal_rules.analyze` على السهم المعروض في أي رادار، فنحصل على نفس
كائن السهم الغني (الشروط، الخطة، الشارت، البيانات، السلوك، التقسيمات...)، ونطعّم عليه إثراء
الرادار (الأخبار، SEC، الشورت المتاح، العلم). ثم نحقن في صفحة كل رادار **مكوّن عرض مستقل**
`window.FDETAIL(faisalObj)` يرسم نفس بطاقات صفحة فيصل تماماً (CSS معزول تحت `.fd` عشان ما يصطدم
بتصميم الرادار المضيف).

الاستخدام في ماسح أي رادار:
    import stock_detail
    row["faisal"] = stock_detail.build(sym, daily_df, info_row, cfg,
                                       news=row.get("news"), dilution=row.get("dilution"),
                                       short_available=row.get("short_available"),
                                       flag=row.get("flag"), country_ar=row.get("country_ar"))
    ...
    html = stock_detail.inject(template_html)   # يحقن CSS+JS قبل </head>

وفي قالب الرادار، بأي مكان يُبنى فيه تفصيل السهم (drawer أو بطاقة مفتوحة):
    ${s.faisal && window.FDETAIL ? FDETAIL(s.faisal) : ""}
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import faisal_rules as _fr


@functools.lru_cache(1)
def _faisal_cfg() -> dict:
    """إعدادات فيصل الجذرية (فيها irtikaz/momentum) — للرادارات اللي لها config خاص بدونها (عبدالله)."""
    return json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))


def build(symbol, d, info, cfg, *, news=None, dilution=None, short_available=None,
          flag=None, country_ar=None, confidence=None, split_zone=None, accum=None,
          name=None, sector=None, country=None):
    """يرجع كائن فيصل الكامل لأي سهم (نفس شكل صف صفحة فيصل)، أو None لو ما كفت البيانات."""
    fcfg = cfg if (cfg and "irtikaz" in cfg and "momentum" in cfg) else _faisal_cfg()
    a = _fr.analyze(d, info or {}, fcfg)
    if a is None:
        return None
    inf = info or {}
    a["symbol"] = symbol
    a["name"] = name or inf.get("shortName") or inf.get("longName") or symbol
    a["sector"] = sector or inf.get("sector")
    a["country"] = country or inf.get("country")
    extra = dict(news=news, dilution=dilution, short_available=short_available,
                 flag=flag, country_ar=country_ar, confidence=confidence,
                 split_zone=split_zone, accum=accum)
    for k, v in extra.items():
        if v is not None:
            a[k] = v
    return a


_MARK = "/*FDETAIL-BUNDLE-v1*/"


def inject(html: str) -> str:
    """يحقن CSS+JS المكوّن المشترك في صفحة الرادار (قبل </head>). آمن للتكرار."""
    if _MARK in html:                       # محقون مسبقاً (مو مجرد استدعاء FDETAIL بالقالب)
        return html
    bundle = "<style>" + _MARK + FDETAIL_CSS + "</style>\n<script>" + _MARK + FDETAIL_JS + "</script>\n"
    if "</head>" in html:
        return html.replace("</head>", bundle + "</head>", 1)
    return bundle + html


# ============================ CSS معزول تحت .fd ============================
FDETAIL_CSS = r"""
.fd{display:flex;flex-direction:column;gap:14px;margin-top:14px}
.fd .fdhdr{font-weight:700;font-size:15px;color:var(--accent);border-top:2px solid var(--accent);padding-top:10px}
.fd .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:var(--shadow)}
.fd .card h3{margin:0 0 10px;font-size:15px}
.fd .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.fd .kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13.5px}
.fd .kv dt{color:var(--muted)} .fd .kv dd{margin:0;font-weight:500}
.fd .check{display:grid;grid-template-columns:22px 1fr 60px;gap:8px;padding:8px 0;border-bottom:1px solid var(--line);align-items:start}
.fd .check:last-child{border-bottom:0}
.fd .check .ic{font-size:15px;line-height:1.4}
.fd .check .t{font-weight:600;font-size:14px}
.fd .check .d{color:var(--muted);font-size:13px}
.fd .check .r{color:var(--warn);font-size:13px;font-weight:500}
.fd .check .p{font-size:12px;color:var(--muted);text-align:left}
.fd .mini{height:5px;background:var(--panel2);border-radius:99px;margin-top:4px;overflow:hidden}
.fd .mini i{display:block;height:100%}
.fd .plan{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.fd .pbox{background:var(--panel2);border-radius:10px;padding:9px 10px}
.fd .pbox span{display:block;color:var(--muted);font-size:12px}
.fd .pbox b{font-size:16px}
.fd .chart svg{width:100%;height:auto;display:block}
.fd .legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted);margin-top:6px}
.fd .legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-left:4px;border-radius:2px}
.fd .newsitem{padding:9px 0;border-bottom:1px solid var(--line)} .fd .newsitem:last-of-type{border-bottom:0}
.fd .newsitem .nt{font-weight:600;font-size:14px}
.fd .news a{color:var(--accent);text-decoration:none}
.fd .news li{margin-bottom:6px;font-size:13.5px}
.fd .links a{color:var(--accent);margin-left:14px;font-size:13.5px;text-decoration:none}
.fd .note{font-size:12.5px;color:var(--muted)}
.fd ul.clean{margin:0;padding-right:18px}
.fd .num,.fd .ltr{direction:ltr;unicode-bidi:isolate;font-variant-numeric:tabular-nums}
.fd .up{color:var(--up)} .fd .down{color:var(--down)}
.fd .pct{font-weight:700}
.fd .badge{font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px}
.fd .badge.ready{background:var(--good-bg);color:var(--good)}
.fd .badge.near{background:var(--warn-bg);color:var(--warn)}
.fd .badge.radar{background:var(--info-bg);color:var(--info)}
.fd .badge.no{background:var(--panel2);color:var(--muted)}
.fd .tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.fd .tag{font-size:11.5px;padding:1px 8px;border-radius:999px;background:var(--panel2);color:var(--muted);border:1px solid var(--line)}
.fd .tag.g{background:var(--good-bg);color:var(--good);border-color:transparent}
.fd .tag.w{background:var(--warn-bg);color:var(--warn);border-color:transparent}
.fd .tag.b{background:var(--bad-bg);color:var(--bad);border-color:transparent}
.fd .tag.i{background:var(--info-bg);color:var(--info);border-color:transparent}
.fd .sent{display:inline-block;font-size:12px;font-weight:700;padding:1px 8px;border-radius:999px;margin-left:6px;white-space:nowrap}
.fd .sent.positive{background:var(--good-bg);color:var(--good)} .fd .sent.negative{background:var(--bad-bg);color:var(--bad)} .fd .sent.neutral{background:var(--panel2);color:var(--muted)}
.fd .newsfull{font-size:13.5px;line-height:1.8;white-space:pre-line;margin:6px 0;max-height:420px;overflow-y:auto;padding-left:6px}
.fd .newsitem .why{font-size:12px;color:var(--muted);margin-top:2px}
@media (max-width:820px){ .fd .grid2{grid-template-columns:1fr} .fd .plan{grid-template-columns:repeat(2,1fr)} }
"""


# ============================ JS: window.FDETAIL(s) ============================
# ملاحظة: نص JS خام — ${...} و backticks تبقى كما هي (ليست f-string).
FDETAIL_JS = r"""
(function(){
  var SENT = {positive:["✅","إيجابي"], negative:["❌","سلبي"], neutral:["➖","محايد"]};
  var STATUS = {ready:"✅ مطابق", near:"🟡 قريب", radar:"👀 رادار", no:"غير مطابق"};
  var MSTATUS = {ready:"🚀 زخم قوي", near:"🟡 زخم محتمل", no:"لا يوجد زخم"};
  function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g, function(c){return({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]);}); }
  function fp(p){ return p==null?"-":(p<1?(+p).toFixed(3):(+p).toFixed(2)); }
  function fpct(p,s){ if(s===undefined)s=true; return p==null?"-":((s&&p>0?"+":"")+(+p).toFixed(1)+"%"); }
  function fM(n){ return n==null?"-":n>=1e9?(n/1e9).toFixed(2)+"B":n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e3?(n/1e3).toFixed(0)+"K":String(n); }
  function cv(name,fb){ var v=getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return v||fb; }
  function scoreColor(s){ return s>=75?"var(--good)":s>=55?"var(--warn)":s>=35?"var(--info)":"var(--muted)"; }
  function tagClass(t){ if(/⚠|امتثال|كسر/.test(t))return "b"; if(/ارتكاز|سحب|Former/.test(t))return "g"; if(/زخم|تقسيم/.test(t))return "w"; return "i"; }
  function dateTxt(t){ return String(t||"").slice(0,16).replace("T"," "); }

  function sentBadge(n){ var s=SENT[n.sentiment]||SENT.neutral; return '<span class="sent '+(n.sentiment||"neutral")+'">'+s[0]+" "+s[1]+'</span>'; }
  function newsHTML(list){
    list = list||[];
    if(!list.length) return '<div class="note">لا توجد أخبار حديثة لهذا السهم</div>';
    return list.map(function(n){
      var why=[].concat((n.why_pos||[]).map(function(w){return "✅ "+w;}),(n.why_neg||[]).map(function(w){return "❌ "+w;}));
      var body=n.text_ar||n.summary_ar||n.summary||"";
      return '<div class="newsitem"><div class="nt">'+sentBadge(n)+" "+esc(n.title_ar||n.title)+'</div>'
        +(why.length?'<div class="why">السبب: '+why.map(esc).join(" · ")+'</div>':"")
        +(body?'<div class="newsfull">'+esc(body)+'</div>':"")
        +'<div class="note">'+esc(n.source||"")+' · <span class="num">'+esc(dateTxt(n.time))+'</span>'+(n.full?"":" · (النص الكامل غير متاح)")+'</div></div>';
    }).join("")+'<div class="note" style="margin-top:6px">✅/❌ تقدير آلي — اقرأ الخبر بنفسك قبل القرار.</div>';
  }
  function checksHTML(list){
    return (list||[]).map(function(c){
      var pc=Math.round((c.progress!=null?c.progress:(c.ok?1:0))*100);
      var col=c.ok?"var(--good)":pc>=50?"var(--warn)":"var(--bad)";
      return '<div class="check"><div class="ic">'+(c.ok?"✅":pc>=50?"🟡":"❌")+'</div>'
        +'<div><div class="t">'+esc(c.title)+'</div><div class="d">'+esc(c.detail)+'</div>'
        +(c.remaining?'<div class="r">⏳ '+esc(c.remaining)+'</div>':"")
        +'<div class="mini"><i style="width:'+pc+'%;background:'+col+'"></i></div></div>'
        +'<div class="p num">'+(+c.got).toFixed(0)+"/"+c.weight+'</div></div>';
    }).join("");
  }
  function chartSVG(s){
    var small=window.innerWidth<600, c=s.chart;
    if(!c||!c.c||!c.c.length) return '<div class="note">لا يوجد شارت</div>';
    if(small&&c.c.length>60){ var k0=c.c.length-60, c2={}; for(var key in c){ c2[key]=Array.isArray(c[key])?c[key].slice(k0):c[key]; } c=c2; }
    var n=c.c.length, W=small?440:800, H=330, VH=50, pad={l:small?6:8,r:small?48:58,t:10,b:22};
    var p=s.plan||{}, levels=[];
    if(p.entry_zone&&p.entry_zone[0]){levels.push(p.entry_zone[0],p.entry_zone[1]);}
    if(s.split_zone&&s.split_zone.applies){levels.push(s.split_zone.half,s.split_zone.sweep);}
    if(p.stop)levels.push(p.stop);
    (p.four_targets||[]).forEach(function(x){levels.push(x);});
    var lo=Math.min.apply(null,c.l.filter(function(x){return x!=null;})), hi=Math.max.apply(null,c.h.filter(function(x){return x!=null;}));
    var vis=levels.filter(function(x){return x!=null&&x<hi*1.6&&x>lo*0.6;});
    lo=Math.min.apply(null,[lo].concat(vis))*0.97; hi=Math.max.apply(null,[hi].concat(vis))*1.03;
    var ph=H-VH-pad.t-pad.b;
    var y=function(v){return pad.t+(hi-v)/(hi-lo)*ph;};
    var cw=(W-pad.l-pad.r)/n;
    var x=function(i){return pad.l+i*cw+cw/2;};
    var up=cv("--up","#16a36a"), dn=cv("--down","#e04848"), line=cv("--line","#e2e5ea"), muted=cv("--muted","#667085");
    var g="";
    for(var k=0;k<=4;k++){ var v=lo+(hi-lo)*k/4; g+='<line x1="'+pad.l+'" x2="'+(W-pad.r)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="'+line+'" stroke-width="1"/><text x="'+(W-pad.r+4)+'" y="'+(y(v)+4)+'" fill="'+muted+'" font-size="11">'+fp(v)+'</text>'; }
    if(p.entry_zone&&p.entry_zone[0]) g+='<rect x="'+pad.l+'" width="'+(W-pad.l-pad.r)+'" y="'+y(p.entry_zone[1])+'" height="'+Math.max(1,y(p.entry_zone[0])-y(p.entry_zone[1]))+'" fill="'+cv("--zone","rgba(19,149,92,.13)")+'"/>';
    var hl=function(v,col,label,dash){ return (v==null||v>hi||v<lo)?"":'<line x1="'+pad.l+'" x2="'+(W-pad.r)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="'+col+'" stroke-width="1.3" '+(dash?'stroke-dasharray="5 4"':"")+'/><text x="'+(pad.l+4)+'" y="'+(y(v)-4)+'" fill="'+col+'" font-size="11" font-weight="600">'+label+" "+fp(v)+'</text>'; };
    g+=hl(p.support,cv("--good","#13955c"),"الدعم",false);
    g+=hl(p.stop,cv("--bad","#d23c3c"),"الوقف",true);
    if(s.split_zone&&s.split_zone.applies){ g+=hl(s.split_zone.half,cv("--warn","#b7791f"),"نصف قمة التقسيم",true); g+=hl(s.split_zone.sweep,cv("--warn","#b7791f"),"مسح 20%",true); }
    (p.four_targets||[]).forEach(function(t,i){ g+=hl(t,cv("--accent","#2f6fde"),"هدف "+(i+1),true); });
    var path=function(arr){ var d="",on=false; (arr||[]).forEach(function(v,i){ if(v==null||v>hi||v<lo){on=false;return;} d+=(on?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1); on=true; }); return d; };
    g+='<path d="'+path(c.ma20)+'" fill="none" stroke="'+cv("--ma20","#e39a1c")+'" stroke-width="1.4"/>';
    g+='<path d="'+path(c.ma50)+'" fill="none" stroke="'+cv("--ma50","#7c5cd6")+'" stroke-width="1.4"/>';
    var vmax=Math.max.apply(null,c.v);
    for(var i=0;i<n;i++){ var o=c.o[i],h=c.h[i],l=c.l[i],cl=c.c[i]; if(cl==null)continue; var col=cl>=o?up:dn, bw=Math.max(1,cw*0.62);
      g+='<line x1="'+x(i)+'" x2="'+x(i)+'" y1="'+y(h)+'" y2="'+y(l)+'" stroke="'+col+'" stroke-width="1"/>';
      g+='<rect x="'+(x(i)-bw/2)+'" width="'+bw+'" y="'+y(Math.max(o,cl))+'" height="'+Math.max(1,Math.abs(y(o)-y(cl)))+'" fill="'+col+'"/>';
      var vh=c.v[i]/vmax*(VH-6); g+='<rect x="'+(x(i)-bw/2)+'" width="'+bw+'" y="'+(H-pad.b-vh)+'" height="'+vh+'" fill="'+col+'" opacity=".45"/>';
    }
    [[0,"start"],[Math.floor(n/2),"middle"],[n-1,"end"]].forEach(function(z){ g+='<text x="'+x(z[0])+'" y="'+(H-6)+'" fill="'+muted+'" font-size="11" text-anchor="'+z[1]+'">'+c.d[z[0]]+'</text>'; });
    return '<svg style="direction:ltr" viewBox="0 0 '+W+" "+H+'" role="img" aria-label="chart '+esc(s.symbol)+'">'+g+'</svg>';
  }
  function confColor(c){ return c.label==="عالية"?"var(--good)":c.label==="متوسطة"?"var(--warn)":"var(--bad)"; }
  function confCard(c){
    if(!c) return "";
    return '<div class="card"><h3>🎯 أداة الثقة — احتمال نجاح التوصية</h3>'
      +'<div class="plan"><div class="pbox"><span>احتمال النجاح</span><b class="num" style="color:'+confColor(c)+'">'+Math.round(c.p*100)+'%</b><span>الثقة: '+esc(c.label)+'</span></div>'
      +'<div class="pbox"><span>✅ معها</span><b style="font-size:13px">'+((c.pos||[]).map(esc).join("، ")||"-")+'</b></div>'
      +'<div class="pbox"><span>⚠ ضدها</span><b style="font-size:13px">'+((c.neg||[]).map(esc).join("، ")||"-")+'</b></div></div>'
      +'<div class="note" style="margin-top:8px">احتمال مو ضمان — التزم بالوقف دائماً.</div></div>';
  }

  window.FDETAIL = function(s){
    if(!s) return "";
    var p=s.plan||{}, I=s.irtikaz||{score:0,status:"no",checks:[],adjustments:[],readiness:[],base:0}, M=s.momentum||{score:0,status:"no",checks:[],warnings:[]};
    var rs = (s.reverse_splits&&s.reverse_splits.length)? s.reverse_splits.map(function(r){return r.ratio+" ("+r.date+"، قبل "+r.days_ago+" يوم)";}).join("<br>") : "لا يوجد خلال سنتين";
    var fr = s.former_runner||{max_pct:0,date:null};
    var H = '<div class="fd">';
    H += '<div class="fdhdr">📐 تحليل فيصل الكامل لـ '+esc(s.symbol)+'</div>';
    // ملخص الجاهزية
    H += '<div class="card"><div class="grid2" style="align-items:center">'
      +'<div><div class="note">جاهزية سهم الارتكاز</div><div class="pct num" style="font-size:30px;color:'+scoreColor(I.score)+'">'+Math.round(I.score)+'%</div><span class="badge '+I.status+'">'+STATUS[I.status]+'</span></div>'
      +'<div><div class="note">قوة الزخم اليوم</div><div class="pct num" style="font-size:30px;color:'+scoreColor(M.score)+'">'+Math.round(M.score)+'%</div><span class="badge '+(M.status==="ready"?"ready":M.status==="near"?"near":"no")+'">'+MSTATUS[M.status]+'</span></div></div>'
      +((I.readiness&&I.readiness.length)?'<div style="margin-top:10px"><b>متى يكون جاهز؟</b><ul class="clean">'+I.readiness.map(function(r){return "<li>"+esc(r)+"</li>";}).join("")+'</ul></div>':"")
      +((s.tags&&s.tags.length)?'<div class="tags" style="margin-top:8px">'+s.tags.map(function(t){return '<span class="tag '+tagClass(t)+'">'+esc(t)+'</span>';}).join("")+'</div>':"")
      +'</div>';
    // الشارت
    H += '<div class="card chart"><h3>الشارت اليومي</h3>'+chartSVG(s)
      +'<div class="legend"><span><i style="background:var(--ma20)"></i>MA20</span><span><i style="background:var(--ma50)"></i>MA50</span><span><i style="background:var(--good)"></i>الدعم</span><span><i style="background:var(--zone);height:9px"></i>منطقة الطلب</span><span><i style="background:var(--bad)"></i>الوقف</span><span><i style="background:var(--accent)"></i>الأهداف</span></div></div>';
    // الثقة
    H += confCard(s.confidence);
    // الخطة
    H += '<div class="card"><h3>📋 خطة فيصل المقترحة (Limit داخل منطقة الطلب)</h3><div class="plan">'
      +'<div class="pbox"><span>الدعم الحقيقي</span><b class="num">'+fp(p.support)+'</b><span>'+esc(p.support_kind||"")+' · لمسات '+(p.support_touches||0)+'</span></div>'
      +'<div class="pbox"><span>منطقة الطلب</span><b class="num">'+fp((p.entry_zone||[])[0])+' – '+fp((p.entry_zone||[])[1])+'</b><span>السعر عنها '+fpct(p.dist_support_pct)+'</span></div>'
      +'<div class="pbox"><span>وقف الخسارة (تحت القاع)</span><b class="num" style="color:var(--bad)">'+fp(p.stop)+'</b><span>كسره = خروج</span></div>'
      +'<div class="pbox"><span>الطلبات المقسمة</span><b class="num">'+((p.orders||[]).map(fp).join(" / ")||"-")+'</b><span>طلب 1 · 2 · 3</span></div>'
      +'<div class="pbox"><span>الهدف الأول</span><b class="num" style="color:var(--good)">'+fp(p.t1)+'</b><span>'+fpct(p.upside_pct)+' · R:R '+(p.rr!=null?p.rr:"-")+'</span></div>'
      +'<div class="pbox"><span>الأهداف الأربعة</span><b class="num">'+((p.four_targets||[]).map(fp).join(" / ")||"-")+'</b><span>الرئيسي '+fp(p.main_target)+'</span></div></div>'
      +'<div class="note" style="margin-top:8px">المقاومات فوق: <span class="num">'+((p.resistances||[]).map(fp).join(" · ")||"-")+'</span></div></div>';
    // منطقة التقسيم
    if(s.split_zone){ var sz=s.split_zone;
      H += '<div class="card"><h3>✂️ منطقة التقسيم</h3><div class="plan">'
        +'<div class="pbox"><span>أعلى سعر يوم التقسيم</span><b class="num">'+fp(sz.split_high)+'</b><span>'+esc(sz.date||"")+' · '+esc(sz.ratio||"")+'</span></div>'
        +'<div class="pbox"><span>÷ 2</span><b class="num">'+fp(sz.half)+'</b></div>'
        +'<div class="pbox"><span>− 20% مسح سيولة</span><b class="num">'+fp(sz.sweep)+'</b></div></div>'
        +'<div style="margin-top:8px;font-weight:600">'+esc(sz.label||"")+' — '+esc(sz.note||"")+'</div></div>';
    }
    // التجميع
    if(s.accum){ var ac=s.accum;
      H += '<div class="card"><h3>'+esc(ac.label||"تجميع")+'</h3><div class="note">نقاط التجميع <b class="num">'+Math.round(ac.score||0)+'</b> · قاعدة <span class="num">'+(ac.base_len||0)+'</span> جلسة</div>'
        +'<ul class="clean" style="margin-top:6px">'+((ac.signs||[]).map(function(x){return "<li>✅ "+esc(x)+"</li>";}).join(""))+'</ul></div>';
    }
    // شروط الارتكاز
    H += '<div class="card"><h3>✔ شروط سهم الارتكاز ('+Math.round(I.base||0)+' من 100 قبل تعديلات السلوك)</h3>'+checksHTML(I.checks)
      +((I.adjustments&&I.adjustments.length)?'<div style="margin-top:10px"><b>تعديلات السلوك:</b><ul class="clean">'+I.adjustments.map(function(a){return '<li><b class="num" style="color:'+(a.pts>0?"var(--good)":"var(--bad)")+'">'+(a.pts>0?"+":"")+a.pts+'</b> '+esc(a.name)+' — <span class="note">'+esc(a.detail)+'</span></li>';}).join("")+'</ul></div>':"")
      +'</div>';
    // الزخم
    H += '<div class="card"><h3>⚡ شروط الزخم</h3>'+checksHTML(M.checks)
      +((M.warnings&&M.warnings.length)?'<div style="color:var(--warn);margin-top:8px">'+M.warnings.map(esc).join("<br>")+'</div>':"")+'</div>';
    // بيانات + سلوك
    var ma=s.ma||{};
    H += '<div class="grid2"><div class="card"><h3>📊 بيانات السهم</h3><dl class="kv">'
      +'<dt>القيمة السوقية</dt><dd class="num">'+fM(s.market_cap)+'</dd>'
      +'<dt>الفلووت</dt><dd class="num">'+fM(s.float)+'</dd>'
      +'<dt>عدد الأسهم</dt><dd class="num">'+fM(s.shares_out)+'</dd>'
      +'<dt>شورت (Short Interest)</dt><dd>'+(s.short_shares!=null?('<span class="num">'+(+s.short_shares).toLocaleString("en-US")+'</span> سهم'):"-")+(s.short_pct_float!=null?' (<span class="num">'+s.short_pct_float+'%</span>)':"")+'</dd>'
      +'<dt>الشورت المتاح للاقتراض</dt><dd>'+(s.short_available&&s.short_available.available!=null?('<span class="num">'+(+s.short_available.available).toLocaleString("en-US")+'</span> سهم <span class="note num">'+esc(s.short_available.date||"")+'</span>'):"-")+'</dd>'
      +'<dt>فوليوم اليوم / المتوسط</dt><dd class="num">'+fM(s.volume)+' / '+fM(s.avg_volume)+' ('+(s.rel_volume!=null?s.rel_volume:"-")+'x)</dd>'
      +'<dt>RSI يومي</dt><dd class="num">'+s.rsi+' (أدنى 10 أيام '+s.rsi_min10+')</dd>'
      +'<dt>MA20 / 50 / 200</dt><dd class="num">'+fp(ma[20])+' / '+fp(ma[50])+' / '+fp(ma[200])+'</dd>'
      +'<dt>قمة/قاع 52 أسبوع</dt><dd class="num">'+fp(s.hi52)+' / '+fp(s.lo52)+' ('+fpct(s.drawdown_pct)+')</dd>'
      +'</dl></div>'
      +'<div class="card"><h3>🧠 السلوك</h3><dl class="kv">'
      +'<dt>Former Runner</dt><dd>'+(fr.max_pct>=100?"✅ ":"")+'<span class="num">+'+fr.max_pct+'%</span> <span class="note">'+esc(fr.date||"")+'</span></dd>'
      +'<dt>التقسيم العكسي</dt><dd class="num">'+rs+'</dd>'
      +'<dt>تحت 1$ (امتثال)</dt><dd>'+(s.below_1_streak>=30?('<span style="color:var(--bad)">'+s.below_1_streak+' جلسة — احتمال تقسيم عكسي</span>'):(s.below_1_streak?s.below_1_streak+" جلسة":"لا"))+'</dd>'
      +'<dt>شموع قروبات</dt><dd>'+(s.pump_candles>=2?('<span style="color:var(--bad)">⚠ '+s.pump_candles+'</span>'):s.pump_candles)+'</dd>'
      +'<dt>التذبذب اليومي</dt><dd class="num">'+s.median_range_pct+'% '+(s.quiet?"(هادئ)":"")+'</dd>'
      +'<dt>نموذج W</dt><dd>'+(s.w?('قاعين '+fp(s.w.low1)+"/"+fp(s.w.low2)+' · '+(s.w.broken?"مكسور ✓":"باقي "+fpct(s.w.to_neck_pct))):"-")+'</dd>'
      +'<dt>سحب سيولة</dt><dd>'+(s.sweep?('إلى '+fp(s.sweep.sweep_low)+' ('+s.sweep.date+') ✓'):"-")+'</dd>'
      +'<dt>صعد من قاع 20 يوم</dt><dd class="num">'+fpct(s.up_from_low20_pct)+'</dd>'
      +'</dl></div></div>';
    // تخفيف SEC
    if(s.dilution){ var dl=s.dilution, lv=dl.level;
      var lvtxt={high:"⛔ خطر عالي — طرح/تسجيل حديث",medium:"⚠ خطر محتمل",low:"خطر خفيف",none:"ما لقى الفحص الآلي ملفات طرح — تأكد بنفسك"}[lv]||"";
      H += '<div class="card"><h3>🏛 خطر التخفيف من SEC (آخر 90 يوم)</h3>'
        +'<div style="font-weight:700;color:'+(lv==="high"?"var(--bad)":lv==="medium"?"var(--warn)":"var(--good)")+'">'+lvtxt+'</div>'
        +((dl.hits&&dl.hits.length)?'<ul class="clean" style="margin-top:6px">'+dl.hits.map(function(h){return '<li><b class="num">'+esc(h.form)+'</b> · <span class="num">'+esc(h.date)+'</span> — '+esc(h.label)+'</li>';}).join("")+'</ul>':"")+'</div>';
    }
    // الأخبار
    H += '<div class="card"><h3>📰 الأخبار</h3>'+newsHTML(s.news)+'<div class="note">الخبر محفز، وليس سبب دخول إذا الشارت غير جاهز.</div></div>';
    // روابط
    H += '<div class="card"><h3>🔗 روابط</h3><div class="links">'
      +'<a href="https://finviz.com/quote.ashx?t='+encodeURIComponent(s.symbol)+'" target="_blank" rel="noopener">Finviz</a>'
      +'<a href="https://www.tradingview.com/chart/?symbol='+encodeURIComponent(s.symbol)+'" target="_blank" rel="noopener">TradingView</a>'
      +'<a href="https://finance.yahoo.com/quote/'+encodeURIComponent(s.symbol)+'" target="_blank" rel="noopener">Yahoo</a>'
      +'<a href="https://iborrowdesk.com/report/'+encodeURIComponent(s.symbol)+'" target="_blank" rel="noopener">الشورت المتاح</a>'
      +'</div></div>';
    H += '</div>';
    return H;
  };
})();
"""
