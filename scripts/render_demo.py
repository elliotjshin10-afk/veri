"""Render the replay harness to a self-contained local page."""
import json, sys
sys.path.insert(0, "src")
from veridis.config import DEMO

d = json.loads((DEMO / "replay.json").read_text())
TRON = "https://tronscan.org/#/address/"
TX = "https://tronscan.org/#/transaction/"

rows = []
for s in d["steps"]:
    flagged = s["band"] == "high_risk"
    caution = s["band"] == "caution"
    pct = int(s["score"] * 100)
    reasons = "".join(f"<li>{r}</li>" for r in s["reasons"]) or "<li class=muted>No elevated signals</li>"
    rows.append(f"""
<div class="row {'flag' if flagged else 'warn' if caution else ''}">
  <div class="k">#{s['k']}</div>
  <div class="when">{s['when']}<br><span class=muted>${s['amount']:,.0f}</span></div>
  <div class="cell block">
    <span class="pill {'bad' if s['blocklist']=='listed' else 'ok'}">{'LISTED' if s['blocklist']=='listed' else 'CLEAN'}</span>
    <div class=muted>no sanctions or scam-list hit</div>
  </div>
  <div class="cell ours">
    <div class="bar"><span style="width:{pct}%"></span></div>
    <div class="scoreline"><b>{s['score']:.2f}</b>
      <span class="pill {'bad' if flagged else 'mid' if caution else 'ok'}">{s['band'].replace('_',' ').upper()}</span></div>
    <ul>{reasons}</ul>
    <div class=muted>dest age {s['dest_age_days']}d &middot; {s['dest_senders_7d']} senders/7d
      &middot; <a href="{TX}{s['tx']}" target=_blank>tx</a></div>
  </div>
</div>""")

flag_note = (f"Model crossed the high-risk threshold at transfer #{d['first_flag_k']}."
             if d["first_flag_k"] else "Model did not cross the high-risk threshold.")
listed_note = (f"This address was not on any public list until {d['listed_at']} "
               f"(source: {d['listed_sources']}) &mdash; after every transfer below."
               if d["listed_at"] else "This address never appeared on a public list.")

html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Veridis replay &mdash; pre-send scam detection</title><style>
*{{box-sizing:border-box}}
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
margin:0;background:#0f1115;color:#e7e9ee}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 20px 60px}}
h1{{font-size:24px;margin:0 0 6px}}
.sub{{color:#9aa3b2;margin-bottom:22px}}
.addr{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
background:#171a21;padding:3px 7px;border-radius:5px;color:#c8cede;text-decoration:none}}
.addr:hover{{color:#fff}}
.banner{{background:#171a21;border:1px solid #242938;border-radius:10px;padding:14px 16px;margin-bottom:22px}}
.head{{display:grid;grid-template-columns:44px 130px 1fr 1.6fr;gap:14px;
padding:10px 14px;color:#9aa3b2;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}
.row{{display:grid;grid-template-columns:44px 130px 1fr 1.6fr;gap:14px;padding:14px;
border-top:1px solid #1d222d;align-items:start}}
.row.flag{{background:rgba(220,60,70,.09)}}
.row.warn{{background:rgba(230,160,40,.07)}}
.k{{color:#6f7787;font-weight:600}}
.cell ul{{margin:8px 0 6px;padding-left:18px}}
.cell li{{margin:2px 0}}
.muted{{color:#79818f;font-size:12px}}
.pill{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.04em}}
.pill.ok{{background:#16301f;color:#4ade80}}
.pill.bad{{background:#3a1519;color:#f87171}}
.pill.mid{{background:#3a2e12;color:#fbbf24}}
.bar{{height:6px;background:#1d222d;border-radius:4px;overflow:hidden;margin-bottom:6px}}
.bar span{{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#ef4444)}}
.scoreline{{display:flex;gap:8px;align-items:center;margin-bottom:4px}}
.panel{{background:#12151b;border:1px solid #1d222d;border-radius:12px;overflow:hidden}}
.foot{{margin-top:22px;background:#171a21;border:1px solid #242938;border-radius:10px;padding:18px}}
.big{{font-size:30px;font-weight:700;color:#f87171}}
a{{color:#7aa2f7}}
</style></head><body><div class=wrap>
<h1>One real victim, replayed transfer by transfer</h1>
<div class=sub>Every address and transaction below is real and verifiable on Tronscan.</div>
<div class=banner>
  <div>Victim <a class=addr href="{TRON}{d['sender']}" target=_blank>{d['sender']}</a>
   &rarr; destination <a class=addr href="{TRON}{d['destination']}" target=_blank>{d['destination']}</a></div>
  <div class=muted style="margin-top:8px">{listed_note}</div>
</div>
<div class=panel>
<div class=head><div>#</div><div>When / amount</div><div>Blocklist check</div><div>Veridis score</div></div>
{''.join(rows)}
</div>
<div class=foot>
  <div class=muted>{flag_note} Money that moved at or after that first flag:</div>
  <div class=big>${d['protected_usd']:,.0f}</div>
  <div class=muted>of ${d['total_usd']:,.0f} total sent to this address.
  A blocklist flagged none of it &mdash; the address was listed only {d['listed_at'] or 'never'}.</div>
  <div class=muted style="margin-top:10px">Generated {d['generated_at']}</div>
</div></div></body></html>"""
(DEMO / "index.html").write_text(html)
print(f"wrote {DEMO/'index.html'}")
