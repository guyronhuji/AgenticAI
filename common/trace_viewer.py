"""Live trace viewer: a browser window that shows model calls, tool calls and
verifier verdicts AS THEY HAPPEN, for the projector in class.

    uv run python common/trace_viewer.py              # serves http://localhost:8765
    uv run python common/trace_viewer.py --port 9000
    uv run python common/trace_viewer.py --dir traces # the folder the demos write to

Leave it running in one terminal; run any demo in another. The page follows
the newest trace file in the folder automatically (every `with trace.run()`
starts a new file), so you never have to touch it between demos.

Standard library only: an HTTP server that serves one HTML page and a JSON
endpoint the page polls twice a second. Nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

TRACE_DIR = pathlib.Path("traces")

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Agent trace</title>
<style>
  :root { --bg:#1b1b1b; --panel:#242424; --ink:#f6f3ec; --dim:#9a9a93; --teal:#5fb3a6; --amber:#e3b341; --red:#e06c6c; --mono:"JetBrains Mono",Consolas,"Courier New",monospace; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--mono); font-size:15px; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid #333; padding:10px 18px; display:flex; gap:28px; align-items:baseline; flex-wrap:wrap; }
  header b { color:var(--teal); font-weight:600; }
  header .file { color:var(--dim); font-size:13px; }
  header button { background:var(--panel); color:var(--ink); border:1px solid #444; font-family:var(--mono); padding:4px 10px; cursor:pointer; }
  #events { padding:12px 18px 60px; display:flex; flex-direction:column; gap:10px; }
  .ev { background:var(--panel); border-left:4px solid var(--dim); padding:10px 14px; border-radius:2px; animation: pop .25s ease-out; }
  @keyframes pop { from { opacity:0; transform:translateY(6px);} to { opacity:1; transform:none;} }
  .ev.llm { border-color:var(--teal); } .ev.tool { border-color:var(--amber); } .ev.nudge { border-color:var(--red); } .ev.verdict.pass { border-color:var(--teal);} .ev.verdict.fail { border-color:var(--red);} .ev.vote { border-color:#9d7cd8; }
  .head { display:flex; gap:16px; align-items:baseline; color:var(--dim); font-size:13px; }
  .head .kind { color:var(--ink); font-weight:600; font-size:14px; }
  .head .model { color:var(--ink); background:#333; padding:1px 8px; border-radius:3px; }
  pre.schema { color:var(--dim); font-size:12px; max-height:360px; overflow:auto; }
  .head .kind.llm { color:var(--teal);} .head .kind.tool { color:var(--amber);} .head .kind.fail { color:var(--red);}
  pre { margin:6px 0 0; white-space:pre-wrap; word-break:break-word; font-family:var(--mono); font-size:14px; line-height:1.4; }
  .thinking { color:var(--dim); font-style:italic; }
  .label { color:var(--dim); font-size:12px; margin-top:6px; }
  .call { color:var(--amber); }
  details summary { cursor:pointer; color:var(--dim); font-size:13px; }
</style></head>
<body>
<header>
  <span><b>agent trace</b></span>
  <span id="stats">waiting for a trace file…</span>
  <span class="file" id="file"></span>
  <button id="pause">pause</button>
  <button id="clear">clear</button>
  <span class="file">font: <button onclick="zoom(1)">+</button> <button onclick="zoom(-1)">−</button></span>
</header>
<div id="events"></div>
<script>
let file=null, since=0, paused=false, totals={llm:0,tool:0,ptok:0,ctok:0,secs:0};
const E=document.getElementById('events');
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function short(s,n){s=String(s??'');return s.length>n?s.slice(0,n)+' …':s;}
function zoom(d){const b=document.body;const cur=parseFloat(getComputedStyle(b).fontSize);b.style.fontSize=(cur+d)+'px';}
function render(e){
  const d=document.createElement('div');
  if(e.kind==='llm_call'){
    totals.llm++; totals.ptok+=e.prompt_tokens||0; totals.ctok+=e.completion_tokens||0; totals.secs+=e.latency_s||0;
    d.className='ev llm';
    let h=`<div class="head"><span class="kind llm">LLM call #${totals.llm}</span><span class="model">${esc(e.role&&e.role!=='tag'?e.role+' · ':'')}${esc(e.model)}</span><span>${e.prompt_tokens||0} in · ${e.completion_tokens||0} out · ${e.latency_s||0}s</span>${e.think?'<span>think='+esc(e.think)+'</span>':''}${e.temperature?'<span>T='+e.temperature+'</span>':''}</div>`;
    if(e.tool_schemas&&e.tool_schemas.length) h+=`<details><summary>tools offered: ${e.tool_schemas.map(t=>esc(t.function&&t.function.name)).join(', ')} (JSON schema the model saw)</summary><pre class="schema">${esc(JSON.stringify(e.tool_schemas,null,2))}</pre></details>`;
    const last=e.last_message||{};
    if(last.role) h+=`<details><summary>context: ${e.n_messages} messages · last = ${esc(last.role)}</summary><pre>${esc(short(last.content,1500))}</pre></details>`;
    if(e.thinking) h+=`<div class="label">thinking</div><pre class="thinking">${esc(short(e.thinking,1200))}</pre>`;
    if(e.tool_calls&&e.tool_calls.length){ for(const c of e.tool_calls){ h+=`<div class="label">emits tool call</div><pre class="call">${esc(c.name)}(${esc(short(JSON.stringify(c.args),1200))})</pre>`; } }
    if(e.content) h+=`<div class="label">reply</div><pre>${esc(short(e.content,2000))}</pre>`;
    else if(!(e.tool_calls&&e.tool_calls.length)) h+=`<div class="label">reply</div><pre class="thinking">(empty)</pre>`;
    d.innerHTML=h;
  } else if(e.kind==='tool_call'){
    totals.tool++; d.className='ev tool';
    d.innerHTML=`<div class="head"><span class="kind tool">RUNTIME executes</span><span>${esc(e.name)}</span></div><pre class="call">${esc(short(JSON.stringify(e.args),800))}</pre><div class="label">result</div><pre>${esc(short(e.result,1500))}</pre>`;
  } else if(e.kind==='verdict'){
    d.className='ev verdict '+(e.passed?'pass':'fail');
    d.innerHTML=`<div class="head"><span class="kind ${e.passed?'':'fail'}">VERIFIER ${e.passed?'PASS':'FAIL'}</span><span>${esc(e.verifier)}</span></div><pre>${esc(short(e.output?JSON.stringify(e.output):(e.label||''),800))}</pre>`;
  } else if(e.kind==='nudge'){
    d.className='ev nudge';
    d.innerHTML=`<div class="head"><span class="kind fail">RUNTIME nudges</span><span>${esc(e.reason)}</span></div><pre>the model returned an empty final reply; the harness asked once for the answer</pre>`;
  } else if(e.kind==='vote'){
    d.className='ev vote';
    d.innerHTML=`<div class="head"><span class="kind">VOTE</span></div><pre>${esc(JSON.stringify(e.votes))} → ${esc(e.winner)}</pre>`;
  } else {
    d.className='ev'; const rest={...e}; delete rest.i; delete rest.t; delete rest.kind;
    d.innerHTML=`<div class="head"><span class="kind">${esc(e.kind)}</span></div><pre>${esc(short(JSON.stringify(rest),800))}</pre>`;
  }
  E.appendChild(d);
}
function stats(){document.getElementById('stats').textContent=`${totals.llm} model calls · ${totals.tool} tool calls · ${totals.ptok} prompt + ${totals.ctok} completion tokens · ${totals.secs.toFixed(1)}s in model`;}
async function tick(){
  if(paused) return;
  try{
    const r=await fetch(`/api/events?file=${encodeURIComponent(file||'')}&since=${since}`); const j=await r.json();
    if(j.file && j.file!==file){ file=j.file; since=0; totals={llm:0,tool:0,ptok:0,ctok:0,secs:0}; E.innerHTML=''; document.getElementById('file').textContent=file; }
    for(const e of j.events){ render(e); since=e.i; }
    if(j.events.length){ stats(); window.scrollTo(0,document.body.scrollHeight); }
  }catch(err){}
}
document.getElementById('pause').onclick=function(){paused=!paused;this.textContent=paused?'resume':'pause';};
document.getElementById('clear').onclick=function(){E.innerHTML='';};
setInterval(tick,500);
</script>
</body></html>
"""


def newest_trace():
    files = sorted(TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def read_events(path, since):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue          # a line still being written
            if e.get("i", 0) > since:
                out.append(e)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # keep the terminal quiet
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/events":
            q = parse_qs(u.query)
            since = int(q.get("since", ["0"])[0] or 0)
            newest = newest_trace()
            if newest is None:
                return self._send(200, json.dumps({"file": None, "events": []}).encode(), "application/json")
            wanted = q.get("file", [""])[0]
            if wanted != newest.name:          # a new demo started: follow it
                since = 0
            events = read_events(newest, since)
            return self._send(200, json.dumps({"file": newest.name, "events": events}, default=str).encode(),
                              "application/json")
        return self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default="traces")
    args = ap.parse_args()
    TRACE_DIR = pathlib.Path(args.dir)
    TRACE_DIR.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"trace viewer: http://localhost:{args.port}   (following newest file in {TRACE_DIR.resolve()}; Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
