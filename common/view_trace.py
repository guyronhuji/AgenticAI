"""Pretty-print a trace file.

    python common/view_trace.py traces/<file>.jsonl            # summary + timeline
    python common/view_trace.py traces/<file>.jsonl --full     # full prompts/replies
    python common/view_trace.py --latest                       # newest trace file
    python common/view_trace.py --follow                       # LIVE: print events as they
                                                               # happen, switching to each new
                                                               # trace file (second terminal
                                                               # on the projector)
    (for a browser window instead:  python common/trace_viewer.py)
"""
import json
import time
import pathlib
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.trace import TRACE_DIR, read  # noqa: E402


def short(s, n=160):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + " ..."


C = {"llm": "\033[36m", "tool": "\033[33m", "ok": "\033[32m", "bad": "\033[31m", "dim": "\033[2m", "end": "\033[0m"}


def fmt_event(e, full=False):
    k = e["kind"]
    if k == "llm_call":
        role = e.get("role"); role = f"{role} · " if role and role != "tag" else ""
        line = (f"{C['llm']}[{e['i']:3d}] LLM{C['end']}  {role}{e.get('model','')}  "
                f"({e.get('prompt_tokens', 0)} in, {e.get('completion_tokens', 0)} out, {e.get('latency_s', 0)}s)")
        if e.get("tool_schemas"):
            names = ", ".join(t["function"]["name"] for t in e["tool_schemas"])
            line += f"\n      {C['dim']}tools offered: {names}" + (("\n" + json.dumps(e["tool_schemas"], indent=2)) if full else "") + C["end"]
        if e.get("thinking"):
            line += f"\n      {C['dim']}thinking: {e['thinking'] if full else short(e['thinking'])}{C['end']}"
        for c in e.get("tool_calls") or []:
            line += f"\n      {C['tool']}-> {c['name']}({short(c['args'], 120)}){C['end']}"
        if e.get("content"):
            line += f"\n      {e['content'] if full else short(e['content'])}"
        return line
    if k == "tool_call":
        res = str(e.get("result"))
        return (f"{C['tool']}[{e['i']:3d}] TOOL {e.get('name')}({short(e.get('args'), 100)}){C['end']}\n      "
                + ("\n      ".join(textwrap.wrap(res, 110)) if full else short(res)))
    if k == "verdict":
        col = C["ok"] if e.get("passed") else C["bad"]
        return f"{col}[{e['i']:3d}] VERDICT {'PASS' if e.get('passed') else 'FAIL'}{C['end']} {e.get('verifier')}  {short(e.get('output') or e.get('label'), 150)}"
    rest = {k2: v for k2, v in e.items() if k2 not in ("i", "t", "kind")}
    return f"[{e['i']:3d}] {k.upper()} {short(rest, 200)}"


def follow():
    """Tail the newest trace file; switch whenever a newer one appears."""
    cur, pos = None, 0
    print(f"following {TRACE_DIR.resolve()}  (Ctrl-C to stop)")
    while True:
        files = sorted(TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if TRACE_DIR.exists() else []
        newest = files[-1] if files else None
        if newest is not None and newest != cur:
            cur, pos = newest, 0
            print(f"\n{C['dim']}===== {cur.name} ====={C['end']}")
        if cur is not None:
            with open(cur, encoding="utf-8") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                if line.strip():
                    try:
                        print(fmt_event(json.loads(line)))
                    except json.JSONDecodeError:
                        pos -= len(line.encode("utf-8")) + 1   # partial line; re-read next time
                        break
        time.sleep(0.3)


def main(argv):
    if "--follow" in argv:
        try:
            follow()
        except KeyboardInterrupt:
            pass
        return
    full = "--full" in argv
    args = [a for a in argv if not a.startswith("--")]
    if "--latest" in argv or not args:
        files = sorted(TRACE_DIR.glob("*.jsonl"))
        if not files:
            sys.exit(f"no traces in {TRACE_DIR}/")
        path = files[-1]
    else:
        path = pathlib.Path(args[0])
    events = read(path)
    calls = [e for e in events if e["kind"] == "llm_call"]
    tools = [e for e in events if e["kind"] == "tool_call"]
    ptok = sum(e.get("prompt_tokens", 0) for e in calls)
    ctok = sum(e.get("completion_tokens", 0) for e in calls)
    lat = sum(e.get("latency_s", 0) for e in calls)
    print(f"{path}\n{len(events)} events | {len(calls)} model calls | "
          f"{len(tools)} tool calls | {ptok} prompt + {ctok} completion tokens | "
          f"{lat:.1f}s in model\n")
    for e in events:
        print(fmt_event(e, full))


if __name__ == "__main__":
    main(sys.argv[1:])
