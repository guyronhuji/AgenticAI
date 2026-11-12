"""Tracing: every model call, tool call and verifier verdict goes to a JSONL
file, one run per file (traces/<run>.jsonl).

Why this exists: agents fail inside loops, and the only way to debug a loop is
to read what actually happened turn by turn. Production systems do this with
OpenTelemetry spans; this is the 40-line version with the same shape.

Usage:
    from common import trace
    with trace.run("math_agent_quiz"):      # opens traces/<timestamp>_math_agent_quiz.jsonl
        ...                                 # every chat()/tool call is logged
    trace.log("note", text="anything")      # ad-hoc events

View a trace:  python common/view_trace.py traces/<file>.jsonl
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import pathlib

TRACE_DIR = pathlib.Path(os.environ.get("TRACE_DIR", "traces"))
_current = None   # module global on purpose: agents may run tools in worker threads


class Tracer:
    def __init__(self, name="run"):
        TRACE_DIR.mkdir(exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = TRACE_DIR / f"{stamp}_{name}.jsonl"
        self.n = 0
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, kind, **fields):
        self.n += 1
        rec = {"i": self.n, "t": _dt.datetime.now().isoformat(timespec="milliseconds"),
               "kind": kind, **fields}
        self._f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def current() -> Tracer | None:
    return _current


@contextlib.contextmanager
def run(name="run", announce=True):
    """Context manager: everything inside is logged to one file."""
    global _current
    prev = _current
    tracer = Tracer(name)
    _current = tracer
    try:
        yield tracer
    finally:
        tracer.close()
        _current = prev
        if announce:
            print(f"[trace] {tracer.n} events -> {tracer.path}")


def log(kind, **fields):
    t = current()
    if t is not None:
        t.log(kind, **fields)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
