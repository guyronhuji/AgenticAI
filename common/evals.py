"""A small eval harness (Week 11), the kind every agent project should have.

Vocabulary (it is just experimental design):
  task     one input with a success criterion
  trial    one attempt at a task; agents are stochastic, so run several
  grader   a function (task, result) -> bool  (deterministic preferred)
  pass@k   P(at least one of k trials succeeds)   -- "can it ever do it?"
  pass^k   P(all k trials succeed)                -- "can I rely on it?"
Both are computed with the exact combinatorial estimator from n >= k trials.

    from common.evals import Task, run_eval, numeric_grader
    tasks = [Task(id="t1", input="What is 2**10?", expected=1024)]
    res = run_eval(tasks, runner=lambda q: solve(q), grader=numeric_grader, trials=3)
    print(res.summary())
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import time
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    input: str
    expected: object = None
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_json(path):
        items = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return [Task(id=str(it.get("id", i)), input=it["question"], expected=it.get("answer"),
                     meta={k: v for k, v in it.items() if k not in ("id", "question", "answer")})
                for i, it in enumerate(items, 1)]


def numeric_grader(task, result, rel_tol=0.005):
    from common.agent import is_correct
    return is_correct(result.get("answer"), task.expected, rel_tol)


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def pass_at_k(n, c, k):
    """P(at least one success in k draws) from n trials with c successes."""
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(n, c, k):
    """P(all k draws succeed) from n trials with c successes."""
    return comb(c, k) / comb(n, k) if comb(n, k) else 0.0


@dataclass
class EvalResult:
    name: str
    trials: int
    rows: list = field(default_factory=list)     # one per (task, trial)

    def per_task(self):
        out = {}
        for r in self.rows:
            d = out.setdefault(r["task"], {"n": 0, "c": 0, "tokens": 0, "calls": 0, "secs": 0.0})
            d["n"] += 1
            d["c"] += int(r["ok"])
            d["tokens"] += r["tokens"]
            d["calls"] += r["calls"]
            d["secs"] += r["secs"]
        return out

    def summary(self, k=None):
        k = k or self.trials
        pt = self.per_task()
        n_tasks = max(1, len(pt))
        tot_trials = sum(d["n"] for d in pt.values()) or 1
        return {
            "config": self.name,
            "tasks": len(pt),
            "trials": self.trials,
            "accuracy": sum(d["c"] for d in pt.values()) / tot_trials,
            f"pass@{k}": sum(pass_at_k(d["n"], d["c"], min(k, d["n"])) for d in pt.values()) / n_tasks,
            f"pass^{k}": sum(pass_pow_k(d["n"], d["c"], min(k, d["n"])) for d in pt.values()) / n_tasks,
            "mean_calls": sum(d["calls"] for d in pt.values()) / tot_trials,
            "mean_tokens": sum(d["tokens"] for d in pt.values()) / tot_trials,
            "mean_secs": sum(d["secs"] for d in pt.values()) / tot_trials,
        }

    def failures(self):
        return [r for r in self.rows if not r["ok"]]

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def run_eval(tasks, runner, grader, trials=1, name="eval", verbose=True,
             on_result=None) -> EvalResult:
    """runner(input) -> result (dict-like with answer/calls/prompt_tokens/
    completion_tokens); grader(task, result) -> bool."""
    res = EvalResult(name=name, trials=trials)
    for t in tasks:
        for trial in range(1, trials + 1):
            t0 = time.perf_counter()
            try:
                r = runner(t.input)
                err = None
            except Exception as e:  # noqa: BLE001 - a crash is a failed trial, not a crashed eval
                r, err = {"answer": None, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}, repr(e)
            secs = time.perf_counter() - t0
            ok = bool(grader(t, r)) if err is None else False
            row = {"config": name, "task": t.id, "trial": trial, "ok": ok,
                   "got": r.get("answer"), "expected": t.expected,
                   "calls": r.get("calls", 0),
                   "tokens": r.get("prompt_tokens", 0) + r.get("completion_tokens", 0),
                   "secs": round(secs, 2), "stopped": r.get("stopped", ""), "error": err,
                   "tier": t.meta.get("tier", "")}
            res.rows.append(row)
            if verbose:
                print(f"  [{t.id:>4}] trial {trial}: {'OK  ' if ok else 'FAIL'} "
                      f"got={str(r.get('answer')):<14} exp={str(t.expected):<14} "
                      f"{r.get('calls', 0)} calls {row['tokens']} tok" + (f"  {err}" if err else ""))
            if on_result:
                on_result(row)
    return res


def plot_accuracy_vs_compute(summaries, path, x="mean_tokens"):
    """One figure: accuracy and pass^k against compute, one point per config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = [s[x] for s in summaries]
    ax.plot(xs, [100 * s["accuracy"] for s in summaries], "o-", label="accuracy (mean over trials)")
    kkey = [k for k in summaries[0] if k.startswith("pass^")][0]
    ax.plot(xs, [100 * s[kkey] for s in summaries], "s--", label=f"{kkey} (all trials correct)")
    for s, xv in zip(summaries, xs):
        ax.annotate(s["config"], (xv, 100 * s["accuracy"]), textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax.set_xlabel(f"{x.replace('_', ' ')} per problem (compute)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Reliability is bought with compute")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path
