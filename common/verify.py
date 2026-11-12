"""Verification building blocks (Week 11).

Three verifier types, in decreasing order of trust:
  programmatic_verify   independent recomputation in the sandbox (highest trust)
  (retrieval-based)     see week11/04_citation_verifier.py
  model_judge           a second model pass with a rubric (lowest trust; use to
                        rank, never to certify)

Plus the two reliability patterns the homework measures:
  self_consistent_solve   sample k answers at T>0, majority vote
  solve_with_verification generator -> verifier -> feedback loop, capped
"""
from __future__ import annotations

from collections import Counter

from common import trace
from common.agent import AgentResult, solve
from common.llm import ask, ask_json
from common.sandbox import run_python

VERIFY_CODE_PROMPT = """You are an independent verifier. Write a short standalone \
Python script that solves the problem below FROM SCRATCH. Use a different method \
or a library where possible; do NOT reuse or restate the proposed answer in the \
derivation. The script must compute the value into a variable named `expected` \
and finish with exactly these lines:

proposed = {answer}
ok = abs(expected - proposed) <= max(1e-9, 0.005 * abs(expected))
print("PASS" if ok else f"FAIL: independent recomputation gives {{expected}}")

Problem: {question}

Reply with ONLY the Python code. No backticks, no commentary."""


def _strip_fences(code):
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```"))


def programmatic_verify(question, answer):
    """Returns (passed, verifier_output). One model call + one sandbox run.
    The verifier sees the QUESTION and the proposed ANSWER, never the
    generator's code or reasoning (correlated errors make a check theater)."""
    try:
        float(str(answer).replace(",", ""))
    except ValueError:
        return False, f"FAIL: proposed answer {answer!r} is not numeric"
    code = _strip_fences(ask(VERIFY_CODE_PROMPT.format(question=question, answer=answer)))
    out = run_python(code, timeout=15)
    passed = out.strip().startswith("PASS")
    trace.log("verdict", verifier="programmatic", passed=passed, output=out[:500])
    return passed, out


def solve_with_verification(question, max_iters=3, **solve_kwargs) -> AgentResult:
    """Generator -> verifier loop. The verifier's output is the external signal
    fed back to the generator; iterations are capped."""
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    feedback, result = None, AgentResult()
    for it in range(1, max_iters + 1):
        result = solve(question, feedback=feedback, **solve_kwargs)
        for k in totals:
            totals[k] += getattr(result, k)
        totals["calls"] += 1                          # the verifier's own call
        if result.answer is None:
            feedback = "No 'ANSWER: <value>' line was produced."
            continue
        passed, out = programmatic_verify(question, result.answer)
        if passed:
            result.stopped = f"verified@{it}"
            break
        feedback = out
    else:
        result.stopped = "unverified"
    for k, v in totals.items():
        setattr(result, k, v)
    return result


def _normalize(ans):
    try:
        return f"{float(ans):.6g}"
    except (TypeError, ValueError):
        return None


def self_consistent_solve(question, k=5, temperature=0.8, **solve_kwargs) -> AgentResult:
    """Sample k solutions at T>0, majority-vote the final answers."""
    agg = AgentResult()
    votes = []
    for _ in range(k):
        r = solve(question, temperature=temperature, **solve_kwargs)
        agg.calls += r.calls
        agg.prompt_tokens += r.prompt_tokens
        agg.completion_tokens += r.completion_tokens
        n = _normalize(r.answer)
        if n is not None:
            votes.append(n)
    if votes:
        winner, count = Counter(votes).most_common(1)[0]
        agg.answer, agg.text = winner, f"votes={votes}"
        agg.stopped = f"vote {count}/{k}"
    trace.log("vote", votes=votes, winner=agg.answer)
    return agg


# ------------------------------------------------------ model-based judge --
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "errors": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "string"},
    },
    "required": ["verdict", "errors", "feedback"],
}

JUDGE_SYSTEM = """You are a verifier. You will be given a QUESTION and a proposed \
ANSWER. Your job is to find errors, not to be agreeable. Check, in order: \
(1) are the numerical steps correct? recompute them; (2) are units consistent; \
(3) does the final answer address the question; (4) are cited facts supported. \
Do not rewrite the answer. Only verify. Reply as JSON."""


def model_judge(question, answer, think=False):
    """LLM-as-judge with a structured verdict. Weakest verifier in the taxonomy:
    self-preference, verbosity and position biases, and no new information.
    Week 11's 06_judge_calibration.py measures how much to trust it."""
    out = ask_json(f"QUESTION:\n{question}\n\nPROPOSED ANSWER:\n{answer}",
                   JUDGE_SCHEMA, system=JUDGE_SYSTEM, think=think, model="judge")
    passed = out.get("verdict") == "PASS"
    trace.log("verdict", verifier="model_judge", passed=passed, output=out)
    return passed, out
