"""The agent loop, written once, reused all of Part IV.

    context = [question]
    while True:
        out = LLM(context)
        if out is an action: context += execute(out)
        else: return out

`run_agent` is the general loop (any tools, any system prompt). `solve` is the
Week 10 math agent: one tool (run_python), final line "ANSWER: <value>".
Every call and tool execution is traced (common/trace.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from common import trace
from common.llm import chat
from common.tools import ToolRegistry, run_python_tool

ANSWER_RE = re.compile(r"ANSWER:\s*([-+]?[\d.,]+(?:[eE][-+]?\d+)?)")


def extract_answer(text):
    """Value on the LAST 'ANSWER:' line; None if absent. Tolerates commas."""
    hits = ANSWER_RE.findall(text or "")
    return hits[-1].replace(",", "") if hits else None


@dataclass
class AgentResult:
    text: str | None = None
    answer: str | None = None
    steps: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    messages: list = field(default_factory=list)
    stopped: str = "final"          # final | max_steps | error
    nudged: bool = False            # the loop had to ask once for a final answer

    @property
    def tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def __getitem__(self, k):      # dict-style access for the eval harness
        return getattr(self, k)

    def get(self, k, default=None):
        return getattr(self, k, default)


def run_agent(question, tools: ToolRegistry, system: str, *, think=False,
              temperature=0.0, max_steps=8, extra_messages=None,
              verbose=False, truncate_tool_output=None) -> AgentResult:
    """The loop. `extra_messages` lets a caller inject feedback (Week 11).
    `truncate_tool_output` caps characters per tool result (Week 13: context budget)."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    messages += list(extra_messages or [])
    res = AgentResult(messages=messages)
    schemas = tools.schemas()

    for step in range(1, max_steps + 1):
        res.steps = step
        reply = chat(messages, tools=schemas, temperature=temperature, think=think)
        res.calls += 1
        res.prompt_tokens += reply.prompt_tokens
        res.completion_tokens += reply.completion_tokens
        if verbose and reply.thinking:
            print(f"--- thinking ({len(reply.thinking)} chars) ---\n{reply.thinking[:600]}")

        if not reply.tool_calls:                       # final answer
            if not (reply.content or "").strip() and not res.nudged:
                # Small models sometimes stop right after reading a tool result,
                # with an empty reply (or the answer stuck in the thinking field).
                # Correct execution, no answer: a harness problem, so the harness
                # nudges once. Visible in the trace as a 'nudge' event.
                res.nudged = True
                messages.append(reply.as_message())
                messages.append({"role": "user", "content":
                                 "You stopped without answering. Using the tool results "
                                 "above, give the final answer now."})
                trace.log("nudge", reason="empty final reply",
                          thinking_len=len(reply.thinking or ""))
                if verbose:
                    print("--- empty final reply; nudging once ---")
                continue
            res.text = reply.content
            res.answer = extract_answer(reply.content)
            messages.append(reply.as_message())
            return res

        messages.append(reply.as_message())            # 1) assistant w/ tool_calls
        for call in reply.tool_calls:
            if verbose:
                print(f"--- tool call: {call.name} ---")
                print(call.args.get("code") or call.args)
            result = tools.call(call.name, call.args)  # 2) WE execute
            if truncate_tool_output and len(result) > truncate_tool_output:
                result = (result[:truncate_tool_output] +
                          f"\n... [truncated {len(result) - truncate_tool_output} chars]")
            if verbose:
                print(f"--- result ---\n{result}")
            messages.append({"role": "tool", "tool_call_id": call.id,   # 3) tool msg
                             "name": call.name, "content": result})

    res.stopped = "max_steps"
    return res


# ------------------------------------------------------ the Week 10 agent --
MATH_SYSTEM = (
    "You are a careful scientific assistant. For any quantitative question do "
    "NOT compute in your head: call the run_python_tool with a short script "
    "that prints the result. Read the tool output. If it is an ERROR, fix the "
    "code and call the tool again. When you have the result, reply in prose "
    "and make the LAST line exactly:\nANSWER: <numeric value only, no units>")

MATH_TOOLS = ToolRegistry()
MATH_TOOLS.register(run_python_tool)


def solve(question, *, temperature=0.0, think=False, max_steps=6,
          feedback=None, verbose=False) -> AgentResult:
    """Week 10 math agent. `feedback` (Week 11) is verifier output from a
    rejected previous attempt: the external signal that makes retry work."""
    extra = None
    if feedback:
        extra = [{"role": "user", "content":
                  "A verifier rejected a previous attempt. Verifier output:\n"
                  f"{feedback}\nSolve the problem again, from scratch, taking "
                  "the verifier's finding into account."}]
    return run_agent(question, MATH_TOOLS, MATH_SYSTEM, think=think,
                     temperature=temperature, max_steps=max_steps,
                     extra_messages=extra, verbose=verbose)


def is_correct(answer_str, expected, rel_tol=0.005):
    """Grade an extracted answer against the expected numeric value."""
    if answer_str is None:
        return False
    try:
        got = float(answer_str)
        expected = float(expected)
    except (TypeError, ValueError):
        return False
    return abs(got - expected) <= max(1e-9, rel_tol * abs(expected))
