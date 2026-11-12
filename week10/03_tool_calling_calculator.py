"""Lecture demo 10.3: function calling with the simplest possible tool.

A calculator published to the model as a JSON schema, and the loop written
out by hand with no helper (common/agent.py is the same loop, generalized).
The model never executes anything: it emits a structured tool_call, OUR
code runs it, and the result goes back as a role="tool" message.

    python week10/03_tool_calling_calculator.py

The last question is deliberately ambiguous: watch the model call the tool
with wrong arguments. The tool ran perfectly and the answer is still wrong.
That sentence is the segue to Week 11.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.llm import chat  # noqa: E402
from common.tools import calc  # noqa: E402

# ---------- the tool schema: the ONLY thing the model sees about calc ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression, e.g. '3*(2**10)'. "
                       "Supports + - * / // % ** and parentheses only.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]


# ---------- the loop, by hand ----------
def run_agent(question, max_steps=5):
    messages = [{"role": "user", "content": question}]
    nudged = False
    for _ in range(max_steps):
        reply = chat(messages, tools=TOOLS)
        if not reply.tool_calls:                              # final answer...
            if not (reply.content or "").strip() and not nudged:
                # ...unless it is empty: small models sometimes stop right after a
                # tool result. The runtime asks once. (common/agent.py does the same.)
                nudged = True
                messages.append(reply.as_message())
                messages.append({"role": "user", "content": "You stopped without answering. "
                                 "Using the tool result above, give the final answer now."})
                trace.log("nudge", reason="empty final reply")
                print("   [runtime] empty reply; nudging once")
                continue
            return reply.content
        messages.append(reply.as_message())                   # 1. assistant: tool_calls=[...]
        for call in reply.tool_calls:
            try:
                result = calc(call.args.get("expression", ""))
            except Exception as e:                            # the model sees errors too
                result = f"ERROR: {e}"
            print(f"   [runtime] calc({call.args.get('expression')!r}) -> {result}")
            trace.log("tool_call", name="calc", args=call.args, result=result)   # the runtime logs what it ran
            messages.append({"role": "tool", "tool_call_id": call.id,   # 2. tool: result
                             "name": "calc", "content": result})
    return "(gave up: too many steps)"


QUESTIONS = [
    "What is 123456 * 654321?",                                   # the 10.1 failure, fixed
    "What is (1 + 1/365)**365, to 6 decimal places?",
    # deliberately ambiguous: 'it' has no clear referent
    "Take the number of days in a leap year, double it, then... actually "
    "just tell me what half of it was.",
]

if __name__ == "__main__":
    with trace.run("tool_calling_calculator"):
        for q in QUESTIONS:
            print("=" * 72)
            print("Q:", q)
            print("A:", run_agent(q))
