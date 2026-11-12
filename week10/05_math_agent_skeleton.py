"""Guided exercise: assemble the math agent. Fill in the TODOs.

Milestones (check each before moving on):
  M1  the model replies at all            (python week10/05_math_agent_skeleton.py "hi")
  M2  you can see msg.tool_calls           (print the Reply object)
  M3  generated code runs in the sandbox and its stdout comes back as a tool message
  M4  the three test questions below come out right
Stretch: on "ERROR:" in the tool output the model gets the traceback and
retries (it already does if your loop is right: the error IS the tool result).
Then compare with common/agent.py, which is the same loop, generalized.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.agent import extract_answer  # noqa: E402
from common.llm import chat  # noqa: E402
from common.sandbox import run_python  # noqa: E402

SYSTEM = (
    "You are a careful scientific assistant. For any quantitative question do "
    "NOT compute in your head: call run_python with a short script that "
    "prints the result. If the tool returns an ERROR, fix the code and call it "
    "again. Finish with a short explanation whose LAST line is exactly:\n"
    "ANSWER: <numeric value only>")

# TODO 1: write the tool schema the model will see. Name it "run_python" with one
# required string parameter "code". Vague descriptions cause wrong calls.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "...",                                   # TODO
        "parameters": {"type": "object", "properties": {}, "required": []},  # TODO
    },
}]


def solve(question, max_steps=6, verbose=True):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(max_steps):
        reply = chat(messages, tools=TOOLS)
        if not reply.tool_calls:
            return reply.content, extract_answer(reply.content)

        # TODO 2: append the assistant message (reply.as_message()) BEFORE the tool results.
        # Forgetting this is the classic bug: the model loses the thread and re-calls forever.

        for call in reply.tool_calls:
            code = call.args.get("code", "")
            if verbose:
                print(f"--- model wrote ---\n{code}")
            # TODO 3: run `code` in the sandbox (run_python) and store it in `result`
            result = "..."
            trace.log("tool_call", name="run_python", args={"code": code}, result=result)  # shows in the trace window
            if verbose:
                print(f"--- sandbox said ---\n{result}")
            # TODO 4: append the tool message: role="tool", tool_call_id=call.id,
            # name="run_python", content=result
    return None, None


TESTS = [
    ("What is 2 to the power 100?", 1267650600228229401496703205376),
    ("How many primes are there below 10000?", 1229),
    ("A ball is dropped from 80 m with g = 9.8 m/s^2. How many seconds until it "
     "hits the ground? Round to 2 decimals.", 4.04),
]

if __name__ == "__main__":
    with trace.run("math_agent_skeleton"):
        if len(sys.argv) > 1:
            text, ans = solve(" ".join(sys.argv[1:]))
            print("FINAL:", text, "\nANSWER:", ans)
        else:
            for q, expected in TESTS:
                text, ans = solve(q, verbose=False)
                print(f"{'OK  ' if ans and abs(float(ans) - expected) < 0.01 * abs(expected) + 1e-9 else 'FAIL'} "
                      f"got={ans} expected={expected}")
