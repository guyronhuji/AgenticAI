"""Lecture demo 10.4 (and Homework 3): prompt-parsed ReAct.

The ORIGINAL agent template (Yao et al., 2022): Thought / Action / Observation
in plain text; the runtime regex-parses "Action:" lines. Compare with
03_tool_calling_calculator.py: same loop, but the plumbing is fragile text
parsing instead of a trained structured format. This is also what "how agents
are trained" looks like from the outside: native tool calling is this
template internalized by fine-tuning on trajectories.

    python week10/04_react_prompt_loop.py

Runs on the BARE model (no native tool-calling bias, no thinking): the prompt
is the only thing holding the format together, which is the point. Two things
a small model does here that are worth narrating: it restarts from the top
instead of continuing from the Observation (the CONTINUE line and the example
in the system prompt are the mitigation, and the runtime adds a loop guard
when it repeats itself), and it sometimes answers with a bare number and no
Action line (PARSE FAILURE: the regex is the whole parser). A third: asked for
a procedure (say, "give the answer in binary") with only a calculator, it tries
to run the algorithm in prose and cannot carry the state from turn to turn,
so it repeats itself; the guard hints once, then terminates with a LOOP
verdict. A procedure needs code as the action (10.5), not a calculator.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.llm import chat  # noqa: E402
from common.tools import calc  # noqa: E402

SYSTEM = """You solve problems by interleaving Thought, Action, Observation.
Available actions:
  calc[expression]   -- evaluate an arithmetic expression
  finish[answer]     -- give the final answer and stop

Rules: write exactly ONE Thought and ONE Action per turn, then stop. The
Observation is written by the runtime, never by you. You will be shown the
transcript so far; CONTINUE it from the last Observation. Do not repeat
earlier steps. As soon as an Observation gives you the answer, finish.

Example transcript:
Question: What is 3 times 2 to the power 10?
Thought: I need 3*(2**10). I should calculate it.
Action: calc[3*(2**10)]
Observation: 3072
Thought: The observation gives the answer.
Action: finish[3072]"""

CONTINUE = "\nContinue the transcript. Write the next Thought and Action only."

ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)


def react_agent(question, max_steps=8, verbose=True):
    transcript = f"Question: {question}\n"
    last_action, repeats, observations = None, 0, []
    for _ in range(max_steps):
        reply = chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": transcript + CONTINUE}],
                     max_tokens=300, stop=["Observation:"], model="bare").content or ""
        # belt and braces: stop=["Observation:"] is the fence; cut anything past it anyway
        reply = reply.split("Observation:")[0]
        if verbose:
            print(reply.strip())
        m = ACTION_RE.search(reply)
        if not m:                                   # THE classic ReAct failure
            return "(PARSE FAILURE: no valid Action found. Discuss!)"
        action, arg = m.group(1).strip(), m.group(2).strip()
        if action == "finish":
            return arg
        if (action, arg) == last_action:            # a loop guard: the runtime notices repetition
            repeats += 1
            if repeats >= 2:                        # hinted once already: end the episode honestly
                trace.log("tool_call", name="loop_guard", args={"arg": arg}, result="LOOP: terminated")
                return (f"(LOOP: the model repeated {action}[{arg}] three times; results so far "
                        f"{observations}. The runtime stopped it. Discuss!)")
            obs = (f"you already ran this; results so far: {observations}. Use them: "
                   "write Action: finish[<answer>]")
            trace.log("tool_call", name="loop_guard", args={"arg": arg}, result=obs)
            if verbose:
                print(f"Observation: {obs}   <- injected by the RUNTIME (repeat detected)")
            transcript += f"{reply.strip()}\nObservation: {obs}\n"
            continue
        last_action, repeats = (action, arg), 0
        if action == "calc":
            try:
                obs = calc(arg)
            except Exception as e:
                obs = f"ERROR: {e}"
        else:
            obs = f"ERROR: unknown action '{action}'"
        observations.append(obs)
        trace.log("tool_call", name=action, args={"arg": arg}, result=obs)
        if verbose:
            print(f"Observation: {obs}   <- injected by the RUNTIME")
        transcript += f"{reply.strip()}\nObservation: {obs}\n"
    return "(gave up: too many steps)"


if __name__ == "__main__":
    q = ("A sample's activity halves every 5730 years. What fraction remains "
         "after 17190 years? Give a decimal.")
    with trace.run("react_prompt_loop"):
        print("Q:", q, "\n")
        print("\nFINAL:", react_agent(q))
        print("(correct: 0.125, three half-lives)")
