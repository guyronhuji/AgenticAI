"""Lecture demo 10.2: sampling (temperature) and the thinking switch.

Part 1 (BARE model, no thinking mode): the same question at T=0 twice and at T=1 five times. T changes the
        sampling rule over an UNCHANGED distribution; the fraction of correct
        samples at T=1 is the `p` that self-consistency (Week 11) amplifies.
Part 2 (AGENT model): the same question with thinking off and on. Thinking is test-time
        compute: more tokens before the answer, visible in a separate field.
        Write down the token counts; Week 11 puts them on a plot.

    python week10/02_sampling_and_thinking.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.llm import chat, model_for  # noqa: E402

Q = ("A bacterial culture doubles every 20 minutes. Starting from a single "
     "cell, how many cells are there after 4 hours? Reply with a short "
     "explanation and then the final line 'ANSWER: <number>'.")
CORRECT = 4096  # 12 doublings
ANS = re.compile(r"ANSWER:\s*([\d,]+)")


def answer(text):
    m = ANS.findall(text or "")
    return m[-1].replace(",", "") if m else None


def one(temperature=0.0, think=False, model="bare"):
    r = chat([{"role": "user", "content": Q}], temperature=temperature, think=think, model=model)
    return answer(r.content), r


if __name__ == "__main__":
    with trace.run("sampling_and_thinking"):
        print(f"--- T = 0, twice on {model_for('bare')} (should be near-identical) ---")
        for _ in range(2):
            a, r = one(0.0)
            print(f"  answer={a}  ({r.completion_tokens} tokens)")

        print("\n--- T = 5, five samples ---")
        hits = 0
        for i in range(5):
            a, r = one(50.0)
            ok = a == str(CORRECT)
            hits += ok
            print(f"  sample {i + 1}: answer={a}  {'OK' if ok else 'WRONG'}  ({r.completion_tokens} tokens)")
        print(f"  fraction correct at T=1: p = {hits}/5   <- write this on the board")

        print("\n--- thinking OFF vs ON (T = 0), on the AGENT model ---")
        for think in (False, True):
            a, r = one(0.0, think=think, model="agent")
            th = f", {len(r.thinking)} chars of thinking" if r.thinking else ", no thinking field"
            print(f"  think={think!s:5}: answer={a}  ({r.completion_tokens} tokens{th})")
        print(f"\n(correct: {CORRECT}. Compare the token costs: that is the price of thinking.)")
