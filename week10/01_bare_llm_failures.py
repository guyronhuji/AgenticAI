"""Lecture demo 10.1: what an LLM is, and three live failures of a bare model.

(a) arithmetic  -> fixed by code execution (10.5)
(b) local data  -> fixed by tools (10.3)
(c) citations   -> fixed by retrieval (10.6) + verification (Week 11)
(d) "did you run it?" -> the model CLAIMS actions it cannot take; this is the
    seam between a model and an agent (10.1, definition of an agent)

    python week10/01_bare_llm_failures.py

Instructor: keep the fabricated citation from (c); it is reused in Week 11.
Everything is traced: afterwards run  python common/view_trace.py --latest
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.llm import ask, model_for  # noqa: E402

FAILURES = [
    ("(a) mental arithmetic",
     "What is 123456 * 654321? Reply with just the number.",
     "correct value: 80779853376. Check the digits; it is usually wrong."),
    ("(b) access to local data",
     "How many rows are in the file measurements.csv in my current directory?",
     "it cannot know. Does it admit that, or invent a number?"),
    ("(c) citations",
     "Give me three peer-reviewed references (authors, year, title, venue) on "
     "using autoencoders for anomaly detection in astronomical light curves.",
     "verify each later; typically at least one is fabricated."),
    ("(d) claimed actions",
     "Run this Python and tell me the output: print(sum(i*i for i in range(1, 2001)))",
     "watch for 'I ran it and the output is...'. Nothing ran. The correct value "
     "is 2668667000. A model emits text; only a RUNTIME executes."),
]

if __name__ == "__main__":
    print(f"bare model: {model_for('bare')} (no tools, no thinking mode)\n")
    with trace.run("bare_llm_failures"):
        for name, prompt, note in FAILURES:
            print("=" * 72)
            print(name)
            print(f"PROMPT: {prompt}")
            print(f"MODEL:  {ask(prompt, model='bare')}")
            print(f"NOTE:   {note}")
