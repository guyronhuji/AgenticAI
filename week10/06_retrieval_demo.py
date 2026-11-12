"""Lecture demo 10.6: retrieval, three ways.

  --mode classic   chunk -> embed -> top-k cosine -> one prompt (v1 RAG)
  --mode hybrid    BM25 + vectors fused with reciprocal-rank fusion
  --mode agentic   search_corpus() is a TOOL inside the agent loop, next to
                   run_python: the model re-queries, then computes

    python week10/06_retrieval_demo.py                # runs all three on the agent model
    python week10/06_retrieval_demo.py --mode hybrid
    python week10/06_retrieval_demo.py --bare         # one-shot modes on the BARE model:
                                                      # provokes the "ignores the context"
                                                      # generation failure for the board

The corpus: 12 short summaries of well-known ML papers (week10/data/corpus).
Question 1 is answerable by any mode. Question 2 is a vocabulary-mismatch
trap (a number: embeddings are bad at numbers, BM25 is exact). Question 3 is
not in the corpus and must produce the refusal sentence. Question 4 needs two
retrievals and a subtraction: only the agentic mode gets it reliably.
Diagnose failures by printing what was retrieved (the index does that).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.agent import run_agent  # noqa: E402
from common.llm import ask, model_for  # noqa: E402
from common.retrieval import RAG_PROMPT, Index, format_hits, make_search_tool  # noqa: E402
from common.tools import ToolRegistry, run_python_tool  # noqa: E402

CORPUS = pathlib.Path(__file__).parent / "data" / "corpus"

QUESTIONS = [
    "What BLEU score did the big Transformer reach on WMT 2014 English-to-German?",      # 28.4
    "Which method improved GSM8K accuracy by 17.9 percentage points?",                   # self-consistency
    "What accuracy does AlphaFold2 achieve on RNA structures?",                          # not in corpus
    "How many more BLEU points did the Transformer score on English-to-French than on "
    "English-to-German? Give the number.",                                               # 41.8 - 28.4 = 13.4
]

AGENT_SYSTEM = (
    "You answer questions about a document corpus. ALWAYS call search_corpus "
    "first; call it again with different words if the passages do not contain "
    "what you need. Use run_python_tool for any arithmetic. Cite passage ids in "
    "brackets. If after searching the corpus does not contain the answer, say "
    "exactly: The sources do not contain this.")


ONE_SHOT_MODEL = "agent"


def one_shot(index, question, mode):
    hits = index.search(question, k=3, mode=mode)
    print(f"   [retrieved via {mode}: {[c.id for c, _ in hits]}]")
    return ask(RAG_PROMPT.format(sources=format_hits(hits), question=question), model=ONE_SHOT_MODEL)


def agentic(index, question):
    tools = ToolRegistry()
    tools.register(make_search_tool(index, k=3, mode="hybrid"))
    tools.register(run_python_tool)
    r = run_agent(question, tools, AGENT_SYSTEM, max_steps=8)
    return f"{r.text}\n   ({r.calls} model calls, {r.tokens} tokens)"


if __name__ == "__main__":
    modes = ["classic", "hybrid", "agentic"]
    if "--mode" in sys.argv:
        modes = [sys.argv[sys.argv.index("--mode") + 1]]
    if "--bare" in sys.argv:
        ONE_SHOT_MODEL = "bare"
    print(f"one-shot modes on {model_for(ONE_SHOT_MODEL)}; agentic mode on {model_for('agent')}")
    index = Index.from_folder(CORPUS)
    print(f"indexed {len(index.chunks)} chunks from {len(set(c.doc for c in index.chunks))} documents\n")
    with trace.run("retrieval_demo"):
        for mode in modes:
            print("#" * 72 + f"\n# mode: {mode}\n" + "#" * 72)
            for q in QUESTIONS:
                print("=" * 72)
                print("Q:", q)
                if mode == "agentic":
                    print("A:", agentic(index, q))
                else:
                    print("A:", one_shot(index, q, "vector" if mode == "classic" else "hybrid"))
