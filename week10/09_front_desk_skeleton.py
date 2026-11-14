"""09: the Front Desk. A router and two specialists (student skeleton: four TODOs).

    python week10/09_front_desk_skeleton.py "What is 17 to the power 13?"   # one question, verbose
    python week10/09_front_desk_skeleton.py                                  # your six questions x 3 trials, with a table
    python week10/09_front_desk_skeleton.py --trials 1                       # quicker
    python week10/09_front_desk_skeleton.py --corpus my_corpus               # your own papers

Fill in TODO 1 to TODO 4, in order. Everything else is given. Until TODO 2 is done
the script stops with "TODO 2"; that is expected.

Three agents on the same loop (common/agent.run_agent):

    router       no tools. One schema-constrained call: compute | literature | neither.
    compute      run_python_tool only. Cannot see the corpus.
    literature   search_corpus only. Cannot run code.

Each specialist starts with a fresh context: it gets the question and its own
system prompt, nothing from the router. `neither` ends the run with a fixed
refusal. The routing decision is logged to the trace as a `route` event.

What to watch: the question that needs BOTH (a number from the corpus, then
arithmetic). Pure routing has no good answer for it. That is the lesson, and
the reason Week 11 talks about orchestrators and verification.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.agent import MATH_SYSTEM, run_agent  # noqa: E402
from common.llm import ask_json, model_for  # noqa: E402
from common.retrieval import Index, make_search_tool  # noqa: E402
from common.tools import ToolRegistry, run_python_tool  # noqa: E402

HERE = pathlib.Path(__file__).parent
REFUSAL = "This is outside what I can answer."

# ---------------------------------------------------------------- the router
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["compute", "literature", "neither"]},
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
}

# TODO 1. Write the router's prompt. It must (a) say that the router decides and does not
# answer, (b) describe each of the three routes in a sentence: what that specialist CAN do
# and what it cannot, (c) end with "Question: " so the question can be appended.
# These descriptions are the only thing the router knows about the specialists.
ROUTER_PROMPT = """You are the front desk of a small research assistant. ...

compute     ...
literature  ...
neither     ...

Question: """


def route(question: str) -> dict:
    # TODO 2. One schema-constrained model call, then log it.
    #   decision = ask_json(ROUTER_PROMPT + question, ROUTE_SCHEMA)
    #   trace.log("route", question=question, route=decision["route"], reason=decision.get("reason", ""))
    #   return decision
    # (If decision["route"] is not one of the three routes, return route "neither".)
    raise NotImplementedError("TODO 2: route()")


# ---------------------------------------------------------- the specialists
LITERATURE_SYSTEM = (
    "You answer questions about a corpus of research papers. ALWAYS call search_corpus first; "
    "call it again with different words if the passages do not contain what you need. "
    "Answer only from the passages and cite passage ids in brackets. If after searching the "
    "corpus does not contain the answer, say exactly: The sources do not contain this.")


def compute_specialist(question: str):
    tools = ToolRegistry()
    tools.register(run_python_tool)
    return run_agent(question, tools, MATH_SYSTEM, max_steps=6)


def literature_specialist(question: str, index: Index):
    tools = ToolRegistry()
    tools.register(make_search_tool(index, k=3, mode="hybrid"))
    return run_agent(question, tools, LITERATURE_SYSTEM, max_steps=8)


# ------------------------------------------------------------ the whole desk
def front_desk(question: str, index: Index, verbose=False) -> dict:
    """Route, dispatch, return {route, reason, text, calls, tokens}. calls includes the router's."""
    decision = route(question)
    if verbose:
        print(f"[router] -> {decision['route']}: {decision.get('reason', '')}")
    # TODO 3. Dispatch. Call compute_specialist(question) or literature_specialist(question, index)
    # according to decision["route"]; each returns an AgentResult r with r.text, r.calls, r.tokens.
    # For "neither", call nobody:  return {**decision, "text": REFUSAL, "calls": 1, "tokens": 0}
    # Otherwise:                   return {**decision, "text": r.text or "", "calls": 1 + r.calls, "tokens": r.tokens}
    raise NotImplementedError("TODO 3: dispatch in front_desk()")


# TODO 4. Your six questions. Two computation, two whose answer is in the corpus, and two
# awkward ones: one that needs a number from the corpus AND THEN arithmetic on it ("both"),
# and one that nobody here can answer ("neither").
# (question, the route you expect, a string the right answer must contain; None = judge by eye)
QUESTIONS = [
    ("What is the standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?", "compute", "2"),          # example: replace
    ("What BLEU score did the big Transformer reach on WMT 2014 English-to-German?", "literature", "28.4"),  # example: replace
    # four more ...
]


def run_table(index: Index, trials: int):
    print(f"model: {model_for('agent')}   trials per question: {trials}\n")
    print(f"{'#':>2}  {'expected':10s} {'routes chosen':34s} {'right':>6s} {'calls':>6s} {'tokens':>7s}")
    tot_right = tot_runs = 0
    for n, (q, expected, must) in enumerate(QUESTIONS, start=1):
        routes, right, calls, toks = [], 0, 0, 0
        for t in range(trials):
            with trace.run(f"frontdesk_q{n}_t{t + 1}", announce=False):
                out = front_desk(q, index)
            routes.append(out["route"])
            ok = must is not None and must.lower() in out["text"].lower()
            right += ok; calls += out["calls"]; toks += out["tokens"]
        tot_right += right; tot_runs += trials
        print(f"{n:>2}  {expected:10s} {', '.join(routes):34s} {right}/{trials:<4d} {calls / trials:>6.1f} {toks / trials:>7.0f}")
        print(f"    {q[:92]}")
    print(f"\nanswers right: {tot_right}/{tot_runs}.   Traces: traces/*frontdesk*  "
          f"(uv run python common/view_trace.py --latest)")
    print("'right' is a crude string match. Read the traces for the awkward questions.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    corpus = HERE / "data" / "corpus"
    trials = 3
    if "--corpus" in argv:
        i = argv.index("--corpus"); corpus = pathlib.Path(argv[i + 1]); del argv[i:i + 2]
    if "--trials" in argv:
        i = argv.index("--trials"); trials = int(argv[i + 1]); del argv[i:i + 2]
    index = Index.from_folder(corpus)
    print(f"corpus: {corpus}  ({len(index.chunks)} chunks)")
    if argv:
        q = " ".join(argv)
        with trace.run("frontdesk"):
            out = front_desk(q, index, verbose=True)
        print("\nFINAL:", out["text"])
        print(f"(route={out['route']}, {out['calls']} model calls incl. the router, {out['tokens']} tokens)")
    else:
        run_table(index, trials)
