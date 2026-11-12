"""The Week 10 math agent, SOLUTION (segments 10.5 and the guided exercise).

NL question -> model writes Python -> sandboxed execution -> ANSWER line.
The agent lives in common/agent.py so Weeks 11 and 12 can reuse it.

One question, verbose:
    python week10/05_math_agent.py "What is the sum of the first 1000 primes?"
Scoreboard on the quiz set (3 trials each, pass^3 reported):
    python week10/05_math_agent.py --quiz week10/data/quiz_10.json
With thinking on:
    python week10/05_math_agent.py --think "..."
On another model (a role name or a tag), e.g. the small non-reasoning one:
    python week10/05_math_agent.py --model bare --quiz week10/data/quiz_10.json
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.agent import solve  # noqa: E402
from common.evals import Task, numeric_grader, run_eval  # noqa: E402
from common.llm import model_for, set_role  # noqa: E402


def run_quiz(path, think=False, trials=3):
    tasks = Task.from_json(path)
    with trace.run("math_agent_quiz"):
        res = run_eval(tasks, runner=lambda q: solve(q, think=think),
                       grader=numeric_grader, trials=trials, name="math_agent")
    s = res.summary()
    print(f"\nSCOREBOARD  accuracy={100 * s['accuracy']:.0f}%  "
          f"pass^{trials}={100 * s[f'pass^{trials}']:.0f}%  "
          f"mean calls={s['mean_calls']:.1f}  mean tokens={s['mean_tokens']:.0f}")
    res.to_csv("quiz_results.csv")
    print("per-trial rows -> quiz_results.csv")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--model" in argv:
        i = argv.index("--model")
        set_role("agent", argv[i + 1])
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    think = "--think" in argv
    print(f"agent model: {model_for('agent')}")
    if "--quiz" in argv and args:
        run_quiz(args[0], think=think)
    elif args:
        with trace.run("math_agent"):
            r = solve(" ".join(args), verbose=True, think=think)
        print("\nFINAL:", r.text)
        print(f"(cost: {r.calls} LLM calls, {r.tokens} tokens, stopped={r.stopped})")
    else:
        print(__doc__)
