"""05b: ONE tool, run_python, and the model writes everything else itself.

Code-as-action (CodeAct, segment 10.4) taken literally: the only tool is a
Python sandbox, and the model is told what the sandbox can do: reach the
network, and save figures. Whatever it needs from the web it fetches with code
it writes; whatever it wants to show, it plots with matplotlib or seaborn and
saves to disk. The harness then lists the new image files in the tool result
and opens them on screen (Preview on macOS). Nothing in the harness knows what
a "search" or a "plot" is.

    python week10/05b_web_search_agent.py "What is the half-life of Co-58 in days, and how many half-lives is one year?"
    python week10/05b_web_search_agent.py "Plot the decay curve of a Co-58 source over one year, with a marker at each half-life."
    python week10/05b_web_search_agent.py --model bare "Who discovered the neutron, and in what year?"
    SANDBOX=docker python week10/05b_web_search_agent.py "..."     # --network none: watch it fail, and read how

Differences from 05: the tool is run_python_web (same sandbox, 60 s timeout
instead of 10, a persistent working directory `sandbox_out/` so saved files
survive, and a docstring that says the network is open and figures are
shown), and the system prompt shows one working recipe for the web (the
Wikipedia search API) so a 4B model does not have to invent an endpoint.

Teaching points: (1) the action space of a code tool is everything the
interpreter can do, including sockets and the file system, so the sandbox is
the boundary, not the tool list; (2) the subprocess sandbox stops accidents,
not adversaries, and here is the network hole in use; (3) a fetched page is
text somebody else wrote, entering the context as a tool result: Week 11,
prompt injection; (4) the model never sees the figure. It sees the filename.
"""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import trace  # noqa: E402
from common.agent import run_agent  # noqa: E402
from common.llm import model_for, set_role  # noqa: E402
from common.sandbox import run_python  # noqa: E402
from common.tools import ToolRegistry  # noqa: E402


FIG_DIR = os.environ.get("SANDBOX_FIGDIR", "sandbox_out")


def run_python_web(code: str) -> str:
    """Execute a Python script in a sandbox and return what it prints. Available:
    numpy, scipy, pandas, matplotlib, seaborn, requests, bs4 (BeautifulSoup), lxml.
    The sandbox HAS network access: scripts can call web APIs and download pages.
    Figures: save with plt.savefig("name.png") in the current directory; saved
    figures are reported back and opened on screen. Never call plt.show().
    Use print() for every value you need back. 60 second limit.

    code: complete Python source code to run
    """
    return run_python(code, timeout=60, workdir=FIG_DIR, open_figures=True)


SYSTEM = """You are a careful scientific assistant with ONE tool, run_python_web, a Python
sandbox that can reach the internet and save figures. You have no other tools: anything you need
from the web, you fetch with code you write.

Rules:
- Never answer a factual question from memory. Write code that looks it up, print
  what you found, read the output, then answer.
- Never compute in your head. Print the arithmetic from code.
- Figures: numpy, scipy, pandas, matplotlib and seaborn are installed. Save every
  figure with plt.savefig("descriptive_name.png", dpi=120); never plt.show(). The
  tool result lists the saved files and they open on the user's screen. Label axes
  with units and print the key numbers behind the plot as well.
- If the tool returns an ERROR, read it, fix the code, call again.
- Cite the URL your answer relies on. If the web does not settle it, say so.
- If the answer is a number, make the LAST line exactly:
ANSWER: <numeric value only, no units>

A recipe that works (Wikipedia, no key). Adapt the query and the page title:

import requests, re, html
UA = {"User-Agent": "course-agent/1.0"}
# 1) search
r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=15, params={
    "action": "query", "list": "search", "srsearch": "cobalt-58 half-life", "srlimit": 3, "format": "json"})
for h in r.json()["query"]["search"]:
    print(h["title"], "|", re.sub(r"<[^>]+>", "", html.unescape(h["snippet"])))
# 2) read one page as plain text (first 3000 characters)
r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=15, params={
    "action": "query", "prop": "extracts", "explaintext": 1, "titles": "Isotopes of cobalt", "format": "json"})
page = next(iter(r.json()["query"]["pages"].values()))
print(page["extract"][:3000])

Other sites work too. BeautifulSoup is installed for parsing HTML:

from bs4 import BeautifulSoup
soup = BeautifulSoup(requests.get(url, headers=UA, timeout=15).text, "lxml")
text = soup.get_text(" ", strip=True)
print(text[:3000])          # pages are long: print only the part you need
# soup.select("table.wikitable tr") etc. for structured pieces

"""

TOOLS = ToolRegistry()
TOOLS.register(run_python_web)

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--model" in argv:
        i = argv.index("--model")
        set_role("agent", argv[i + 1])
        del argv[i:i + 2]
    think = "--think" in argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(0)
    print(f"agent model: {model_for('agent')}   tools: ['run_python_web']")
    t0 = time.time()
    with trace.run("codeact_web_agent"):
        r = run_agent(" ".join(args), TOOLS, SYSTEM, think=think, max_steps=10,
                      verbose=True, truncate_tool_output=4000)
    print("\nFINAL:", r.text)
    print(f"(cost: {r.calls} LLM calls, {r.tokens} tokens, stopped={r.stopped})")
    figdir = pathlib.Path(FIG_DIR).resolve()
    made = sorted(str(p) for p in figdir.glob("*")
                  if p.suffix.lower() in (".png", ".pdf", ".svg", ".jpg", ".jpeg")
                  and p.stat().st_mtime >= t0 - 1) if figdir.is_dir() else []
    if made:
        print("\nfigures from this run:")
        for f in made:
            print("  " + f)
