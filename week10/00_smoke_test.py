"""Setup homework: prove your local model works. Submit the printed output.

    python week10/00_smoke_test.py

This is a DIAGNOSTIC smoke test: it checks the server is up, that the
configured model is pulled, that it answers, that it can call a tool, and
whether it has a thinking mode. Each failure message names the fix.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.llm import BACKEND, BASE_URL, MODEL, ROLES, chat  # noqa: E402
from common.tools import ToolRegistry, calc  # noqa: E402


def fail(msg):
    print(f"\nSMOKE TEST FAILED: {msg}")
    sys.exit(1)


def main():
    print(f"backend:  {BACKEND}\nendpoint: {BASE_URL}")
    print(f"models:   agent={ROLES['agent']}  bare={ROLES['bare']}  judge={ROLES['judge']}")

    if BACKEND == "ollama":
        try:
            import ollama
            client = ollama.Client(host=BASE_URL)
            names = [m.model for m in client.list().models]
        except Exception as e:
            fail(f"cannot reach Ollama at {BASE_URL} ({e}). Start the Ollama app "
                 "(or `ollama serve`), then retry.")
        print(f"pulled:   {names}")
        for role in ("agent", "bare"):
            tag = ROLES[role]
            if tag not in names and tag + ":latest" not in names:
                fail(f"the {role} model {tag!r} is not pulled. Run: ollama pull {tag}")

    # 1) plain chat
    t0 = time.perf_counter()
    r = chat([{"role": "user", "content": "Reply with exactly one short sentence "
               "confirming you are running, and name one thing neural networks are bad at."}])
    print(f"reply:    {r.content!s:.200}")
    print(f"timing:   {time.perf_counter() - t0:.1f}s (first call includes model load), "
          f"{r.completion_tokens} tokens generated")
    if not r.content:
        fail("empty reply")

    # 1b) the bare model answers too (used by the failure demos; no tools, no thinking)
    r = chat([{"role": "user", "content": "What is 123456 * 654321? Reply with just the number."}],
             model="bare")
    print(f"bare:     {ROLES['bare']} says {str(r.content).strip()[:40]!r} "
          f"(correct is 80779853376; being wrong here is expected and useful)")

    # 2) tool call
    tools = ToolRegistry()
    tools.register(calc)
    r = chat([{"role": "user", "content": "Use the calc tool to compute 123456 * 654321."}],
             tools=tools.schemas())
    if r.tool_calls:
        c = r.tool_calls[0]
        print(f"tool:     model called {c.name}({c.args}) -> {tools.call(c.name, c.args)}")
    else:
        print("tool:     WARNING: the model answered in text instead of calling the tool. "
              "Either the model tag does not support tools (pick one from the setup "
              "guide table) or Ollama is out of date (update it). Week 10 needs this.")

    # 3) structured output (JSON schema). Not available on Ollama's MLX runner.
    from common.llm import ask_json, structured_output_available
    out = ask_json("What is the boiling point of water at 1 atm?",
                   {"type": "object", "properties": {"value": {"type": "number"},
                                                      "unit": {"type": "string"}},
                    "required": ["value", "unit"]})
    if structured_output_available() is False:
        print(f"schema:   NOT supported by this runner (prompted-JSON fallback gave {out}). "
              "This happens with MLX builds on Apple silicon: pull the GGUF build instead, "
              "e.g. ollama pull gemma4:e4b-it-q4_K_M, and set LLM_MODEL to it.")
    elif "_parse_error" in out:
        print(f"schema:   WARNING: reply was not valid JSON: {out['_parse_error'][:80]!r}")
    else:
        print(f"schema:   supported ({out})")

    # 4) thinking mode (optional capability)
    r = chat([{"role": "user", "content": "What is 17 * 23? One line."}], think=True)
    if r.thinking:
        print(f"thinking: supported ({len(r.thinking)} chars of reasoning before the answer)")
    else:
        print("thinking: not exposed by this model (fine; Week 11 has a fallback)")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
