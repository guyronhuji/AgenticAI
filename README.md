# Agentic AI: course code and slides

Code and slides for the Agentic AI part of *AI Methods for Science*. This repository is kept up to date during the course.
Before each class, run `git pull` to get the latest version.

```bash
git clone https://github.com/guyronhuji/AgenticAI.git
cd AgenticAI
git pull          # later, to update
```

## What's here

| Folder | Contents |
|---|---|
| `slides/` | Lecture decks (PDF and PowerPoint) |
| `common/` | Shared building blocks used by every week: LLM client, agent loop, tools, sandbox, retrieval, tracing, verifiers |
| `week10/` | Week 10 demos and exercises |

## Setup

Every script runs against a local model served by [Ollama](https://ollama.com). You don't need a cloud account.

```bash
# 1. Install Ollama, then pull the model for your hardware:
ollama pull gemma4:e4b-it-q4_K_M   # 16 GB laptops (default)   | 8 GB: ollama pull gemma4:e2b-it-q4_K_M
ollama pull granite4:micro         # the small "bare" model used in the failure demos
ollama pull nomic-embed-text       # optional: embeddings without sentence-transformers

# 2. Python environment with uv (astral.sh/uv): installs Python if needed, creates .venv,
#    and installs the locked dependencies. Prefix commands with `uv run`.
uv sync                            # add `--extra embeddings` for local sentence-transformers (pulls PyTorch)
                                   # add `--extra notebook` for the 06a embedding-map notebook

# 3. Smoke test against your model (this is the setup homework; submit the output):
uv run python week10/00_smoke_test.py
```

If you prefer pip, run `pip install -r requirements.txt` in a venv of your own. `pyproject.toml` and `uv.lock` are the source of truth.

Run everything **from the repository root** with `uv run python <script>`.
Every demo writes a trace; read it with `uv run python common/view_trace.py --latest`,
or watch it live with `uv run python common/trace_viewer.py` (browser, http://localhost:8765).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama`, `openai` (any OpenAI-compatible host), `mock` (scripted; for tests) |
| `LLM_MODEL` | `gemma4:e4b-it-q4_K_M` | the agent model (tool calling + switchable thinking) |
| `BARE_MODEL` | `granite4:micro` | the bare model: no thinking, no tools; used where a plain LLM should fail |
| `LLM_BASE_URL` | `http://localhost:11434` | or a hosted endpoint |
| `LLM_API_KEY` | `ollama` | hosted key |
| `LLM_NUM_CTX` | `16384` | context window requested from Ollama |
| `SANDBOX` | `subprocess` | `docker` for a real boundary (`--network none`) |
| `SANDBOX_FIGDIR` | `sandbox_out` | where figures from sandboxed code are saved |
| `TRACE_DIR` | `traces` | where JSONL traces go |
| `CROSSREF_MAILTO` | example address | your own address for CrossRef's polite pool |
| `EMBED_BACKEND` | `auto` | `st` (sentence-transformers), `ollama`, `hash` |

## Week 10: lesson segment to code

| Segment | File | What it is |
|---|---|---|
| Setup | `week10/00_smoke_test.py` | Diagnostic smoke test (server, model, tool call, thinking) |
| 10.1 What is an agent | `week10/01_bare_llm_failures.py` | Four live failures of a bare model |
| 10.2 Local inference | `week10/02_sampling_and_thinking.py` | T=0 vs T=1; thinking off vs on with token costs |
| 10.3 Function calling | `week10/03_tool_calling_calculator.py` | Calculator tool + the loop by hand |
| 10.4 Agent patterns | `week10/04_react_prompt_loop.py` | Prompt-parsed ReAct |
| 10.5 Python as the tool | `week10/05_math_agent_skeleton.py` | Your exercise skeleton (`05_math_agent.py` is the reference solution; `--quiz` runs the scoreboard) |
| 10.5b Code as action | `week10/05b_web_search_agent.py` | One tool, `run_python_web`: the model writes the web search and the plots itself; figures land in `sandbox_out/` |
| 10.6 Retrieval | `week10/06_retrieval_demo.py` + `data/corpus/` | Classic, hybrid, and agentic retrieval over paper summaries |
| 10.6a Embedding map | `week10/06a_embedding_map.ipynb` | Notebook: sentence embeddings, cosine heatmap, interactive UMAP map (needs `uv sync --extra notebook`) |
| 10.7 MCP | `week10/07_mcp_server.py`, `08_mcp_client_agent.py` | Course tools as an MCP server; the agent loop consuming any server |
| HW 10 | `week10/make_corpus_from_pdfs.py` | Turns your PDFs into the text corpus the retrieval code expects |
| Week 10 exercise | `week10/09_FRONT_DESK_EXERCISE.md`, `week10/09_front_desk_skeleton.py` | The Front Desk: build a router in front of two specialists. Read the sheet, fill in the four TODOs in the skeleton. Practice, not handed in |

Note: `common/sandbox.py` with `SANDBOX=subprocess` is a teaching sandbox. It guards against accidents, not adversaries.
