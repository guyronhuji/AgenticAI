"""Shared building blocks for Part IV (v2).

llm.py        one chat() for three backends (ollama | openai | mock)
trace.py      JSONL tracing of every model and tool call
view_trace.py read a trace; --follow tails it live
trace_viewer.py browser dashboard that shows traces live (projector)
tools.py      tool registry + the standard course tools
sandbox.py    sandboxed execution of model-written Python
agent.py      the agent loop; the Week 10 math agent
verify.py     verifiers, self-consistency, verifier loop, LLM judge
evals.py      tasks, trials, graders, pass@k / pass^k, plots
retrieval.py  classic, hybrid and agentic retrieval
"""
