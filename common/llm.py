"""Shared LLM client for the whole course (v2).

The agent loop never talks to a vendor SDK directly. It calls `chat()` with
plain-dict messages in ONE canonical format (the OpenAI chat format, which is
the industry lingua franca) and gets back one normalized `Reply`. A backend
adapter translates to whatever the provider wants. Three backends:

  LLM_BACKEND=ollama   (default) Ollama's native /api/chat via the `ollama`
                       package: reliable tool calls, a `think` switch, and
                       JSON-schema structured outputs.
  LLM_BACKEND=openai   any OpenAI-compatible hosted endpoint (the fallback for
                       weak laptops): set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
  LLM_BACKEND=mock     a scripted fake model, so the harness, tracing and eval
                       code can be tested on any machine in seconds with no
                       model at all. (Testing your harness without a model is
                       a real engineering practice, not a toy.)

Canonical message dicts:
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str|None,
   "tool_calls": [{"id": str, "type": "function",
                   "function": {"name": str, "arguments": <JSON string>}}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field

BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
# Three model ROLES. Different demos need different models:
#   agent  the tool-calling model with a switchable thinking mode; every agent loop,
#          verifier, retrieval and MCP demo, the workshop. Explicit GGUF build on purpose:
#          the bare "gemma4:e4b" tag is an MLX build on Apple silicon whose runner cannot
#          do schema-constrained output.
#   bare   a small model with NO thinking mode and no tools passed, for the demos that
#          must show a plain LLM failing: bare-LLM failures, sampling, prompt-parsed
#          ReAct, sycophancy, self-consistency. A model that thinks (or that is strong)
#          would get these right and the demo would teach nothing.
#   judge  the model used as LLM-as-judge; ideally NOT the agent model (self-preference).
# Override any of them with an environment variable; LLM_MODEL also sets the agent role.
MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b-it-q4_K_M")
ROLES = {
    "agent": MODEL,
    "bare": os.environ.get("BARE_MODEL", "granite4:micro"),
    # judge defaults to the bare model: a different family from the generator, already
    # pulled, cheap. Weak on purpose; 11.4 calibrates it. JUDGE_MODEL=gemma4:12b-it-q4_K_M
    # for a stronger judge if the machine has the RAM.
    "judge": os.environ.get("JUDGE_MODEL", os.environ.get("BARE_MODEL", "granite4:micro")),
}


def set_role(role: str, tag_or_role: str):
    """Point a role at another tag (or at another role), e.g. set_role("agent", "bare")
    to run an agent demo on the small non-reasoning model so that it fails more."""
    ROLES[role] = ROLES.get(tag_or_role, tag_or_role)


def model_for(role: str) -> str:
    """Resolve a role name ('agent' | 'bare' | 'judge') or a literal tag."""
    return ROLES.get(role, role)
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "http://localhost:11434" if BACKEND == "ollama" else "https://api.openai.com/v1",
)
# Ollama silently falls back to a small context unless told otherwise.
# Agents stuff tool output into the window, so raise it.
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "16384"))


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def as_message(self) -> dict:
        """The canonical assistant message to append to the conversation."""
        msg = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                for c in self.tool_calls]
        return msg

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


def _json_fallback_messages(messages, schema):
    """Prompted JSON: used when the runner cannot constrain decoding to a schema."""
    out = [dict(m) for m in messages]
    note = ("\n\nReply with ONLY a JSON object matching this JSON schema, no prose, "
            "no code fences:\n" + json.dumps(schema))
    for m in reversed(out):
        if m["role"] == "user":
            m["content"] = (m.get("content") or "") + note
            break
    return out


def _parse_json_reply(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"_parse_error": text}


# ---------------------------------------------------------------- backends --
class OllamaBackend:
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=BASE_URL)
        self._think_unsupported = False
        self.schema_unsupported = False   # set when the runner answers 501 to format=

    @staticmethod
    def _to_native(messages):
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or "",
                            "tool_calls": [{"function": {
                                "name": c["function"]["name"],
                                "arguments": json.loads(c["function"]["arguments"])}}
                                for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"],
                            "tool_name": m.get("name", "")})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def chat(self, messages, tools, temperature, think, schema, max_tokens, stop=None, model=None):
        kwargs = dict(model=model or MODEL, messages=self._to_native(messages),
                      options={"temperature": temperature, "num_ctx": NUM_CTX})
        if max_tokens:
            kwargs["options"]["num_predict"] = max_tokens
        if stop:
            kwargs["options"]["stop"] = list(stop)
        if tools:
            kwargs["tools"] = tools
        if schema and not self.schema_unsupported:
            kwargs["format"] = schema
        elif schema:
            kwargs["messages"] = _json_fallback_messages(kwargs["messages"], schema)
        if think is not None and not self._think_unsupported:
            kwargs["think"] = think
        try:
            r = self.client.chat(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "structured output" in msg and "format" in kwargs:
                # MLX runner on macOS (or a build without xgrammar): fall back to
                # asking for JSON in the prompt. Weaker guarantee, same interface.
                self.schema_unsupported = True
                kwargs.pop("format")
                kwargs["messages"] = _json_fallback_messages(kwargs["messages"], schema)
                r = self.client.chat(**kwargs)
            elif "think" in msg and "think" in kwargs:   # model without a thinking mode
                self._think_unsupported = True
                kwargs.pop("think")
                r = self.client.chat(**kwargs)
            else:
                raise
        m = r.message
        calls = [ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=c.function.name,
                          args=dict(c.function.arguments or {}))
                 for c in (m.tool_calls or [])]
        return Reply(content=m.content or None, thinking=m.thinking or None,
                     tool_calls=calls,
                     prompt_tokens=r.prompt_eval_count or 0,
                     completion_tokens=r.eval_count or 0)


class OpenAIBackend:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def chat(self, messages, tools, temperature, think, schema, max_tokens, stop=None, model=None):
        clean = [{k: v for k, v in m.items() if k != "thinking"} for m in messages]
        kwargs = dict(model=model or MODEL, messages=clean, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
        if schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "out", "schema": schema}}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = list(stop)
        if think:  # providers differ; this is the most common spelling
            kwargs["reasoning_effort"] = think if isinstance(think, str) else "medium"
        r = self.client.chat.completions.create(**kwargs)
        m = r.choices[0].message
        calls = []
        for c in (m.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_malformed": c.function.arguments}
            calls.append(ToolCall(id=c.id, name=c.function.name, args=args))
        u = r.usage
        return Reply(content=m.content, thinking=getattr(m, "reasoning", None),
                     tool_calls=calls,
                     prompt_tokens=(u.prompt_tokens if u else 0) or 0,
                     completion_tokens=(u.completion_tokens if u else 0) or 0)


class MockBackend:
    """Scripted model. `set_mock_script([...])` queues replies; each item is
    {"content": "..."} or {"tool_calls": [{"name": ..., "args": {...}}]}.
    Alternatively `set_mock_handler(fn)` where fn(messages, tools) -> item,
    for behaviour that depends on the conversation."""
    script: list = []
    handler = None

    def chat(self, messages, tools, temperature, think, schema, max_tokens, stop=None, model=None):
        if MockBackend.handler is not None:
            item = MockBackend.handler(messages, tools)
        elif MockBackend.script:
            item = MockBackend.script.pop(0)
        else:
            item = {"content": "(mock: no scripted reply)"}
        calls = [ToolCall(id=f"call_{i}", name=c["name"], args=c.get("args", {}))
                 for i, c in enumerate(item.get("tool_calls", []))]
        ptoks = sum(len((m.get("content") or "").split()) for m in messages)
        ctoks = len((item.get("content") or "").split()) + 10 * len(calls)
        return Reply(content=item.get("content"), thinking=item.get("thinking"),
                     tool_calls=calls, prompt_tokens=ptoks, completion_tokens=ctoks)


def set_mock_script(items):
    MockBackend.handler = None
    MockBackend.script = list(items)


def set_mock_handler(fn):
    MockBackend.handler = fn


_backend = None


def get_backend():
    global _backend
    if _backend is None:
        _backend = {"ollama": OllamaBackend, "openai": OpenAIBackend,
                    "mock": MockBackend}[BACKEND]()
    return _backend


# ------------------------------------------------------------------ public --
def chat(messages, *, tools=None, temperature=0.0, think=False, schema=None,
         max_tokens=None, stop=None, model="agent") -> Reply:
    """One model call. `think` is False/True/'low'|'medium'|'high' where the
    model supports it; `schema` is a JSON schema that constrains the output;
    `stop` is a list of strings at which generation halts; `model` is a role name
    ('agent' | 'bare' | 'judge') or a literal tag."""
    from common import trace  # local import: trace imports nothing from here
    t0 = time.perf_counter()
    tag = model_for(model)
    reply = get_backend().chat(messages, tools, temperature, think, schema, max_tokens, stop, tag)
    reply.latency_s = time.perf_counter() - t0
    trace.log("llm_call", model=tag, role=(model if model in ROLES else "tag"), backend=BACKEND,
              think=think, temperature=temperature, n_messages=len(messages),
              last_message=messages[-1], tools=[t["function"]["name"] for t in (tools or [])],
              tool_schemas=list(tools or []),          # the JSON the model saw, verbatim
              content=reply.content, thinking=reply.thinking,
              tool_calls=[{"name": c.name, "args": c.args} for c in reply.tool_calls],
              prompt_tokens=reply.prompt_tokens,
              completion_tokens=reply.completion_tokens,
              latency_s=round(reply.latency_s, 3))
    return reply


def ask(prompt, system=None, **kwargs) -> str:
    """Convenience: one prompt in, reply text out."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs).content or ""


def ask_json(prompt, schema, system=None, **kwargs) -> dict:
    """Structured output: the reply is parsed JSON matching `schema`."""
    text = ask(prompt, system=system, schema=schema, **kwargs)
    return _parse_json_reply(text)


def structured_output_available() -> bool | None:
    """True/False once a schema call has been made on the Ollama backend; None before."""
    b = get_backend()
    if isinstance(b, OllamaBackend):
        return not b.schema_unsupported
    return None
