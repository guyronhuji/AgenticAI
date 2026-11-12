"""Tool registry: Python functions published to the model as JSON schemas.

A "tool" is nothing but a function signature the model can see. The registry
builds the schema from the function's type hints and docstring, executes
calls the model emits, traces them, and turns exceptions into strings the
model can read (an error message IS information for the next step).

    from common.tools import ToolRegistry, calc, run_python_tool
    tools = ToolRegistry()
    tools.register(calc)
    tools.register(run_python_tool)
    tools.schemas()            # -> list of {"type": "function", ...}
    tools.call("calc", {"expression": "2**10"})

Also here: the course's standard tools (safe calculator, sandboxed Python,
unit conversion via pint, arXiv search).
"""
from __future__ import annotations

import ast
import inspect
import json
import operator
import os
import typing

from common import trace
from common.sandbox import run_python

_PY2JSON = {int: "integer", float: "number", str: "string", bool: "boolean",
            list: "array", dict: "object"}


def schema_from_function(fn, name=None, description=None) -> dict:
    """Build an OpenAI-style tool schema from a function's signature and docstring.
    Parameter descriptions come from lines like `name: description` in the docstring."""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    main, _, params_doc = doc.partition("\n\n")
    pdesc = {}
    for line in params_doc.splitlines():
        if ":" in line:
            k, _, v = line.strip().partition(":")
            pdesc[k.strip()] = v.strip()
    props, required = {}, []
    hints = typing.get_type_hints(fn)
    for pname, p in sig.parameters.items():
        t = hints.get(pname, str)
        props[pname] = {"type": _PY2JSON.get(t, "string")}
        if pname in pdesc:
            props[pname]["description"] = pdesc[pname]
        if p.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            props[pname]["default"] = p.default
    return {"type": "function", "function": {
        "name": name or fn.__name__,
        "description": description or main.strip(),
        "parameters": {"type": "object", "properties": props, "required": required}}}


class ToolRegistry:
    def __init__(self):
        self._fns, self._schemas = {}, {}

    def register(self, fn=None, *, name=None, description=None, schema=None):
        """Use as tools.register(fn) or as a decorator @tools.register."""
        if fn is None:
            return lambda f: self.register(f, name=name, description=description, schema=schema)
        s = schema or schema_from_function(fn, name, description)
        self._fns[s["function"]["name"]] = fn
        self._schemas[s["function"]["name"]] = s
        return fn

    def schemas(self) -> list[dict]:
        return list(self._schemas.values())

    def names(self):
        return list(self._fns)

    def call(self, name: str, args: dict) -> str:
        """Execute a tool call. Always returns a string; errors become text."""
        if name not in self._fns:
            result = f"ERROR: unknown tool '{name}'. Available: {self.names()}"
        else:
            try:
                out = self._fns[name](**(args or {}))
                result = out if isinstance(out, str) else json.dumps(out, default=str)
            except TypeError as e:
                result = f"ERROR: bad arguments for {name}: {e}"
            except Exception as e:  # noqa: BLE001 - the model should see it
                result = f"ERROR: {type(e).__name__}: {e}"
        trace.log("tool_call", name=name, args=args, result=result[:4000])
        return result


# --------------------------------------------------------- standard tools --
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def calc(expression: str) -> str:
    """Evaluate a plain arithmetic expression such as '3*(2**10)'. Supports + - * / // % ** and parentheses only; no names, no function calls.

    expression: the arithmetic expression to evaluate
    """
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10000:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only plain arithmetic is allowed")
    if len(expression) > 500:
        raise ValueError("expression too long")
    return str(ev(ast.parse(expression.strip(), mode="eval")))


def run_python_tool(code: str) -> str:
    """Execute a short Python script in a sandbox and return what it prints. Use print() for every value you need back.

    code: complete Python source code to run
    """
    return run_python(code)


FIGURE_DIR = os.environ.get("SANDBOX_FIGDIR", "sandbox_out")


def run_python_figures_tool(code: str) -> str:
    """Execute a Python script in a sandbox with numpy, scipy, pandas, matplotlib and seaborn available, and return what it prints. To make a figure, save it with plt.savefig("name.png") in the current directory: saved figures are reported back and opened on screen. Do not call plt.show(). Use print() for every value you need back.

    code: complete Python source code to run
    """
    return run_python(code, timeout=60, workdir=FIGURE_DIR, open_figures=True)


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a physical quantity between units using pint, e.g. 5 'mile' to 'km'. Fails if the dimensions differ.

    value: the numeric magnitude
    from_unit: unit of the input, e.g. 'mile', 'eV', 'm/s**2'
    to_unit: unit wanted, e.g. 'kilometer', 'joule', 'ft/s**2'
    """
    import pint
    ureg = pint.UnitRegistry()
    q = ureg.Quantity(float(value), from_unit).to(to_unit)
    return f"{q.magnitude:.10g} {to_unit}"


def arxiv_query(query: str = "", author: str = "", newest_first: bool = False, max_results: int = 5) -> dict:
    """Build the arXiv API request. Separate so students (and tests) can see the query
    the model's arguments turn into: arXiv's search syntax is fielded (au:, ti:, all:),
    terms must be ANDed explicitly, and results are relevance-ranked unless asked otherwise."""
    parts = []
    if author.strip():
        parts.append(f'au:"{author.strip()}"')
    for term in query.replace(",", " ").split():
        parts.append(f"all:{term}")
    if not parts:
        raise ValueError("give a query, an author, or both")
    params = {"search_query": " AND ".join(parts),
              "max_results": max(1, min(int(max_results), 10))}
    if newest_first:
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"
    return params


def arxiv_search(query: str = "", author: str = "", newest_first: bool = False, max_results: int = 5) -> str:
    """Search arXiv. Put the author's name in `author` (not in `query`); use newest_first=true for 'latest' or 'most recent'. Returns title, authors, date, arXiv id and an abstract snippet per hit.

    query: topic words, e.g. 'beta decay neutron'; leave empty when searching by author only
    author: author name as written on papers, e.g. 'Guy Ron'
    newest_first: true to sort by submission date, newest first (for 'latest')
    max_results: number of results, 1-10
    """
    import xml.etree.ElementTree as ET
    import requests
    params = arxiv_query(query, author, newest_first, max_results)
    r = requests.get("http://export.arxiv.org/api/query", params=params, timeout=20)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    lines = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        title = " ".join(e.findtext("a:title", "", ns).split())
        authors = [x.findtext("a:name", "", ns) for x in e.findall("a:author", ns)]
        shown = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        date = e.findtext("a:published", "", ns)[:10]
        aid = e.findtext("a:id", "", ns).rsplit("/", 1)[-1]
        summ = " ".join(e.findtext("a:summary", "", ns).split())[:160]
        lines.append(f"- {title} ({shown}; {date}) arXiv:{aid}\n  {summ}")
    return "\n".join(lines) or f"(no results for search_query={params['search_query']!r})"


def crossref_mailto():
    """CrossRef's polite pool wants a real contact address. Set CROSSREF_MAILTO."""
    return os.environ.get("CROSSREF_MAILTO", "ai-methods-course@example.edu")
