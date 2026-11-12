"""Lecture demo 10.7: the Week 10 agent loop consuming tools from ANY MCP server.

The loop is unchanged. What changed: the tool schemas are DISCOVERED from a
server at runtime (session.list_tools()) instead of being written in this file,
and tool calls are routed over the protocol (session.call_tool()).

    python week10/08_mcp_client_agent.py --list
    python week10/08_mcp_client_agent.py "Convert 26.2 miles to km, then tell me how many seconds a 3:30:00 marathon takes."
    python week10/08_mcp_client_agent.py --server /path/to/other_server.py "..."

Point --server at any stdio MCP server (an arXiv server, a database server,
your lab's instrument server from Week 12 project E) and the same agent works.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from common import trace  # noqa: E402
from common.agent import run_agent  # noqa: E402
from common.tools import ToolRegistry  # noqa: E402

DEFAULT_SERVER = pathlib.Path(__file__).parent / "07_mcp_server.py"
SYSTEM = ("You are a careful scientific assistant with tools. Use them for every "
          "computation and lookup; never guess numbers, titles or authors. Read each "
          "tool's parameter descriptions and fill the right field (for example an "
          "author's name goes in `author`, not in `query`). Reply with a short "
          "explanation and a final line 'ANSWER: <the answer>' (a number, or a title "
          "and arXiv id for a paper question).")


def mcp_tool_to_schema(t):
    """An MCP tool listing is already the JSON schema from 10.3, by design."""
    return {"type": "function", "function": {
        "name": t.name, "description": t.description or "",
        "parameters": t.input_schema or {"type": "object", "properties": {}}}}


async def main(server_path, question, list_only=False):
    params = StdioServerParameters(command=sys.executable, args=[str(server_path)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            print(f"[mcp] server exposes {len(listing.tools)} tools:")
            for t in listing.tools:
                print(f"   - {t.name}: {(t.description or '')[:70]}")
            if list_only:
                import json
                print(json.dumps(mcp_tool_to_schema(listing.tools[0]), indent=2))
                return

            # Bridge: every MCP tool becomes an entry in our ToolRegistry whose
            # implementation forwards the call over the protocol.
            loop = asyncio.get_running_loop()
            tools = ToolRegistry()
            for t in listing.tools:
                def make(name):
                    def fn(**kwargs):
                        fut = asyncio.run_coroutine_threadsafe(session.call_tool(name, kwargs), loop)
                        res = fut.result(timeout=60)
                        return "\n".join(getattr(c, "text", str(c)) for c in res.content)
                    return fn
                tools.register(make(t.name), schema=mcp_tool_to_schema(t))

            # run_agent is synchronous; run it in a worker thread so the MCP
            # session's event loop stays free to serve tool calls.
            with trace.run("mcp_client_agent"):
                r = await asyncio.to_thread(run_agent, question, tools, SYSTEM, verbose=True)
            print("\nFINAL:", r.text)
            print(f"({r.calls} calls, {r.tokens} tokens)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    server = DEFAULT_SERVER
    if "--server" in argv:
        i = argv.index("--server")
        server = pathlib.Path(argv[i + 1])
        del argv[i:i + 2]
    list_only = "--list" in argv
    q = " ".join(a for a in argv if not a.startswith("--")) or \
        "Convert 26.2 miles to kilometres and report the value."
    asyncio.run(main(server, q, list_only))
