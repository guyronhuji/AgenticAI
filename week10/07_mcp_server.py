"""Lecture demo 10.7: the course tools as a Model Context Protocol server.

Without MCP every tool is wired to every runtime by hand (M tools x N apps).
With MCP a server exposes tools over JSON-RPC and ANY client (our loop, an
IDE, a chat app) can discover and call them. The tool schema the client sees
is the same JSON you wrote by hand in 10.3.

Run standalone to see the tools it exports (uses stdio, so run via the client):
    python week10/08_mcp_client_agent.py --list

Or plug it into any MCP client, e.g. Claude Desktop / Claude Code:
    {"command": "python", "args": ["/abs/path/week10/07_mcp_server.py"]}

Requires: pip install "mcp>=2,<3"
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from common import tools as course_tools  # noqa: E402
from common.sandbox import run_python  # noqa: E402

server = MCPServer("course-tools", instructions="Tools from the AI Methods for Science course.")


@server.tool()
def run_python_tool(code: str) -> str:
    """Execute a short Python script in a sandbox and return what it prints."""
    return run_python(code)


@server.tool()
def calc(expression: str) -> str:
    """Evaluate a plain arithmetic expression such as '3*(2**10)'."""
    return course_tools.calc(expression)


@server.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a physical quantity between units, e.g. 5 'mile' to 'km'."""
    return course_tools.unit_convert(value, from_unit, to_unit)


@server.tool()
def arxiv_search(query: str = "", author: str = "", newest_first: bool = False, max_results: int = 5) -> str:
    """Search arXiv. Put the author's name in `author`, topic words in `query`; newest_first=true for 'latest'. Returns title, authors, date, id, snippet per hit."""
    return course_tools.arxiv_search(query, author, newest_first, max_results)


if __name__ == "__main__":
    server.run("stdio")
