"""Serve the documentation search tool over MCP, so any MCP client can search the RapidFire AI docs.

    python mcp_server.py                                 # stdio transport
    claude mcp add rapidfire-docs -- python mcp_server.py   # e.g. register it with Claude Code
"""

from mcp.server.mcpserver import MCPServer

import common
from retriever import DocSearch, format_docs

mcp = MCPServer('rapidfire-docs')
search = DocSearch(common.CORPUS_DIR, common.make_embeddings(common.load_api_key()))


@mcp.tool()
def search_docs(query: str, k: int = 5) -> str:
    """Search the RapidFire AI documentation. Returns the most relevant passages, each labeled with its file and line range."""
    return format_docs(search.search(query, k=k))


if __name__ == '__main__':
    mcp.run()
