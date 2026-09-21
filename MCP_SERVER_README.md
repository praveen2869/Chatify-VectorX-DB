# VectorX Database MCP Server

A FastMCP server that provides Model Context Protocol access to your VectorX vector database containing web content from the Crawl4AI RAG system.

## Installation

```bash
pip install -r mcp_requirements.txt
```

Set `OPENAI_API_KEY`, `VECTORX_API_TOKEN`, and `VECTORX_ENCRYPTION_KEY` in a `.env` file, then run:

```bash
python vectorx_mcp_server.py
```

Available tools include semantic document search, database statistics, URL lookup, source lookup, AI-powered questions, and title/summary search.
