#!/usr/bin/env python3
"""Test script for the VectorX MCP Server."""

import asyncio
import sys
sys.path.insert(0, '.')

from vectorx_mcp_server import (
    get_database_stats,
    list_all_urls,
    search_documents,
    get_documents_by_url,
    ask_question_about_content,
    search_by_title_or_summary,
)

async def test_mcp_server():
    print('=' * 60)
    print('Testing VectorX MCP Server Functions')
    print('=' * 60)
    print('\n1. Testing get_database_stats()...')
    print(get_database_stats())
    print('\n2. Testing list_all_urls()...')
    print(list_all_urls())
    print('\n3. Testing search_documents()...')
    print(await search_documents('API authentication and security', max_results=3))
    print('\n4. Testing search_by_title_or_summary()...')
    print(search_by_title_or_summary('API', limit=3))

if __name__ == '__main__':
    asyncio.run(test_mcp_server())
