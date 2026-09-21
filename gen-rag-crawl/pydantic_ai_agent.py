from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
from litellm import AsyncOpenAI
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from db import VectorXCollection

load_dotenv()
model = OpenAIModel(os.getenv('LLM_MODEL', 'gpt-4o-mini'))
@dataclass
class PydanticAIDeps:
    collection: VectorXCollection
    openai_client: AsyncOpenAI

pydantic_ai_agent = Agent(model, system_prompt='Use retrieve_relevant_documentation for every user question and answer only from retrieved content.', deps_type=PydanticAIDeps, retries=2)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    try:
        response = await openai_client.embeddings.create(model='text-embedding-3-small', input=text)
        return response.data[0].embedding
    except Exception:
        return [0] * 1536

@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    results = ctx.deps.collection.query(query_embeddings=[await get_embedding(user_query, ctx.deps.openai_client)], n_results=5, include=['documents', 'metadatas'])
    if not results['documents'][0]: return 'No relevant documentation found.'
    return '\n\n---\n\n'.join(f"# {meta.get('title', 'Unknown Title')}\n\n{doc}\n\nSource: {meta.get('url', 'Unknown Source')}" for doc, meta in zip(results['documents'][0], results['metadatas'][0]))

@pydantic_ai_agent.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    results = ctx.deps.collection.get(include=['metadatas'])
    return sorted(set(meta.get('url', 'Unknown URL') for meta in results['metadatas'] if meta and meta.get('url')))

@pydantic_ai_agent.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    results = ctx.deps.collection.get(where={'url': url}, include=['documents', 'metadatas'])
    if not results['documents']: return f'No content found for URL: {url}'
    sorted_results = sorted(zip(results['documents'], results['metadatas']), key=lambda x: x[1].get('chunk_number', 0))
    return '\n\n'.join([f"# {sorted_results[0][1].get('title', 'Documentation Page').split(' - ')[0]}\n"] + [doc for doc, _ in sorted_results])
