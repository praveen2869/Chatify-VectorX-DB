import os
import json
import asyncio
import sys
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

if sys.platform == 'win32':
    try: asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError: pass
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from db import init_collection
load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
vectorx_collection = None

def get_vectorx_collection():
    global vectorx_collection
    if vectorx_collection is None: vectorx_collection = init_collection()
    return vectorx_collection

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    chunks=[]; start=0
    while start < len(text):
        end=min(start+chunk_size, len(text))
        if end < len(text):
            chunk=text[start:end]; code_block=chunk.rfind('```'); last_break=chunk.rfind('\n\n'); last_period=chunk.rfind('. ')
            if code_block > chunk_size*0.3: end=start+code_block
            elif last_break > chunk_size*0.3: end=start+last_break
            elif last_period > chunk_size*0.3: end=start+last_period+1
        chunk=text[start:end].strip()
        if chunk: chunks.append(chunk)
        start=max(start+1, end)
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    try:
        response=await openai_client.chat.completions.create(model=os.getenv('LLM_MODEL','gpt-4o-mini'), messages=[{'role':'system','content':"Return a JSON object with concise 'title' and 'summary' keys."},{'role':'user','content':f'URL: {url}\n\nContent:\n{chunk[:1000]}...'}], response_format={'type':'json_object'})
        return json.loads(response.choices[0].message.content)
    except Exception: return {'title':'Error processing title','summary':'Error processing summary'}

async def get_embedding(text: str) -> List[float]:
    try: return (await openai_client.embeddings.create(model='text-embedding-3-small', input=text)).data[0].embedding
    except Exception: return [0] * 1536

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    extracted=await get_title_and_summary(chunk,url); embedding=await get_embedding(chunk)
    return ProcessedChunk(url, chunk_number, extracted['title'], extracted['summary'], chunk, {'source':urlparse(url).netloc,'chunk_size':len(chunk),'crawled_at':datetime.now(timezone.utc).isoformat(),'url_path':urlparse(url).path}, embedding)

async def insert_chunk(chunk: ProcessedChunk):
    get_vectorx_collection().add([chunk.content],[chunk.embedding],[{'url':chunk.url,'chunk_number':chunk.chunk_number,'title':chunk.title,'summary':chunk.summary,**chunk.metadata}],[f'{chunk.url}_{chunk.chunk_number}'])

async def process_and_store_document(url: str, markdown: str):
    chunks=await asyncio.gather(*[process_chunk(chunk,i,url) for i,chunk in enumerate(chunk_text(markdown))]); await asyncio.gather(*[insert_chunk(chunk) for chunk in chunks])

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    browser_config=BrowserConfig(headless=True, verbose=False, browser_type='chromium', extra_args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])
    crawl_config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, process_iframes=False, remove_overlay_elements=True, page_timeout=30000)
    semaphore=asyncio.Semaphore(min(max_concurrent,3))
    async def process_url(url):
        async with semaphore:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result=await crawler.arun(url=url, config=crawl_config)
                if result.success and result.markdown and result.markdown.raw_markdown: await process_and_store_document(url,result.markdown.raw_markdown)
    await asyncio.gather(*[process_url(url) for url in urls], return_exceptions=True)

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    try:
        response=requests.get(sitemap_url); response.raise_for_status(); root=ElementTree.fromstring(response.content); namespace={'ns':'http://www.sitemaps.org/schemas/sitemap/0.9'}; return [loc.text for loc in root.findall('.//ns:loc',namespace)]
    except Exception: return []

if __name__ == '__main__': asyncio.run(crawl_parallel(['https://docs.crawl4ai.com/']))
