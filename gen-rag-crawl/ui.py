from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
from datetime import datetime
import pytz

import streamlit as st
import json
import logfire
from openai import AsyncOpenAI

from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart,
    TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart, ModelMessagesTypeAdapter,
)
from pydantic_ai_agent import pydantic_ai_agent, PydanticAIDeps
from db import init_collection
from crawler import crawl_parallel, get_urls_from_sitemap
from dotenv import load_dotenv

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vectorx_collection = None

def get_vectorx_collection():
    global vectorx_collection
    if vectorx_collection is None:
        vectorx_collection = init_collection()
    return vectorx_collection

logfire.configure(send_to_logfire=False)

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal["user", "model"]
    timestamp: str
    content: str


def format_sitemap_url(url: str) -> str:
    """Format URL to ensure proper sitemap URL structure."""
    url = url.rstrip("/")
    if not url.endswith("sitemap.xml"):
        url = f"{url}/sitemap.xml"
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def get_db_stats():
    """Get statistics and information about the current database."""
    try:
        results = get_vectorx_collection().get(include=["metadatas"])
        if not results["metadatas"]:
            return None
        urls = set(); domains = set(); last_crawled_times = []
        for meta in results["metadatas"]:
            if meta and isinstance(meta, dict):
                url = meta.get("url", "Unknown URL")
                if url != "Unknown URL": urls.add(url)
                source = meta.get("source", "Unknown source")
                if source != "Unknown source": domains.add(source)
                crawled_at = meta.get("crawled_at", "")
                if crawled_at: last_crawled_times.append(crawled_at)
        last_updated = ""
        if last_crawled_times:
            last_updated = max(last_crawled_times)
            try:
                dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                dt = dt.astimezone(datetime.now().astimezone().tzinfo)
                last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                pass
        return {"urls": list(urls), "domains": list(domains), "doc_count": len(results["ids"]), "last_updated": last_updated}
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return None


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {"messages": [], "processing_complete": False, "urls_processed": set(), "is_processing": False, "current_progress": 0, "total_urls": 0, "suggested_questions": None}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value


def initialize_with_existing_data():
    """Check for existing data and initialize session state accordingly."""
    stats = get_db_stats()
    if stats and stats["doc_count"] > 0:
        st.session_state.processing_complete = True
        st.session_state.urls_processed = set(stats["urls"])
        return stats
    return None


def display_message_part(part):
    """Display a single part of a message in the Streamlit UI."""
    if part.part_kind == "system-prompt":
        with st.chat_message("system"): st.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"): st.markdown(part.content)
    elif part.part_kind == "text":
        with st.chat_message("assistant"): st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    """Run the agent with streaming text for the user_input prompt."""
    deps = PydanticAIDeps(collection=get_vectorx_collection(), openai_client=openai_client)
    async with pydantic_ai_agent.run_stream(user_input, deps=deps, message_history=st.session_state.messages[:-1]) as result:
        partial_text = ""; message_placeholder = st.empty()
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk; message_placeholder.markdown(partial_text)
        filtered_messages = [msg for msg in result.new_messages() if not (hasattr(msg, "parts") and any(part.part_kind == "user-prompt" for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)
        st.session_state.messages.append(ModelResponse(parts=[TextPart(content=partial_text)]))


async def process_url(url: str):
    """Process a single URL or sitemap URL."""
    try:
        progress_container = st.empty()
        with progress_container.container():
            formatted_url = format_sitemap_url(url); st.write(f"🔄 Processing {formatted_url}..."); st.write("📑 Attempting to fetch sitemap...")
            urls = get_urls_from_sitemap(formatted_url)
            if urls:
                st.write(f"📎 Found {len(urls)} URLs in sitemap"); progress_bar = st.progress(0, text="Processing URLs..."); st.session_state.total_urls = len(urls)
                status_placeholder = st.empty(); status_placeholder.text("⏳ Crawling web pages..."); await crawl_parallel(urls)
                status_placeholder.text("⚙️ Chunking documents..."); await asyncio.sleep(0.1); status_placeholder.text("🧮 Computing embeddings..."); await asyncio.sleep(0.1); status_placeholder.text("💾 Storing in database..."); await asyncio.sleep(0.1); progress_bar.progress(100, text="Processing complete!"); status_placeholder.empty()
            else:
                st.write("❌ No sitemap found or empty sitemap."); st.write("🔍 Attempting to process as single URL..."); original_url = url.rstrip("/sitemap.xml"); st.session_state.total_urls = 1; status_placeholder = st.empty(); status_placeholder.text("⏳ Crawling webpage..."); await crawl_parallel([original_url]); status_placeholder.empty()
            try:
                doc_count = len(get_vectorx_collection().get()["ids"]); st.success(f"""✅ Processing complete!\n\nDocuments in database: {doc_count}\nLast processed URL: {url}\n\nYou can now start asking questions about the content.""")
            except Exception as e: st.error(f"Unable to get document count: {str(e)}")
    except Exception as e: st.error(f"Error processing URL: {str(e)}")


def generate_contextual_questions(collection) -> list[str]:
    """Generate contextual questions based on the content in VectorX."""
    try:
        results = collection.get(include=["documents", "metadatas"], limit=10)
        if not results["documents"]: return []
        content_summary = "\n".join(results["documents"][:3]); domains = {meta.get("source", "") for meta in results["metadatas"] if meta and meta.get("source")}
        messages = [{"role": "system", "content": "You are a helpful AI that generates relevant questions based on a corpus of documents. Generate 4-5 specific questions that can be answered from the provided content. Questions should be diverse and specific to the actual content."}, {"role": "user", "content": f"Based on content from these domains: {', '.join(domains)} and this sample content: {content_summary[:1000]}...\nGenerate 4-5 specific, contextual questions that could be answered from this knowledge base. Format as a simple list with each question on a new line starting with a hyphen. Make questions specific to the actual content, not generic."}]
        from openai import OpenAI
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=200)
        return [q.strip("- ").strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
    except Exception as e:
        print(f"Error generating contextual questions: {e}")
        return ["What are the main topics covered in these documents?", "Can you summarize the key points from the loaded content?", "What specific information can I find in these documents?", "What are the most important concepts discussed in this content?"]


async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System", page_icon="🤖", layout="wide")
    initialize_session_state(); existing_data = initialize_with_existing_data(); st.title("Dynamic RAG Chat System")
    if existing_data:
        st.success("💡 System is ready with existing knowledge base!")
        with st.expander("Knowledge Base Information", expanded=True):
            st.markdown(f"""### Current Knowledge Base Stats:\n- 📚 Number of documents: {existing_data['doc_count']}\n- 🌐 Number of sources: {len(existing_data['domains'])}\n- 🕒 Last updated: {existing_data['last_updated']}\n\n### Sources include:\n{', '.join(existing_data['domains'])}\n\n### You can ask questions about:\n- Any content from the processed websites\n- Specific information from any of the loaded pages\n- Technical details, documentation, or other content from these sources\n\n### Loaded URLs:\n""")
            for url in existing_data["urls"]: st.write(f"- {url}")
    else: st.info("👋 Welcome! Start by adding a website to create your knowledge base.")
    input_col, chat_col = st.columns([1, 2])
    with input_col:
        st.subheader("Add Content to RAG System"); st.write("Enter a website URL to process. The system will:"); st.write("1. First try to find and process the sitemap (automatically appending '/sitemap.xml')"); st.write("2. If no sitemap is found, process the URL as a single page")
        url_input = st.text_input("Website URL", key="url_input", placeholder="example.com or https://example.com")
        if url_input: st.caption(f"Will try: {format_sitemap_url(url_input)}")
        col1, col2 = st.columns(2)
        with col1: process_button = st.button("Process URL", disabled=st.session_state.is_processing, type="primary")
        with col2:
            if st.button("Clear Database", disabled=st.session_state.is_processing, type="secondary"):
                try:
                    all_ids = get_vectorx_collection().get()["ids"]
                    if all_ids: get_vectorx_collection().delete(ids=all_ids)
                    st.session_state.processing_complete = False; st.session_state.urls_processed = set(); st.session_state.messages = []; st.session_state.suggested_questions = None; st.success("Database cleared successfully!"); st.rerun()
                except Exception as e: st.error(f"Error clearing database: {str(e)}")
        if process_button and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing = True; await process_url(url_input); st.session_state.urls_processed.add(url_input); st.session_state.processing_complete = True; st.session_state.is_processing = False; st.session_state.suggested_questions = None; st.rerun()
            else: st.warning("This URL has already been processed!")
        if st.session_state.urls_processed:
            st.subheader("Processed URLs:"); urls_list = list(st.session_state.urls_processed)
            for url in urls_list[:3]: st.write(f"✓ {url}")
            remaining = len(urls_list) - 3
            if remaining > 0:
                st.write(f"_...and {remaining} more_")
                with st.expander("Show all URLs"):
                    for url in urls_list[3:]: st.write(f"✓ {url}")
    with chat_col:
        if st.session_state.processing_complete:
            chat_container = st.container()
            with chat_container:
                st.subheader("Chat Interface")
                with st.expander("📝 Suggested Questions", expanded=False):
                    if existing_data and existing_data["doc_count"] > 0:
                        if st.session_state.suggested_questions is None: st.session_state.suggested_questions = generate_contextual_questions(get_vectorx_collection())
                        st.markdown("Try asking:")
                        for question in st.session_state.suggested_questions: st.markdown(f"- {question}")
                        if st.button("🔄 Refresh Suggestions"): st.session_state.suggested_questions = generate_contextual_questions(get_vectorx_collection()); st.rerun()
                    else: st.markdown("Process some URLs to get contextual question suggestions.")
                messages_container = st.container()
                st.markdown("""<style>.stChatMessageContent { max-height: 400px; overflow-y: auto; }</style>""", unsafe_allow_html=True)
                with messages_container:
                    for msg in st.session_state.messages:
                        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                            for part in msg.parts: display_message_part(part)
                st.markdown("<div style='padding: 3rem;'></div>", unsafe_allow_html=True)
        else:
            if existing_data: st.info("The knowledge base is ready! Start asking questions below.")
            else: st.info("Please process a URL first to start chatting!")
    if st.session_state.processing_complete:
        user_input = st.chat_input("Ask a question about the processed content...", disabled=st.session_state.is_processing)
        if user_input:
            st.session_state.messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
            with st.chat_message("user"): st.markdown(user_input)
            with st.chat_message("assistant"): await run_agent_with_streaming(user_input)
            st.markdown("""<script>function scrollToBottom(){const messages=document.querySelector('.stChatMessageContent');if(messages){messages.scrollTop=messages.scrollHeight;}}setTimeout(scrollToBottom,100);</script>""", unsafe_allow_html=True)
        if st.button("Clear Chat History"): st.session_state.messages = []; st.rerun()
    st.markdown("---")
    if existing_data: st.markdown(f"System Status: 🟢 Ready with {existing_data['doc_count']} documents from {len(existing_data['domains'])} sources")
    else: st.markdown("System Status: 🟡 Waiting for content")


if __name__ == "__main__":
    asyncio.run(main())
