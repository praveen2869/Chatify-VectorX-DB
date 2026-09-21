# Crawl4AI RAG System

A Retrieval-Augmented Generation system that crawls websites, extracts content, stores embeddings in VectorX, and provides a Streamlit chatbot.

## Prerequisites

- Python 3.8+
- OpenAI API key
- VectorX API token and encryption key

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
VECTORX_API_TOKEN=your_vectorx_api_token_here
VECTORX_ENCRYPTION_KEY=your_vectorx_encryption_key_here
LLM_MODEL=gpt-4o-mini
```

## Usage

```bash
cd gen-rag-crawl
streamlit run ui.py
```

The system crawls a sitemap or single URL, chunks the content, creates embeddings, stores it in VectorX, and provides a chat interface for questions about the collected documentation.
