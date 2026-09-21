import os
from typing import List, Dict, Any, Optional
from vecx.vectorx import VectorX
from dotenv import load_dotenv

load_dotenv()
_vectorx_client = None
_vectorx_index = None
_encryption_key = None
_index_name = 'pydantic_ai_docs'

def get_vectorx_client():
    global _vectorx_client
    if _vectorx_client is None:
        api_token = os.getenv('VECTORX_API_TOKEN')
        if not api_token:
            raise ValueError('VECTORX_API_TOKEN environment variable is required')
        _vectorx_client = VectorX(token=api_token)
    return _vectorx_client

def get_encryption_key():
    global _encryption_key
    if _encryption_key is None:
        stored_key = os.getenv('VECTORX_ENCRYPTION_KEY')
        if stored_key:
            _encryption_key = stored_key
        else:
            client = get_vectorx_client()
            _encryption_key = client.generate_key()
            print('Generated encryption key: ' + _encryption_key)
    return _encryption_key

def init_vectorx_index():
    global _vectorx_index
    if _vectorx_index is None:
        client = get_vectorx_client(); key = get_encryption_key()
        try:
            _vectorx_index = client.get_index(name=_index_name, key=key)
        except Exception:
            client.create_index(name=_index_name, dimension=1536, key=key, space_type='cosine')
            _vectorx_index = client.get_index(name=_index_name, key=key)
    return _vectorx_index

class VectorXCollection:
    def __init__(self): self.index = init_vectorx_index()
    def add(self, documents, embeddings, metadatas, ids):
        vectors_data = [{'id': doc_id, 'vector': embedding, 'meta': {'content': doc, **metadata}} for doc, embedding, metadata, doc_id in zip(documents, embeddings, metadatas, ids)]
        self.index.upsert(vectors_data)
    def query(self, query_embeddings, n_results=5, include=None, where=None):
        filter_dict = self._convert_where_to_filter(where) if where else None
        results = self.index.query(vector=query_embeddings[0] if query_embeddings else [], top_k=n_results, filter=filter_dict, include_vectors=False)
        documents = [[]]; metadatas = [[]]; ids = [[]]
        for result in results:
            content = result.get('meta', {}).get('content', '')
            documents[0].append(content); meta = result.get('meta', {}).copy(); meta.pop('content', None); metadatas[0].append(meta); ids[0].append(result.get('id', ''))
        return {'documents': documents, 'metadatas': metadatas, 'ids': ids}
    def get(self, ids=None, where=None, include=None, limit=None):
        if ids:
            documents=[]; metadatas=[]; result_ids=[]
            for doc_id in ids:
                try:
                    vector_data = self.index.get_vector(doc_id)
                    if vector_data:
                        documents.append(vector_data.get('meta', {}).get('content', '')); meta=vector_data.get('meta', {}).copy(); meta.pop('content', None); metadatas.append(meta); result_ids.append(doc_id)
                except Exception: pass
            return {'documents': documents, 'metadatas': metadatas, 'ids': result_ids}
        info = self.index.describe(); total_vectors = info.get('count', 0)
        if not total_vectors: return {'documents': [], 'metadatas': [], 'ids': []}
        import random
        random_vector = [random.uniform(-0.1, 0.1) for _ in range(info.get('dimension', 1536))]
        results = self.index.query(vector=random_vector, top_k=min(total_vectors, limit or 200, 200), include_vectors=False)
        documents=[]; metadatas=[]; result_ids=[]
        for result in results:
            documents.append(result.get('meta', {}).get('content', '')); meta=result.get('meta', {}).copy(); meta.pop('content', None); metadatas.append(meta); result_ids.append(result.get('id', ''))
        return {'documents': documents, 'metadatas': metadatas, 'ids': result_ids}
    def delete(self, ids=None, where=None):
        if ids:
            for doc_id in ids: self.index.delete_vector(doc_id)
        elif where: self.index.delete_with_filter(self._convert_where_to_filter(where))
    def _convert_where_to_filter(self, where):
        return {key: value if isinstance(value, dict) else {'eq': value} for key, value in where.items()}

def init_collection(): return VectorXCollection()
