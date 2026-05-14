"""
Pipeline 2: Basic RAG (Vector Embeddings + LLM)
=================================================
Industry standard RAG: chunk documents, embed into ChromaDB using
local sentence-transformers, retrieve similar chunks, send to Groq LLM.
"""

import time
import json
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.base import BasePipeline, PipelineResult, calculate_cost
from pipelines.llm_only import _call_groq_with_retry


def chunk_text(text, chunk_size=None, overlap=None):
    """Split text into overlapping chunks."""
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class BasicRAGPipeline(BasePipeline):
    """Pipeline 2: ChromaDB vector search + Groq LLM."""
    
    def __init__(self):
        super().__init__("Basic RAG")
        self.client = None
        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self._indexed = False
    
    def initialize(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        
        # Load local embedding model (free, no API key needed)
        print(f"   Loading embedding model: {config.EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        try:
            self.collection = self.chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
            if self.collection.count() > 0:
                self._indexed = True
                print(f"Basic RAG initialized (ChromaDB: {self.collection.count()} chunks)")
            else:
                print(f"Basic RAG initialized (ChromaDB: empty)")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Basic RAG initialized (new ChromaDB collection)")
    
    def index_documents(self, dataset_path=None):
        """Chunk and index all documents into ChromaDB."""
        self.ensure_initialized()
        if self._indexed:
            print(f"ChromaDB already has {self.collection.count()} chunks. Skip.")
            return
        if dataset_path is None:
            dataset_path = os.path.join(config.DATASET_DIR, "wikipedia_ai_dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Indexing {data['total_articles']} articles...")
        all_chunks, all_ids, all_metadata = [], [], []
        for i, article in enumerate(data["articles"]):
            chunks = chunk_text(article["text"])
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"doc_{i}_chunk_{j}")
                all_metadata.append({
                    "title": article["title"],
                    "source": article.get("url", ""),
                    "chunk_index": j
                })
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Batch embed with sentence-transformers (much faster than API)
        batch_size = 500
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            batch_texts = all_chunks[start:end]
            
            # Local embedding -- no rate limits!
            embeddings = self.embedder.encode(batch_texts, show_progress_bar=False).tolist()
            
            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                ids=all_ids[start:end],
                metadatas=all_metadata[start:end]
            )
            print(f"   Indexed {end}/{len(all_chunks)} chunks...")
        
        self._indexed = True
        print(f"Indexing complete! {len(all_chunks)} chunks")
    
    def _retrieve_context(self, question, top_k=None):
        if top_k is None:
            top_k = config.RAG_TOP_K
        
        # Embed query locally
        query_emb = self.embedder.encode([question]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        chunks = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        context_parts = []
        for chunk, meta in zip(chunks, metadatas):
            source = meta.get("title", "Unknown")
            context_parts.append(f"[Source: {source}]\n{chunk}")
        return "\n\n---\n\n".join(context_parts), chunks
    
    def query(self, question):
        self.ensure_initialized()
        if not self._indexed:
            return PipelineResult(
                pipeline_name=self.name, question=question,
                answer="Error: Not indexed. Run index_documents() first.",
                metadata={"error": "not_indexed"}
            )
        
        start_time = time.time()
        try:
            context, chunks = self._retrieve_context(question)
            
            response = _call_groq_with_retry(
                self.client,
                config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable AI assistant. Answer using ONLY the provided context. If the context doesn't contain enough information, say so but try your best."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:"}
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            
            latency = time.time() - start_time
            answer = response.choices[0].message.content
            usage = response.usage
            pt = usage.prompt_tokens if usage else len(context) // 4
            ct = usage.completion_tokens if usage else len(answer) // 4
            tt = usage.total_tokens if usage else pt + ct
            
            return PipelineResult(
                pipeline_name=self.name, question=question, answer=answer,
                prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                latency_seconds=round(latency, 3),
                cost_usd=calculate_cost(pt, ct),
                context_text=context[:1000],
                context_tokens=len(context) // 4,
                metadata={
                    "model": config.LLM_MODEL,
                    "num_chunks": len(chunks),
                    "top_k": config.RAG_TOP_K
                }
            )
        except Exception as e:
            return PipelineResult(
                pipeline_name=self.name, question=question,
                answer=f"Error: {e}",
                latency_seconds=round(time.time() - start_time, 3),
                metadata={"error": str(e)}
            )
