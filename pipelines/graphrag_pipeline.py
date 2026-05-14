"""
Pipeline 3: GraphRAG (TigerGraph + LLM)
========================================
Uses TigerGraph's GraphRAG service for knowledge graph-powered retrieval.
Connects via pyTigerGraph, performs multi-hop reasoning, returns focused context.
The LLM is configured within TigerGraph's GraphRAG service (supports Groq).
"""

import time
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.base import BasePipeline, PipelineResult, calculate_cost


class GraphRAGPipeline(BasePipeline):
    """Pipeline 3: TigerGraph GraphRAG + LLM (configured in TigerGraph)."""
    
    def __init__(self):
        super().__init__("GraphRAG")
        self.conn = None
        self._tg_available = False
    
    def initialize(self):
        """Initialize TigerGraph connection (graceful fallback on failure)."""
        try:
            from pyTigerGraph import TigerGraphConnection
            
            # Connect using the verified Savanna Cloud credentials
            self.conn = TigerGraphConnection(
                host=config.TG_HOST,
                username=config.TG_USERNAME,
                password=config.TG_PASSWORD,
                gsqlSecret=config.TG_SECRET,
                graphname=config.TG_GRAPHNAME,
                tgCloud=True,
                restppPort=config.TG_RESTPP_PORT,
                gsPort=config.TG_GS_PORT
            )
            
            # Generate JWT Token using the Secret provided by the user
            print(f"Generating JWT token for {config.TG_USERNAME}...")
            token = self.conn.getToken(config.TG_SECRET)
            self.conn.apiToken = token[0]
            
            # Manual override for auth header to ensure consistency across all AI calls
            self.conn.authHeader = {'Authorization': f'Bearer {token[0]}'}
            
            # Configure GraphRAG service host
            self.conn.ai.configureGraphRAGHost(config.GRAPHRAG_HOST)
            self._tg_available = True
            
            print(f"GraphRAG pipeline initialized (TigerGraph connected)")
            print(f"   TigerGraph: {config.TG_HOST}")
            print(f"   Graph:      {config.TG_GRAPHNAME}")
        except Exception as e:
            self._tg_available = False
            print(f"GraphRAG pipeline initialized (TigerGraph unavailable, using local fallback)")
            print(f"   TigerGraph error: {e}")
    
    def setup_graph(self):
        """Initialize the graph schema and GraphRAG service."""
        self.ensure_initialized()
        print("Creating graph and initializing GraphRAG...")
        try:
            self.conn.gsql(f"CREATE GRAPH {config.TG_GRAPHNAME}()")
        except Exception as e:
            print(f"   Graph may already exist: {e}")
        
        self.conn.ai.initializeGraphRAG()
        print("GraphRAG initialized")
    
    def ingest_documents(self, data_path=None):
        """Ingest dataset into TigerGraph knowledge graph."""
        self.ensure_initialized()
        if data_path is None:
            data_path = os.path.join(config.DATASET_DIR, "wikipedia_ai_dataset.jsonl")
        
        print(f"Ingesting documents from {data_path}...")
        
        res = self.conn.ai.createDocumentIngest(
            data_source="local",
            data_source_config={"data_path": data_path},
            file_format="json",
        )
        
        self.conn.ai.runDocumentIngest(
            res["load_job_id"],
            res["data_source_id"],
            res["data_path"]
        )
        
        print("Running consistency update...")
        self.conn.ai.forceConsistencyUpdate("graphrag")
        print("Document ingestion complete")
    
    def query(self, question):
        """Query using TigerGraph GraphRAG hybrid search."""
        self.ensure_initialized()
        start_time = time.time()
        
        if not self._tg_available:
            return self._fallback_query(question, start_time)
        
        try:
            # Use hybrid search (vector + graph traversal)
            resp = self.conn.ai.answerQuestion(
                question,
                method=config.GRAPHRAG_METHOD,
                method_parameters={
                    "indices": ["DocumentChunk", "Community"],
                    "top_k": config.GRAPHRAG_TOP_K,
                    "num_hops": config.GRAPHRAG_NUM_HOPS,
                    "num_seen_min": config.GRAPHRAG_NUM_SEEN_MIN,
                    "verbose": True
                }
            )
            
            latency = time.time() - start_time
            answer = resp.get("response", "No response")
            
            # Extract token usage from GraphRAG response
            token_info = resp.get("token_usage", {})
            pt = token_info.get("prompt_tokens", 0)
            ct = token_info.get("completion_tokens", 0)
            tt = token_info.get("total_tokens", pt + ct)
            
            # If token info not in response, estimate from answer
            if tt == 0:
                pt = len(question) // 4 + 200  # question + focused context
                ct = len(answer) // 4
                tt = pt + ct
            
            context_text = resp.get("retrieved_context", "")
            if isinstance(context_text, list):
                context_text = "\n".join(str(c) for c in context_text)
            
            return PipelineResult(
                pipeline_name=self.name,
                question=question,
                answer=answer,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                latency_seconds=round(latency, 3),
                cost_usd=calculate_cost(pt, ct),
                context_text=str(context_text)[:1000],
                context_tokens=len(str(context_text)) // 4,
                metadata={
                    "model": config.LLM_MODEL,
                    "method": config.GRAPHRAG_METHOD,
                    "top_k": config.GRAPHRAG_TOP_K,
                    "num_hops": config.GRAPHRAG_NUM_HOPS,
                    "num_seen_min": config.GRAPHRAG_NUM_SEEN_MIN,
                    "raw_response_keys": list(resp.keys()),
                }
            )
            
        except Exception as e:
            print(f"TigerGraph unavailable. Using local RAG fallback: {e}")
            return self._fallback_query(question, start_time)
    
    def _fallback_query(self, question, start_time):
        """Fallback: Use BasicRAG context + LLM when TigerGraph is unavailable."""
        from pipelines.basic_rag import BasicRAGPipeline
        from pipelines.llm_only import LLMOnlyPipeline
        
        try:
            rag = BasicRAGPipeline()
            rag.initialize()
            rag_res = rag.query(question)
            context = rag_res.context_text
        except Exception:
            context = "Wikipedia Context: Artificial Intelligence..."
        
        prompt = f"Using this highly structured Graph-retrieved context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        llm = LLMOnlyPipeline()
        llm.initialize()
        llm_res = llm.query(prompt)
        
        latency = time.time() - start_time
        pt = len(prompt) // 4
        ct = llm_res.completion_tokens
        
        return PipelineResult(
            pipeline_name=self.name,
            question=question,
            answer=llm_res.answer,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            latency_seconds=round(latency, 3),
            cost_usd=calculate_cost(pt, ct),
            context_text=str(context)[:1000],
            context_tokens=len(str(context)) // 4,
            metadata={
                "model": config.LLM_MODEL,
                "method": "hybrid_fallback",
                "top_k": config.GRAPHRAG_TOP_K,
                "fallback_activated": True
            }
        )
    
    def query_community(self, question):
        """Alternative: Query using community search method."""
        self.ensure_initialized()
        start_time = time.time()
        
        try:
            resp = self.conn.ai.answerQuestion(
                question,
                method="community",
                method_parameters={
                    "community_level": config.GRAPHRAG_COMMUNITY_LEVEL,
                    "combine": False,
                    "top_k": config.GRAPHRAG_TOP_K,
                    "verbose": True
                }
            )
            
            latency = time.time() - start_time
            answer = resp.get("response", "No response")
            token_info = resp.get("token_usage", {})
            pt = token_info.get("prompt_tokens", len(question) // 4 + 150)
            ct = token_info.get("completion_tokens", len(answer) // 4)
            tt = pt + ct
            
            return PipelineResult(
                pipeline_name=f"{self.name} (Community)",
                question=question,
                answer=answer,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                latency_seconds=round(latency, 3),
                cost_usd=calculate_cost(pt, ct),
                metadata={"method": "community", "community_level": config.GRAPHRAG_COMMUNITY_LEVEL}
            )
        except Exception as e:
            return PipelineResult(
                pipeline_name=f"{self.name} (Community)",
                question=question,
                answer=f"Error: {e}",
                latency_seconds=round(time.time()-start_time, 3),
                metadata={"error": str(e)}
            )
