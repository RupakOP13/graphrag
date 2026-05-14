import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.graphrag_pipeline import GraphRAGPipeline

print("==================================================")
print("🚀 TIGERGRAPH GRAPHRAG INGESTION SCRIPT")
print("==================================================")

try:
    p3 = GraphRAGPipeline()
    print("1. Initializing connection to Savanna Cloud...")
    p3.initialize()
    
    print("\n2. Setting up Graph Schema (GraphRAGHackathon)...")
    p3.setup_graph()
    
    print("\n3. Starting Data Ingestion (This may take several minutes)...")
    # This will read from dataset/data/wikipedia_ai_dataset.jsonl
    p3.ingest_documents()
    
    print("\n✅ INGESTION COMPLETE!")
    print("You are now ready to run the full benchmark.")

except Exception as e:
    print(f"\n❌ ERROR during ingestion: {e}")
    print("Make sure your Docker GraphRAG microservices are fully running.")
