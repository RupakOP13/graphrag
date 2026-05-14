import os
import sys
import requests

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

GRAPHRAG_HOST = config.GRAPHRAG_HOST  # http://localhost:8000
GRAPH_NAME = config.TG_GRAPHNAME      # GraphRAGHackathon

# The dataset file path INSIDE the Docker container (mounted via docker-compose volume)
CONTAINER_DATA_PATH = "/data/wikipedia_ai_dataset.jsonl"

print("=" * 50)
print("TIGERGRAPH GRAPHRAG INGESTION SCRIPT")
print("=" * 50)
print(f"GraphRAG service: {GRAPHRAG_HOST}")
print(f"Graph: {GRAPH_NAME}")
print(f"Data path (in container): {CONTAINER_DATA_PATH}")
print()

# Step 1: Create the ingest job
print("Step 1: Creating ingest job...")
create_payload = {
    "data_source": "local",
    "data_source_config": {
        "data_path": CONTAINER_DATA_PATH
    },
    "file_format": "json",
    "loader_config": {},
    "chunker": "character-recursive",
    "chunker_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "embedder": "OpenAI",
    "embedder_config": {}
}

try:
    r = requests.post(
        f"{GRAPHRAG_HOST}/{GRAPH_NAME}/graphrag/create_ingest",
        json=create_payload,
        timeout=60
    )
    print(f"  Status: {r.status_code}")
    
    if r.status_code != 200:
        print(f"  Error: {r.text[:500]}")
        sys.exit(1)
    
    job_info = r.json()
    print(f"  Job created: {job_info}")
    
    load_job_id = job_info.get("load_job_id", "")
    data_source_id = job_info.get("data_source_id", "")
    data_path = job_info.get("data_path", CONTAINER_DATA_PATH)
    
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Step 2: Run the ingest job
print("\nStep 2: Running ingest job (this takes several minutes)...")
run_payload = {
    "load_job_id": load_job_id,
    "data_source_id": data_source_id,
    "data_path": data_path
}

try:
    r = requests.post(
        f"{GRAPHRAG_HOST}/{GRAPH_NAME}/graphrag/ingest",
        json=run_payload,
        timeout=600  # 10 min timeout for large dataset
    )
    print(f"  Status: {r.status_code}")
    
    if r.status_code != 200:
        print(f"  Error: {r.text[:500]}")
        sys.exit(1)
    
    result = r.json()
    print(f"  Ingest result: {result}")
    
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Step 3: Force consistency update
print("\nStep 3: Running consistency update (entity extraction)...")
try:
    r = requests.post(
        f"{GRAPHRAG_HOST}/{GRAPH_NAME}/graphrag/forceConsistencyUpdate",
        json={"method": "graphrag"},
        timeout=300
    )
    print(f"  Status: {r.status_code} - {r.text[:200]}")
except Exception as e:
    print(f"  Consistency update: {e}")

print("\n" + "=" * 50)
print("INGESTION COMPLETE!")
print("You can now run the full benchmark.")
print("=" * 50)
