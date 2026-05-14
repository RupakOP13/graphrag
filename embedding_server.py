import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-3-small"

@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    # Handle single string or list of strings
    texts = [req.input] if isinstance(req.input, str) else req.input
    
    # Generate embeddings
    embeddings_list = model.encode(texts).tolist()
    
    # Format to OpenAI format
    data = []
    for i, emb in enumerate(embeddings_list):
        data.append({
            "object": "embedding",
            "embedding": emb,
            "index": i
        })
        
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
