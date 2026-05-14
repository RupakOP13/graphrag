import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# TigerGraph GraphRAG schema expects 1536-dim vectors (OpenAI standard)
# We pad our 384-dim embeddings to 1536 with zeros
TARGET_DIM = 1536
MODEL_DIM = 384

class EmbeddingRequest(BaseModel):
    input: Any  # Accept string, list of strings, or list of token ints
    model: str = "text-embedding-3-small"

def decode_input(raw_input) -> List[str]:
    """Convert any input format to a list of strings."""
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list) and len(raw_input) > 0:
        # List of ints = token IDs
        if isinstance(raw_input[0], int):
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                return [enc.decode(raw_input)]
            except Exception:
                return [" ".join(str(t) for t in raw_input)]
        # List of lists of ints = batch of token sequences
        if isinstance(raw_input[0], list):
            texts = []
            for seq in raw_input:
                if seq and isinstance(seq[0], int):
                    try:
                        import tiktoken
                        enc = tiktoken.get_encoding("cl100k_base")
                        texts.append(enc.decode(seq))
                    except Exception:
                        texts.append(" ".join(str(t) for t in seq))
                else:
                    texts.append(str(seq))
            return texts
        # List of strings
        return [str(s) for s in raw_input]
    return [str(raw_input)]

def pad_to_1536(embedding: list) -> list:
    """Pad a 384-dim embedding to 1536 dims by repeating it 4x (1536/384=4)."""
    # Repeat the embedding to fill 1536 dimensions
    repeated = (embedding * 4)[:TARGET_DIM]
    return repeated

@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    texts = decode_input(req.input)
    raw_embeddings = model.encode(texts).tolist()

    data = []
    for i, emb in enumerate(raw_embeddings):
        padded = pad_to_1536(emb)
        data.append({
            "object": "embedding",
            "embedding": padded,
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
