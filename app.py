
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os
from typing import List, Optional

app = FastAPI(title="SHL Assessment Recommender")

# Global variables
model = None
products_data = None
embeddings_matrix = None

class QueryRequest(BaseModel):
    query: str

class RecommendationItem(BaseModel):
    url: str
    name: str
    adaptive_support: str = Field(..., serialization_alias="adaptive support")
    description: str
    duration: int
    remote_support: str = Field(..., serialization_alias="remote support")
    test_type: List[str] = Field(..., serialization_alias="test type")

class RecommendationResponse(BaseModel):
    recommended_assessments: List[RecommendationItem] = Field(..., serialization_alias="recommended assessments")

@app.on_event("startup")
async def load_resources():
    global model, products_data, embeddings_matrix
    import gc
    
    # Load model with minimal memory footprint strategy if possible
    # We are using a small model (80MB), but the overhead can be high.
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Load embeddings
    if os.path.exists("product_embeddings.pkl"):
        print("Loading embeddings...")
        with open("product_embeddings.pkl", "rb") as f:
            products_data = pickle.load(f)
            embeddings_matrix = np.array([p['vector'] for p in products_data])
            norm = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            embeddings_matrix = embeddings_matrix / (norm + 1e-9)
        print(f"Loaded {len(products_data)} items.")
    else:
        print("product_embeddings.pkl not found.")
        
    gc.collect()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: QueryRequest):
    if model is None or embeddings_matrix is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded.")
    
    # Encode query
    query_vec = model.encode([request.query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    
    # Cosine similarity
    scores = np.dot(embeddings_matrix, query_vec)
    
    # Get top k (fixed to 10 max per requirement)
    top_indices = np.argsort(scores)[::-1][:10]
    
    results = []
    # If no results found (unlikely), return at least 1? The logic below returns whatever is top.
    
    for idx in top_indices:
        p = products_data[idx]
        # Handle cases where new fields might not exist yet during transition
        item = RecommendationItem(
            url=p.get('url', ''),
            name=p.get('name', p.get('title', 'Unknown')),
            adaptive_support=p.get('adaptive_support', 'No'),
            description=p.get('description', '')[:200], # Trucate if too long? No, spec says string.
            duration=p.get('duration', 0),
            remote_support=p.get('remote_support', 'Yes'),
            test_type=p.get('test_type', p.get('test_types', []))
        )
        results.append(item)
    
    if not results and len(products_data) > 0:
         # Fallback if similarity failed completely?? Shouldn't happen.
         pass
         
    return RecommendationResponse(recommended_assessments=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
