
import json
from sentence_transformers import SentenceTransformer
import pickle
import os

INPUT_FILE = "shl_products.json"
OUTPUT_FILE = "product_embeddings.pkl"

def create_embeddings():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found. Please run scrape_catalog_full.py first.")
        return

    with open(INPUT_FILE, "r") as f:
        products = json.load(f)
    
    print(f"Loaded {len(products)} products.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sentences = []
    for p in products:
        # Create a rich representation for embedding
        # Title + Test Types + Description (truncated)
        text = p.get('name', '')
        
        types = p.get('test_type', [])
        if types:
            text += " " + " ".join(types)
            
        desc = p.get('description', '')
        if desc:
            text += " " + desc
            
        sentences.append(text)
        
    print("Generating embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    print("Embeddings generated.")
    
    # Save combined data
    data_with_embeddings = []
    for i, p in enumerate(products):
        p['vector'] = embeddings[i]
        data_with_embeddings.append(p)
        
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data_with_embeddings, f)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_embeddings()
