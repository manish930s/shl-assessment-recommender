
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os
from collections import defaultdict

EMBEDDINGS_FILE = "product_embeddings.pkl"
DATASET_FILE = "Gen_AI Dataset.xlsx"

def get_slug(url):
    if pd.isna(url): return ""
    url = url.split('?')[0]
    url = url.strip().rstrip('/')
    return url.split('/')[-1].lower()

def evaluate_recall():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings file not found.")
        return

    print("Loading resources...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        products = pickle.load(f)
    
    # Precompute slugs for products
    for p in products:
        p['slug'] = get_slug(p['url'])
        
    embeddings_matrix = np.array([p['vector'] for p in products])
    norm = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_matrix = embeddings_matrix / (norm + 1e-9)
    
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("Loading Train Set...")
    try:
        df = pd.read_excel(DATASET_FILE, sheet_name='Train-Set')
    except:
        df = pd.read_excel(DATASET_FILE, sheet_name='Train Set')

    # Group by Query
    query_groups = defaultdict(set)
    for _, row in df.iterrows():
        q = row['Query']
        url = row['Assessment_url']
        slug = get_slug(url)
        if slug:
            query_groups[q].add(slug)
            
    print(f"Loaded {len(query_groups)} unique queries.")
    
    recalls = []
    
    print("\n--- Evaluation ---")
    for query, true_slugs in query_groups.items():
        if not true_slugs:
            continue
            
        # Encode query
        query_vec = model.encode([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        
        # Search
        scores = np.dot(embeddings_matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:10]
        
        retrieved_slugs = {products[i]['slug'] for i in top_indices}
        
        # Calculate Recall
        intersection = true_slugs.intersection(retrieved_slugs)
        recall = len(intersection) / len(true_slugs)
        recalls.append(recall)
        
        print(f"Query: {query[:50]}... | Recall: {recall:.2f}")
        # Debug top 1 match
        top_product = products[top_indices[0]]
        print(f"  Top 1: {top_product.get('name', 'N/A')} ({top_product['slug']})")
        # Check description presence
        desc_len = len(top_product.get('description', ''))
        print(f"  Desc Len: {desc_len}")

    mean_recall = np.mean(recalls) if recalls else 0
    print(f"\nMean Recall@10: {mean_recall:.4f}")

if __name__ == "__main__":
    evaluate_recall()
