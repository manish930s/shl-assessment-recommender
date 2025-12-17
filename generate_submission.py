
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os

EMBEDDINGS_FILE = "product_embeddings.pkl"
DATASET_FILE = "Gen_AI Dataset.xlsx"
OUTPUT_FILE = "submission.csv"

def generate_submission():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings file not found.")
        return

    print("Loading resources...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        products = pickle.load(f)
    
    embeddings_matrix = np.array([p['vector'] for p in products])
    # Normalize
    norm = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_matrix = embeddings_matrix / (norm + 1e-9)
    
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("Loading Test Set...")
    # Sheet names: 'Train-Set', 'Test-Set'?
    # I saw 'Train-Set' earlier. Assumption: 'Test-Set' exists as per instructions.
    # Instruction says: "Unlabeled test set: This dataset contains a set of 9 queries"
    try:
        df = pd.read_excel(DATASET_FILE, sheet_name='Test-Set')
    except:
        # Fallback if name is different. I'll check sheets lightly.
        xl = pd.ExcelFile(DATASET_FILE)
        # Find sheet that is not Train-Set
        sheets = [s for s in xl.sheet_names if 'Test' in s]
        if sheets:
            df = pd.read_excel(DATASET_FILE, sheet_name=sheets[0])
        else:
            print("Could not find Test set sheet.")
            return

    results = []
    
    print(f"Generating predictions for {len(df)} queries...")
    
    for _, row in df.iterrows():
        query = row['Query']
        
        # Encode query
        query_vec = model.encode([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        
        # Search
        scores = np.dot(embeddings_matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:10] # Top 10
        
        for idx in top_indices:
            p_url = products[idx]['url']
            results.append({
                "Query": query,
                "Assessment_url": p_url
            })
            
    # Save to CSV
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved submission to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_submission()
