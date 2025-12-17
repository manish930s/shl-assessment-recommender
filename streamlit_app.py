
import streamlit as st
import requests
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Set page configuration
st.set_page_config(
    page_title="SHL Smart Recommender",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #333333;
    }
    .card h3 {
        margin-top: 0;
        color: #1f1f1f;
    }
    .card p {
        color: #333333;
        margin-bottom: 0.5rem;
    }
    .card a {
        color: #007bff;
        text-decoration: none;
        font-weight: bold;
    }
    .metric-badge {
        background-color: #f1f3f5;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        color: #343a40;
        margin-right: 0.5rem;
        display: inline-block;
        margin-top: 5px;
        border: 1px solid #dee2e6;
    }
    .meta-info {
        font-size: 0.85rem;
        color: #555555;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🎯 SHL Assessment Recommender")
st.markdown("""
    <div style='background-color: #e3f2fd; color: #333333; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #2196f3; margin-bottom: 2rem;'>
        <strong>Welcome!</strong> This tool helps you find the perfect SHL assessment for your hiring needs.
        Simply describe the role, skills, or behaviors you are looking for.
    </div>
""", unsafe_allow_html=True)

# Load Resources (Cached)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open("product_embeddings.pkl", "rb") as f:
        products = pickle.load(f)
    embeddings = np.array([p['vector'] for p in products])
    # Normalize
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norm + 1e-9)
    return model, products, embeddings

try:
    model, products, embeddings_matrix = load_resources()
    st.sidebar.success("Models Loaded Successfully! ✅")
    st.sidebar.info(f"Database contains {len(products)} assessments.")
except Exception as e:
    st.error(f"Error loading resources: {e}. Please ensure data pipeline has run.")
    st.stop()

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area("Describe your ideal candidate or job requirements:", 
                        placeholder="e.g. 'Looking for a Java developer who leads teams effectively and communicates well.'",
                        height=100)
    
    search_clicked = st.button("Find Assessments 🔍")

with col2:
    st.markdown("### Search Settings")
    top_k = st.slider("Number of recommendations:", 5, 20, 10)
    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("- Mention specific job titles")
    st.markdown("- Include soft skills (e.g. 'leadership')")
    st.markdown("- Include technical skills (e.g. 'Python')")

# Search Logic
if search_clicked and query:
    with st.spinner("Analyzing requirements..."):
        # Encode
        query_vec = model.encode([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        
        # Similarity
        scores = np.dot(embeddings_matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "product": products[idx],
                "score": scores[idx]
            })
            
    # Display Results
    st.markdown(f"### Top {top_k} Recommendations")
    
    for item in results:
        p = item['product']
        score = item['score']
        
        name = p.get('name', p.get('title', 'Unknown'))
        desc = p.get('description', 'No description available.')
        duration = p.get('duration', 0)
        remote = p.get('remote_support', 'N/A')
        adaptive = p.get('adaptive_support', 'N/A')
        
        # Truncate description for display
        if not desc:
            desc_display = "<span style='color: #999; font-style: italic;'>No description available from catalog.</span>"
        elif len(desc) > 200:
            desc_display = desc[:200] + "..."
        else:
            desc_display = desc
        
        # Determine Card Color based on score
        border_color = "#28a745" if score > 0.5 else "#ffc107"
        
        types = p.get('test_type', p.get('test_types', []))
        types_html = "".join([f"<span class='metric-badge'>{t}</span>" for t in types])
        
        duration_display = f"{duration} mins" if duration > 0 else "Varies/Not Listed"
        
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {border_color};">
            <h3>{name}</h3>
            <p style="color: #6c757d; font-size: 0.9rem;">Relevance Score: <strong>{score:.2f}</strong></p>
            <p>{desc_display}</p>
            <div class="meta-info">
                <strong>Duration:</strong> {duration_display} | 
                <strong>Remote:</strong> {remote} | 
                <strong>Adaptive:</strong> {adaptive}
            </div>
            <div style="margin-bottom: 0.5rem;">{types_html}</div>
            <a href="{p['url']}" target="_blank">View Assessment on SHL.com &rarr;</a>
        </div>
        """, unsafe_allow_html=True)
        
elif search_clicked:
    st.warning("Please enter a query first.")
