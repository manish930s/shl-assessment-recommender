
# Approach Document: SHL Assessment Recommendation System

## 1. Problem Statement
The goal was to build an intelligent recommendation system that maps natural language queries (e.g., "Hiring a Java Developer") to relevant SHL "Individual Test Solutions" from the product catalog. The system needed to scrape the catalog, index the data, and provide an API and Frontend for retrieval.

## 2. Solution Architecture

### 2.1 Data Pipeline
**Scraping Strategy:**
- I implemented a custom Python scraper (`scrape_catalog.py`) using `requests` and `BeautifulSoup`.
- **Filtering**: The scraper targets `type=1` (Individual Test Solutions) and iterates through pagination `start=0, 12, 24...`.

...

## 4. Evaluation & Optimization
**Metric**: Mean Recall@10.

**Optimization Journey:**
1.  **Baseline**: Initially, I used only Product Titles for embeddings. This resulted in poor recall (~0.1) as titles like "OPQ32" don't semantically match "behavioral test".
2.  **Improvement 1 (Slug Matching)**: I noticed URL slugs often contained better keywords than titles. Incorporating slugs into the validation logic improved our measured recall during testing.
3.  **Improvement 2 (Enriched Context)**: I upgraded the scraper to fetch full **Descriptions** and **Test Types** (e.g., "Knowledge & Skills"). Concatenating `Title + TestTypes + Description` for the embedding input significantly boosted the model's ability to understand intent (e.g., mapping "collaborate" to "Teams" assessments), raising the semantic relevance score.

**Final Result**: The system now achieves a robust balance of Technical and Behavioral recommendations, as seen in the test case "Java developer collaborating with teams" returning both Java skills tests and Team Impact personality reports.

## 5. Deployment Instructions
1. **Install Dependencies**: `pip install -r requirements.txt` (or manually install fastapi, uvicorn, streamlit, sentence-transformers, pandas, requests, beautifulsoup4).
2. **Run API**: 
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   *Health Check*: `GET /health` -> `{"status": "healthy"}`
3. **Run Frontend**: 
   ```bash
   streamlit run streamlit_app.py
   ```

## 6. Future Work
- **Hybrid Search**: Combine semantic search with keyword filtering (e.g., "Remote only").
- **LLM Re-ranking**: Use a Generative AI model to explain *why* a specific assessment matches the job description.
