
# SHL Assessment Recommendation System

## 📋 Prerequisites
- Python 3.9 or higher
- Internet connection (for scraping and loading models)

## 🛠️ Installation

1.  **Install Dependencies**:
    Open a terminal in the project folder and run:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

### 1. Start the API Backend
This runs the recommendation engine.
```bash
uvicorn app:app --port 8000 --reload
```
*   The API will be live at `http://127.0.0.1:8000`.
*   Documentation/Swagger UI: `http://127.0.0.1:8000/docs`
*   Health Check: `http://127.0.0.1:8000/health`

### 2. Start the Frontend (User Interface)
Open a **new** terminal window (keep the API running) and run:
```bash
streamlit run streamlit_app.py
```
*   This will automatically open the web interface in your browser (usually `http://localhost:8501`).

## 📊 Data Pipeline (Optional)
If you need to regenerate the data or embeddings:

1.  **Scrape Data**:
    ```bash
    python scrape_catalog.py
    ```
    *Creates `shl_products.json`.*

2.  **Generate Embeddings**:
    ```bash
    python create_embeddings.py
    ```
    *Creates `product_embeddings.pkl`.*

## 🧪 Evaluation & Submission
*   **Evaluate Model**:
    ```bash
    python evaluate_model.py
    ```
    *Calculates Mean Recall@10 on the Training Set.*

*   **Generate Submission**:
    ```bash
    python generate_submission.py
    ```
    *Creates `submission.csv` for the Test Set.*
