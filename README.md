# Duplicate Question Retrieval Engine (QQP)

A production-grade Information Retrieval system for natural language duplicate question retrieval built on the **Quora Question Pairs (QQP)** dataset. The system indexes ~538,000 unique questions and evaluates four distinct retrieval strategies:
1. **TF-IDF**: Sparse lexical representation with sublinear TF scaling and cosine similarity.
2. **BM25 (Okapi)**: Probabilistic lexical ranking with term saturation and document-length normalization.
3. **Sentence Transformers (SBERT)**: Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2` with L2-normalized embeddings.
4. **Hybrid RRF**: Reciprocal Rank Fusion ($k=60$) combining lexical exact-match signals with dense semantic representations.

---

## Architecture Overview

```
                                      ┌────────────────┐
                                      │   User Query   │
                                      └───────┬────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
              ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
              │    TF-IDF    │        │     BM25     │        │    SBERT     │
              │ Sparse Cosine│        │ BM25 Okapi   │        │ 384-d Dense  │
              └───────┬──────┘        └───────┬──────┘        └───────┬──────┘
                      │ Top 100               │ Top 100               │ Top 100
                      └───────────────────────┼───────────────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Reciprocal Rank Fusion  │
                                 │     RRF(d), k = 60      │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Streamlit Search UI     │
                                 └─────────────────────────┘
```

---

## Key Engineering Decisions

- **Leakage-Aware Component Splitting**: Question pairs naturally form connected components of duplicate clusters. Standard random splitting causes questions from the same cluster to appear in both training and test sets, artificially inflating metrics. We use graph-based connected-component splitting so entire duplicate clusters reside strictly in either train, validation, or test sets.
- **Transitive Closure**: If Question A is duplicate to B, and B is duplicate to C, all three form a mutual duplicate clique. Transitive closure prevents penalizing valid duplicates as false negatives during evaluation.
- **Selective Lexical Preprocessing**: Common stop words are removed, but critical question words (*what, how, why, when, where*) and negations (*not, no, never*) are preserved to prevent semantic distortion.
- **Dense Exact Search**: Embeddings are L2-normalized during index creation, allowing exact cosine similarity to be computed via dot product `np.dot` in ~50–150 ms without approximation error.

---

## Directory Structure

```
ir_project/
├── app.py                             # Interactive Streamlit search UI
├── requirements.txt                   # Pinned project dependencies
├── README.md                          # Project documentation
│
├── data/                              # Dataset directory
│   └── quora_duplicate_questions.tsv  # Downloaded QQP dataset (~58 MB)
│
├── artifacts/                         # Serialized indexes and experimental results
│   ├── corpus.pkl                     # Unique question DataFrame (~538k questions)
│   ├── duplicate_map.pkl              # Duplicate adjacency map with transitive closure
│   ├── splits.pkl                     # Leakage-aware train/val/test splits
│   ├── tfidf_vectorizer.pkl           # Fitted TfidfVectorizer
│   ├── tfidf_matrix.pkl               # Sparse TF-IDF matrix
│   ├── bm25_index.pkl                 # BM25Okapi index
│   ├── sbert_embeddings.npy           # 384-dimensional dense embeddings (~789 MB)
│   ├── comparison_table.csv           # Performance metrics table across methods
│   ├── pr_curve.png                   # Precision-Recall curve
│   └── eval_results.json              # Detailed evaluation results
│
├── src/                               # Core source modules
│   ├── config.py                      # Central configuration & hyperparameters
│   ├── data_loader.py                 # TSV loading, schema validation, cleaning
│   ├── corpus.py                      # Corpus extraction & duplicate graph builder
│   ├── splitter.py                    # Connected-component split & query sampling
│   ├── preprocessing.py               # Text normalization and selective stop-word filtering
│   ├── evaluation.py                  # P@K, R@K, MAP@K, nDCG@K & PR-curve generation
│   └── retrievers/                    # Retrieval modules
│       ├── base_retriever.py          # Abstract base retriever interface
│       ├── tfidf_retriever.py         # TF-IDF sparse retriever
│       ├── bm25_retriever.py          # BM25 Okapi retriever
│       ├── sbert_retriever.py         # SBERT dense retriever
│       └── hybrid_retriever.py        # Reciprocal Rank Fusion hybrid retriever
│
├── scripts/                           # Runnable pipeline scripts
│   ├── 01_download_data.py            # Download QQP dataset
│   ├── 02_build_corpus.py             # Build unique corpus & splits
│   ├── 03_build_indexes.py            # Construct TF-IDF, BM25, and SBERT indexes
│   ├── 04_evaluate.py                 # Run benchmark evaluation across all methods
│   └── run_all.py                     # End-to-end automated execution script
│
└── tests/                             # Automated test suite (pytest)
    ├── test_preprocessing.py          # Preprocessing unit tests
    ├── test_corpus.py                 # Graph & splitting unit tests
    ├── test_retrievers.py             # TF-IDF, BM25, and RRF logic tests
    ├── test_evaluation.py             # Metric calculations validation
    └── test_integration.py            # End-to-end integration tests
```

---

## Setup & Reproduction Instructions

### 1. Environment Setup

```bash
# Clone the repository and navigate to root
cd ir_project

# Create and activate virtual environment
python -m venv .venv

# Windows activation:
.venv\Scripts\activate

# Linux/macOS activation:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK tokenizers and stop words
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run the Complete Pipeline

To run the entire pipeline automatically (download data, build corpus and splits, construct indexes, and run the evaluation benchmark):

```bash
python scripts/run_all.py
```

Or execute stage-by-stage:
```bash
python scripts/01_download_data.py   # Download & validate dataset
python scripts/02_build_corpus.py    # Extract unique questions & splits
python scripts/03_build_indexes.py   # Build TF-IDF, BM25, and SBERT indexes
python scripts/04_evaluate.py        # Evaluate on test queries
```

### 3. Launch the Interactive Search Interface

```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

### 4. Run Automated Tests

```bash
pytest tests/ -v
```

---

## Evaluation Metrics

Evaluated at $K \in \{5, 10\}$ over unseen test queries:
- **Precision@K**: Fraction of top-$K$ retrieved questions that are known duplicates.
- **Recall@K**: Fraction of all known duplicates retrieved in top-$K$.
- **MAP@K (Mean Average Precision)**: Macro-averaged AP penalizing lower ranks.
- **nDCG@K (Normalized Discounted Cumulative Gain)**: Graded relevance score with logarithmic discount.
- **Query Latency**: Wall-clock retrieval time per query (mean, p50, p95).
