"""Central configuration for the duplicate question retrieval project."""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RAW_DATA_PATH = os.path.join(DATA_DIR, "quora_duplicate_questions.tsv")

# Corpus artifacts
CORPUS_PATH = os.path.join(ARTIFACTS_DIR, "corpus.pkl")
DUPLICATE_MAP_PATH = os.path.join(ARTIFACTS_DIR, "duplicate_map.pkl")
SPLITS_PATH = os.path.join(ARTIFACTS_DIR, "splits.pkl")
CORPUS_IDS_PATH = os.path.join(ARTIFACTS_DIR, "corpus_ids.pkl")

# Index artifacts
TFIDF_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
BM25_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "bm25_index.pkl")
SBERT_EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "sbert_embeddings.npy")

# Evaluation artifacts
EVAL_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "eval_results.json")
COMPARISON_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "comparison_table.csv")
PR_CURVE_PATH = os.path.join(ARTIFACTS_DIR, "pr_curve.png")

# Model configuration
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIM = 384

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# RRF parameters
RRF_K = 60

# Retrieval parameters
DEFAULT_TOP_K = 10
CANDIDATE_DEPTH = 100  # candidates retrieved per method for RRF

# Evaluation parameters
EVAL_K_VALUES = [5, 10]
NUM_EVAL_QUERIES = 1000  # number of test queries to evaluate

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
