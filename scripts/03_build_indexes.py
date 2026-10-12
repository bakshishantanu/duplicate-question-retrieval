"""Build all retrieval indexes (TF-IDF, BM25, SBERT)."""
import os
import sys
import time
import logging
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.config import (
    ARTIFACTS_DIR, CORPUS_PATH, CORPUS_IDS_PATH,
    TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH,
    BM25_INDEX_PATH, SBERT_EMBEDDINGS_PATH
)
from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.sbert_retriever import SBERTRetriever


def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"Corpus file not found at {CORPUS_PATH}!")
        print("Please run `python scripts/02_build_corpus.py` first.")
        sys.exit(1)

    print("Step 1: Loading unique question corpus...")
    corpus_df = joblib.load(CORPUS_PATH)
    print(f"Loaded corpus: {len(corpus_df)} questions")

    # Ensure corpus_ids.pkl is present
    if not os.path.exists(CORPUS_IDS_PATH):
        joblib.dump(corpus_df['qid'].values, CORPUS_IDS_PATH)

    # 1. TF-IDF
    print("\n-------------------------------------------")
    print("Step 2: Checking / Building TF-IDF Index (sparse cosine)...")
    print("-------------------------------------------")
    if os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(TFIDF_VECTORIZER_PATH):
        print(f"TF-IDF index already exists at {TFIDF_MATRIX_PATH}. Skipping build.")
        tfidf_time = 0.0
    else:
        t0 = time.time()
        tfidf = TFIDFRetriever()
        tfidf.build_index(corpus_df)
        tfidf.save_index(TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH)
        tfidf_time = time.time() - t0
        print(f"TF-IDF index built and saved in {tfidf_time:.1f}s")

    # 2. BM25
    print("\n-------------------------------------------")
    print("Step 3: Checking / Building BM25 Index (BM25 Okapi)...")
    print("-------------------------------------------")
    if os.path.exists(BM25_INDEX_PATH):
        print(f"BM25 index already exists at {BM25_INDEX_PATH}. Skipping build.")
        bm25_time = 0.0
    else:
        t0 = time.time()
        bm25 = BM25Retriever()
        bm25.build_index(corpus_df)
        bm25.save_index(BM25_INDEX_PATH)
        bm25_time = time.time() - t0
        print(f"BM25 index built and saved in {bm25_time:.1f}s")

    # 3. SBERT
    print("\n-------------------------------------------")
    print("Step 4: Building Sentence Transformer Embeddings...")
    print("-------------------------------------------")
    if os.path.exists(SBERT_EMBEDDINGS_PATH):
        print(f"SBERT embeddings already exist at {SBERT_EMBEDDINGS_PATH}. Skipping build.")
        sbert_time = 0.0
    else:
        import torch
        batch_size = 512 if torch.cuda.is_available() else 256
        t0 = time.time()
        sbert = SBERTRetriever()
        sbert.build_index(corpus_df, batch_size=batch_size)
        sbert.save_index(SBERT_EMBEDDINGS_PATH)
        sbert_time = time.time() - t0
        print(f"SBERT embeddings built and saved in {sbert_time:.1f}s")

    print("\n===========================================")
    print("All indexes built and persisted successfully!")
    print(f"  TF-IDF time: {tfidf_time:.1f}s")
    print(f"  BM25 time:   {bm25_time:.1f}s")
    print(f"  SBERT time:  {sbert_time:.1f}s")
    print(f"Artifacts located in: {ARTIFACTS_DIR}")
    print("===========================================")


if __name__ == "__main__":
    main()
