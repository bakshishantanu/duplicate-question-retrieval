"""Integration tests verifying artifact integrity and retriever querying."""
import os
import pytest
import joblib
from src.config import CORPUS_PATH, TFIDF_VECTORIZER_PATH, BM25_INDEX_PATH, SBERT_EMBEDDINGS_PATH, TFIDF_MATRIX_PATH, CORPUS_IDS_PATH
from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.sbert_retriever import SBERTRetriever
from src.retrievers.hybrid_retriever import HybridRetriever


@pytest.mark.skipif(not os.path.exists(CORPUS_PATH), reason="Artifacts not built yet")
def test_full_pipeline_retrieval():
    corpus_df = joblib.load(CORPUS_PATH)
    corpus_ids = joblib.load(CORPUS_IDS_PATH)

    tfidf = TFIDFRetriever()
    tfidf.load_index(TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, corpus_ids)

    bm25 = BM25Retriever()
    bm25.load_index(BM25_INDEX_PATH)

    sbert = SBERTRetriever()
    sbert.load_index(SBERT_EMBEDDINGS_PATH, corpus_ids)

    hybrid = HybridRetriever({'TF-IDF': tfidf, 'BM25': bm25, 'SBERT': sbert})

    query = "How to learn Python programming?"
    for name, ret in [('TF-IDF', tfidf), ('BM25', bm25), ('SBERT', sbert), ('Hybrid RRF', hybrid)]:
        results = ret.retrieve(query, top_k=5)
        assert len(results) == 5, f"{name} did not return top 5 results"
        assert all(isinstance(r[0], int) and isinstance(r[1], float) for r in results)
