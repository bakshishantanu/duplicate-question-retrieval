"""Unit tests for retrieval modules."""
import pytest
import pandas as pd
from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever


@pytest.fixture
def toy_corpus():
    return pd.DataFrame({
        'qid': [1, 2, 3, 4, 5],
        'question_text': [
            'How to learn Python programming',
            'What is the best way to study Python',
            'How to cook Italian pasta at home',
            'Best restaurants in New York City',
            'Tips for learning Python fast',
        ],
        'preprocessed_text': [
            'learn python program',
            'best way studi python',
            'cook italian pasta home',
            'best restaur new york citi',
            'tip learn python fast',
        ],
    })


def test_tfidf_retrieval(toy_corpus):
    retriever = TFIDFRetriever(max_features=100)
    retriever.build_index(toy_corpus)
    results = retriever.retrieve("learn python", top_k=3)
    assert len(results) > 0
    top_qids = [r[0] for r in results]
    assert 1 in top_qids or 2 in top_qids or 5 in top_qids


def test_bm25_retrieval(toy_corpus):
    retriever = BM25Retriever()
    retriever.build_index(toy_corpus)
    results = retriever.retrieve("learn python", top_k=3)
    assert len(results) > 0
    top_qids = [r[0] for r in results]
    assert 1 in top_qids or 2 in top_qids or 5 in top_qids


def test_self_exclusion(toy_corpus):
    retriever = TFIDFRetriever(max_features=100)
    retriever.build_index(toy_corpus)
    results = retriever.retrieve("How to learn Python programming", top_k=5, exclude_qids={1})
    top_qids = [r[0] for r in results]
    assert 1 not in top_qids


def test_hybrid_rrf_synthetic():
    class DummyRetriever:
        def __init__(self, rankings):
            self.rankings = rankings
        def retrieve(self, query, top_k=10, exclude_qids=None):
            return self.rankings[:top_k]

    r1 = DummyRetriever([(1, 0.9), (2, 0.8), (3, 0.7)])
    r2 = DummyRetriever([(2, 0.95), (1, 0.85), (4, 0.6)])

    hybrid = HybridRetriever({'r1': r1, 'r2': r2}, k=60, candidate_depth=10)
    res = hybrid.retrieve("query", top_k=5)
    result_qids = [r[0] for r in res]

    # Documents 1 and 2 appear at rank 1 and 2 across both retrievers, so they must be top 2
    assert set(result_qids[:2]) == {1, 2}
    assert 3 in result_qids or 4 in result_qids
