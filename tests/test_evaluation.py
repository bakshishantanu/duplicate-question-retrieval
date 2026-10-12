"""Unit tests for evaluation metrics."""
from src.evaluation import precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k


def test_precision_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6, 7, 8}
    assert abs(precision_at_k(retrieved, relevant, 3) - 2 / 3) < 1e-9
    assert precision_at_k([4, 5], {1, 2}, 2) == 0.0


def test_recall_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6, 7, 8}
    assert abs(recall_at_k(retrieved, relevant, 3) - 2 / 5) < 1e-9
    assert recall_at_k([1, 2], {1, 2}, 2) == 1.0


def test_average_precision_at_k():
    retrieved = [1, 4, 2, 3]
    relevant = {1, 2, 3}
    # Hits at rank 1 (P@1=1/1), rank 3 (P@3=2/3), rank 4 (P@4=3/4)
    expected_ap = (1.0 + 2.0 / 3.0 + 3.0 / 4.0) / 3.0
    assert abs(average_precision_at_k(retrieved, relevant, 4) - expected_ap) < 1e-6


def test_ndcg_at_k_perfect():
    retrieved = [1, 2, 3]
    relevant = {1, 2, 3}
    assert abs(ndcg_at_k(retrieved, relevant, 3) - 1.0) < 1e-9


def test_ndcg_at_k_worst():
    retrieved = [4, 5, 6]
    relevant = {1, 2, 3}
    assert ndcg_at_k(retrieved, relevant, 3) == 0.0


def test_ndcg_at_k_partial():
    retrieved = [4, 1, 5]
    relevant = {1}
    ndcg = ndcg_at_k(retrieved, relevant, 3)
    assert 0.0 < ndcg < 1.0
