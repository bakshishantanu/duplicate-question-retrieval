"""Evaluation metrics, comparison tables, and Precision-Recall visualization."""
import json
import logging
import time
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Individual Query Metrics ───

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K for a single query."""
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K for a single query."""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(relevant)


def average_precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Average Precision@K for a single query."""
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    hits = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(len(relevant), k)


def dcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Discounted Cumulative Gain @ K for binary relevance."""
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because ranks are 1-indexed (log2(1+1) for rank 1)
    return dcg


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ─── Aggregate Evaluation ───

def evaluate_retriever(
    retriever: Any,
    eval_queries: List[Tuple[int, Set[int]]],
    corpus_df: pd.DataFrame,
    k_values: Optional[List[int]] = None,
    max_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a retriever over a query set.

    Args:
        retriever: Object implementing retrieve(query, top_k, exclude_qids).
        eval_queries: List of (query_qid, set_of_relevant_qids).
        corpus_df: Corpus DataFrame with 'qid' and 'question_text'.
        k_values: List of cutoffs (default [5, 10]).
        max_queries: Max number of queries to evaluate.

    Returns:
        Dict with macro-averaged metrics and query latencies.
    """
    if k_values is None:
        k_values = [5, 10]

    qid_to_text = dict(zip(corpus_df['qid'], corpus_df['question_text']))
    queries = eval_queries[:max_queries] if max_queries is not None else eval_queries

    all_metrics = {
        k: {'precision': [], 'recall': [], 'ap': [], 'ndcg': []}
        for k in k_values
    }
    latencies = []
    max_k = max(k_values)

    for query_qid, relevant_qids in queries:
        query_text = qid_to_text.get(query_qid, "")
        if not query_text:
            continue

        t0 = time.perf_counter()
        results = retriever.retrieve(
            query_text,
            top_k=max_k,
            exclude_qids={query_qid}  # Always exclude self-match
        )
        latency = time.perf_counter() - t0
        latencies.append(latency)

        retrieved_ids = [qid for qid, _ in results]

        for k in k_values:
            all_metrics[k]['precision'].append(precision_at_k(retrieved_ids, relevant_qids, k))
            all_metrics[k]['recall'].append(recall_at_k(retrieved_ids, relevant_qids, k))
            all_metrics[k]['ap'].append(average_precision_at_k(retrieved_ids, relevant_qids, k))
            all_metrics[k]['ndcg'].append(ndcg_at_k(retrieved_ids, relevant_qids, k))

    results = {}
    for k in k_values:
        results[f'P@{k}'] = float(np.mean(all_metrics[k]['precision'])) if all_metrics[k]['precision'] else 0.0
        results[f'R@{k}'] = float(np.mean(all_metrics[k]['recall'])) if all_metrics[k]['recall'] else 0.0
        results[f'MAP@{k}'] = float(np.mean(all_metrics[k]['ap'])) if all_metrics[k]['ap'] else 0.0
        results[f'nDCG@{k}'] = float(np.mean(all_metrics[k]['ndcg'])) if all_metrics[k]['ndcg'] else 0.0

    if latencies:
        results['avg_latency_ms'] = float(np.mean(latencies) * 1000)
        results['p50_latency_ms'] = float(np.percentile(latencies, 50) * 1000)
        results['p95_latency_ms'] = float(np.percentile(latencies, 95) * 1000)
    else:
        results['avg_latency_ms'] = 0.0
        results['p50_latency_ms'] = 0.0
        results['p95_latency_ms'] = 0.0

    results['num_queries'] = len(latencies)
    return results


def generate_comparison_table(
    all_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> str:
    """Generate a clean pandas comparison table and optional CSV."""
    rows = []
    for method, metrics in all_results.items():
        row = {'Method': method}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved comparison table to {save_path}")

    return df.to_string(index=False)


def generate_pr_curve(
    retrievers: Dict[str, Any],
    eval_queries: List[Tuple[int, Set[int]]],
    corpus_df: pd.DataFrame,
    max_k: int = 50,
    max_queries: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Generate Precision-Recall curve across methods by varying K."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    qid_to_text = dict(zip(corpus_df['qid'], corpus_df['question_text']))
    queries = eval_queries[:max_queries] if max_queries is not None else eval_queries
    k_range = list(range(1, max_k + 1))

    plt.figure(figsize=(9, 6), dpi=150)
    colors = {
        'TF-IDF': '#e74c3c',
        'BM25': '#3498db',
        'SBERT': '#2ecc71',
        'Hybrid RRF': '#9b59b6'
    }
    markers = {
        'TF-IDF': 'o',
        'BM25': 's',
        'SBERT': '^',
        'Hybrid RRF': 'D'
    }

    for method_name, retriever in retrievers.items():
        logger.info(f"Computing PR curve points for {method_name}...")
        # Pre-retrieve max_k results for each query once
        retrieved_cache = []
        for query_qid, relevant_qids in queries:
            query_text = qid_to_text.get(query_qid, "")
            if not query_text:
                retrieved_cache.append([])
                continue
            res = retriever.retrieve(query_text, top_k=max_k, exclude_qids={query_qid})
            retrieved_cache.append([r[0] for r in res])

        precisions = []
        recalls = []

        for k in k_range:
            k_precisions = []
            k_recalls = []
            for (_, relevant_qids), retrieved_ids in zip(queries, retrieved_cache):
                k_precisions.append(precision_at_k(retrieved_ids, relevant_qids, k))
                k_recalls.append(recall_at_k(retrieved_ids, relevant_qids, k))

            precisions.append(float(np.mean(k_precisions)) if k_precisions else 0.0)
            recalls.append(float(np.mean(k_recalls)) if k_recalls else 0.0)

        plt.plot(
            recalls,
            precisions,
            label=method_name,
            color=colors.get(method_name, '#333333'),
            marker=markers.get(method_name, 'o'),
            markevery=max(1, max_k // 10),
            linewidth=2,
            markersize=5,
        )

    plt.xlabel('Recall@K', fontsize=12)
    plt.ylabel('Precision@K', fontsize=12)
    plt.title('Precision–Recall Curve Across Retrieval Methods', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PR curve to {save_path}")

    plt.close()
