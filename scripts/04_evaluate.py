"""Run full retrieval evaluation pipeline."""
import os
import sys
import json
import logging
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.config import (
    ARTIFACTS_DIR, CORPUS_PATH, DUPLICATE_MAP_PATH, SPLITS_PATH, CORPUS_IDS_PATH,
    TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, BM25_INDEX_PATH, SBERT_EMBEDDINGS_PATH,
    EVAL_RESULTS_PATH, COMPARISON_TABLE_PATH, PR_CURVE_PATH,
    EVAL_K_VALUES, NUM_EVAL_QUERIES
)
from src.splitter import get_eval_queries
from src.evaluation import evaluate_retriever, generate_comparison_table, generate_pr_curve
from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.sbert_retriever import SBERTRetriever
from src.retrievers.hybrid_retriever import HybridRetriever


def main():
    print("Step 1: Loading corpus and evaluation split...")
    corpus_df = joblib.load(CORPUS_PATH)
    duplicate_map = joblib.load(DUPLICATE_MAP_PATH)
    splits = joblib.load(SPLITS_PATH)
    corpus_ids = joblib.load(CORPUS_IDS_PATH)

    eval_queries = get_eval_queries(splits, duplicate_map, split_name='test')
    print(f"Total available test evaluation queries: {len(eval_queries)}")
    print(f"Using up to {NUM_EVAL_QUERIES} queries for evaluation benchmark.")

    print("\nStep 2: Loading retriever indexes...")
    tfidf = TFIDFRetriever()
    tfidf.load_index(TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, corpus_ids)

    bm25 = BM25Retriever()
    bm25.load_index(BM25_INDEX_PATH)

    sbert = SBERTRetriever()
    sbert.load_index(SBERT_EMBEDDINGS_PATH, corpus_ids)

    hybrid = HybridRetriever({'TF-IDF': tfidf, 'BM25': bm25, 'SBERT': sbert})

    methods = {
        'TF-IDF': tfidf,
        'BM25': bm25,
        'SBERT': sbert,
        'Hybrid RRF': hybrid,
    }

    all_results = {}
    print("\nStep 3: Running benchmark across methods...")
    for name, retriever in methods.items():
        print(f"\n--> Evaluating {name}...")
        results = evaluate_retriever(
            retriever=retriever,
            eval_queries=eval_queries,
            corpus_df=corpus_df,
            k_values=EVAL_K_VALUES,
            max_queries=NUM_EVAL_QUERIES
        )
        all_results[name] = results
        for k in EVAL_K_VALUES:
            print(f"    MAP@{k}:  {results.get(f'MAP@{k}', 0.0):.4f}")
            print(f"    nDCG@{k}: {results.get(f'nDCG@{k}', 0.0):.4f}")
            print(f"    P@{k}:    {results.get(f'P@{k}', 0.0):.4f}")
            print(f"    R@{k}:    {results.get(f'R@{k}', 0.0):.4f}")
        print(f"    Avg Latency: {results['avg_latency_ms']:.1f} ms")

    print("\nStep 4: Generating performance comparison table...")
    table_str = generate_comparison_table(all_results, save_path=COMPARISON_TABLE_PATH)
    print("\n" + table_str)

    print("\nStep 5: Generating Precision-Recall curve graph...")
    generate_pr_curve(
        retrievers=methods,
        eval_queries=eval_queries,
        corpus_df=corpus_df,
        max_k=30,
        max_queries=min(300, len(eval_queries)),
        save_path=PR_CURVE_PATH
    )

    print("\nStep 6: Saving raw JSON evaluation metrics...")
    with open(EVAL_RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete! Output files generated in: {ARTIFACTS_DIR}")
    print(f"  - Table: {COMPARISON_TABLE_PATH}")
    print(f"  - Graph: {PR_CURVE_PATH}")
    print(f"  - Metrics: {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
