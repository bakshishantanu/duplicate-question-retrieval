"""Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set, Any
from src.config import RRF_K, CANDIDATE_DEPTH
from src.retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion."""

    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        k: int = RRF_K,
        candidate_depth: int = CANDIDATE_DEPTH
    ):
        """
        Args:
            retrievers: Dict mapping method name to retriever instance.
                       e.g., {'tfidf': tfidf_ret, 'bm25': bm25_ret, 'sbert': sbert_ret}
            k: RRF smoothing constant (default 60).
            candidate_depth: Number of top candidates to fetch per method.
        """
        self.retrievers = retrievers
        self.k = k
        self.candidate_depth = candidate_depth

    def build_index(self, corpus_df) -> None:
        """Build indexes for all underlying retrievers."""
        for name, retriever in self.retrievers.items():
            logger.info(f"Building index for underlying retriever: {name}")
            retriever.build_index(corpus_df)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_qids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents using Reciprocal Rank Fusion."""
        # 1. Fetch ranked lists from each retriever
        all_rankings: Dict[str, List[Tuple[int, float]]] = {}
        for name, retriever in self.retrievers.items():
            results = retriever.retrieve(
                query,
                top_k=self.candidate_depth,
                exclude_qids=exclude_qids
            )
            all_rankings[name] = results

        # 2. Accumulate RRF scores
        rrf_scores: Dict[int, float] = defaultdict(float)
        method_count: Dict[int, int] = defaultdict(int)
        best_rank: Dict[int, int] = defaultdict(lambda: float('inf'))

        for name, results in all_rankings.items():
            for rank_0, (qid, _score) in enumerate(results):
                rank_1 = rank_0 + 1
                contribution = 1.0 / (self.k + rank_1)
                rrf_scores[qid] += contribution
                method_count[qid] += 1
                if rank_1 < best_rank[qid]:
                    best_rank[qid] = rank_1

        if not rrf_scores:
            return []

        # 3. Deterministic ranking:
        # primary: descending RRF score
        # secondary: descending method consensus count
        # tertiary: best rank achieved
        # quaternary: stable qid
        candidates = list(rrf_scores.keys())
        candidates.sort(key=lambda qid: (
            -rrf_scores[qid],
            -method_count[qid],
            best_rank[qid],
            qid
        ))

        return [(qid, rrf_scores[qid]) for qid in candidates[:top_k]]

    def retrieve_with_details(
        self,
        query: str,
        top_k: int = 10,
        exclude_qids: Optional[Set[int]] = None
    ) -> Dict[str, Any]:
        """Retrieve top-k documents with full diagnostic details for the UI.

        Returns:
            Dict containing:
                - 'results': List of (qid, score)
                - 'individual_rankings': Dict mapping method to candidate list
                - 'contributions': Dict mapping qid to method rank & contribution info
        """
        all_rankings: Dict[str, List[Tuple[int, float]]] = {}
        for name, retriever in self.retrievers.items():
            results = retriever.retrieve(
                query,
                top_k=self.candidate_depth,
                exclude_qids=exclude_qids
            )
            all_rankings[name] = results

        rrf_scores: Dict[int, float] = defaultdict(float)
        contributions: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        method_count: Dict[int, int] = defaultdict(int)
        best_rank: Dict[int, int] = defaultdict(lambda: float('inf'))

        for name, results in all_rankings.items():
            for rank_0, (qid, orig_score) in enumerate(results):
                rank_1 = rank_0 + 1
                contribution = 1.0 / (self.k + rank_1)
                rrf_scores[qid] += contribution
                contributions[qid][name] = {
                    'rank': rank_1,
                    'original_score': orig_score,
                    'rrf_contribution': contribution,
                }
                method_count[qid] += 1
                if rank_1 < best_rank[qid]:
                    best_rank[qid] = rank_1

        candidates = list(rrf_scores.keys())
        candidates.sort(key=lambda qid: (
            -rrf_scores[qid],
            -method_count[qid],
            best_rank[qid],
            qid
        ))

        top_candidates = candidates[:top_k]
        return {
            'results': [(qid, rrf_scores[qid]) for qid in top_candidates],
            'individual_rankings': all_rankings,
            'contributions': {qid: contributions[qid] for qid in top_candidates},
        }
