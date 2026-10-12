"""BM25 Okapi retrieval using rank_bm25."""
import logging
from typing import List, Tuple, Optional, Set
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from src.config import BM25_K1, BM25_B
from src.preprocessing import preprocess_text
from src.retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 Okapi lexical retrieval."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: Optional[np.ndarray] = None

    def build_index(self, corpus_df) -> None:
        """Build BM25 index from tokenized corpus.

        Args:
            corpus_df: DataFrame with 'qid' and either 'preprocessed_tokens' or 'question_text'.
        """
        logger.info("Building BM25 index...")
        self.corpus_ids = np.array(corpus_df['qid'].values, dtype=int)

        if 'preprocessed_tokens' in corpus_df.columns:
            tokenized_corpus = corpus_df['preprocessed_tokens'].tolist()
        else:
            tokenized_corpus = [
                preprocess_text(text, return_tokens=True)
                for text in corpus_df['question_text'].values
            ]

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")
        logger.info(f"Average document length: {self.bm25.avgdl:.2f} tokens")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_qids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents by BM25 score."""
        if self.bm25 is None or self.corpus_ids is None:
            raise ValueError("BM25 index has not been built or loaded.")

        tokenized_query = preprocess_text(query, return_tokens=True)
        if not tokenized_query:
            return []

        scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)

        fetch_k = top_k + (len(exclude_qids) if exclude_qids else 0) + 10
        if fetch_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(scores, -fetch_k)[-fetch_k:]
            sorted_candidates = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
            top_indices = sorted_candidates

        results = []
        for idx in top_indices:
            qid = int(self.corpus_ids[idx])
            if exclude_qids and qid in exclude_qids:
                continue
            score = float(scores[idx])
            if score > 0:
                results.append((qid, score))
            if len(results) >= top_k:
                break

        return results

    def save_index(self, path: str) -> None:
        """Save BM25 index state to disk."""
        joblib.dump({
            'bm25': self.bm25,
            'corpus_ids': self.corpus_ids,
            'k1': self.k1,
            'b': self.b,
        }, path)
        logger.info(f"Saved BM25 index to {path}")

    def load_index(self, path: str) -> None:
        """Load BM25 index from disk."""
        data = joblib.load(path)
        self.bm25 = data['bm25']
        self.corpus_ids = np.array(data['corpus_ids'], dtype=int)
        self.k1 = data.get('k1', self.k1)
        self.b = data.get('b', self.b)
        logger.info(f"Loaded BM25 index ({len(self.corpus_ids)} documents)")
