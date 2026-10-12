"""TF-IDF retrieval with sparse cosine similarity."""
import logging
from typing import List, Tuple, Optional, Set
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text
from src.retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class TFIDFRetriever(BaseRetriever):
    """TF-IDF sparse retrieval using cosine similarity."""

    def __init__(self, max_features: int = 50000, sublinear_tf: bool = True):
        """
        Args:
            max_features: Maximum vocabulary size.
            sublinear_tf: Use 1 + log(TF) instead of raw TF.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self.tfidf_matrix = None
        self.corpus_ids: Optional[np.ndarray] = None

    def build_index(self, corpus_df) -> None:
        """Build TF-IDF matrix from corpus text.

        Args:
            corpus_df: DataFrame with 'qid' and 'preprocessed_text' or 'question_text'.
        """
        logger.info("Building TF-IDF index...")
        self.corpus_ids = np.array(corpus_df['qid'].values, dtype=int)

        if 'preprocessed_text' in corpus_df.columns:
            texts = corpus_df['preprocessed_text'].values
        else:
            texts = [
                preprocess_text(t, remove_stopwords=True, apply_stemming=True)
                for t in corpus_df['question_text'].values
            ]

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_qids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents by cosine similarity to query."""
        if self.tfidf_matrix is None or self.corpus_ids is None:
            raise ValueError("Index has not been built or loaded.")

        processed_query = preprocess_text(query, remove_stopwords=True, apply_stemming=True)
        if not processed_query:
            return []

        query_vector = self.vectorizer.transform([processed_query])
        # Vectorizer normalizes rows by L2 norm, so dot product with matrix transpose is cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        fetch_k = top_k + (len(exclude_qids) if exclude_qids else 0) + 10
        if fetch_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for fast top-k selection on large corpus
            candidate_indices = np.argpartition(similarities, -fetch_k)[-fetch_k:]
            sorted_candidates = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
            top_indices = sorted_candidates

        results = []
        for idx in top_indices:
            qid = int(self.corpus_ids[idx])
            if exclude_qids and qid in exclude_qids:
                continue
            score = float(similarities[idx])
            if score > 0:
                results.append((qid, score))
            if len(results) >= top_k:
                break

        return results

    def save_index(self, vectorizer_path: str, matrix_path: str) -> None:
        """Persist vectorizer and matrix to disk."""
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.tfidf_matrix, matrix_path)
        logger.info(f"Saved TF-IDF index to {vectorizer_path}, {matrix_path}")

    def load_index(
        self,
        vectorizer_path: str,
        matrix_path: str,
        corpus_ids: Optional[np.ndarray] = None
    ) -> None:
        """Load previously saved vectorizer and matrix."""
        self.vectorizer = joblib.load(vectorizer_path)
        self.tfidf_matrix = joblib.load(matrix_path)
        if corpus_ids is not None:
            self.corpus_ids = np.array(corpus_ids, dtype=int)
        logger.info(f"Loaded TF-IDF index: {self.tfidf_matrix.shape}")
