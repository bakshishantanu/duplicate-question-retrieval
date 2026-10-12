"""Sentence Transformer (SBERT) dense semantic retrieval."""
import logging
from typing import List, Tuple, Optional, Set
import numpy as np
from src.config import SBERT_MODEL_NAME
from src.preprocessing import preprocess_for_sbert
from src.retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class SBERTRetriever(BaseRetriever):
    """Dense semantic retrieval using Sentence Transformers."""

    def __init__(self, model_name: str = SBERT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_ids: Optional[np.ndarray] = None

    def _load_model(self):
        """Lazy-load the SBERT model with robust device fallback."""
        if self.model is None:
            logger.info(f"Loading SBERT model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            import torch

            target_device = self.device
            if target_device is None:
                target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

            try:
                logger.info(f"Using device for SBERT: {target_device}")
                self.model = SentenceTransformer(self.model_name, device=target_device)
            except Exception as e:
                logger.warning(f"Failed to load SBERT on {target_device} ({e}), falling back to cpu")
                self.model = SentenceTransformer(self.model_name, device='cpu')
            logger.info("SBERT model loaded successfully")

    def build_index(self, corpus_df, batch_size: int = 256) -> None:
        """Encode all corpus questions and store normalized embeddings.

        Args:
            corpus_df: DataFrame with 'qid' and 'question_text' columns.
            batch_size: Batch size for model encoding.
        """
        self._load_model()
        self.corpus_ids = np.array(corpus_df['qid'].values, dtype=int)
        texts = [preprocess_for_sbert(t) for t in corpus_df['question_text'].values]

        logger.info(f"Encoding {len(texts)} questions with SBERT ({self.model_name})...")
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize so dot product equals cosine similarity
        ).astype(np.float32)

        logger.info(f"Generated embeddings shape: {self.corpus_embeddings.shape}")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_qids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents by semantic cosine similarity."""
        if self.corpus_embeddings is None or self.corpus_ids is None:
            raise ValueError("Corpus embeddings have not been built or loaded.")

        self._load_model()
        query_text = preprocess_for_sbert(query)
        if not query_text:
            return []

        # Encode query to L2-normalized vector
        try:
            query_embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        except Exception as e:
            logger.warning(f"Error during GPU encoding ({e}). Falling back to CPU...")
            self.model = self.model.to('cpu')
            query_embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

        # Dot product with normalized corpus embeddings = cosine similarity
        similarities = np.dot(self.corpus_embeddings, query_embedding.T).flatten()

        fetch_k = top_k + (len(exclude_qids) if exclude_qids else 0) + 10
        if fetch_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            candidate_indices = np.argpartition(similarities, -fetch_k)[-fetch_k:]
            sorted_candidates = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
            top_indices = sorted_candidates

        results = []
        for idx in top_indices:
            qid = int(self.corpus_ids[idx])
            if exclude_qids and qid in exclude_qids:
                continue
            score = float(similarities[idx])
            results.append((qid, score))
            if len(results) >= top_k:
                break

        return results

    def save_index(self, embeddings_path: str) -> None:
        """Save dense embeddings to .npy file."""
        np.save(embeddings_path, self.corpus_embeddings)
        logger.info(f"Saved SBERT embeddings to {embeddings_path}")

    def load_index(self, embeddings_path: str, corpus_ids: Optional[np.ndarray] = None) -> None:
        """Load embeddings from .npy file."""
        self.corpus_embeddings = np.load(embeddings_path)
        if corpus_ids is not None:
            self.corpus_ids = np.array(corpus_ids, dtype=int)
        logger.info(f"Loaded SBERT embeddings: {self.corpus_embeddings.shape}")
