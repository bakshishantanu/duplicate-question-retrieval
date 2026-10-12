"""Abstract base class for all retrievers."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set


class BaseRetriever(ABC):
    """Base interface for duplicate question retrieval methods."""

    @abstractmethod
    def build_index(self, corpus_df) -> None:
        """Build the retrieval index from the corpus DataFrame."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_qids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for a query.

        Args:
            query: The search query text.
            top_k: Number of ranked results to return.
            exclude_qids: Optional set of question IDs to exclude from results.

        Returns:
            List of (qid, score) tuples, sorted by descending score.
        """
        pass
