"""Leakage-aware splitting based on connected components."""
import logging
from typing import Dict, Set, List, Tuple
import networkx as nx
import numpy as np
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO

logger = logging.getLogger(__name__)


def split_by_components(
    corpus_df,
    duplicate_map: Dict[int, Set[int]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """Split corpus into train/val/test by assigning entire connected
    components to a single split to prevent leakage.

    Args:
        corpus_df: DataFrame with 'qid' column.
        duplicate_map: {qid: set of duplicate qids}.
        train_ratio, val_ratio, test_ratio: Split proportions.
        random_seed: For reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to an
        array of qids belonging to that split.
    """
    rng = np.random.RandomState(random_seed)

    # Build graph for component detection
    G = nx.Graph()
    all_qids = set(corpus_df['qid'].values)
    G.add_nodes_from(all_qids)

    for qid, dups in duplicate_map.items():
        for dup_qid in dups:
            if dup_qid in all_qids:
                G.add_edge(qid, dup_qid)

    components = list(nx.connected_components(G))
    rng.shuffle(components)

    total_questions = len(corpus_df)
    train_target = int(total_questions * train_ratio)
    val_target = int(total_questions * (train_ratio + val_ratio))

    train_qids, val_qids, test_qids = [], [], []
    cumulative = 0

    for component in components:
        cumulative += len(component)
        if cumulative <= train_target:
            train_qids.extend(component)
        elif cumulative <= val_target:
            val_qids.extend(component)
        else:
            test_qids.extend(component)

    splits = {
        'train': np.array(sorted(train_qids), dtype=int),
        'val': np.array(sorted(val_qids), dtype=int),
        'test': np.array(sorted(test_qids), dtype=int),
    }

    logger.info(
        f"Split sizes - Train: {len(splits['train'])}, "
        f"Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    # Verify no overlap across splits
    assert len(set(splits['train']) & set(splits['val'])) == 0, "Train-Val overlap detected"
    assert len(set(splits['train']) & set(splits['test'])) == 0, "Train-Test overlap detected"
    assert len(set(splits['val']) & set(splits['test'])) == 0, "Val-Test overlap detected"

    return splits


def get_eval_queries(
    splits: Dict[str, np.ndarray],
    duplicate_map: Dict[int, Set[int]],
    split_name: str = 'test',
    min_duplicates: int = 1
) -> List[Tuple[int, Set[int]]]:
    """Get evaluation queries: questions that have at least one known
    duplicate in the corpus.

    Args:
        splits: Output from split_by_components.
        duplicate_map: {qid: set of duplicate qids}.
        split_name: Which split to draw queries from ('test' or 'val').
        min_duplicates: Minimum number of known duplicates required.

    Returns:
        List of (query_qid, set_of_relevant_qids) tuples.
    """
    corpus_qids = set()
    for name, qids in splits.items():
        corpus_qids.update(qids)

    eval_queries = []
    for qid in splits[split_name]:
        if qid in duplicate_map:
            # Relevant docs = known duplicates that are in the corpus
            relevant = (duplicate_map[qid] & corpus_qids) - {qid}
            if len(relevant) >= min_duplicates:
                eval_queries.append((qid, relevant))

    logger.info(f"Evaluation queries from '{split_name}' split: {len(eval_queries)}")
    return eval_queries
