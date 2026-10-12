"""Build unique-question corpus and duplicate adjacency map."""
import logging
from collections import defaultdict
from typing import Dict, Set
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique questions from question pairs.

    Args:
        df: Cleaned QQP DataFrame with qid1, question1, qid2, question2 columns.

    Returns:
        DataFrame with columns [qid, question_text], one row per unique question.
    """
    logger.info("Extracting unique questions from pair DataFrame...")
    q1 = df[['qid1', 'question1']].rename(columns={'qid1': 'qid', 'question1': 'question_text'})
    q2 = df[['qid2', 'question2']].rename(columns={'qid2': 'qid', 'question2': 'question_text'})

    all_questions = pd.concat([q1, q2], ignore_index=True)

    # Deduplicate by qid - keep first occurrence
    corpus_df = all_questions.drop_duplicates(subset='qid', keep='first').reset_index(drop=True)
    corpus_df['qid'] = corpus_df['qid'].astype(int)

    logger.info(f"Built corpus: {len(corpus_df)} unique questions from {len(df)} pairs")
    return corpus_df


def build_duplicate_map(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """Build duplicate adjacency map with transitive closure.

    Args:
        df: QQP DataFrame with qid1, qid2, is_duplicate columns.

    Returns:
        Dictionary mapping each qid to the set of all qids that are
        (transitively) duplicate to it.
    """
    logger.info("Building duplicate graph with transitive closure...")
    G = nx.Graph()
    dup_pairs = df[df['is_duplicate'] == 1]

    for _, row in dup_pairs.iterrows():
        G.add_edge(int(row['qid1']), int(row['qid2']))

    logger.info(f"Duplicate graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Transitive closure via connected components
    duplicate_map: Dict[int, Set[int]] = defaultdict(set)
    components = list(nx.connected_components(G))

    for component in components:
        comp_set = set(component)
        for qid in comp_set:
            # A question's duplicates are all other questions in its component
            duplicate_map[qid] = comp_set - {qid}

    logger.info(f"Connected components: {len(components)}")
    if components:
        logger.info(f"Largest component size: {max(len(c) for c in components)}")
    logger.info(f"Questions with known duplicates: {len(duplicate_map)}")

    return dict(duplicate_map)
