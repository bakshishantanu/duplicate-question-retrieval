"""Unit tests for corpus construction and splitting."""
import pytest
import pandas as pd
from src.corpus import build_corpus, build_duplicate_map
from src.splitter import split_by_components, get_eval_queries


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [0, 1, 2, 3],
        'qid1': [1, 3, 5, 2],
        'qid2': [2, 4, 6, 7],
        'question1': ['Q1 text', 'Q3 text', 'Q5 text', 'Q2 duplicate of 7 text'],
        'question2': ['Q2 text', 'Q4 text', 'Q6 text', 'Q7 text'],
        'is_duplicate': [1, 0, 1, 1],
    })


def test_corpus_unique_count(sample_df):
    corpus = build_corpus(sample_df)
    assert len(corpus) == 7  # qid 1, 2, 3, 4, 5, 6, 7
    assert corpus['qid'].is_unique


def test_duplicate_map_transitive_closure(sample_df):
    # (1, 2) is dup, (2, 7) is dup -> 1, 2, 7 must be mutual duplicates
    dm = build_duplicate_map(sample_df)
    assert 2 in dm[1]
    assert 7 in dm[1]
    assert 1 in dm[2]
    assert 7 in dm[2]
    assert 1 in dm[7]
    assert 2 in dm[7]
    assert 3 not in dm
    assert 4 not in dm


def test_splits_no_overlap(sample_df):
    corpus = build_corpus(sample_df)
    dm = build_duplicate_map(sample_df)
    splits = split_by_components(corpus, dm, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

    all_qids = set()
    for qids in splits.values():
        all_qids.update(qids)
    assert all_qids == set(corpus['qid'])

    assert len(set(splits['train']) & set(splits['val'])) == 0
    assert len(set(splits['train']) & set(splits['test'])) == 0
    assert len(set(splits['val']) & set(splits['test'])) == 0


def test_get_eval_queries(sample_df):
    corpus = build_corpus(sample_df)
    dm = build_duplicate_map(sample_df)
    splits = split_by_components(corpus, dm, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    queries = get_eval_queries(splits, dm, split_name='test')
    for qid, rel_set in queries:
        assert qid not in rel_set
        assert len(rel_set) >= 1
