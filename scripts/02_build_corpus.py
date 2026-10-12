"""Build corpus, duplicate map with transitive closure, and leakage-aware splits."""
import os
import sys
import logging
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.config import (
    ARTIFACTS_DIR, CORPUS_PATH, DUPLICATE_MAP_PATH, SPLITS_PATH, CORPUS_IDS_PATH
)
from src.data_loader import load_qqp
from src.corpus import build_corpus, build_duplicate_map
from src.splitter import split_by_components
from src.preprocessing import preprocess_text


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Step 1: Loading raw dataset...")
    df = load_qqp()

    print("\nStep 2: Building unique question corpus...")
    corpus_df = build_corpus(df)

    print("\nStep 3: Preprocessing corpus text for lexical search (TF-IDF & BM25)...")
    corpus_df['preprocessed_text'] = corpus_df['question_text'].apply(
        lambda t: preprocess_text(t, remove_stopwords=True, apply_stemming=True)
    )

    print("\nStep 4: Building duplicate graph & transitive closure...")
    duplicate_map = build_duplicate_map(df)

    print("\nStep 5: Performing leakage-aware connected-component split...")
    splits = split_by_components(corpus_df, duplicate_map)

    print("\nStep 6: Saving serialized artifacts...")
    joblib.dump(corpus_df, CORPUS_PATH)
    joblib.dump(duplicate_map, DUPLICATE_MAP_PATH)
    joblib.dump(splits, SPLITS_PATH)
    joblib.dump(corpus_df['qid'].values, CORPUS_IDS_PATH)

    print(f"\nArtifacts successfully written to {ARTIFACTS_DIR}:")
    print(f"  Corpus: {len(corpus_df)} questions -> {CORPUS_PATH}")
    print(f"  Duplicate map: {len(duplicate_map)} entries -> {DUPLICATE_MAP_PATH}")
    print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])} -> {SPLITS_PATH}")


if __name__ == "__main__":
    main()
