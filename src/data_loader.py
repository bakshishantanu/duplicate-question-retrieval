"""Load and validate the QQP dataset."""
import logging
import hashlib
import pandas as pd
from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {'id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'}


def _generate_stable_qid(text: str) -> int:
    """Generate a stable positive 32-bit integer ID from question text."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)


def load_qqp(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load QQP dataset from TSV file.

    Returns:
        Cleaned DataFrame with validated schema and stable IDs.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        ValueError: If required question/target columns are missing.
    """
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='skip')

    # Basic column check
    if 'question1' not in df.columns or 'question2' not in df.columns or 'is_duplicate' not in df.columns:
        raise ValueError(f"Required columns (question1, question2, is_duplicate) missing. Found: {list(df.columns)}")

    # Synthesize qid1/qid2/id if missing (e.g. GLUE version fallback)
    if 'qid1' not in df.columns:
        logger.warning("qid1 column missing; synthesizing stable IDs from question1 text hashes.")
        df['qid1'] = df['question1'].astype(str).apply(_generate_stable_qid)
    if 'qid2' not in df.columns:
        logger.warning("qid2 column missing; synthesizing stable IDs from question2 text hashes.")
        df['qid2'] = df['question2'].astype(str).apply(_generate_stable_qid)
    if 'id' not in df.columns:
        df['id'] = range(len(df))

    logger.info(f"Loaded {len(df)} rows")

    # Report statistics before cleaning
    null_counts = df[['question1', 'question2', 'is_duplicate']].isnull().sum()
    logger.info(f"Null counts:\n{null_counts}")

    # Drop rows with null questions
    initial_count = len(df)
    df = df.dropna(subset=['question1', 'question2']).copy()
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with null question text")

    # Strip whitespace and convert to string
    df['question1'] = df['question1'].astype(str).str.strip()
    df['question2'] = df['question2'].astype(str).str.strip()

    # Drop any rows where questions became empty after stripping
    df = df[(df['question1'].str.len() > 0) & (df['question2'].str.len() > 0)].copy()

    # Ensure integer types
    df['id'] = df['id'].astype(int)
    df['qid1'] = df['qid1'].astype(int)
    df['qid2'] = df['qid2'].astype(int)
    df['is_duplicate'] = df['is_duplicate'].astype(int)

    # Verify is_duplicate is binary
    assert df['is_duplicate'].isin([0, 1]).all(), "is_duplicate contains non-binary values"

    # Remove exact self-duplicate pairs (trivial matches)
    self_dupes = df[df['question1'] == df['question2']]
    if len(self_dupes) > 0:
        logger.warning(f"Removing {len(self_dupes)} self-duplicate pairs")
        df = df[df['question1'] != df['question2']].copy()

    # Log summary statistics
    n_dup = int(df['is_duplicate'].sum())
    n_non_dup = len(df) - n_dup
    logger.info(f"After cleaning: {len(df)} pairs ({n_dup} duplicate, {n_non_dup} non-duplicate)")
    logger.info(f"Duplicate ratio: {n_dup / len(df):.3f}")

    return df
