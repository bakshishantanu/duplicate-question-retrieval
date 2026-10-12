"""Download and verify the QQP dataset."""
import os
import sys
import urllib.request
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RAW_DATA_PATH
from src.data_loader import load_qqp

OFFICIAL_QQP_URL = "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset():
    """Download the QQP dataset from the official Quora CDN."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_DATA_PATH):
        print(f"Dataset already exists at {RAW_DATA_PATH}")
        return

    print(f"Downloading QQP dataset from {OFFICIAL_QQP_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(OFFICIAL_QQP_URL, headers=headers)

    try:
        with urllib.request.urlopen(req) as response, open(RAW_DATA_PATH, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading QQP") as pbar:
                while True:
                    buffer = response.read(1024 * 64)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    pbar.update(len(buffer))
        print(f"\nSuccessfully downloaded dataset to {RAW_DATA_PATH}")
    except Exception as e:
        if os.path.exists(RAW_DATA_PATH):
            os.remove(RAW_DATA_PATH)
        print(f"Automatic download failed: {e}")
        print("\nPlease manually download 'quora_duplicate_questions.tsv' from:")
        print("  https://www.kaggle.com/datasets/quora/question-pairs-dataset")
        print(f"and place it in: {DATA_DIR}")
        sys.exit(1)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        download_dataset()

    print("\nValidating dataset schema and statistics...")
    df = load_qqp()
    print("\nDataset loaded and cleaned successfully!")
    print(f"  Total valid pairs: {len(df)}")
    print(f"  Unique qid1 values: {df['qid1'].nunique()}")
    print(f"  Unique qid2 values: {df['qid2'].nunique()}")
    print(f"  Duplicate pairs (label=1): {df['is_duplicate'].sum()}")
    print(f"  Non-duplicate pairs (label=0): {len(df) - df['is_duplicate'].sum()}")
    print("\nSample rows:")
    print(df[['id', 'qid1', 'qid2', 'is_duplicate', 'question1', 'question2']].head())


if __name__ == "__main__":
    main()
