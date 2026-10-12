import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH, CORPUS_PATH, EVAL_RESULTS_PATH, PR_CURVE_PATH, ARTIFACTS_DIR
)

# Output directory for Overleaf project figures
OUTPUT_DIR = os.path.join("overleaf_project", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set clean publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def generate_label_distribution():
    """Figure 1: Duplicate label distribution in cleaned QQP pairs."""
    pos_count = 149249
    neg_count = 255019
    total = pos_count + neg_count

    pos_pct = (pos_count / total) * 100
    neg_pct = (neg_count / total) * 100

    categories = ['Non-Duplicate\n(is_duplicate = 0)', 'Duplicate\n(is_duplicate = 1)']
    counts = [neg_count, pos_count]
    percentages = [neg_pct, pos_pct]
    colors = ['#4A90E2', '#50E3C2']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor='#333333', linewidth=1.2)

    ax.set_ylabel('Number of Question Pairs')
    ax.set_title('Quora Question Pairs: Class Distribution (Cleaned Corpus)')
    ax.set_ylim(0, 300000)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.annotate(f'{count:,}\n({pct:.2f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_question_lengths():
    """Figure 2: Distribution of question token lengths in the corpus."""
    corpus_df = joblib.load(CORPUS_PATH)
    lengths = corpus_df['question_text'].str.split().str.len()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(lengths, bins=range(1, 45), color='#34495E', edgecolor='white', linewidth=0.8, density=True)

    median_len = lengths.median()
    mean_len = lengths.mean()
    p95_len = lengths.quantile(0.95)

    ax.axvline(median_len, color='#E74C3C', linestyle='--', linewidth=1.5, label=f'Median: {median_len:.0f} tokens')
    ax.axvline(mean_len, color='#F39C12', linestyle=':', linewidth=1.5, label=f'Mean: {mean_len:.1f} tokens')
    ax.axvline(p95_len, color='#2ECC71', linestyle='-.', linewidth=1.5, label=f'95th %ile: {p95_len:.0f} tokens')

    ax.set_xlabel('Question Length (Whitespace Tokens)')
    ax.set_ylabel('Density')
    ax.set_title('Corpus Question Length Distribution (N = 537,905)')
    ax.set_xlim(1, 40)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "question_lengths.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_precision_recall_plot():
    """Figure 3: High-resolution Precision-Recall curve across 4 methods."""
    # Copy existing verified artifact if available, or load and re-render
    import shutil
    src_pr = PR_CURVE_PATH
    out_path = os.path.join(OUTPUT_DIR, "precision_recall.png")
    if os.path.exists(src_pr):
        shutil.copyfile(src_pr, out_path)
        print(f"Copied PR curve: {out_path}")
    else:
        print("Warning: pr_curve.png not found!")


def generate_quality_latency_plot():
    """Figure 4: MAP@10 vs Median Latency trade-off frontier."""
    with open(EVAL_RESULTS_PATH, 'r') as f:
        data = json.load(f)

    methods = ['TF-IDF', 'BM25', 'SBERT', 'Hybrid RRF']
    colors = {'TF-IDF': '#3498DB', 'BM25': '#E67E22', 'SBERT': '#2ECC71', 'Hybrid RRF': '#9B59B6'}
    markers = {'TF-IDF': 'o', 'BM25': 's', 'SBERT': '^', 'Hybrid RRF': 'D'}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for m in methods:
        p50 = data[m]['p50_latency_ms']
        map10 = data[m]['MAP@10']
        ax.scatter(p50, map10, color=colors[m], marker=markers[m], s=120, label=m, edgecolor='black', zorder=5)

        # Annotate
        offset_x = 15 if m != 'Hybrid RRF' else -110
        offset_y = 0.004 if m != 'BM25' else -0.008
        ax.annotate(f"{m}\n({p50:.1f}ms, {map10:.4f})",
                    xy=(p50, map10),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    fontsize=9, fontweight='medium')

    ax.set_xlabel('Median Query Latency (ms)')
    ax.set_ylabel('Mean Average Precision (MAP@10)')
    ax.set_title('Retrieval Quality vs. Query Latency Trade-off')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0.57, 0.73)
    ax.set_xlim(0, 1150)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "quality_latency.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_label_distribution()
    generate_question_lengths()
    generate_precision_recall_plot()
    generate_quality_latency_plot()
    print("All report figures generated successfully.")
