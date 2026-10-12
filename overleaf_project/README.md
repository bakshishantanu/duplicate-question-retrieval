# Overleaf LaTeX Project: Duplicate Question Retrieval Engine

This directory contains the complete, compilable, publication-ready Overleaf project for **Project 8: Duplicate Question Retrieval** (Quora Question Pairs dataset).

---

## 1. Quick Start: Uploading to Overleaf

1. **Create a ZIP Archive:**
   Select all files and folders inside `overleaf_project/`:
   - `main.tex`
   - `references.bib`
   - `figures/` (contains all 4 high-resolution PNG plots)
   - Compress them into a single file, e.g. `qqp_ir_report.zip`.

2. **Upload to Overleaf:**
   - Go to [Overleaf](https://www.overleaf.com/).
   - Click **New Project** $\to$ **Upload Project**.
   - Select your `qqp_ir_report.zip` file.

3. **Compiler Settings:**
   - Overleaf will automatically detect `main.tex` as the root document.
   - Compiler: **pdfLaTeX** (default).
   - TeX Live version: **2024** (or latest default).
   - Click **Recompile** to generate the final PDF.

---

## 2. Project Directory Structure

```
overleaf_project/
├── main.tex                 # Complete 7-page technical report
├── references.bib           # 10 verified academic references (IEEE format)
├── README.md                # Overleaf compilation guide and artifact inventory
└── figures/
    ├── label_distribution.png   # Class distribution bar chart (clean QQP pairs)
    ├── question_lengths.png     # Question token length histogram (N = 537,905)
    ├── precision_recall.png     # Cutoff-based PR curve across all 4 methods
    └── quality_latency.png      # MAP@10 vs median query latency trade-off frontier
```

---

## 3. Editable Placeholders to Complete Before Submission

In `main.tex`, update the author metadata block (lines 35–41):

```latex
\author{
  \textbf{Student Name} \\                      % <-- Your full name
  Registration Number: \texttt{[REG-XXXXXX]} \\  % <-- Your student / roll number
  Department of Computer Science and Engineering \\
  \texttt{student@institution.edu}              % <-- Your university email
}
```

No empirical results or system figures are placeholders: all metrics, counts, formulas, and curves reflect the real experiments run on the 537,905-question corpus.

---

## 4. Key Sections & Content Summary

| Section | Focus & Key Contents |
| :--- | :--- |
| **Title \& Abstract** | 215-word abstract detailing QQP retrieval framing, 537k corpus, four methods, empirical findings, and open-source artifacts. |
| **1. Introduction** | Problem motivation in CQA platforms, lexical vs. semantic trade-offs, vocabulary mismatch, and RRF rank fusion rationale. |
| **2. About Dataset** | Exact QQP counts (404,268 clean pairs, 537,905 unique questions), transitive closure graph closure ($N=60,449$ components), component-based leakage prevention, label distribution figure, and question length histogram. |
| **3. Methodology** | High-level TikZ architecture diagram (offline indexing vs online querying), conservative lexical preprocessing, TF-IDF, BM25, SBERT bi-encoder, and RRF ($k=60$) mathematical formulations. |
| **4. Experimental Analysis** | 1,000 unseen test query benchmark, mathematical metric definitions, main comparison table, cutoff Precision-Recall curve, MAP@10 vs. latency trade-off scatter plot, qualitative case studies, and error analysis. |
| **5. Conclusion** | Empirical findings summary, production viability, and justified future work (cross-encoders, ANN/FAISS indexing, metric learning). |
| **References** | Verified citations for BM25, SBERT, RRF, QQP, TF-IDF, and foundational IR literature. |

---

## 5. Summary of Empirical Results Included in the Report

Evaluated on **1,000 unseen test queries** against the $N = 537,905$ question corpus:

| Retrieval Method | MAP@5 | MAP@10 | nDCG@5 | nDCG@10 | P@10 | R@10 | p50 Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (Sparse Cosine)** | 0.6047 | 0.5955 | 0.6699 | 0.6800 | 0.2231 | 0.6977 | 100.0 ms | 108.7 ms |
| **BM25 (Okapi)** | 0.6039 | 0.5965 | 0.6688 | 0.6811 | 0.2247 | 0.7051 | 795.0 ms | 870.4 ms |
| **SBERT (`all-MiniLM-L6-v2`)** | **0.7078** | **0.7036** | **0.7653** | **0.7774** | **0.2716** | **0.7786** | **44.0 ms** | 118.6 ms |
| **Hybrid RRF ($k=60$)** | 0.6765 | 0.6657 | 0.7386 | 0.7469 | 0.2538 | 0.7538 | 981.5 ms | 1,048.5 ms |

---

## 6. How Figures Were Generated

All figures in `figures/` are generated directly from the real experimental data:
- `figures/label_distribution.png`: Generated from cleaned QQP class counts (149,249 positive, 255,019 negative).
- `figures/question_lengths.png`: Computed from the 537,905 question text lengths in `artifacts/corpus.pkl`.
- `figures/precision_recall.png`: Evaluated across cutoffs $K \in [1, 30]$ on test queries.
- `figures/quality_latency.png`: Plotted from `artifacts/eval_results.json`.

To regenerate the figures at any time:
```bash
python scripts/generate_report_figures.py
```
