# Claude Prompt: Overleaf-Ready LaTeX Report — Duplicate Question Retrieval

## How to use this file
Give Claude this entire prompt and attach `IR_report formatt.docx`. If available, also attach the implementation plan and actual evaluation outputs. The DOCX is a **format/structure reference only**. Its example uses Cranfield data and example metrics; those must not be copied into the QQP report.

---

# MASTER PROMPT FOR CLAUDE

Act as an academic technical writer, Information Retrieval researcher, and LaTeX engineer. Generate a complete, polished, compilable Overleaf project for **Project 8: Duplicate Question Retrieval**. Return the actual source files, not just advice or an outline.

## 1. Required deliverables

Provide:
- `main.tex`: complete report.
- `references.bib`: verified BibTeX references.
- Any additional `.tex` files only if modularization helps.
- Any external figure files actually required, or clearly list those that must be generated from project outputs.
- A short README with Overleaf upload and compilation instructions.
- A final list of any remaining placeholders or empirical artifacts that must be supplied.

Prefer a self-contained `main.tex` with TikZ/PGFPlots diagrams where practical. Use actual Python-generated plots when those plots depend on experimental results.

## 2. Authoritative problem statement — do not change scope

### Project title
**Project 8: Duplicate Question Retrieval**

### Problem
Build a semantic duplicate-question retrieval system that accepts a natural-language query, searches a corpus of existing questions, retrieves candidate questions, ranks them by relevance, and displays the top results in a usable search interface.

Equivalent questions may use different words and sentence structures. Questions that share keywords may nevertheless have different intent. Exact string matching and keyword overlap alone are insufficient.

The system combines lexical and semantic retrieval and evaluates whether fusing their rankings improves retrieval quality.

**The main deliverable is a search-and-ranking engine, not merely a binary classifier that predicts whether two supplied questions are duplicates.**

### Dataset
Use the Quora Question Pairs (QQP) dataset. It contains question pairs with a binary `is_duplicate` label. The commonly distributed training split has approximately 404,000 pairs, but the exact version, row count, schema, and statistics must be verified from the data actually used.

Possible fields include `id`, `qid1`, `qid2`, `question1`, `question2`, and `is_duplicate`; include only fields that actually exist.

The application must transform pair data into a searchable corpus of unique questions derived from both question columns. Preserve original question text and stable IDs. Use positive labels to construct known duplicate relationships. **Do not assume an unobserved pair is a confirmed negative.** Explain the limits of incomplete relevance judgments.

### Mandatory methods and deliverables
1. Text preprocessing for lexical retrieval: cleaning, normalization, tokenization, selective stop-word handling, and stemming if justified.
2. TF-IDF retrieval, typically with cosine similarity.
3. BM25 lexical retrieval.
4. Dense semantic retrieval using a pretrained Sentence Transformer; consider `sentence-transformers/all-MiniLM-L6-v2` as the initial model if the implementation uses it.
5. Reciprocal Rank Fusion (RRF) to combine TF-IDF, BM25, and Sentence Transformer rankings.
6. Functional Streamlit search interface.
7. Evaluation with MAP, nDCG@K, Precision@K, Recall@K, and query latency.
8. Model-comparison table and Precision–Recall graph.
9. Reproducible preprocessing, indexing, evaluation, and run instructions.

RRF formula:

\[
\operatorname{RRF}(d)=\sum_{m=1}^{M}\frac{1}{k+r_m(d)}
\]

Here, \(d\) is a candidate question, \(r_m(d)\) is its one-based rank in retrieval system \(m\), \(M\) is the number of systems, and \(k\) is a smoothing constant, commonly initialized at 60. A candidate absent from a ranking contributes zero from that ranking. Explain why RRF fuses ranks rather than directly adding incompatible raw scores.

## 3. Use the attached DOCX as the report structure reference

Follow its high-level organization and approximate page allocation:

1. Title, student details, abstract, keywords.
2. **1. Introduction** — approximately 1 page.
3. **2. About Dataset** — approximately 1 page, including a graph/table and example.
4. **3. Methodology** — approximately 2 pages.
5. **4. Experimental Setup and Performance Analysis** — approximately 2 pages.
6. **5. Conclusion** — approximately 1 page.
7. References.

The DOCX contains Cranfield-specific material and illustrative performance numbers. **Do not reuse Cranfield data, Cranfield results, or its claimed improvements.** Preserve the structure, not the example's empirical content.

Use editable author placeholders:
- Student name(s)
- Registration number(s)
- Email address(es)
- Course/instructor/institution only if supplied or required

Do not invent personal or institutional details.

Suggested title: **Comparative Performance Evaluation of Lexical, Semantic, and Hybrid Retrieval for Duplicate Question Retrieval**.

### Abstract
Write about 150–250 words covering the retrieval problem, QQP dataset, four methods, evaluation protocol, search interface, and contributions. Report empirical findings only when actual results exist. If they are unavailable, use a clearly identified draft placeholder rather than inventing results.

### Keywords
Information Retrieval; Duplicate Question Retrieval; Quora Question Pairs; TF-IDF; BM25; Sentence Transformers; Reciprocal Rank Fusion; Semantic Search; MAP; nDCG.

---

# 4. Exact content expected in each section

## Section 1 — Introduction (about 1 page)

Explain:
- Why Information Retrieval matters for search and Q&A systems.
- Lexical versus semantic retrieval.
- TF-IDF and BM25 strengths: efficient keyword matching, useful baselines, and interpretability.
- Lexical limitations: vocabulary mismatch, paraphrases, synonyms, and keyword overlap without equivalent intent.
- Sentence Transformer strengths: dense sentence representations and semantic matching.
- Dense retrieval limitations: related topics can be mistaken for duplicates; negation and subtle intent differences can be missed.
- Why RRF is appropriate: it fuses rankings without requiring raw score calibration.
- Project objectives and contributions.

Explicit objectives:
1. Construct a unique-question corpus from QQP.
2. Implement TF-IDF, BM25, and Sentence Transformer retrieval independently.
3. Combine their rankings with RRF.
4. Evaluate all four methods under a consistent protocol.
5. Build an interactive search interface.
6. Compare retrieval quality and latency.

Do not claim RRF wins before the experiments support that claim.

## Section 2 — About Dataset (about 1 page)

### 2.1 Dataset description
Describe the actual QQP version used. Report measured:
- Pair count before and after cleaning.
- Missing/malformed row counts.
- Duplicate-row count, if checked.
- Number of unique questions in the corpus.
- Positive/negative pair counts and proportions.
- Train/validation/test split sizes.
- Relevant data limitations.

Include a compact schema table. Only include columns present in the actual file.

### 2.2 Pair dataset versus retrieval corpus
Explain why pairs must be converted into unique individual questions. Include a small illustrative table with example pairs and labels, followed by an explanation of the corpus and known duplicate relationships. Label examples as illustrative unless they are explicitly drawn from the dataset.

Explain that missing pair annotations do not prove non-duplicate status.

### 2.3 Dataset visuals
Include:
- **Figure: Duplicate-label distribution** — bar chart of positive and negative labels using actual counts/proportions.
- **Table or figure: Dataset/corpus summary** — pair count, cleaned rows, unique questions, and split sizes. Use a table if a chart would compare incompatible scales.
- **Optional question-length histogram** — only if it helps characterize short-text retrieval.

Do not include decorative charts.

## Section 3 — Methodology (about 2 pages)

### 3.1 Architecture diagram — mandatory
Create a readable, publication-quality diagram using TikZ (preferred) or another Overleaf-compatible technique. It must show offline index building separately from online query handling.

Required nodes:
- QQP pairs.
- Cleaning and unique-question corpus.
- TF-IDF sparse matrix/index.
- BM25 index.
- Precomputed Sentence Transformer embeddings.
- User query and input validation.
- Three independent retrieval paths.
- Candidate ranked lists.
- RRF fusion.
- Final top-K ranking.
- Streamlit results interface.

Suggested conceptual flow:

```text
OFFLINE
QQP pairs -> clean/validate -> unique-question corpus
                         |          |           |
                         v          v           v
                    TF-IDF index  BM25 index  dense embeddings

ONLINE
User query -> validation/preprocessing
                |          |          |
                v          v          v
             TF-IDF      BM25     Sentence Transformer
                |          |          |
                +----------+----------+
                           |
                       RRF fusion
                           |
                   final ranked top-K
                           |
                    Streamlit UI
```

Draw the real figure in TikZ, with labelled arrows, clear separation of offline and online operations, and a descriptive caption. Avoid tiny text or overcrowded boxes.

### 3.2 Preprocessing and corpus construction
Describe missing-value handling, stable IDs, exact repeated-question deduplication, preservation of original text, lexical normalization, tokenization, stop-word handling, optional stemming, and leakage-aware data splitting.

Explain why aggressive lexical preprocessing should not automatically be applied to Sentence Transformer inputs. Negation and question words can carry intent.

State the actual split method and its limitations. Recurring questions and duplicate relationships may leak across a random pair-row split. Do not claim leakage is eliminated unless the implementation's protocol justifies that claim.

### 3.3 TF-IDF retrieval
Explain the vector-space representation, sparse vectors, query vectorization, cosine similarity, candidate ranking, strengths, and weaknesses. Include a formula:

\[
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)
\log\left(\frac{N}{\operatorname{df}(t)}\right)
\]

If scikit-learn uses a smoothed IDF formula, describe the actual configuration/formula rather than pretending this simplified expression is exact.

### 3.4 BM25 retrieval
Explain term frequency, inverse document frequency, term-frequency saturation, length normalization, tokenization consistency, candidate ranking, parameters, and why BM25 differs from TF-IDF. Use the actual implementation and settings.

### 3.5 Sentence Transformer retrieval
Describe the actual model identifier, one-time corpus embedding, embedding persistence, query embedding, cosine similarity, top-K retrieval, advantages for paraphrases, and limitations for subtle intent differences. If `all-MiniLM-L6-v2` was not actually used, do not state that it was.

### 3.6 RRF
Include the formula, define variables, and report the actual smoothing constant, candidate depth, missing-candidate policy, and tie-breaking policy. Explain that RRF does not guarantee an improvement; that is an empirical question.

### 3.7 Technology stack
Include a concise table of the technologies actually used and their roles, potentially Python, Pandas, NumPy, scikit-learn, `rank_bm25`, `sentence-transformers`, Matplotlib, Streamlit, pytest, and local artifact serialization. Do not list unused tools as if implemented. Do not add a vector database or cloud stack without a concrete need.

## Section 4 — Experimental Setup and Performance Analysis (about 2 pages)

### 4.1 Experimental setup
Report the actual:
- Dataset source/version and cleaning rules.
- Corpus-construction approach.
- Split strategy and number of evaluation queries.
- Method for constructing known relevance judgments.
- Models, model versions, and preprocessing.
- Top-K candidate depths and RRF settings.
- Hardware, software, CPU/GPU details.
- Latency measurement procedure.
- Whether loading time is included.
- Reproducibility controls, seeds, and saved configuration.

Explain incomplete relevance judgments. If labels do not exhaustively identify every relevant candidate, describe metrics as performance against known positives, not perfect ground truth.

### 4.2 Required evaluation metrics
Define each metric and ensure its formula matches the actual code.

**Precision@K**

\[
P@K=\frac{\text{relevant results in top }K}{K}
\]

State how the denominator is handled if fewer than K results are returned.

**Recall@K**

\[
R@K=\frac{\text{known relevant results in top }K}
{\text{total known relevant results for the query}}
\]

Explain how queries with no known relevant results are handled in aggregate results.

**Average Precision / MAP**

\[
AP@K=\frac{\sum_{i=1}^{K}P@i\cdot rel_i}{\min(R,K)}
\]

Use this expression only if \(R\) is the number of known relevant candidates and it matches the implementation's convention. Otherwise give the actual convention. Define macro-averaging for MAP and the treatment of queries without positives.

\[
MAP@K=\frac{1}{|Q'|}\sum_{q\in Q'}AP@K(q)
\]

Define \(Q'\) explicitly.

**DCG / nDCG**

For binary relevance, one common formulation is:

\[
DCG@K=\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}
\]

\[
nDCG@K=\frac{DCG@K}{IDCG@K}
\]

Explain ideal ranking and zero-IDCG handling.

**Latency**
Report a defined statistic such as median milliseconds/query and preferably p95 latency. Separate offline index-build time from online query time.

Use the metrics actually implemented. Never present a formula that disagrees with the code.

### 4.3 Model-comparison table — mandatory
Compare TF-IDF, BM25, Sentence Transformer, and Hybrid RRF using the same queries and relevance criteria.

Recommended columns:

| Model | MAP@10 | nDCG@10 | Precision@10 | Recall@10 | Median latency (ms) |
|---|---:|---:|---:|---:|---:|
| TF-IDF | measured | measured | measured | measured | measured |
| BM25 | measured | measured | measured | measured | measured |
| Sentence Transformer | measured | measured | measured | measured | measured |
| Hybrid RRF | measured | measured | measured | measured | measured |

This table is a specification, not a source of values. Do not insert fabricated numbers or leave literal words such as “measured” in a final empirical report. If results are not available, make a clearly marked draft version that compiles and identify the values still needed.

### 4.4 Precision–Recall graph — mandatory
Use actual evaluation outputs. For retrieval, a defensible plot is precision versus recall at cutoffs such as K = 1, 3, 5, 10, 20, and 50. Label it as a cutoff-based retrieval Precision–Recall curve.

Alternatively, use a conventional threshold-based classification Precision–Recall curve only if pairwise duplicate scores and binary labels are evaluated with an explicit threshold sweep. Explain the difference.

Compare the four methods on shared axes if readable. Label axes, legend, and caption. Prefer importing an actual plot generated by Python, such as `figures/precision_recall.png` or `.pdf`.

### 4.5 Optional extra analysis
Only if supported by actual data, include one of:
- MAP@10 or nDCG@10 comparison chart.
- Latency-versus-quality plot.
- Ablation table.
- Qualitative example showing ranked results from each retriever.

Avoid repeating identical information in multiple visuals.

### 4.6 Interpretation
Discuss actual results, including:
- Best method by metric.
- Whether RRF improves ranking quality.
- Quality/latency trade-offs.
- Lexical and semantic failure cases.
- Whether the methods provide complementary signals.
- Limitations and potential biases.

Every comparative claim must be supported by the table or figures. Do not assert consistent RRF superiority unless measured results establish it.

## Section 5 — Conclusion (about 1 page)
Summarize the objective, implemented methods, architecture, actual findings, limitations, and justified future work. Possible future work includes cross-encoder reranking, model fine-tuning, approximate nearest-neighbour indexing, improved relevance judgments, and query-level error analysis—but identify unimplemented items as future work. Do not introduce new results here.

## References
Use verified BibTeX entries and a consistent academic style, preferably IEEE numeric citations unless the supplied course requirements say otherwise. Include relevant sources for:
- BM25.
- Sentence-BERT / Sentence Transformers.
- Reciprocal Rank Fusion.
- QQP dataset source.
- TF-IDF / IR fundamentals where relevant.
- Evaluation methodology or tools where appropriate.

Foundational references to verify before inclusion:
- Robertson and Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” *Foundations and Trends in Information Retrieval*, 2009.
- Reimers and Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” EMNLP-IJCNLP, 2019.
- Cormack, Clarke, and Buettcher, “Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods,” SIGIR, 2009.
- An authoritative QQP dataset source/publication.

Verify titles, authors, year, venue, pages, DOI/URL, and source validity. Do not fabricate references. Cite claims where they appear.

---

# 5. Required figures and tables checklist

## Mandatory figures
1. **System architecture diagram** — offline corpus/index construction and online retrieval, three retrievers, RRF, UI.
2. **QQP duplicate-label distribution** — actual counts or proportions.
3. **Precision–Recall graph** — actual measured results and a clearly defined construction.

## Recommended figures
4. Question-length histogram, if informative.
5. Quality-versus-latency plot, if measured.
6. Qualitative top-result comparison across the four methods, if based on real queries/results.

## Mandatory tables
1. Dataset schema and measured statistics.
2. Technology stack and roles.
3. Experimental configuration.
4. Performance comparison table with MAP@10, nDCG@10, Precision@10, Recall@10, and latency where available.
5. Optional error-analysis table if useful.

Every figure/table needs a unique number, informative caption, `\label{...}`, and an in-text reference. Use actual data. Do not add visuals just for decoration.

---

# 6. LaTeX / Overleaf requirements

## Compiler and packages
Target Overleaf with pdfLaTeX unless there is a compelling reason to use another compiler. Use a small, compatible package set. Potential packages:

- `geometry`
- `graphicx`
- `microtype`
- `amsmath`, `amssymb`
- `booktabs`
- `tabularx`
- `array`
- `caption`
- `xcolor`
- `enumitem`
- `tikz`
- `pgfplots`
- `hyperref`
- `url`
- `cite` or a compatible numeric citation package

Set a supported PGFPlots compatibility version. Avoid package conflicts and unused packages. For an IEEE numeric bibliography, `\bibliographystyle{IEEEtran}` and `\bibliography{references}` are acceptable if the chosen citation setup is compatible.

## Formatting
- Professional undergraduate technical-report layout.
- Consistent section numbering, margins, and typography.
- Equations properly formatted.
- Tables using `booktabs` where appropriate.
- Figures that fit the page width.
- Proper captions, labels, cross-references, and citations.
- No manually hardcoded figure numbers in prose.
- No overflowing tables/diagrams.
- Avoid pdfLaTeX-problematic Unicode symbols; use LaTeX math commands.
- No invented affiliations, logos, or author details.
- Match the DOCX's high-level section organization and approximate lengths.

## Artifact naming
Use consistent names such as:

```text
main.tex
references.bib
README.md
figures/
    label_distribution.png
    question_lengths.png
    precision_recall.png
    quality_latency.png
```

Only reference files that are supplied or explicitly listed as generated artifacts. Draw the architecture directly in TikZ unless an external source is genuinely more practical. Explain which plots come from Python and which diagrams are drawn in LaTeX.

---

# 7. Starter LaTeX skeleton

Use and expand this skeleton. Do not return it unchanged; generate complete, valid source.

```latex
\documentclass[11pt,a4paper]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{hyperref}
\usepackage{url}
\usepackage{cite}

\pgfplotsset{compat=1.18}
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue
}

\title{Comparative Performance Evaluation of Lexical, Semantic, and Hybrid Retrieval for Duplicate Question Retrieval}
\author{
  Student Name \\
  Registration Number \\
  Email Address
}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
% Problem, QQP, methods, protocol, contributions, actual findings only.
\end{abstract}

\noindent\textbf{Keywords:}
Information Retrieval; Duplicate Question Retrieval; QQP;
TF-IDF; BM25; Sentence Transformers; Reciprocal Rank Fusion.

\section{Introduction}

\section{About Dataset}
\subsection{Dataset Description}
\subsection{Question-Pair Data and Retrieval Corpus}

\section{Methodology}
\subsection{System Architecture}
\subsection{Preprocessing and Corpus Construction}
\subsection{TF-IDF Retrieval}
\subsection{BM25 Retrieval}
\subsection{Sentence Transformer Retrieval}
\subsection{Reciprocal Rank Fusion}
\subsection{Implementation Stack}

\section{Experimental Setup and Performance Analysis}
\subsection{Experimental Setup}
\subsection{Evaluation Metrics}
\subsection{Quantitative Comparison}
\subsection{Precision--Recall Analysis}
\subsection{Discussion and Error Analysis}

\section{Conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

---

# 8. Connect the report to real implementation outputs

Recommend a small, reproducible data-to-report workflow:

1. A data-analysis script saves dataset statistics as JSON/CSV.
2. The evaluation script saves per-model metrics as CSV/JSON.
3. The plotting script saves the Precision–Recall graph as PDF/PNG.
4. The report uses those actual values and imports the generated figure.
5. The final report records the evaluation settings and environment.

Do not add a complex automatic report-generation system without a clear need. If suggesting a results CSV, use a schema such as model, cutoff, MAP, nDCG, precision, recall, median latency, and p95 latency, adjusted to the actual implementation.

Do not claim that raw CSV values can be included in standard LaTeX automatically unless a suitable package or conversion step is actually provided. A simple documented process is sufficient.

---

# 9. Academic integrity and evidence rules

- Write original prose and cite sources properly.
- Cite the actual QQP source and model used.
- Do not copy the Cranfield example's data or results.
- Do not claim a plagiarism/similarity percentage or that a detector has been run.
- Do not fabricate dataset statistics, model metrics, latency, graph coordinates, hardware measurements, references, or qualitative results.
- Clearly distinguish prior published findings from this project's own findings.
- State evaluation limitations honestly, especially incomplete relevance judgments.
- Avoid unsupported claims such as “RRF always performs best.”
- Keep terminology consistent.

---

# 10. Final self-audit

Before returning the project, verify:

## Content
- [ ] The report is about QQP Duplicate Question Retrieval, not Cranfield.
- [ ] It frames the task as retrieval and ranking, not only classification.
- [ ] TF-IDF, BM25, Sentence Transformer, and RRF are all explained.
- [ ] Unique-question corpus construction is described.
- [ ] Leakage and incomplete relevance judgments are addressed.
- [ ] The real search interface is documented.
- [ ] Experimental settings are reproducible.
- [ ] Metric definitions match the implementation.
- [ ] Model comparison and Precision–Recall graph are included or visibly marked pending.
- [ ] Every performance claim is supported by actual evidence.

## Figures and tables
- [ ] Architecture diagram is readable.
- [ ] Dataset plots use actual counts.
- [ ] Captions and labels are present.
- [ ] All figures and tables are referenced in the prose.
- [ ] No fabricated empirical plots or results appear.

## LaTeX
- [ ] `main.tex` compiles in Overleaf using the stated compiler.
- [ ] `references.bib` exists and citations resolve.
- [ ] Packages are compatible.
- [ ] Cross-references resolve.
- [ ] No unavailable image files are silently referenced.
- [ ] Tables and diagrams do not overflow.
- [ ] Structure follows the supplied DOCX.

## Deliverables
- [ ] Provide all source files and a project tree.
- [ ] List all generated empirical artifacts needed.
- [ ] Provide Overleaf upload/compile instructions.
- [ ] Identify every remaining placeholder.

Now generate the complete Overleaf-ready LaTeX project. Do not return only an outline or general advice.
