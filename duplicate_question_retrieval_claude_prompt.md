# TASK: Create a Complete, Implementation-Ready Plan for Project 8 — Duplicate Question Retrieval

## 1. Your Role

Act as a senior Machine Learning Engineer, Information Retrieval Engineer, NLP Researcher, and Technical Project Architect.

Your task is to study the problem statement below and produce a comprehensive implementation plan that I can follow from an empty project directory to a fully implemented, tested, evaluated, and documented application.

I do not want a generic roadmap, a superficial list of technologies, or a tutorial that discusses information retrieval in isolation. I want an actionable engineering blueprint containing concrete implementation decisions, architecture, algorithms, mathematical formulations, file structures, dependencies, commands, testing procedures, evaluation protocols, and acceptance criteria.

**Your most important responsibility is scope discipline.** You must implement the exact project described below. Do not silently replace the project with a different NLP task, introduce unnecessary infrastructure, or prioritize fashionable technologies over the stated requirements.

Every major design decision must include:
1. What will be implemented.
2. Why it is necessary for this project.
3. Why the selected method or technology is preferable to relevant alternatives.
4. What advantages it provides.
5. What limitations and trade-offs it introduces.
6. How it will be implemented.
7. How its correctness and effectiveness will be verified.

Assume I am a Computer Science undergraduate with knowledge of Python, basic machine learning, and data structures, but I need clear implementation instructions, including the reasoning behind decisions. Explain advanced information retrieval concepts without oversimplifying the engineering details.

---

# PART I — AUTHORITATIVE PROBLEM STATEMENT

## 2. Project Title

**Project 8: Duplicate Question Retrieval**

## 3. Core Problem Statement

Develop a semantic duplicate-question retrieval system that can identify and retrieve questions expressing the same or substantially equivalent intent from a large question-and-answer database.

The system must accept a natural-language question from a user, search a corpus of existing questions, retrieve relevant candidates, rank those candidates by relevance, and present the highest-ranked results through a usable search interface.

The central challenge is that semantically equivalent questions can use different vocabulary, word order, and sentence structures. Conversely, questions sharing many words may ask fundamentally different things.

For example:

- "How can I learn Python quickly?"
- "What is the fastest way to learn Python?"
- "How do I become a Python developer?"

The first two questions may be semantically equivalent, whereas the third has a related topic but a potentially different intent.

The system must therefore go beyond exact string matching and simple keyword overlap.

It must combine lexical information retrieval with semantic similarity techniques and investigate whether a hybrid ranking strategy improves retrieval quality.

**The primary deliverable is a functioning retrieval engine, not merely a binary classifier that predicts whether two supplied questions are duplicates.**

Binary duplicate labels from the dataset will support the construction of relevance judgments, evaluation, and optional supervised extensions. They do not replace the requirement to implement an actual search-and-ranking pipeline.

## 4. Dataset: Quora Question Pairs

Use the Quora Question Pairs (QQP) dataset.

The project brief identifies a dataset containing more than 400,000 question pairs with binary duplicate labels. The commonly distributed training split contains approximately 404,000 pairs, although the exact count depends on the dataset version.

Typical columns include:

- `id`: Pair identifier.
- `qid1`: Identifier of the first question, when available.
- `qid2`: Identifier of the second question, when available.
- `question1`: First question's text.
- `question2`: Second question's text.
- `is_duplicate`: Binary label indicating whether the pair is considered a duplicate.

The exact schema must be verified against the downloaded dataset rather than assumed.

The label semantics are:

- `is_duplicate = 1`: The pair is labelled as duplicate.
- `is_duplicate = 0`: The pair is labelled as non-duplicate.

The dataset consists of question pairs, while the application must search a corpus of individual questions. Converting the pair-oriented dataset into a unique-question corpus is therefore a mandatory design step.

The implementation plan must explain how to:
1. Load and inspect the dataset.
2. Handle missing, malformed, and duplicated records.
3. Construct a unique-question corpus from both question columns.
4. Preserve stable question IDs and original question text.
5. Build mappings from questions to known duplicate questions.
6. Use pair labels appropriately during evaluation.
7. Prevent train-test leakage.
8. Document the limitations of the available relevance judgments.

Do not assume that every question pair absent from the dataset is a confirmed negative. An unobserved pair is not necessarily a non-duplicate pair.

The precise dataset URL must be verified or obtained from the original project brief. Do not invent a download URL or silently substitute a different dataset.

## 5. Mandatory Functional Requirements

The finished application must satisfy all the following requirements.

### FR-1: Question preprocessing

Implement appropriate text preprocessing for lexical retrieval, including:

- Text cleaning and normalization.
- Tokenization.
- Selective stop-word handling.
- Stemming, if justified by experiments.

Do not blindly apply every preprocessing operation to every model.

Maintain a distinction between:
- Original question text, used for display and semantic encoding.
- Normalized lexical representations, used for lexical retrieval.

Explain the impact of removing negation, punctuation, question words, and other potentially meaningful tokens.

### FR-2: Unique-question corpus and indexing

Build a searchable collection of individual questions from the dataset's two question columns.

Create stable IDs and mappings that allow retrieved results to be associated with their original text and relevant metadata.

The index must support efficient retrieval of candidate questions without reconstructing the entire corpus on every search.

### FR-3: TF-IDF retrieval

Implement TF-IDF vectorization and lexical similarity ranking.

The implementation should use sparse representations where appropriate and a clearly defined similarity measure, such as cosine similarity.

Explain why TF-IDF is a useful baseline, its advantages over raw word counts, and its limitations compared with semantic retrieval.

### FR-4: BM25 retrieval

Implement BM25 as an independent lexical retrieval system.

Use a practical Python implementation, such as `rank_bm25`, unless research establishes a better choice for the project's requirements.

Explain term-frequency saturation, inverse document frequency, and document-length normalization, including the relevance of these properties to short questions.

Do not treat BM25 as interchangeable with TF-IDF. Evaluate both separately.

### FR-5: Sentence Transformer retrieval

Implement dense semantic retrieval using a pretrained Sentence Transformer model.

Use `sentence-transformers/all-MiniLM-L6-v2` as the initial candidate model unless there is a strong, explicitly justified reason to select another model.

The implementation must:
- Encode each indexed question once.
- Persist its embedding.
- Encode incoming queries at search time.
- Compute similarity between the query embedding and indexed question embeddings.
- Retrieve the top-ranked candidates.
- Exclude the query itself or its equivalent corpus entry where appropriate.

Explain why sentence embeddings can retrieve paraphrases with limited word overlap and why semantic similarity does not guarantee true duplicate intent.

Discuss the relevant alternatives, such as averaging Word2Vec/GloVe embeddings, fine-tuning a transformer, using an LLM API, or using a cross-encoder.

Do not require training a transformer from scratch.

### FR-6: Hybrid ranking using Reciprocal Rank Fusion

Implement Reciprocal Rank Fusion (RRF) to combine the rankings produced by TF-IDF, BM25, and Sentence Transformer retrieval.

Use the following formulation:

RRF(d) = sum over retrieval systems m of 1 / (k + rank_m(d))

where:
- d is a candidate question.
- m identifies a retrieval system.
- rank_m(d) is the candidate's rank in that system.
- k is the smoothing constant.

Use a conventional initial value such as k = 60, then determine whether tuning is worthwhile using validation data.

Candidates missing from a particular ranking contribute zero for that ranking.

The plan must explicitly explain why rank fusion is preferable to naively adding raw scores with incompatible scales.

The system must retrieve candidates from all three systems, merge candidate IDs, calculate RRF scores, and produce one final ranked list.

### FR-7: Search interface

Implement a functional search interface that allows users to:

- Enter a natural-language question.
- Select TF-IDF, BM25, Sentence Transformer, or Hybrid RRF.
- Choose the number of results to display.
- View ranked candidate questions.
- Inspect meaningful relevance information or model scores, with correct labels.
- See the performance comparison table.
- View the Precision–Recall graph.
- Inspect search latency.

Use Streamlit for the initial implementation unless a concrete requirement justifies a different framework.

The interface must connect to the actual retrieval pipeline. A static mock-up does not satisfy this requirement.

### FR-8: Performance evaluation

Implement an evaluation pipeline that compares the retrieval approaches using consistent query sets and relevance definitions.

At minimum, investigate:

- Mean Average Precision (MAP), preferably MAP@K for a clearly specified cutoff.
- Normalized Discounted Cumulative Gain (nDCG@K).
- Precision@K.
- Recall@K.
- Query latency.

Evaluate at consistent cutoffs, such as K = 5 and K = 10, where appropriate.

Also generate the required Precision–Recall graph.

The plan must explain the evaluation protocol in detail, particularly the limitations of pairwise labels when used as retrieval relevance judgments.

### FR-9: Performance comparison table

Generate a table containing the measured performance of:

1. TF-IDF.
2. BM25.
3. Sentence Transformer.
4. Hybrid RRF.

A cross-encoder extension may be included as an optional fifth row if justified.

Populate the table using actual experimental results. Do not invent metrics, claim improvements before benchmarking, or use fabricated placeholder values in the final report.

### FR-10: Precision–Recall visualization

Generate a Precision–Recall graph.

Define precisely how the graph is constructed:
- For retrieval, explain how precision and recall vary with the number of retrieved candidates.
- If using a binary classification-style curve, explain the score threshold and ground-truth labels used.

Ensure that the graph corresponds to real evaluation data.

### FR-11: Reproducibility and documentation

Provide:
- A reproducible dependency specification.
- Dataset preparation instructions.
- Index construction instructions.
- Model execution instructions.
- Evaluation instructions.
- Search-interface launch instructions.
- A clear README.
- Documented experimental settings.
- Tests for critical functionality.

The system should be reproducible on a typical modern laptop, without requiring a dedicated GPU.

---

# PART II — TECHNICAL DESIGN CONSTRAINTS

## 6. Recommended Initial Technology Stack

Treat the following as the default stack, not as an excuse to introduce additional technologies unnecessarily.

- Python: Main programming language.
- Pandas: Dataset loading and processing.
- NumPy: Numerical operations.
- scikit-learn: TF-IDF, sparse operations, similarity calculations, and relevant evaluation utilities.
- `rank_bm25`: Initial BM25 implementation.
- `sentence-transformers`: Dense question embeddings.
- Matplotlib: Performance plots and Precision–Recall visualization.
- Streamlit: Interactive search interface.
- `joblib`, NumPy serialization, or another suitable local serialization mechanism: Persisting indexes and artifacts.
- pytest: Automated tests.
- Git: Version control.

Prefer exact dense similarity search initially, provided measured latency and memory consumption are acceptable.

FAISS, dedicated vector databases, distributed computation, cloud infrastructure, REST API frameworks, and additional frontend technologies are not required for the initial version.

Introduce an additional technology only when:
1. A mandatory requirement cannot be satisfied reasonably without it; or
2. Measured results establish a concrete need for it.

For every additional dependency, explain its purpose, alternatives, cost, complexity, and measurable benefit.

## 7. Scope Boundaries: Features That Must Not Replace the Core Project

The following are optional extensions, not mandatory components of the initial implementation:

- Cross-encoder reranking.
- Fine-tuning Sentence Transformers on QQP.
- Approximate nearest-neighbour indexing.
- FAISS or a vector database.
- Learned ranking models.
- LLM-based pair classification.
- Neural networks trained from scratch.
- Cloud deployment.
- User accounts, authentication, and multi-user management.
- Distributed data processing.
- A React/Next.js frontend.
- A separate production API server.
- Real-time ingestion and index updates.

You may discuss an extension if it has a clear benefit, but the required retrieval pipeline, evaluation, comparison table, graph, and search interface must be completed first.

Do not allow optional enhancements to expand the minimum viable project beyond its original requirements.

## 8. Critical Data and Evaluation Constraints

Treat data leakage and evaluation validity as first-class engineering concerns.

The QQP dataset contains recurring questions and duplicate relationships. A naïve random split of pair rows may cause questions, duplicate relationships, or closely related examples to appear across training and evaluation partitions.

Your plan must:
- Explain the leakage risks.
- Choose a defensible split strategy appropriate for the intended retrieval scenario.
- Keep evaluation queries separate from their ground-truth answer relationships in the index where required by that strategy.
- Ensure that the query itself is not counted as a successful retrieved duplicate.
- Explain how connected components or question-ID grouping may help prevent leakage.
- Explain the limitations of connected-component splitting, including the possibility of large components.
- Avoid fitting learned preprocessing or ranking parameters on the test set.
- Tune hyperparameters only on training or validation data as appropriate.
- Evaluate every retrieval model on a consistent test protocol.

If the dataset version does not contain question IDs, explain how to construct stable question identities and what limitations this introduces.

Do not claim that a pair is irrelevant simply because no positive relationship was observed.

If complete relevance judgments are unavailable, describe the evaluation as being based on known positive pairs or incomplete relevance judgments, and discuss the consequences for recall and ranking metrics.

Do not silently change the evaluation protocol to make results look better.

---

# PART III — YOUR REQUIRED OUTPUT

## 9. Begin with a Requirements Traceability Matrix

Before producing the implementation roadmap, create a table with these columns:

- Requirement ID.
- Exact requirement.
- Planned module or component.
- Implementation method.
- Verification method.
- Acceptance criterion.
- Status: mandatory or optional.

Include every functional requirement from FR-1 through FR-11.

This matrix becomes the authoritative checklist for the entire project.

Every subsequent stage must map back to this matrix. Do not add features that have no clear relationship to a mandatory requirement unless they are explicitly marked optional.

## 10. Provide a Detailed Architecture

Describe the end-to-end architecture from dataset ingestion to the final search result.

Include:
- The data flow.
- Module responsibilities.
- The relationship between pairwise data and the unique-question corpus.
- The three independent retrieval paths.
- The RRF fusion component.
- The evaluation pipeline.
- The Streamlit interface.
- Artifact storage and loading.
- Error handling and invalid-input behavior.

Include a Mermaid architecture diagram if appropriate.

Explain each major component and the interfaces between components.

Be explicit about which artifacts are created offline and which operations occur at query time.

## 11. Provide the Exact Project Directory Structure

Produce a realistic, modular directory tree for the entire application.

For every file and directory, explain:
- Its purpose.
- Its expected inputs.
- Its expected outputs.
- Its dependencies.
- Which stage creates it.

Keep the project modular without introducing excessive abstraction or an unnecessarily complicated package hierarchy.

Make sure every file mentioned later in the plan exists in the proposed structure.

Do not describe a file that has no defined role, and do not use inconsistent filenames across sections.

## 12. Provide a Complete Implementation Roadmap

Break the implementation into sequential stages.

For every stage, provide the following sections.

### A. Objective

Explain the exact outcome of the stage and which requirements it satisfies.

### B. Why this stage is necessary

Explain its place in the complete system and the consequences of skipping it.

### C. Chosen method and alternatives

Explain the selected technology or algorithm, why it is preferable here, which alternatives were considered, and the relevant trade-offs.

Do not give generic statements such as "Python is popular" or "this model is accurate." Tie the reasoning to the dataset size, retrieval task, computational constraints, implementation complexity, or evaluation requirements.

### D. Detailed implementation procedure

Give an ordered, practical sequence of steps.

Include:
- Specific files to create or modify.
- Function and class responsibilities.
- Inputs and outputs.
- Important implementation details.
- Mathematical formulas where relevant.
- Edge cases.
- Expected artifacts.
- How components connect to earlier and later stages.

Do not merely say "implement BM25" or "evaluate the model." Explain the operations required to complete those tasks.

### E. Commands and dependencies

Where applicable, provide:
- Environment creation commands.
- Package installation commands.
- Script execution commands.
- Test commands.
- Expected command-line behavior.

Assume Windows is a likely development environment, but keep the project cross-platform where practical. Explain the purpose of each important command in simple language.

Do not invent package APIs or unverified command-line options.

### F. Verification and tests

Define tests that establish whether the implementation is correct.

Include unit tests, integration tests, and data-validation checks where appropriate.

Specify the expected behavior rather than inventing model-performance scores.

### G. Acceptance criteria

Provide objective conditions that must be satisfied before moving to the next stage.

Avoid arbitrary performance targets that have no empirical justification.

### H. Failure modes and recovery

Explain likely problems, their causes, diagnostic steps, and appropriate fixes.

### I. Completion checkpoint

End each stage with a checklist of artifacts, tests, and decisions that must be completed before continuing.

## 13. Cover These Stages in Order

The roadmap must cover, at minimum:

1. Requirements analysis, environment setup, and repository initialization.
2. Dataset acquisition, schema verification, exploratory analysis, and data cleaning.
3. Unique-question corpus construction, ID mapping, duplicate graph construction, and leakage-aware splitting.
4. Text preprocessing and comparison of lexical preprocessing strategies.
5. TF-IDF indexing and retrieval.
6. BM25 indexing and retrieval.
7. Sentence Transformer embedding generation, persistence, and semantic retrieval.
8. Reciprocal Rank Fusion implementation and hybrid candidate ranking.
9. Retrieval evaluation, metric implementation, comparison table, and Precision–Recall graph.
10. Streamlit search interface and integration with the actual retrieval pipeline.
11. Automated testing, reproducibility, error handling, and performance measurement.
12. Documentation, final validation, and project completion.

You may add necessary sub-stages, but preserve this order unless a technical dependency makes a change necessary. Explain any reordering.

## 14. Explain the Algorithms at Implementation Level

For each mandatory algorithm, explain:

- The mathematical intuition.
- The exact inputs and outputs.
- The representation of questions.
- The ranking or similarity calculation.
- The relevant hyperparameters.
- How candidates are selected.
- How ties and missing candidates are handled.
- How the implementation is validated.
- Its advantages and disadvantages compared with the other mandatory methods.

Specifically cover:
- TF-IDF with cosine similarity.
- BM25.
- Sentence Transformer embeddings with cosine similarity.
- Reciprocal Rank Fusion.
- Precision@K.
- Recall@K.
- Average Precision and MAP.
- DCG and nDCG.
- The construction and interpretation of the Precision–Recall graph.

Use a consistent definition of each metric throughout the plan.

## 15. Define a Fair Benchmarking Protocol

Specify how to compare the retrieval models fairly.

The plan must state:
- How evaluation queries are selected.
- How ground-truth duplicate questions are identified.
- How query self-matches are excluded.
- How corpus leakage is prevented.
- Whether results are macro-averaged across queries.
- Which cutoff values are used.
- How candidate depth is selected for RRF.
- How model-loading and index-loading time are treated.
- How query latency is measured.
- Which machine and environment details are recorded.
- How results are saved for the comparison table and graph.

Separate offline index-building time from online query latency.

If relevant positive questions are not completely annotated, explain why reported recall is only recall against the known relevance judgments.

Do not report results that have not been measured.

## 16. Include an Explicit RRF Design

Specify:
- The top-K candidate depth for each retrieval system.
- How candidate IDs are merged.
- The RRF formula and smoothing constant.
- How absent candidates are handled.
- How ties are resolved deterministically.
- Whether retrieval scores are retained for diagnostics.
- How the fusion strategy is evaluated against its individual components.

Choose a reasonable initial configuration and explain how it will be validated.

Do not introduce a trainable fusion model unless it is clearly marked optional.

## 17. Define the Search Interface and Its Integration

Describe the complete query lifecycle:

1. The user submits a question.
2. Input validation and conservative normalization occur.
3. The selected retriever generates candidates.
4. The selected ranking or fusion method orders the candidates.
5. Self-matches and invalid results are removed.
6. The top-K results are mapped to original question text.
7. Results and appropriate diagnostic information are rendered.
8. Query latency is recorded.

Explain how indexes and models are loaded and cached.

Specify how empty queries, missing artifacts, malformed input, and search failures are handled.

Ensure that the interface displays real results from the implemented retrieval system.

## 18. Provide a Final Verification Matrix

At the end of the plan, map every requirement to:

- Its implementing module.
- Its test or benchmark.
- Its expected artifact.
- Its acceptance criterion.
- The evidence needed to demonstrate completion.

This final matrix must agree with the original requirements matrix.

Include a final go/no-go checklist for declaring the project complete.

## 19. Distinguish Mandatory Work from Optional Improvements

Clearly separate:
- Mandatory minimum viable implementation.
- Improvements worth considering after baseline completion.
- Advanced research extensions.

For each optional extension, explain:
- The problem it addresses.
- Its benefit relative to the mandatory implementation.
- Its additional dependencies and complexity.
- How its value would be measured.
- Why it should not block project completion.

Prioritize completing and validating the required system over adding advanced features.

---

# PART IV — STRICT SCOPE AND QUALITY RULES

## 20. Non-Negotiable Rules

1. **Do not change the problem statement.** The objective is semantic duplicate-question retrieval using the QQP dataset, with lexical retrieval, dense semantic retrieval, and hybrid ranking.

2. **Do not replace retrieval with classification.** A binary duplicate classifier may be discussed as an optional extension, but it cannot substitute for the search engine.

3. **Do not omit any mandatory retrieval method.** Implement TF-IDF, BM25, Sentence Transformer retrieval, and RRF.

4. **Do not omit the deliverables.** The search interface, performance comparison table, and Precision–Recall graph are mandatory.

5. **Do not invent dataset details, benchmark scores, APIs, or experimental results.** Identify assumptions and verify implementation details.

6. **Do not introduce unnecessary infrastructure.** Avoid adding a database, cloud deployment, microservices, Docker, a frontend framework, or a vector database without a concrete requirement.

7. **Do not implement optional enhancements before the baseline.** The mandatory pipeline must work first.

8. **Do not ignore evaluation leakage.** Explicitly address recurring questions, duplicate relationships, self-matches, and incomplete relevance judgments.

9. **Do not use the test set for model selection.** Keep model selection and final evaluation separate.

10. **Do not combine incompatible raw retrieval scores without justification.** Use RRF as the required hybrid approach.

11. **Do not use generic implementation instructions.** Every stage must contain specific steps, files, interfaces, verification procedures, and acceptance criteria.

12. **Do not omit trade-offs.** Explain why a method is appropriate here and where an alternative might be superior.

13. **Do not make the project unnecessarily research-heavy.** The primary objective is a completed, reproducible software system with credible experimental evaluation.

14. **Do not produce code for the entire application as a substitute for the implementation plan.** This task is to design the plan. Include short illustrative snippets only where they clarify an important design decision.

15. **Do not silently omit difficult requirements.** If a requirement presents a genuine technical or evaluation challenge, explain it and propose a defensible solution.

16. **Do not contradict earlier sections.** Filenames, model choices, interfaces, metrics, dependencies, and execution commands must remain consistent throughout the plan.

## 21. Research and Verification

If you have access to web search, use official documentation and original research where needed to verify:
- Sentence Transformers model behavior and recommended semantic-search practices.
- scikit-learn TF-IDF implementation details.
- The chosen BM25 implementation.
- The definitions of retrieval evaluation metrics.
- Streamlit model and resource caching.
- Any APIs or package versions mentioned in the plan.

Prefer official documentation and primary research over unsourced blog posts.

Cite important external claims. Clearly distinguish verified facts from engineering recommendations and assumptions.

Do not spend time researching unrelated frameworks, alternative datasets, or advanced infrastructure that the project does not require.

## 22. Required Final Response Structure

Produce the implementation plan in this exact high-level order:

1. Executive summary and explicit project boundaries.
2. Requirements traceability matrix.
3. Architecture and end-to-end data flow.
4. Technology selection and comparison with alternatives.
5. Project directory structure.
6. Dataset construction and leakage-aware evaluation design.
7. Detailed implementation stages, including all mandatory algorithms.
8. Benchmarking methodology and metric definitions.
9. Search interface and integration design.
10. Testing, reproducibility, and error handling.
11. Final verification matrix.
12. Completion checklist and optional future extensions.

The response must be detailed enough that I can use it as the authoritative engineering blueprint while implementing the project incrementally.

Do not shorten the plan merely to keep the response brief. Do not repeat the same generic explanation in every section; instead, provide precise cross-references where appropriate.

If the full plan exceeds the available response budget, divide it into clearly numbered parts and explicitly state which sections remain. Preserve technical depth and consistency rather than compressing or silently omitting requirements.

---

# FINAL INSTRUCTION

Before presenting your answer, perform a self-audit.

Verify that:
- Every mandatory requirement has an implementation stage.
- Every implementation stage has a verification method and acceptance criteria.
- All three retrieval methods and RRF are implemented independently and compared fairly.
- The unique-question corpus is correctly derived from question pairs.
- Data leakage and incomplete relevance judgments are addressed.
- The search interface, comparison table, and Precision–Recall graph are explicitly included.
- All technology choices have project-specific justifications and alternatives.
- Optional features cannot displace mandatory work.
- The final plan is internally consistent and practically executable.

If you identify a conflict between an implementation choice and the authoritative problem statement, revise the choice rather than changing the requirement.

Now produce the complete implementation plan.
