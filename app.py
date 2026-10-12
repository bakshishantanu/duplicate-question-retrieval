"""Interactive Streamlit Search Interface for Duplicate Question Retrieval."""
import os
import sys
import time
import json
import streamlit as st
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CORPUS_PATH, CORPUS_IDS_PATH, TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH,
    BM25_INDEX_PATH, SBERT_EMBEDDINGS_PATH, EVAL_RESULTS_PATH,
    COMPARISON_TABLE_PATH, PR_CURVE_PATH, DEFAULT_TOP_K
)
from src.retrievers.tfidf_retriever import TFIDFRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.sbert_retriever import SBERTRetriever
from src.retrievers.hybrid_retriever import HybridRetriever

# ─── Page Configuration ───
st.set_page_config(
    page_title="Duplicate Question Retrieval Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.1rem;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Resource Loaders ───
@st.cache_resource(show_spinner="Loading corpus into memory...")
def load_corpus_data():
    if not os.path.exists(CORPUS_PATH):
        return None, None
    corpus_df = joblib.load(CORPUS_PATH)
    corpus_ids = corpus_df['qid'].values
    qid_to_text = dict(zip(corpus_df['qid'], corpus_df['question_text']))
    return corpus_df, qid_to_text


@st.cache_resource(show_spinner="Loading retrieval indexes & models...")
def load_retrievers():
    if not (os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(BM25_INDEX_PATH) and os.path.exists(SBERT_EMBEDDINGS_PATH)):
        return None

    corpus_df, _ = load_corpus_data()
    corpus_ids = corpus_df['qid'].values

    # 1. TF-IDF
    tfidf = TFIDFRetriever()
    tfidf.load_index(TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, corpus_ids)

    # 2. BM25
    bm25 = BM25Retriever()
    bm25.load_index(BM25_INDEX_PATH)

    # 3. SBERT
    sbert = SBERTRetriever()
    sbert.load_index(SBERT_EMBEDDINGS_PATH, corpus_ids)

    # 4. Hybrid
    hybrid = HybridRetriever({'TF-IDF': tfidf, 'BM25': bm25, 'SBERT': sbert})

    return {
        'Hybrid RRF': hybrid,
        'TF-IDF': tfidf,
        'BM25': bm25,
        'SBERT': sbert,
    }


@st.cache_data
def load_evaluation_metrics():
    if os.path.exists(EVAL_RESULTS_PATH):
        with open(EVAL_RESULTS_PATH, 'r') as f:
            return json.load(f)
    return None


# ─── Main Application ───
def main():
    st.markdown('<div class="main-title">🔎 Duplicate Question Retrieval Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Semantic & Lexical Question Search powered by TF-IDF, BM25, Sentence Transformers, and Reciprocal Rank Fusion (RRF).</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Search Configuration")
    method = st.sidebar.selectbox(
        "Retrieval Method",
        ["Hybrid RRF", "SBERT", "BM25", "TF-IDF"],
        index=0,
        help="Select between hybrid fusion, dense semantic representation, or classical lexical models."
    )

    top_k = st.sidebar.slider(
        "Top Results (K)",
        min_value=1,
        max_value=50,
        value=DEFAULT_TOP_K,
        help="Number of duplicate candidates to retrieve."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Benchmark & Analysis")
    show_table = st.sidebar.checkbox("Performance Table", value=True)
    show_pr = st.sidebar.checkbox("Precision–Recall Curve", value=True)

    # Load resources
    corpus_df, qid_to_text = load_corpus_data()
    retrievers = load_retrievers()

    if corpus_df is None or retrievers is None:
        st.warning("⚠️ Corpus or index artifacts were not found in the `artifacts/` directory.")
        st.info("Please build the retrieval pipeline by executing:")
        st.code("python scripts/run_all.py", language="bash")
        return

    st.sidebar.success(f"Corpus loaded: **{len(corpus_df):,}** unique questions")

    # Search Bar & Example Queries
    sample_queries = [
        "How can I learn Python quickly?",
        "What is the best way to lose weight naturally?",
        "How does machine learning differ from deep learning?",
        "Why is the sky blue in the daytime?",
        "What are the best books for learning data science?"
    ]

    selected_sample = st.selectbox("Or choose an example question:", ["-- Custom Query --"] + sample_queries)
    default_text = "" if selected_sample == "-- Custom Query --" else selected_sample

    query = st.text_input(
        "Enter your question query:",
        value=default_text,
        placeholder="e.g. How do I start studying machine learning?",
        max_chars=300
    )

    tab_search, tab_eval, tab_info = st.tabs(["🔍 Search Results", "📈 Benchmark Metrics", "ℹ️ System Architecture"])

    with tab_search:
        if query:
            clean_query = query.strip()
            if len(clean_query) < 3:
                st.warning("Please enter a question with at least 3 characters.")
            else:
                retriever = retrievers[method]
                t0 = time.perf_counter()

                details = None
                if method == "Hybrid RRF":
                    details = retriever.retrieve_with_details(clean_query, top_k=top_k)
                    results = details['results']
                else:
                    results = retriever.retrieve(clean_query, top_k=top_k)

                latency_ms = (time.perf_counter() - t0) * 1000

                # Summary Metrics Row
                mcol1, mcol2, mcol3 = st.columns([1, 1, 2])
                with mcol1:
                    st.metric("Results Retrieved", len(results))
                with mcol2:
                    st.metric("Query Latency", f"{latency_ms:.1f} ms")
                with mcol3:
                    st.metric("Ranking Algorithm", method)

                st.markdown(f"### Ranked Candidates for: *\"{clean_query}\"*")

                if not results:
                    st.info("No matching questions found.")
                else:
                    for rank, (qid, score) in enumerate(results, 1):
                        q_text = qid_to_text.get(qid, "[Question text not found]")

                        with st.container():
                            col_text, col_score = st.columns([0.82, 0.18])
                            with col_text:
                                st.markdown(f"**{rank}.** {q_text}")
                                if details and qid in details['contributions']:
                                    contribs = details['contributions'][qid]
                                    badge_items = []
                                    for m_name, m_info in contribs.items():
                                        badge_items.append(f"**{m_name}**: rank #{m_info['rank']} (+{m_info['rrf_contribution']:.4f})")
                                    st.caption("Consensus: " + " | ".join(badge_items))
                            with col_score:
                                label = "RRF Score" if method == "Hybrid RRF" else "Similarity"
                                st.metric(label, f"{score:.4f}")
                            st.divider()
        else:
            st.info("Type a question above or pick one from the example dropdown to start searching.")

    with tab_eval:
        st.subheader("Experimental Performance Evaluation")
        st.markdown("""
        All methods are evaluated on an unseen test split using **Connected-Component Splitting** to ensure zero data leakage.
        Relevance is established by ground-truth duplicate clusters with transitive closure.
        """)

        eval_data = load_evaluation_metrics()
        if eval_data:
            eval_rows = []
            for m_name, metrics in eval_data.items():
                row = {'Method': m_name}
                for k, v in metrics.items():
                    if isinstance(v, float):
                        row[k] = f"{v:.4f}" if "latency" not in k else f"{v:.1f}"
                    else:
                        row[k] = v
                eval_rows.append(row)

            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)

            if os.path.exists(PR_CURVE_PATH):
                st.markdown("#### Precision–Recall Curves Across Methods")
                st.image(PR_CURVE_PATH, caption="Precision vs. Recall at varying ranking depths K (1 to 30)")
        else:
            st.info("Evaluation results not yet generated. Run `python scripts/04_evaluate.py` to produce full benchmark tables and graphs.")

    with tab_info:
        st.subheader("System Architecture & Design Decisions")
        st.markdown("""
        ### Key Engineering Pillars
        1. **Leakage-Aware Splitting:**
           Connected components of duplicate questions are kept strictly intact within either train, val, or test sets. No duplicates of test queries ever leak into the candidate training index.
        2. **Transitive Closure Adjacency:**
           If Question A is duplicate to B, and B is duplicate to C, all three form a mutual duplicate clique, preventing false negatives during evaluation.
        3. **Lexical Normalization:**
           Conservative normalization preserving essential question words (*what, how, why*) and negations (*not, no*) while applying Porter stemming.
        4. **Dense Semantic Embeddings:**
           `sentence-transformers/all-MiniLM-L6-v2` produces L2-normalized 384-dimensional embeddings. Cosine similarities are computed through vector dot products.
        5. **Reciprocal Rank Fusion (RRF):**
           $$RRF(d) = \sum_{m} \\frac{1}{k + \\text{rank}_m(d)}, \\quad k=60$$
           Combines lexical exact-match signals with dense semantic representations in a parameter-free, rank-based manner.
        """)


if __name__ == "__main__":
    main()
