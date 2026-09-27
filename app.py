# app.py

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from pipeline import graph

st.set_page_config(page_title="Malaysian Legal RAG Assistant", page_icon="⚖️", layout="centered")

# ---------------------------------------------------------------------------
# Styling — palette and fonts live in .streamlit/config.toml; this covers
# what config.toml can't reach: the header seal, example-question chips,
# and the source-citation tags.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    footer {visibility: hidden;}

    .seal {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        border: 1.5px solid #96762E;
        color: #96762E;
        font-size: 1.3rem;
        margin-right: 0.6rem;
        flex-shrink: 0;
    }
    .header-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.1rem;
    }
    .header-row h1 {
        margin: 0;
        padding: 0;
        font-size: 1.9rem;
    }
    .subtitle {
        color: #5b6274;
        margin-bottom: 1.5rem;
        font-size: 1.02rem;
    }
    hr.rule {
        border: none;
        border-top: 1px solid rgba(28, 43, 68, 0.15);
        margin: 0 0 1.4rem 0;
    }
    .stButton > button {
        border: 1px solid rgba(28, 43, 68, 0.2);
        background-color: transparent;
        border-radius: 8px;
        font-size: 0.92rem;
        padding: 0.55rem 0.9rem;
    }
    .stButton > button:hover {
        border-color: #96762E;
        color: #96762E;
    }
    .source-tag {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #96762E;
        border: 1px solid #96762E;
        border-radius: 4px;
        padding: 0.05rem 0.4rem;
        margin-right: 0.4rem;
    }
    .disclaimer {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        background-color: rgba(150, 118, 46, 0.08);
        border: 1px solid rgba(150, 118, 46, 0.35);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 1.4rem;
        font-size: 0.85rem;
        line-height: 1.45;
        color: #5b6274;
    }
    .disclaimer strong {
        color: #96762E;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header-row"><span class="seal">⚖️</span><h1>Malaysian Legal RAG Assistant</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">RAG-Fusion over the Employment Act 1955, Companies Act 2016, and the PDPA 2010</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown(
    '<div class="disclaimer">⚠️ <span><strong>AI-generated content.</strong> '
    'Answers are AI-generated from retrieved legal excerpts and may be incomplete or incorrect. '
    'This is not legal advice — verify against the official legislation or consult a qualified lawyer.</span></div>',
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "How much annual leave after 3 years of service?",
    "Does a private company need an AGM?",
    "Can personal data leave Malaysia?",
]

SOURCES = [
    {
        "title": "Employment Act 1955 (Act 265)",
        "desc": "Governs the employment relationship in Malaysia — leave entitlements, termination notice, working hours, and related employee protections.",
        "url": "https://lom.agc.gov.my/",
    },
    {
        "title": "Companies Act 2016 (Act 777)",
        "desc": "Governs the incorporation, governance, and administration of companies registered in Malaysia.",
        "url": "https://lom.agc.gov.my/",
    },
    {
        "title": "Personal Data Protection Act 2010 (Act 709)",
        "desc": "Governs how personal data is processed in the course of commercial transactions in Malaysia.",
        "url": "https://lom.agc.gov.my/",
    },
]


def render_sources(docs, answer_text=""):
    with st.expander("Show retrieved sources"):
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page")
            page_display = page + 1 if isinstance(page, int) else "?"
            cited = f"[{i}]" in answer_text
            tag = f'<span class="source-tag">[{i}]</span> page {page_display}'
            if not cited:
                tag += ' <span style="color:#5b6274; font-size:0.78rem;">(retrieved, not cited)</span>'
            st.markdown(tag, unsafe_allow_html=True)
            st.caption(doc.page_content[:300] + "...")


def render_queries(queries):
    with st.expander("How this was researched"):
        st.caption("RAG-Fusion: the question is expanded into several search queries, each retrieved separately, then combined by rank.")
        for q in queries:
            st.markdown(f"- {q}")


ask_tab, sources_tab, eval_tab = st.tabs(["Ask", "Sources", "Evaluation"])

# ---------------------------------------------------------------------------
# Ask — chat-style history, persists for the session
# ---------------------------------------------------------------------------
with ask_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.messages:
        _, clear_col = st.columns([5, 1])
        if clear_col.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    chat_box = st.container(height=480)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("queries"):
                        render_queries(msg["queries"])
                    if msg.get("sources"):
                        render_sources(msg["sources"], msg["content"])

        if not st.session_state.messages:
            st.caption("Try asking:")
            cols = st.columns(3)
            for col, example in zip(cols, EXAMPLE_QUESTIONS):
                if col.button(example, use_container_width=True):
                    st.session_state.chat_input = example
                    st.rerun()

    question = st.chat_input(
    "Ask about employment law, company law, or data protection in Malaysia...",
    key="chat_input",
    )   

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_box:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating an answer..."):
                    state = graph.invoke({"question": question})
                st.write(state["answer"])
                render_queries(state["generated_queries"])
                render_sources(state["fused_documents"], state["answer"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": state["answer"],
            "queries": state["generated_queries"],
            "sources": state["fused_documents"],
        })


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
with sources_tab:
    st.write("")
    for source in SOURCES:
        st.markdown(f"**{source['title']}**")
        st.write(source["desc"])
        st.markdown(f"[Read the full text ↗]({source['url']})")
        st.write("")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
with eval_tab:
    st.write("")
    try:
        results = pd.read_csv("eval/eval_results_800_240.csv")
        hit = results["retrieval_hit"].dropna()

        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval hit rate", f"{hit.mean():.0%}")
        c2.metric("Correctness", f"{results['correct'].mean():.0%}")
        c3.metric("Groundedness", f"{results['grounded'].mean():.0%}")
        st.caption(f"Scored against {len(results)} hand-written question/answer pairs, each grounded in a specific section citation.")

        with st.expander("Methodology and known limitations"):
            st.markdown(
                "- **Retrieval hit** checks whether the correct section was retrieved, matched against "
                "each answer's cited section marker.\n"
                "- **Correctness** measures whether the generated answer matched the referenced answer.\n"
                "- **Groundedness** measures whether the answer was actually supported by the retrieved document.\n"
                "- **Correctness** and **groundedness** are graded by an LLM-as-judge comparing the "
                "generated answer against the retrieved context and the reference answer.\n"
                "- One question — whether a First Schedule exclusion overrides a general annual-leave "
                "provision — isn't reliably resolved: the query-generation step doesn't consistently "
                "search exclusion schedules unless the question hints at one. When unresolved, the "
                "system hedges rather than asserting a wrong answer."
            )
    except FileNotFoundError:
        st.caption("Run the eval harness to populate this tab.")
