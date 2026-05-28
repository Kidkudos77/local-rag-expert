"""
rag_chain.py — RAG Chain with streaming, memory, re-ranking, and citation finding.
"""

import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.3-70b-versatile"
TOP_K       = 8   # Retrieve more, then re-rank down to 5

PROMPT_TEMPLATE = """\
You are a helpful expert research assistant.
{history_section}
Answer the question below based ONLY on the context provided.
If the answer is not in the context, say:
"I don't have enough information in the provided documents to answer that."

Context:
{context}

Question: {question}
Answer:"""

COMPARISON_PROMPT = """\
You are a research assistant that specializes in comparing academic papers.
{history_section}
Using ONLY the context below (which comes from multiple papers), answer the comparison question.
Be specific about which paper says what. Reference paper names when possible.

Context:
{context}

Comparison question: {question}
Answer:"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    sections = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def format_history(messages: list, n: int = 4) -> str:
    """Format last n Q&A pairs as context for the LLM."""
    if not messages:
        return ""
    recent = [m for m in messages if m["role"] in ("user", "assistant")][-n * 2:]
    if not recent:
        return ""
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:400]}")
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


def _cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def rerank_docs(question: str, docs: list, top_k: int = 5) -> list:
    """
    Re-rank retrieved chunks by computing cosine similarity between
    the question embedding and each chunk embedding.
    Returns top_k most relevant chunks.
    """
    if len(docs) <= top_k:
        return docs

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    question_vec = embeddings.embed_query(question)
    chunk_vecs   = embeddings.embed_documents([d.page_content for d in docs])

    scored = [
        (doc, _cosine_similarity(question_vec, vec))
        for doc, vec in zip(docs, chunk_vecs)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def find_citation(answer: str, source_docs: list) -> tuple:
    """
    Find the single sentence in source_docs most likely to have produced the answer.
    Uses word overlap as a simple but effective heuristic.
    Returns (sentence, source_filename).
    """
    answer_words = set(answer.lower().split())
    best_sentence = ""
    best_source   = ""
    best_score    = 0

    for doc in source_docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        # Split on sentence boundaries
        sentences = [s.strip() for s in doc.page_content.replace('\n', ' ').split('.') if len(s.strip()) > 30]
        for sent in sentences:
            sent_words = set(sent.lower().split())
            score = len(answer_words & sent_words)
            if score > best_score:
                best_score    = score
                best_sentence = sent
                best_source   = source

    return best_sentence, best_source


# ── Retrieval ─────────────────────────────────────────────────────────────────

def get_retriever(top_k: int = TOP_K):
    db = st.session_state.get("chroma_db")
    if db is None:
        raise ValueError("No vector store found. Please ingest documents first.")
    return db.as_retriever(search_kwargs={"k": top_k})


# ── Streaming query (main chat function) ─────────────────────────────────────

def get_source_docs(question: str) -> list:
    """Retrieve and re-rank source documents for a question."""
    retriever = get_retriever(top_k=TOP_K)
    raw_docs  = retriever.invoke(question)
    return rerank_docs(question, raw_docs, top_k=5)


def stream_query(question: str, source_docs: list, history: list = None, comparison_mode: bool = False):
    """
    Stream the LLM response token by token.
    Yields string chunks — use with st.write_stream().
    """
    history = history or []
    context = format_docs(source_docs)
    hist    = format_history(history)

    template = COMPARISON_PROMPT if comparison_mode else PROMPT_TEMPLATE
    prompt   = ChatPromptTemplate.from_template(template)
    llm      = ChatGroq(model=LLM_MODEL)
    chain    = prompt | llm | StrOutputParser()

    for chunk in chain.stream({
        "context":         context,
        "question":        question,
        "history_section": hist,
    }):
        yield chunk


# ── Non-streaming fallback (used by Paper Summaries tab) ─────────────────────

def query(question: str, history: list = None, comparison_mode: bool = False) -> tuple:
    source_docs = get_source_docs(question)
    response    = "".join(stream_query(question, source_docs, history, comparison_mode))
    return response, source_docs
