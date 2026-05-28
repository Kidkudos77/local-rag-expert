"""
rag_chain.py — Retrieval-Augmented Generation Chain
Uses Groq for LLM inference and HuggingFace for embeddings.
Supports both local persistent ChromaDB and in-memory mode for cloud.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "llama-3.3-70b-versatile"
TOP_K        = 5

PROMPT_TEMPLATE = """\
You are a helpful expert assistant. Answer the user's question based ONLY on the
context provided below. Do not use any outside knowledge.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents to answer that."

Context:
{context}

---

Question: {question}

Answer:"""


def format_docs(docs: list) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # If we have an in-memory db stored in session state (cloud mode), use it
    if "chroma_db" in st.session_state and st.session_state["chroma_db"] is not None:
        db = st.session_state["chroma_db"]
    elif os.path.exists(CHROMA_PATH):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )
    else:
        raise ValueError("No vector store found. Please ingest documents first.")

    return db.as_retriever(search_kwargs={"k": TOP_K})


def query(question: str) -> tuple:
    retriever   = get_retriever()
    source_docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm    = ChatGroq(model=LLM_MODEL)

    chain = (
        {"context": lambda _: format_docs(source_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer, source_docs
