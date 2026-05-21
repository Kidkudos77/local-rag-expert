"""
app.py — Streamlit Frontend

What this does:
  - Sidebar: accepts file uploads, triggers ingestion pipeline
  - Main area: persistent chat interface backed by the RAG chain

Run with: streamlit run app.py
"""

import os
import shutil

import streamlit as st

from ingest import ingest, CHROMA_PATH
from rag_chain import query

# --- Page Config ---
st.set_page_config(
    page_title="Local RAG Expert",
    page_icon="🧠",
    layout="wide",
)

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.title("🧠 RAG Expert")
    st.caption("Private. Local. Grounded.")

    st.divider()
    st.subheader("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Files stay on your machine. Nothing is sent to the cloud.",
    )

    ingest_btn = st.button(
        "⚙️  Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_btn:
        # Save uploads to a temp folder
        DOCS_FOLDER = "documents"
        if os.path.exists(DOCS_FOLDER):
            shutil.rmtree(DOCS_FOLDER)
        os.makedirs(DOCS_FOLDER)

        for file in uploaded_files:
            dest = os.path.join(DOCS_FOLDER, file.name)
            with open(dest, "wb") as f:
                f.write(file.getbuffer())

        with st.spinner(f"Ingesting {len(uploaded_files)} file(s)... this may take a minute."):
            try:
                ingest(DOCS_FOLDER)
                st.success(f"✅ Ready! {len(uploaded_files)} file(s) indexed.")
                # Clear chat history when new docs are loaded
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    # Show current status
    st.divider()
    if os.path.exists(CHROMA_PATH):
        st.success("✅ Vector store is ready")
    else:
        st.warning("⚠️ No documents ingested yet")

    st.divider()
    st.caption(
        "**Stack:** LangChain · ChromaDB · Ollama (Llama 3) · Streamlit"
    )


# --- Main Area: Chat Interface ---
st.title("Chat with Your Documents")
st.caption("Answers are grounded in your uploaded documents only — no hallucination from outside knowledge.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):

    if not os.path.exists(CHROMA_PATH):
        st.error("Please upload and ingest documents first using the sidebar.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
