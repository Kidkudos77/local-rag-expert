"""
app.py — Streamlit Frontend
Run with: streamlit run app.py
"""

import os
import shutil

import streamlit as st

from ingest import ingest, CHROMA_PATH
from rag_chain import query
from analyzer import analyze_text, summarize_paper
from extractor import (
    extract_text_from_pdf,
    extract_text_from_uploaded,
    find_section,
    get_available_sections,
    get_raw_preview,
    build_section_map,
    list_ingested_pdfs,
)

st.set_page_config(
    page_title="Local RAG Expert",
    page_icon="🧠",
    layout="wide",
)

# --- Sidebar ---
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
        DOCS_FOLDER = "documents"
        if os.path.exists(DOCS_FOLDER):
            shutil.rmtree(DOCS_FOLDER)
        os.makedirs(DOCS_FOLDER)

        for file in uploaded_files:
            dest = os.path.join(DOCS_FOLDER, file.name)
            with open(dest, "wb") as f:
                f.write(file.getbuffer())

        st.info(f"Processing {len(uploaded_files)} file(s)...")
        progress_bar = st.progress(0, text="Starting ingestion...")
        status_text  = st.empty()

        def update_progress(current, total):
            pct = int((current / total) * 100)
            progress_bar.progress(pct, text=f"Embedding chunks... {current}/{total}")
            status_text.caption(f"{pct}% complete")

        try:
            db = ingest(DOCS_FOLDER, progress_callback=update_progress)
            st.session_state["chroma_db"] = db
            progress_bar.progress(100, text="✅ Done!")
            status_text.empty()
            st.success(f"✅ Ready! {len(uploaded_files)} file(s) indexed.")
            st.session_state.messages = []
            st.session_state.pop("section_text", None)
            st.session_state.pop("summaries", None)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ingestion failed: {e}")

    st.divider()
    if os.path.exists(CHROMA_PATH) or st.session_state.get("chroma_db") is not None:
        st.success("✅ Vector store is ready")
    else:
        st.warning("⚠️ No documents ingested yet")
    st.divider()
    st.caption("**Stack:** LangChain · ChromaDB · Ollama (Llama 3) · Streamlit")


# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "💬 Chat with Documents",
    "🔬 Analyze Text",
    "📝 Paper Summaries",
])


# ── Tab 1: RAG Chat ──────────────────────────────────────────────────────────
with tab1:
    st.header("Chat with Your Documents")
    st.caption("Retrieves answers grounded strictly in your ingested documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents...", key="rag_input"):
        if not os.path.exists(CHROMA_PATH) and st.session_state.get("chroma_db") is None:
            st.error("Please upload and ingest documents first using the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    response, source_docs = query(prompt)
                    st.markdown(response)
                    with st.expander("📄 Sources used"):
                        for i, doc in enumerate(source_docs, 1):
                            source = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Chunk {i}** — `{source}`")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    err_msg = f"Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})


# ── Tab 2: Analyze Text ──────────────────────────────────────────────────────
with tab2:
    st.header("🔬 Analyze Text")
    st.caption("Break down any section of a research paper into plain English.")

    source_mode = st.radio(
        "Choose your source",
        ["📂 From ingested documents", "📤 Upload a new PDF", "✏️ Paste text manually"],
        horizontal=True,
    )

    extracted_text = ""

    if source_mode == "📂 From ingested documents":
        ingested = list_ingested_pdfs("documents")
        if not ingested:
            st.warning("No documents ingested yet. Use the sidebar to ingest files first.")
        else:
            selected_file = st.selectbox("Select a document", ingested)
            file_path = os.path.join("documents", selected_file)

            with st.spinner("Reading document..."):
                if selected_file.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        extracted_text = f.read()

            if extracted_text:
                with st.expander("🔍 Raw extracted text (use to check headings)", expanded=False):
                    st.text(get_raw_preview(extracted_text, 2000))

                available_sections = get_available_sections(extracted_text)
                section_choice = st.selectbox("Select a section", available_sections)

                if st.button("📥 Extract Section", key="extract_a"):
                    section_text = find_section(extracted_text, section_choice)
                    if section_text:
                        st.session_state["section_text"]   = section_text
                        st.session_state["section_choice"] = section_choice
                    else:
                        st.warning("Section not found. Try 'Paste text manually'.")

    elif source_mode == "📤 Upload a new PDF":
        analyze_upload = st.file_uploader("Upload a PDF to analyze", type=["pdf"], key="analyze_uploader")
        if analyze_upload:
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_uploaded(analyze_upload)

            if extracted_text:
                with st.expander("🔍 Raw extracted text (use to check headings)", expanded=False):
                    st.text(get_raw_preview(extracted_text, 2000))

                available_sections = get_available_sections(extracted_text)
                section_choice = st.selectbox("Select a section", available_sections)

                if st.button("📥 Extract Section", key="extract_b"):
                    section_text = find_section(extracted_text, section_choice)
                    if section_text:
                        st.session_state["section_text"]   = section_text
                        st.session_state["section_choice"] = section_choice
                    else:
                        st.warning("Section not found. Try 'Paste text manually'.")

    elif source_mode == "✏️ Paste text manually":
        section_choice = st.selectbox(
            "What are you analyzing?",
            ["Title", "Abstract", "Introduction", "Methods",
             "Results", "Discussion / Conclusion", "Custom section"],
        )
        pasted = st.text_area("Paste your text here", height=200,
                              placeholder="Paste any title, abstract, or section...")
        if pasted.strip():
            st.session_state["section_text"]   = pasted.strip()
            st.session_state["section_choice"] = section_choice

    if st.session_state.get("section_text"):
        st.divider()
        with st.expander("📄 Section text to be analyzed", expanded=False):
            preview = st.session_state["section_text"][:1500]
            st.text(preview + ("..." if len(st.session_state["section_text"]) > 1500 else ""))

        if st.button("🔍 Analyze", use_container_width=False):
            with st.spinner("Analyzing with Llama 3..."):
                try:
                    result = analyze_text(
                        st.session_state["section_text"],
                        st.session_state["section_choice"],
                    )
                    st.markdown("### Plain-English Breakdown")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")


# ── Tab 3: Paper Summaries ───────────────────────────────────────────────────
with tab3:
    st.header("📝 Paper Summaries")
    st.caption("Generate a plain-English abstract for each ingested paper.")

    ingested = list_ingested_pdfs("documents")

    if not ingested:
        st.warning("No documents ingested yet. Use the sidebar to ingest files first.")
    else:
        st.write(f"**{len(ingested)} document(s) available:**")

        # Initialize summaries cache in session state
        if "summaries" not in st.session_state:
            st.session_state["summaries"] = {}

        # Option to summarize all at once or one at a time
        col1, col2 = st.columns([1, 3])
        with col1:
            summarize_all = st.button(
                "⚡ Summarize All",
                use_container_width=True,
                help="Generates summaries for all ingested documents"
            )

        if summarize_all:
            for filename in ingested:
                if filename not in st.session_state["summaries"]:
                    file_path = os.path.join("documents", filename)
                    with st.spinner(f"Summarizing {filename}..."):
                        try:
                            if filename.endswith(".pdf"):
                                text = extract_text_from_pdf(file_path)
                            else:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    text = f.read()
                            summary = summarize_paper(text, filename)
                            st.session_state["summaries"][filename] = summary
                        except Exception as e:
                            st.session_state["summaries"][filename] = f"Error: {e}"

        st.divider()

        # Display each paper with its summary
        for filename in ingested:
            with st.expander(f"📄 {filename}", expanded=True):
                if filename in st.session_state["summaries"]:
                    st.markdown(st.session_state["summaries"][filename])
                else:
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        if st.button("Generate", key=f"sum_{filename}"):
                            file_path = os.path.join("documents", filename)
                            with st.spinner("Summarizing..."):
                                try:
                                    if filename.endswith(".pdf"):
                                        text = extract_text_from_pdf(file_path)
                                    else:
                                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                            text = f.read()
                                    summary = summarize_paper(text, filename)
                                    st.session_state["summaries"][filename] = summary
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                    with col_b:
                        st.caption("Click Generate to create a plain-English summary.")
