"""
app.py — Local RAG Document Expert
Tabs: Chat | Analyze | Summaries | Research Tools | Annotations
"""

import os
import shutil
from datetime import datetime

import numpy as np
import streamlit as st

from ingest import ingest
from rag_chain import get_source_docs, stream_query, find_citation
from analyzer import (
    analyze_text, summarize_paper, generate_literature_review,
    build_glossary, identify_research_gaps, compare_papers, analyze_image,
)
from extractor import (
    extract_text_from_pdf, extract_text_from_uploaded,
    find_section, get_available_sections, get_raw_preview,
    build_section_map, list_ingested_pdfs,
)
from exporter import export_as_markdown, export_as_txt, export_annotations_as_markdown

st.set_page_config(page_title="Local RAG Expert", page_icon="🧠", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "messages":    [],
    "chroma_db":   None,
    "summaries":   {},
    "annotations": [],
    "section_text":   "",
    "section_choice": "Abstract",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Expert")
    st.caption("Private. Local. Grounded.")
    st.divider()
    st.subheader("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    ingest_btn = st.button("⚙️ Ingest Documents", disabled=not uploaded_files, use_container_width=True)

    if ingest_btn:
        DOCS_FOLDER = "documents"
        if os.path.exists(DOCS_FOLDER):
            shutil.rmtree(DOCS_FOLDER)
        os.makedirs(DOCS_FOLDER)

        for file in uploaded_files:
            with open(os.path.join(DOCS_FOLDER, file.name), "wb") as f:
                f.write(file.getbuffer())

        progress_bar = st.progress(0, text="Starting ingestion...")
        status_text  = st.empty()

        def update_progress(current, total):
            pct = int((current / total) * 100)
            progress_bar.progress(pct, text=f"Embedding chunks... {current}/{total}")
            status_text.caption(f"{pct}% complete")

        try:
            db = ingest(DOCS_FOLDER, progress_callback=update_progress)
            st.session_state["chroma_db"]  = db
            st.session_state["messages"]   = []
            st.session_state["summaries"]  = {}
            st.session_state.pop("section_text", None)
            progress_bar.progress(100, text="✅ Done!")
            status_text.empty()
            st.success(f"✅ {len(uploaded_files)} file(s) indexed.")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ingestion failed: {e}")

    st.divider()
    if st.session_state["chroma_db"] is not None:
        st.success("✅ Vector store is ready")
    else:
        st.warning("⚠️ No documents ingested yet")

    st.divider()

    # Export chat history
    if st.session_state["messages"]:
        st.subheader("📤 Export Chat")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Markdown",
                data=export_as_markdown(st.session_state["messages"]),
                file_name="chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇️ Text",
                data=export_as_txt(st.session_state["messages"]),
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
        st.divider()

    st.caption("**Stack:** LangChain · ChromaDB · Groq (LLaMA 3.3) · Streamlit")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat",
    "🔬 Analyze",
    "📝 Summaries",
    "📚 Research Tools",
    "📌 Annotations",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Chat with Your Documents")

    # Chat controls
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        comparison_mode = st.toggle("🔀 Cross-paper comparison mode", value=False,
                                     help="Compares information across all ingested papers")
    with col_b:
        show_citations = st.toggle("🔖 Show citation highlights", value=True,
                                    help="Highlights the exact sentence that answers each question")
    with col_c:
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

    st.divider()

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_citations and msg.get("citation"):
                with st.expander("🔖 Best matching citation"):
                    st.markdown(f"**From:** `{msg['citation_source']}`")
                    st.info(f'*"...{msg["citation"]}..."*')
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources used"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}** — `{os.path.basename(src['source'])}`")
                        st.caption(src["content"][:300] + "...")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state["chroma_db"] is None:
            st.error("Please ingest documents first using the sidebar.")
            st.stop()

        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            try:
                source_docs = get_source_docs(prompt)
                history     = st.session_state["messages"][:-1]  # exclude current question

                # Stream the response
                response = st.write_stream(
                    stream_query(prompt, source_docs, history, comparison_mode)
                )

                # Find citation
                citation, citation_source = find_citation(response, source_docs)

                # Show citation
                if show_citations and citation:
                    with st.expander("🔖 Best matching citation"):
                        st.markdown(f"**From:** `{citation_source}`")
                        st.info(f'*"...{citation}..."*')

                # Show sources
                with st.expander("📄 Sources used"):
                    for i, doc in enumerate(source_docs, 1):
                        src = os.path.basename(doc.metadata.get("source", "unknown"))
                        st.markdown(f"**Chunk {i}** — `{src}`")
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()

                # Save to history
                st.session_state["messages"].append({
                    "role":           "assistant",
                    "content":        response,
                    "citation":       citation,
                    "citation_source": citation_source,
                    "sources":        [{"source": d.metadata.get("source",""), "content": d.page_content} for d in source_docs],
                })

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err})


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔬 Analyze Text")
    st.caption("Break down any section of a paper into plain English, or analyze a figure/chart.")

    analyze_mode = st.radio("Mode", ["📄 Text section", "🖼️ Image / Figure"], horizontal=True)

    # ── Image analysis mode ──────────────────────────────────────────────────
    if analyze_mode == "🖼️ Image / Figure":
        st.caption("Upload a figure, chart, or table from a paper for plain-English analysis.")
        img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key="img_upload")
        img_question = st.text_input("What do you want to know about this figure?",
                                      placeholder="Describe this figure and explain what the results show.")

        if img_file and st.button("🔍 Analyze Figure"):
            with st.spinner("Analyzing image with vision model..."):
                try:
                    result = analyze_image(img_file.getvalue(), img_question)
                    st.markdown("### Figure Analysis")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Image analysis failed: {e}")

    # ── Text section mode ────────────────────────────────────────────────────
    else:
        source_mode = st.radio(
            "Choose your source",
            ["📂 From ingested documents", "📤 Upload a new PDF", "✏️ Paste text manually"],
            horizontal=True,
        )

        extracted_text = ""

        if source_mode == "📂 From ingested documents":
            ingested = list_ingested_pdfs("documents")
            if not ingested:
                st.warning("No documents ingested yet.")
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
                    with st.expander("🔍 Raw extracted text", expanded=False):
                        st.text(get_raw_preview(extracted_text, 2000))
                    available = get_available_sections(extracted_text)
                    section_choice = st.selectbox("Select a section", available)
                    if st.button("📥 Extract Section", key="extract_a"):
                        text = find_section(extracted_text, section_choice)
                        if text:
                            st.session_state["section_text"]   = text
                            st.session_state["section_choice"] = section_choice
                        else:
                            st.warning("Section not found. Try 'Paste text manually'.")

        elif source_mode == "📤 Upload a new PDF":
            upload = st.file_uploader("Upload a PDF", type=["pdf"], key="analyze_upload")
            if upload:
                with st.spinner("Extracting text..."):
                    extracted_text = extract_text_from_uploaded(upload)
                if extracted_text:
                    with st.expander("🔍 Raw extracted text", expanded=False):
                        st.text(get_raw_preview(extracted_text, 2000))
                    available = get_available_sections(extracted_text)
                    section_choice = st.selectbox("Select a section", available)
                    if st.button("📥 Extract Section", key="extract_b"):
                        text = find_section(extracted_text, section_choice)
                        if text:
                            st.session_state["section_text"]   = text
                            st.session_state["section_choice"] = section_choice
                        else:
                            st.warning("Section not found. Try 'Paste text manually'.")

        elif source_mode == "✏️ Paste text manually":
            section_choice = st.selectbox("What are you analyzing?", list(["Title","Abstract","Introduction",
                "Methods","Results","Discussion / Conclusion","Custom section"]))
            pasted = st.text_area("Paste your text here", height=200)
            if pasted.strip():
                st.session_state["section_text"]   = pasted.strip()
                st.session_state["section_choice"] = section_choice

        # Analysis output
        if st.session_state.get("section_text"):
            st.divider()
            with st.expander("📄 Section text to analyze", expanded=False):
                st.text(st.session_state["section_text"][:1500] + "...")

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔍 Analyze"):
                    with st.spinner("Analyzing..."):
                        try:
                            result = analyze_text(
                                st.session_state["section_text"],
                                st.session_state["section_choice"],
                            )
                            st.markdown("### Plain-English Breakdown")
                            st.markdown(result)
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
            with col2:
                # Annotate button — saves passage to annotations
                note = st.text_input("Add a note to save this passage as an annotation:", key="analyze_note")
                if st.button("📌 Save as Annotation"):
                    st.session_state["annotations"].append({
                        "doc":       st.session_state["section_choice"],
                        "passage":   st.session_state["section_text"][:500],
                        "note":      note,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    st.success("Annotation saved!")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SUMMARIES
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📝 Paper Summaries")
    st.caption("Generate plain-English abstracts for each ingested paper.")

    ingested = list_ingested_pdfs("documents")
    if not ingested:
        st.warning("No documents ingested yet.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("⚡ Summarize All", use_container_width=True):
                for filename in ingested:
                    if filename not in st.session_state["summaries"]:
                        fp = os.path.join("documents", filename)
                        with st.spinner(f"Summarizing {filename}..."):
                            try:
                                text = extract_text_from_pdf(fp) if filename.endswith(".pdf") else open(fp, "r", encoding="utf-8", errors="ignore").read()
                                st.session_state["summaries"][filename] = summarize_paper(text, filename)
                            except Exception as e:
                                st.session_state["summaries"][filename] = f"Error: {e}"

        st.divider()

        for filename in ingested:
            with st.expander(f"📄 {filename}", expanded=True):
                if filename in st.session_state["summaries"]:
                    st.markdown(st.session_state["summaries"][filename])
                else:
                    if st.button("Generate", key=f"sum_{filename}"):
                        fp = os.path.join("documents", filename)
                        with st.spinner("Summarizing..."):
                            try:
                                text = extract_text_from_pdf(fp) if filename.endswith(".pdf") else open(fp, "r", encoding="utf-8", errors="ignore").read()
                                st.session_state["summaries"][filename] = summarize_paper(text, filename)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH TOOLS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📚 Research Tools")

    tool = st.selectbox("Select a tool", [
        "📖 Literature Review Generator",
        "📘 Glossary Builder",
        "🔍 Research Gap Identifier",
        "⚖️ Cross-Paper Comparison",
        "🗺️ Paper Relationship Map",
    ])

    ingested = list_ingested_pdfs("documents")

    def _load_texts(files: list) -> dict:
        texts = {}
        for f in files:
            fp = os.path.join("documents", f)
            try:
                texts[f] = extract_text_from_pdf(fp) if f.endswith(".pdf") else open(fp, "r", encoding="utf-8", errors="ignore").read()
            except:
                pass
        return texts

    def _ensure_summaries(files: list):
        """Make sure all files have summaries generated."""
        texts = _load_texts(files)
        for f in files:
            if f not in st.session_state["summaries"] and f in texts:
                with st.spinner(f"Generating summary for {f}..."):
                    st.session_state["summaries"][f] = summarize_paper(texts[f], f)

    # ── Literature Review ──────────────────────────────────────────────────
    if tool == "📖 Literature Review Generator":
        st.subheader("Literature Review Generator")
        st.caption("Selects from your ingested papers and writes a structured literature review section.")

        if not ingested:
            st.warning("No documents ingested yet.")
        else:
            selected = st.multiselect("Select papers to include", ingested, default=ingested)
            if st.button("✍️ Generate Literature Review") and selected:
                _ensure_summaries(selected)
                sums = {f: st.session_state["summaries"][f] for f in selected if f in st.session_state["summaries"]}
                if sums:
                    with st.spinner("Writing literature review..."):
                        try:
                            result = generate_literature_review(sums)
                            st.markdown("### Generated Literature Review")
                            st.markdown(result)
                            st.download_button("⬇️ Download", result.encode(), "literature_review.md", "text/markdown")
                        except Exception as e:
                            st.error(f"Failed: {e}")
                else:
                    st.warning("Generate summaries first in the Summaries tab.")

    # ── Glossary Builder ───────────────────────────────────────────────────
    elif tool == "📘 Glossary Builder":
        st.subheader("Glossary Builder")
        st.caption("Extracts and defines all technical terms across your ingested papers.")

        if not ingested:
            st.warning("No documents ingested yet.")
        else:
            selected = st.multiselect("Select papers", ingested, default=ingested)
            if st.button("📘 Build Glossary") and selected:
                texts = _load_texts(selected)
                with st.spinner("Extracting terms..."):
                    try:
                        result = build_glossary(texts)
                        st.markdown("### Technical Glossary")
                        st.markdown(result)
                        st.download_button("⬇️ Download", result.encode(), "glossary.md", "text/markdown")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # ── Research Gap Identifier ────────────────────────────────────────────
    elif tool == "🔍 Research Gap Identifier":
        st.subheader("Research Gap Identifier")
        st.caption("Analyzes multiple papers and surfaces what questions remain unanswered.")

        if not ingested:
            st.warning("No documents ingested yet.")
        else:
            selected = st.multiselect("Select papers", ingested, default=ingested)
            if st.button("🔍 Identify Gaps") and selected:
                _ensure_summaries(selected)
                sums = {f: st.session_state["summaries"][f] for f in selected if f in st.session_state["summaries"]}
                if sums:
                    with st.spinner("Analyzing research gaps..."):
                        try:
                            result = identify_research_gaps(sums)
                            st.markdown("### Research Gaps Analysis")
                            st.markdown(result)
                            st.download_button("⬇️ Download", result.encode(), "research_gaps.md", "text/markdown")
                        except Exception as e:
                            st.error(f"Failed: {e}")

    # ── Cross-Paper Comparison ─────────────────────────────────────────────
    elif tool == "⚖️ Cross-Paper Comparison":
        st.subheader("Cross-Paper Comparison")
        st.caption("Ask a specific comparison question across your ingested papers.")

        if not ingested:
            st.warning("No documents ingested yet.")
        else:
            selected = st.multiselect("Select papers to compare", ingested, default=ingested)
            cmp_question = st.text_input("What do you want to compare?",
                                          placeholder="How do these papers differ in their methodology?")
            if st.button("⚖️ Compare") and selected and cmp_question:
                texts = _load_texts(selected)
                with st.spinner("Comparing papers..."):
                    try:
                        result = compare_papers(cmp_question, texts)
                        st.markdown("### Comparative Analysis")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # ── Paper Relationship Map ─────────────────────────────────────────────
    elif tool == "🗺️ Paper Relationship Map":
        st.subheader("Paper Relationship Map")
        st.caption("Visualizes thematic similarity between your ingested papers.")

        if not ingested or len(ingested) < 2:
            st.warning("Ingest at least 2 documents to generate a relationship map.")
        else:
            if st.button("🗺️ Generate Map"):
                _ensure_summaries(ingested)
                sums = {f: st.session_state["summaries"].get(f, "") for f in ingested}

                with st.spinner("Computing paper similarities..."):
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        import plotly.graph_objects as go
                        import networkx as nx

                        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        names    = list(sums.keys())
                        vecs     = np.array(embedder.embed_documents(list(sums.values())))

                        # Cosine similarity matrix
                        norms   = np.linalg.norm(vecs, axis=1, keepdims=True)
                        normed  = vecs / np.where(norms == 0, 1, norms)
                        sim_mat = normed @ normed.T

                        # Build network graph
                        G = nx.Graph()
                        short_names = [os.path.basename(n).replace(".pdf","").replace(".txt","")[:25] for n in names]
                        for name in short_names:
                            G.add_node(name)

                        threshold = 0.3  # only draw edges above this similarity
                        for i in range(len(names)):
                            for j in range(i + 1, len(names)):
                                sim = float(sim_mat[i, j])
                                if sim > threshold:
                                    G.add_edge(short_names[i], short_names[j], weight=sim)

                        pos = nx.spring_layout(G, seed=42)

                        # Build plotly figure
                        edge_traces = []
                        for u, v, data in G.edges(data=True):
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            sim_val = data.get("weight", 0)
                            edge_traces.append(go.Scatter(
                                x=[x0, x1, None], y=[y0, y1, None],
                                mode="lines",
                                line=dict(width=sim_val * 5, color=f"rgba(100,150,255,{sim_val:.2f})"),
                                hoverinfo="none",
                            ))

                        node_x = [pos[n][0] for n in G.nodes()]
                        node_y = [pos[n][1] for n in G.nodes()]
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode="markers+text",
                            text=list(G.nodes()), textposition="top center",
                            marker=dict(size=20, color="#4C9BE8", line=dict(width=2, color="white")),
                            hovertemplate="<b>%{text}</b><extra></extra>",
                        )

                        fig = go.Figure(
                            data=edge_traces + [node_trace],
                            layout=go.Layout(
                                title="Paper Relationship Map (edge thickness = similarity)",
                                showlegend=False,
                                hovermode="closest",
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=500,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Papers with thicker edges share more thematic content. Similarity threshold: 0.30")

                    except Exception as e:
                        st.error(f"Map generation failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANNOTATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📌 Annotations")
    st.caption("Save passages and notes from your research. Annotations last for this session.")

    # Add new annotation manually
    with st.expander("➕ Add new annotation", expanded=False):
        ann_doc     = st.text_input("Document name / source")
        ann_passage = st.text_area("Passage to annotate", height=100)
        ann_note    = st.text_area("Your note", height=80)
        if st.button("Save Annotation") and ann_passage:
            st.session_state["annotations"].append({
                "doc":       ann_doc or "Manual entry",
                "passage":   ann_passage,
                "note":      ann_note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.success("Annotation saved!")
            st.rerun()

    st.divider()

    annotations = st.session_state["annotations"]
    if not annotations:
        st.info("No annotations yet. Add them here or via the Analyze tab.")
    else:
        st.write(f"**{len(annotations)} annotation(s)**")

        # Export annotations
        st.download_button(
            "⬇️ Export All Annotations",
            data=export_annotations_as_markdown(annotations),
            file_name="annotations.md",
            mime="text/markdown",
        )
        st.divider()

        for i, ann in enumerate(annotations):
            with st.expander(f"📌 {ann.get('doc','Unknown')} — {ann.get('timestamp','')}", expanded=True):
                st.markdown(f"**Passage:**")
                st.info(ann.get("passage", ""))
                st.markdown(f"**Note:** {ann.get('note','')}")
                if st.button("🗑️ Delete", key=f"del_ann_{i}"):
                    st.session_state["annotations"].pop(i)
                    st.rerun()
