"""
app.py — Local RAG Document Expert
Tabs: Chat | Analyze | Summaries | Research Tools | Annotations | Writing Assistant
"""

import os
import shutil
from datetime import datetime

import numpy as np
import streamlit as st

from ingest import ingest
from rag_chain import get_source_docs, stream_query, find_citation, get_all_sources
from analyzer import (
    analyze_text, summarize_paper, generate_literature_review,
    build_glossary, identify_research_gaps, compare_papers,
    ask_primary_paper, suggest_questions, improve_writing,
    format_citations, find_similar_paper, analyze_image,
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
    "messages":       [],
    "chroma_db":      None,
    "summaries":      {},
    "annotations":    [],
    "section_text":   "",
    "section_choice": "Abstract",
    "questions":      [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

DOCS_FOLDER = "documents"


# ── Shared helpers ────────────────────────────────────────────────────────────
def _load_texts(files: list) -> dict:
    texts = {}
    for f in files:
        fp = os.path.join(DOCS_FOLDER, f)
        try:
            texts[f] = extract_text_from_pdf(fp) if f.endswith(".pdf") else open(fp, "r", encoding="utf-8", errors="ignore").read()
        except:
            pass
    return texts


def _ensure_summaries(files: list):
    texts = _load_texts(files)
    for f in files:
        if f not in st.session_state["summaries"] and f in texts:
            with st.spinner(f"Generating summary for {f}..."):
                st.session_state["summaries"][f] = summarize_paper(texts[f], f)


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

    # Save files immediately on upload so they persist across reruns
    # Clear old staged files first to prevent duplicates from previous sessions
    if uploaded_files:
        if os.path.exists(DOCS_FOLDER):
            shutil.rmtree(DOCS_FOLDER)
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        for file in uploaded_files:
            dest = os.path.join(DOCS_FOLDER, file.name)
            with open(dest, "wb") as f:
                f.write(file.getbuffer())

    # Show staged files
    staged = [f for f in os.listdir(DOCS_FOLDER) if f.endswith((".pdf", ".txt"))] if os.path.exists(DOCS_FOLDER) else []
    if staged:
        st.caption(f"📂 {len(staged)} file(s) staged:")
        for fname in staged:
            st.caption(f"  • {fname}")

    col_ingest, col_clear = st.columns([3, 1])
    with col_ingest:
        ingest_btn = st.button("⚙️ Ingest Documents", disabled=not staged, use_container_width=True)
    with col_clear:
        if st.button("🗑️", help="Clear staged files", disabled=not staged):
            if os.path.exists(DOCS_FOLDER):
                shutil.rmtree(DOCS_FOLDER)
            st.rerun()

    if ingest_btn and staged:
        progress_bar = st.progress(0, text="Starting ingestion...")
        status_text  = st.empty()

        def update_progress(current, total):
            pct = int((current / total) * 100)
            progress_bar.progress(pct, text=f"Embedding chunks... {current}/{total}")
            status_text.caption(f"{pct}% complete")

        try:
            db = ingest(DOCS_FOLDER, progress_callback=update_progress)
            st.session_state["chroma_db"]      = db
            st.session_state["messages"]         = []
            st.session_state["summaries"]        = {}
            st.session_state["questions"]        = []
            st.session_state["ingested_files"]   = staged.copy()
            st.session_state.pop("section_text", None)
            progress_bar.progress(100, text="✅ Done!")
            status_text.empty()
            st.success(f"✅ {len(staged)} file(s) indexed.")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ingestion failed: {e}")

    st.divider()
    if st.session_state["chroma_db"] is not None:
        st.success("✅ Vector store is ready")
        loaded = st.session_state.get("ingested_files", [])
        if loaded:
            st.caption(f"**{len(loaded)} paper(s) loaded:**")
            for p in loaded:
                st.caption(f"  📄 {p}")
    else:
        st.warning("⚠️ No documents ingested yet")

    st.divider()

    # Export chat history
    if st.session_state["messages"]:
        st.subheader("📤 Export Chat")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Markdown", data=export_as_markdown(st.session_state["messages"]),
                               file_name="chat_history.md", mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("⬇️ Text", data=export_as_txt(st.session_state["messages"]),
                               file_name="chat_history.txt", mime="text/plain", use_container_width=True)
        st.divider()

    st.caption("**Stack:** LangChain · ChromaDB · Groq (LLaMA 3.3) · Streamlit")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat",
    "🔬 Analyze",
    "📝 Summaries",
    "📚 Research Tools",
    "📌 Annotations",
    "✍️ Writing Assistant",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Chat with Your Documents")

    col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])
    with col_a:
        comparison_mode = st.toggle("🔀 Cross-paper comparison", value=False)
    with col_b:
        show_citations  = st.toggle("🔖 Citation highlights", value=True)
    with col_c:
        # Primary paper mode
        ingested = list_ingested_pdfs(DOCS_FOLDER)
        primary_options = ["None (search all papers)"] + ingested
        primary_choice  = st.selectbox("📌 Primary paper", primary_options, label_visibility="collapsed",
                                        help="Pin one paper as primary source")
    with col_d:
        if st.button("🗑️ Clear"):
            st.session_state["messages"] = []
            st.rerun()

    # Question suggester — show suggested questions if available
    if st.session_state["questions"]:
        with st.expander("💡 Suggested questions (click to ask)", expanded=False):
            for q in st.session_state["questions"]:
                if st.button(q, key=f"sq_{q[:30]}"):
                    st.session_state["_auto_prompt"] = q
                    st.rerun()

    st.divider()

    # Render history
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

    # Handle auto-prompt from question suggester
    auto_prompt = st.session_state.pop("_auto_prompt", None)
    prompt = st.chat_input("Ask a question about your documents...") or auto_prompt

    if prompt:
        if st.session_state["chroma_db"] is None:
            st.error("Please ingest documents first.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                history = st.session_state["messages"][:-1]

                # Primary paper mode — answer from primary + compare others
                if primary_choice != "None (search all papers)":
                    texts = _load_texts(ingested)
                    primary_text  = texts.get(primary_choice, "")
                    other_texts   = {k: v for k, v in texts.items() if k != primary_choice}
                    with st.spinner("Answering from primary paper..."):
                        response = ask_primary_paper(prompt, primary_text, primary_choice, other_texts)
                    st.markdown(response)
                    source_docs = []
                    citation = citation_source = ""
                else:
                    source_docs = get_source_docs(prompt)
                    response = st.write_stream(stream_query(prompt, source_docs, history, comparison_mode))
                    citation, citation_source = find_citation(response, source_docs)

                    if show_citations and citation:
                        with st.expander("🔖 Best matching citation"):
                            st.markdown(f"**From:** `{citation_source}`")
                            st.info(f'*"...{citation}..."*')

                    with st.expander("📄 Sources used"):
                        for i, doc in enumerate(source_docs, 1):
                            src = os.path.basename(doc.metadata.get("source", "unknown"))
                            st.markdown(f"**Chunk {i}** — `{src}`")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

                st.session_state["messages"].append({
                    "role":            "assistant",
                    "content":         response,
                    "citation":        citation if primary_choice == "None (search all papers)" else "",
                    "citation_source": citation_source if primary_choice == "None (search all papers)" else "",
                    "sources":         [{"source": d.metadata.get("source",""), "content": d.page_content} for d in source_docs],
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
    analyze_mode = st.radio("Mode", ["📄 Text section", "🖼️ Image / Figure"], horizontal=True)

    if analyze_mode == "🖼️ Image / Figure":
        img_file     = st.file_uploader("Upload a figure or chart", type=["png","jpg","jpeg","webp"], key="img_upload")
        img_question = st.text_input("What do you want to know?", placeholder="Describe this figure and explain what the results show.")
        if img_file and st.button("🔍 Analyze Figure"):
            with st.spinner("Analyzing image..."):
                try:
                    st.markdown("### Figure Analysis")
                    st.markdown(analyze_image(img_file.getvalue(), img_question))
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        source_mode = st.radio("Source", ["📂 From ingested documents", "📤 Upload a new PDF", "✏️ Paste text manually"], horizontal=True)
        extracted_text = ""

        if source_mode == "📂 From ingested documents":
            ingested = list_ingested_pdfs(DOCS_FOLDER)
            if not ingested:
                st.warning("No documents ingested yet.")
            else:
                sel = st.selectbox("Select a document", ingested)
                fp  = os.path.join(DOCS_FOLDER, sel)
                with st.spinner("Reading..."):
                    extracted_text = extract_text_from_pdf(fp) if sel.endswith(".pdf") else open(fp, "r", encoding="utf-8", errors="ignore").read()
                if extracted_text:
                    with st.expander("🔍 Raw text", expanded=False):
                        st.text(get_raw_preview(extracted_text, 2000))
                    section_choice = st.selectbox("Select a section", get_available_sections(extracted_text))
                    if st.button("📥 Extract Section", key="ext_a"):
                        t = find_section(extracted_text, section_choice)
                        if t:
                            st.session_state["section_text"]   = t
                            st.session_state["section_choice"] = section_choice
                        else:
                            st.warning("Section not found. Try paste mode.")

        elif source_mode == "📤 Upload a new PDF":
            upload = st.file_uploader("Upload PDF", type=["pdf"], key="analyze_upload")
            if upload:
                with st.spinner("Extracting..."):
                    extracted_text = extract_text_from_uploaded(upload)
                if extracted_text:
                    with st.expander("🔍 Raw text", expanded=False):
                        st.text(get_raw_preview(extracted_text, 2000))
                    section_choice = st.selectbox("Select a section", get_available_sections(extracted_text))
                    if st.button("📥 Extract Section", key="ext_b"):
                        t = find_section(extracted_text, section_choice)
                        if t:
                            st.session_state["section_text"]   = t
                            st.session_state["section_choice"] = section_choice
                        else:
                            st.warning("Section not found. Try paste mode.")

        elif source_mode == "✏️ Paste text manually":
            section_choice = st.selectbox("What are you analyzing?", ["Title","Abstract","Introduction","Methods","Results","Discussion / Conclusion","Custom section"])
            pasted = st.text_area("Paste your text here", height=200)
            if pasted.strip():
                st.session_state["section_text"]   = pasted.strip()
                st.session_state["section_choice"] = section_choice

        if st.session_state.get("section_text"):
            st.divider()
            with st.expander("📄 Section to analyze", expanded=False):
                st.text(st.session_state["section_text"][:1500] + "...")

            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("🔍 Analyze"):
                    with st.spinner("Analyzing..."):
                        try:
                            st.markdown("### Plain-English Breakdown")
                            st.markdown(analyze_text(st.session_state["section_text"], st.session_state["section_choice"]))
                        except Exception as e:
                            st.error(f"Failed: {e}")
            with c2:
                note = st.text_input("Add a note to save as annotation:", key="analyze_note")
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
    ingested = list_ingested_pdfs(DOCS_FOLDER)

    if not ingested:
        st.warning("No documents ingested yet.")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("⚡ Summarize All", use_container_width=True):
                texts = _load_texts(ingested)
                for f in ingested:
                    if f not in st.session_state["summaries"] and f in texts:
                        with st.spinner(f"Summarizing {f}..."):
                            try:
                                st.session_state["summaries"][f] = summarize_paper(texts[f], f)
                            except Exception as e:
                                st.session_state["summaries"][f] = f"Error: {e}"
        with c2:
            if st.button("💡 Suggest Questions", use_container_width=True, help="Generate 10 questions to ask in the Chat tab"):
                texts = _load_texts(ingested)
                with st.spinner("Generating questions..."):
                    try:
                        st.session_state["questions"] = suggest_questions(texts)
                        st.success("Questions ready — check the Chat tab!")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        st.divider()

        for f in ingested:
            with st.expander(f"📄 {f}", expanded=True):
                if f in st.session_state["summaries"]:
                    st.markdown(st.session_state["summaries"][f])
                else:
                    if st.button("Generate", key=f"sum_{f}"):
                        fp = os.path.join(DOCS_FOLDER, f)
                        with st.spinner("Summarizing..."):
                            try:
                                text = extract_text_from_pdf(fp) if f.endswith(".pdf") else open(fp, "r", encoding="utf-8", errors="ignore").read()
                                st.session_state["summaries"][f] = summarize_paper(text, f)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH TOOLS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📚 Research Tools")
    ingested = list_ingested_pdfs(DOCS_FOLDER)

    tool = st.selectbox("Select a tool", [
        "📖 Literature Review Generator",
        "📘 Glossary Builder",
        "🔍 Research Gap Identifier",
        "⚖️ Cross-Paper Comparison",
        "🗺️ Paper Relationship Map",
        "🔎 Similarity Search",
        "📋 Citation Formatter",
    ])

    if not ingested:
        st.warning("No documents ingested yet.")
    else:
        # ── Literature Review ──────────────────────────────────────────────
        if tool == "📖 Literature Review Generator":
            st.caption("Writes a structured literature review from your ingested papers.")
            selected = st.multiselect("Select papers", ingested, default=ingested)
            if st.button("✍️ Generate") and selected:
                _ensure_summaries(selected)
                sums = {f: st.session_state["summaries"][f] for f in selected if f in st.session_state["summaries"]}
                if sums:
                    with st.spinner("Writing..."):
                        try:
                            result = generate_literature_review(sums)
                            st.markdown("### Literature Review")
                            st.markdown(result)
                            st.download_button("⬇️ Download", result.encode(), "literature_review.md", "text/markdown")
                        except Exception as e:
                            st.error(f"Failed: {e}")

        # ── Glossary ───────────────────────────────────────────────────────
        elif tool == "📘 Glossary Builder":
            st.caption("Extracts and defines all technical terms across your papers.")
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

        # ── Research Gaps ──────────────────────────────────────────────────
        elif tool == "🔍 Research Gap Identifier":
            st.caption("Identifies what questions remain unanswered across your papers.")
            selected = st.multiselect("Select papers", ingested, default=ingested)
            if st.button("🔍 Identify Gaps") and selected:
                _ensure_summaries(selected)
                sums = {f: st.session_state["summaries"][f] for f in selected if f in st.session_state["summaries"]}
                if sums:
                    with st.spinner("Analyzing..."):
                        try:
                            result = identify_research_gaps(sums)
                            st.markdown("### Research Gaps")
                            st.markdown(result)
                            st.download_button("⬇️ Download", result.encode(), "research_gaps.md", "text/markdown")
                        except Exception as e:
                            st.error(f"Failed: {e}")

        # ── Cross-Paper Comparison ─────────────────────────────────────────
        elif tool == "⚖️ Cross-Paper Comparison":
            st.caption("Ask a specific comparison question across multiple papers.")
            selected    = st.multiselect("Select papers", ingested, default=ingested)
            cmp_question = st.text_input("Comparison question", placeholder="How do these papers differ in their methodology?")
            if st.button("⚖️ Compare") and selected and cmp_question:
                texts = _load_texts(selected)
                with st.spinner("Comparing..."):
                    try:
                        result = compare_papers(cmp_question, texts)
                        st.markdown("### Comparative Analysis")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # ── Paper Relationship Map ─────────────────────────────────────────
        elif tool == "🗺️ Paper Relationship Map":
            st.caption("Visualizes thematic similarity between papers.")
            if len(ingested) < 2:
                st.warning("Ingest at least 2 documents to generate a map.")
            elif st.button("🗺️ Generate Map"):
                _ensure_summaries(ingested)
                sums = {f: st.session_state["summaries"].get(f, "") for f in ingested}
                with st.spinner("Computing similarities..."):
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        import plotly.graph_objects as go
                        import networkx as nx

                        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        names    = list(sums.keys())
                        vecs     = np.array(embedder.embed_documents(list(sums.values())))
                        norms    = np.linalg.norm(vecs, axis=1, keepdims=True)
                        normed   = vecs / np.where(norms == 0, 1, norms)
                        sim_mat  = normed @ normed.T

                        G           = nx.Graph()
                        short_names = [os.path.basename(n).replace(".pdf","").replace(".txt","")[:25] for n in names]
                        for name in short_names:
                            G.add_node(name)
                        for i in range(len(names)):
                            for j in range(i + 1, len(names)):
                                sim = float(sim_mat[i, j])
                                if sim > 0.3:
                                    G.add_edge(short_names[i], short_names[j], weight=sim)

                        pos = nx.spring_layout(G, seed=42)
                        edge_traces = []
                        for u, v, data in G.edges(data=True):
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            sim_val = data.get("weight", 0)
                            edge_traces.append(go.Scatter(x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                                line=dict(width=sim_val*5, color=f"rgba(100,150,255,{sim_val:.2f})"), hoverinfo="none"))

                        node_trace = go.Scatter(
                            x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                            mode="markers+text", text=list(G.nodes()), textposition="top center",
                            marker=dict(size=20, color="#4C9BE8", line=dict(width=2, color="white")),
                            hovertemplate="<b>%{text}</b><extra></extra>",
                        )
                        fig = go.Figure(data=edge_traces + [node_trace],
                            layout=go.Layout(title="Paper Relationship Map", showlegend=False, hovermode="closest",
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Thicker edges = more similar content. Threshold: 0.30")
                    except Exception as e:
                        st.error(f"Map failed: {e}")

        # ── Similarity Search ──────────────────────────────────────────────
        elif tool == "🔎 Similarity Search":
            st.caption("Paste any paragraph and find which ingested paper it's most similar to.")
            _ensure_summaries(ingested)
            passage = st.text_area("Paste a passage", height=150, placeholder="Paste any paragraph here...")
            if st.button("🔎 Find Similar Paper") and passage.strip():
                sums = {f: st.session_state["summaries"].get(f,"") for f in ingested}
                with st.spinner("Searching..."):
                    try:
                        result = find_similar_paper(passage, sums)
                        st.markdown("### Similarity Analysis")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # ── Citation Formatter ─────────────────────────────────────────────
        elif tool == "📋 Citation Formatter":
            st.caption("Format your retrieved sources as APA, MLA, or IEEE citations.")
            style = st.selectbox("Citation style", ["APA", "MLA", "IEEE"])
            st.info("Ask a question in the Chat tab first, then come here to format the sources from that answer.")
            if st.session_state["messages"]:
                last_assistant = next((m for m in reversed(st.session_state["messages"]) if m["role"] == "assistant" and m.get("sources")), None)
                if last_assistant:
                    st.write(f"**Sources from your last answer:**")
                    for src in last_assistant["sources"]:
                        st.caption(f"• {os.path.basename(src['source'])}")
                    if st.button(f"📋 Format as {style}"):
                        from langchain_core.documents import Document
                        docs = [Document(page_content=s["content"], metadata={"source": s["source"]}) for s in last_assistant["sources"]]
                        with st.spinner("Formatting citations..."):
                            try:
                                result = format_citations(docs, style)
                                st.markdown(f"### {style} Citations")
                                st.markdown(result)
                                st.download_button("⬇️ Download", result.encode(), f"citations_{style.lower()}.md", "text/markdown")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                else:
                    st.warning("No answers with sources found. Ask a question in the Chat tab first.")
            else:
                st.warning("No chat history yet. Ask a question in the Chat tab first.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANNOTATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📌 Annotations")
    st.caption("Save passages and notes from your research.")

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
            st.success("Saved!")
            st.rerun()

    st.divider()
    annotations = st.session_state["annotations"]
    if not annotations:
        st.info("No annotations yet. Add them here or via the Analyze tab.")
    else:
        st.write(f"**{len(annotations)} annotation(s)**")
        st.download_button("⬇️ Export All", data=export_annotations_as_markdown(annotations),
                           file_name="annotations.md", mime="text/markdown")
        st.divider()
        for i, ann in enumerate(annotations):
            with st.expander(f"📌 {ann.get('doc','Unknown')} — {ann.get('timestamp','')}", expanded=True):
                st.markdown("**Passage:**")
                st.info(ann.get("passage",""))
                st.markdown(f"**Note:** {ann.get('note','')}")
                if st.button("🗑️ Delete", key=f"del_{i}"):
                    st.session_state["annotations"].pop(i)
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — WRITING ASSISTANT
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("✍️ Writing Assistant")
    st.caption("Improve your writing using your ingested papers as sources, or find papers similar to your draft.")

    ingested = list_ingested_pdfs(DOCS_FOLDER)
    if not ingested:
        st.warning("No documents ingested yet. Ingest papers first to use as writing context.")
    else:
        writing_tool = st.radio("What would you like to do?",
            ["📝 Improve my draft", "🔎 Find relevant papers for my draft"], horizontal=True)

        draft = st.text_area("Paste your draft paragraph or section here", height=200,
                              placeholder="Paste your own writing here...")

        if writing_tool == "📝 Improve my draft":
            selected = st.multiselect("Use these papers as context", ingested, default=ingested)
            if st.button("✨ Improve Draft") and draft.strip() and selected:
                texts = _load_texts(selected)
                with st.spinner("Improving your draft..."):
                    try:
                        result = improve_writing(draft, texts)
                        st.markdown("### Improved Draft")
                        st.markdown(result)
                        st.download_button("⬇️ Download", result.encode(), "improved_draft.md", "text/markdown")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        elif writing_tool == "🔎 Find relevant papers for my draft":
            if st.button("🔎 Find Relevant Papers") and draft.strip():
                _ensure_summaries(ingested)
                sums = {f: st.session_state["summaries"].get(f,"") for f in ingested}
                with st.spinner("Finding relevant papers..."):
                    try:
                        result = find_similar_paper(draft, sums)
                        st.markdown("### Relevance Analysis")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Failed: {e}")
