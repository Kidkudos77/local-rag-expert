"""
ingest.py — Document Ingestion Pipeline
"""

import os
import shutil
import time

from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100
BATCH_SIZE   = 10   # Embed this many chunks per call — faster than one-by-one


def load_documents(folder_path: str) -> list:
    docs = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if pdf_files:
        pdf_loader = PyPDFDirectoryLoader(folder_path)
        try:
            docs.extend(pdf_loader.load())
            print(f"  Loaded {len(pdf_files)} PDF(s)")
        except Exception as e:
            print(f"  PDF loader error: {e}")

    txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
    try:
        txt_docs = txt_loader.load()
        docs.extend(txt_docs)
        if txt_docs:
            print(f"  Loaded {len(txt_docs)} text file(s)")
    except Exception as e:
        print(f"  Text loader error: {e}")

    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def create_vector_store(chunks: list, progress_callback=None) -> Chroma:
    """
    Embed chunks in batches and write to ChromaDB.
    progress_callback(current, total) fires after each batch so the UI
    can update a progress bar in real time.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Windows-safe deletion: retry up to 5 times if files are locked
    if os.path.exists(CHROMA_PATH):
        for _ in range(5):
            try:
                shutil.rmtree(CHROMA_PATH)
                break
            except PermissionError:
                time.sleep(1)

    total = len(chunks)
    db = None

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
            )
        else:
            db.add_documents(batch)

        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, total), total)

    print(f"  Stored {total} chunks in ChromaDB at '{CHROMA_PATH}/'")
    return db


def ingest(folder_path: str = "documents", progress_callback=None) -> Chroma:
    """
    Full pipeline: load → split → embed → store.
    progress_callback(current, total) fires during the embedding step.
    """
    print(f"\nIngesting documents from '{folder_path}'...")

    print("Step 1/3: Loading documents...")
    docs = load_documents(folder_path)
    if not docs:
        raise ValueError(f"No PDFs or .txt files found in '{folder_path}'.")
    print(f"  Total pages/docs loaded: {len(docs)}")

    print("Step 2/3: Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  Total chunks created: {len(chunks)}")

    print("Step 3/3: Embedding and storing in ChromaDB...")
    db = create_vector_store(chunks, progress_callback=progress_callback)

    print("\nIngestion complete!\n")
    return db


if __name__ == "__main__":
    ingest("documents")
