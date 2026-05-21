"""
ingest.py — Document Ingestion Pipeline

What this does:
  1. Loads all PDFs and .txt files from a folder
  2. Splits them into overlapping chunks
  3. Embeds each chunk using Ollama's nomic-embed-text model
  4. Stores the embeddings + text in a local ChromaDB vector store

Run standalone: python ingest.py
Or call ingest(path) from app.py.
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
CHROMA_PATH = "chroma_db"       # Where ChromaDB stores its files on disk
EMBED_MODEL = "nomic-embed-text" # The Ollama embedding model
CHUNK_SIZE = 800                 # Max tokens per chunk
CHUNK_OVERLAP = 100              # How many tokens overlap between adjacent chunks
                                 # Overlap prevents losing context at chunk boundaries


def load_documents(folder_path: str) -> list:
    """
    Load all PDFs and .txt files from a directory.
    Returns a list of LangChain Document objects.
    """
    docs = []

    # Load PDFs
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if pdf_files:
        pdf_loader = PyPDFDirectoryLoader(folder_path)
        try:
            docs.extend(pdf_loader.load())
            print(f"  Loaded {len(pdf_files)} PDF(s)")
        except Exception as e:
            print(f"  PDF loader error: {e}")

    # Load .txt files
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
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — keeping chunks semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Embed each chunk and store in ChromaDB.

    OllamaEmbeddings sends text to the locally-running nomic-embed-text model,
    which returns a high-dimensional vector for each chunk. ChromaDB stores those
    vectors and the original text so we can do similarity search later.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Always rebuild from scratch so old documents don't pollute new uploads
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}/'")
    return db


def ingest(folder_path: str = "documents") -> Chroma:
    """
    Full ingestion pipeline: load → split → embed → store.
    Returns the ChromaDB vector store object.
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
    db = create_vector_store(chunks)

    print("\nIngestion complete!\n")
    return db


if __name__ == "__main__":
    # Quick test: put some PDFs or .txt files in a 'documents/' folder,
    # then run: python ingest.py
    ingest("documents")
