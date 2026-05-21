"""
rag_chain.py — Retrieval-Augmented Generation Chain

What this does:
  1. Connects to the ChromaDB vector store built by ingest.py
  2. On each question, retrieves the top-5 most relevant chunks (similarity search)
  3. Stuffs those chunks into a prompt alongside the question
  4. Sends the prompt to a local Ollama LLM
  5. Returns the grounded answer as a string

The chain is built using LangChain Expression Language (LCEL), which is the
industry-standard way to compose retrieval + LLM pipelines.
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "llama3"          # Swap to "mistral" or any Ollama model you've pulled
TOP_K        = 5                 # Number of chunks to retrieve per query

# --- Prompt Template ---
# This is the most important engineering decision in any RAG system.
# "Answer ONLY from the context" prevents hallucination by grounding the LLM.
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
    """
    Join retrieved chunks into a single context string.
    Including the source filename helps the LLM attribute information correctly.
    """
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def get_retriever():
    """
    Load ChromaDB and return a retriever.
    The retriever does a cosine similarity search between the query embedding
    and all stored chunk embeddings, returning the top-k closest matches.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": TOP_K})


def build_rag_chain():
    """
    Assemble the full RAG pipeline using LCEL (LangChain Expression Language).

    Pipeline structure:
      question (str)
          ↓
      {"context": retriever → format_docs, "question": passthrough}
          ↓
      prompt template (formats question + context into a chat message)
          ↓
      ChatOllama (sends to local LLM, streams back a response)
          ↓
      StrOutputParser (extracts the text from the LLM response object)
    """
    retriever = get_retriever()
    prompt    = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm       = ChatOllama(model=LLM_MODEL)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def query(question: str) -> str:
    """
    Run a question through the RAG chain and return the answer string.
    Called by app.py on each user message.
    """
    chain = build_rag_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    # Quick CLI test after running ingest.py
    test_q = "What is the main topic of these documents?"
    print(f"Q: {test_q}")
    print(f"A: {query(test_q)}")
