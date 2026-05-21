"""
rag_pipeline.py
---------------
Builds the FAISS index from PDFs in pdf_files/.
Run this ONCE locally before deploying, or via `python -m src.rag_pipeline`.

FIXES applied:
1. Removed broken FAISS.load_local() call inside load_pdf() — that referenced
   an undefined `embeddings` variable and made no sense (loading inside the
   builder). The function now only loads PDFs.
2. Replaced safe_chunks = chunks[:100] hack with batched embedding. The index
   now contains ALL chunks, processed in batches of 64 to stay within memory
   limits on free-tier hardware.
3. Removed the unused load_vector_db() duplicate function (the runtime loader
   lives in rag_module.py).
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "pdf_files")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 64  # tune down to 32 if you hit memory limits


def load_pdfs():
    """Load every PDF in pdf_files/ and return a list of Documents."""
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    all_docs = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDFs found in {PDF_DIR}")

    for pdf in pdf_files:
        path = os.path.join(PDF_DIR, pdf)
        print(f"Loading {pdf}...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  -> {len(docs)} pages")

    print(f"Total pages loaded: {len(all_docs)}")
    return all_docs


def split_text(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings():
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_index_batched(chunks, embeddings, batch_size=EMBED_BATCH_SIZE):
    """
    Build FAISS index in batches to stay within memory limits.

    Why batched: free-tier hardware (e.g. HF Spaces) OOMs when embedding
    thousands of chunks at once. We embed `batch_size` at a time and merge
    into a growing index. Result: ALL chunks are indexed, not just the
    first 100 as in the previous version.
    """
    if not chunks:
        raise ValueError("No chunks to embed")

    total = len(chunks)
    print(f"Building FAISS index in batches of {batch_size} (total chunks: {total})")

    # Seed the index with the first batch
    first_batch = chunks[:batch_size]
    db = FAISS.from_documents(first_batch, embeddings)
    print(f"  Batch 1: {len(first_batch)} chunks indexed")

    # Add remaining batches
    for i in range(batch_size, total, batch_size):
        batch = chunks[i:i + batch_size]
        partial = FAISS.from_documents(batch, embeddings)
        db.merge_from(partial)
        batch_num = (i // batch_size) + 1
        print(f"  Batch {batch_num}: {len(batch)} chunks indexed "
              f"(total so far: {min(i + batch_size, total)}/{total})")

    return db


if __name__ == "__main__":
    print("Starting RAG pipeline...")
    print(f"PDF directory: {PDF_DIR}")
    print(f"Index output: {INDEX_DIR}")

    documents = load_pdfs()
    chunks = split_text(documents)
    embeddings = create_embeddings()
    db = build_index_batched(chunks, embeddings)

    print(f"\nSaving FAISS index to {INDEX_DIR}...")
    db.save_local(INDEX_DIR)
    print("Done.")
