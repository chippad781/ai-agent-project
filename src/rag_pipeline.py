from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time

def load_pdf():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FAISS.load_local(
        os.path.join(BASE_DIR, "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True
    )
    folder_path = os.path.join(BASE_DIR, "pdf_files")
    pdf_files = os.listdir(folder_path)
    all_docs = []
    for pdf in pdf_files:
        if pdf.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, pdf))
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded documents: {len(all_docs)}")
    return all_docs

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
    chunks = splitter.split_documents(documents)
    return chunks

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_db(chunks, embeddings):
    print("Creating embeddings safely...")

    safe_chunks = chunks[:100]

    db = FAISS.from_documents(safe_chunks, embeddings)

    return db

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings
    )

    return db

if __name__ == "__main__":
    print("🚀 Starting RAG pipeline...")

    documents = load_pdf()
    print(f"Loaded {len(documents)} documents")

    chunks = split_text(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = create_embeddings()

    db = create_vector_db(chunks, embeddings)

    print("💾 Saving FAISS index...")
    db.save_local("faiss_index")

    print("✅ Done!")
