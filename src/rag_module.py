from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

_qa = None

def _get_qa():
    global _qa
    if _qa is not None:
        return _qa

    print("Initializing RAG system (lazy)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_path = os.path.join(BASE_DIR, "faiss_index")
    print("Loading FAISS from:", faiss_path)
    print("Path exists?", os.path.exists(faiss_path))

    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    _qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    print("RAG system ready")
    return _qa


def handle_rag_query(query):
    try:
        print("STEP 1: Received query")
        qa = _get_qa()
        print("STEP 2: Calling QA chain")
        result = qa.invoke({"query": query})
        print("STEP 3: Got result")

        if isinstance(result, dict):
            return result.get("result", str(result))
        return str(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", str(e))
        return f"RAG Error: {str(e)}"