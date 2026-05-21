"""
rag_module.py
-------------
RAG pipeline: retrieves relevant chunks from FAISS and generates an answer
using Groq's llama-3.1-8b-instant.

FIXES applied:
1. Added conversational memory using LangChain's ConversationBufferMemory,
   keyed by session_id. Each user session has its own memory buffer that
   stores the last N exchanges.
2. Switched from RetrievalQA (stateless) to ConversationalRetrievalChain,
   which incorporates chat history into the retrieval and generation.
3. Added bounded memory (window of last 5 exchanges) to prevent context
   bloat on long conversations.
4. Added a get_or_create_chain() helper so the FAISS index and embeddings
   load once per process, but memory is per-session.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

# Bounded window of last K exchanges to keep context size manageable
MEMORY_WINDOW = 5

# Module-level singletons (loaded once per process)
_embeddings = None
_vectorstore = None
_llm = None

# Per-session memory: { session_id: ConversationBufferWindowMemory }
_session_memory = {}

# Per-session chain: { session_id: ConversationalRetrievalChain }
_session_chain = {}


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Loading embeddings model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print(f"Loading FAISS index from {FAISS_PATH}")
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_PATH}. "
                f"Run `python -m src.rag_pipeline` first to build it."
            )
        _vectorstore = FAISS.load_local(
            FAISS_PATH,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def _get_or_create_memory(session_id):
    """Get or create a memory buffer for this session."""
    if session_id not in _session_memory:
        _session_memory[session_id] = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    return _session_memory[session_id]


def _get_or_create_chain(session_id):
    """Get or create a conversational chain for this session."""
    if session_id not in _session_chain:
        retriever = _get_vectorstore().as_retriever(search_kwargs={"k": 2})
        memory = _get_or_create_memory(session_id)
        _session_chain[session_id] = ConversationalRetrievalChain.from_llm(
            llm=_get_llm(),
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
        )
    return _session_chain[session_id]


def handle_rag_query(query, session_id="default"):
    """
    Answer a query using RAG with conversational memory.

    `session_id` defaults to "default" so existing callers keep working,
    but new callers should pass a unique ID per user/session to get isolated
    memory.
    """
    try:
        print(f"RAG query (session={session_id}): {query}")
        chain = _get_or_create_chain(session_id)
        result = chain.invoke({"question": query})

        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("RAG ERROR:", str(e))
        return f"RAG Error: {str(e)}"


def reset_session(session_id):
    """Clear memory for a session (useful for a 'new chat' button)."""
    _session_memory.pop(session_id, None)
    _session_chain.pop(session_id, None)
