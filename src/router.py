import os
from langchain_groq import ChatGroq

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    return _llm

def route(user_input):
    prompt = f"""You are a query router. Classify the user query into exactly one category:

- SQL: structured data queries — counts, averages, filtering, top-N, listing rows from a movies database
- RAG: explanations, definitions, comparisons, conceptual questions, document-based answers

Respond with exactly one word: SQL or RAG.

Examples:
Query: How many action movies are there? → SQL
Query: Top 5 movies by rating → SQL
Query: List all movies from 2020 → SQL
Query: What is the plot of Inception? → RAG
Query: Compare drama and comedy genres → RAG
Query: Explain what RAG is → RAG

Query: {user_input}
Answer:"""

    try:
        response = _get_llm().invoke(prompt).content.strip().upper()
        if "SQL" in response:
            return "SQL"
        return "RAG"
    except Exception as e:
        print(f"Router error, defaulting to RAG: {e}")
        return "RAG"