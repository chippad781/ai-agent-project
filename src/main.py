"""
main.py
-------
FastAPI entry point.

FIXES applied:
1. Added optional session_id to QueryRequest so callers can maintain
   conversation context. Falls back to "default" if not provided.
2. Added /reset endpoint to clear a session's memory.
3. Wired session_id through to handle_rag_query so RAG calls have memory.
4. Better error responses (don't leak stack traces to the client).
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.router import route
from src.sql_module import handle_sql_query
from src.rag_module import handle_rag_query, reset_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI startup complete — models load lazily on first request")
    print("GROQ_API_KEY set?", bool(os.getenv("GROQ_API_KEY")))
    yield


app = FastAPI(lifespan=lifespan, title="AI Agent (RAG + SQL)")


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ResetRequest(BaseModel):
    session_id: str


@app.get("/")
def health():
    return {"status": "ok", "service": "ai-agent"}


@app.post("/query")
def query(data: QueryRequest):
    try:
        decision = route(data.question)
        print(f"ROUTER decision: {decision}")

        if decision == "SQL":
            result = handle_sql_query(data.question)
        else:
            result = handle_rag_query(data.question, session_id=data.session_id)

        return {"response": result, "route": decision}

    except Exception as e:
        # Log full trace server-side, return a clean message to the client
        import traceback
        traceback.print_exc()
        return {"error": "Something went wrong processing your query."}


@app.post("/reset")
def reset(data: ResetRequest):
    """Clear conversation memory for a session."""
    reset_session(data.session_id)
    return {"status": "reset", "session_id": data.session_id}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
