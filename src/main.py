from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from src.router import route
from src.sql_module import handle_sql_query
from src.rag_module import handle_rag_query
import os
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI startup complete - model loads lazily")
    print("GROQ_API_KEY set?", bool(os.getenv("GROQ_API_KEY")))
    yield

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(data: QueryRequest):
    print("QUERY FUNCTION RUNNING")
    try:
        user_input = data.question
        decision = route(user_input)

        if decision == "SQL":
            result = handle_sql_query(user_input)
        else:
            result = handle_rag_query(user_input)

        return {"response": result}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)