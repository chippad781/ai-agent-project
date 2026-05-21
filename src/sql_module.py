"""
sql_module.py
-------------
Generates SQL from a natural-language question and executes it against the
movies SQLite database.

FIXES applied:
1. Added _sanitize_sql() — strips markdown fences, trailing semicolons, and
   common LLM artifacts.
2. Added is_safe_select() — rejects any query that isn't a single SELECT.
   This is the primary defense against the LLM generating DESTRUCTIVE SQL
   (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE).
3. Opened the connection in read-only mode using sqlite3 URI form, so even
   if a destructive query slips past the allowlist, the database refuses to
   execute it. Defense in depth.
4. Lazy-loaded the LLM (one global instance) instead of recreating per call.
"""

import os
import re
import sqlite3
from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "movies.db")

_llm = None


def _get_llm():
    """Lazy-init the LLM so we don't create a new one per query."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def _sanitize_sql(raw):
    """Strip markdown fences, leading/trailing whitespace, trailing semicolon."""
    if not raw:
        return ""
    # Remove markdown code fences if the LLM added them
    raw = re.sub(r"^```(?:sql)?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw.strip())
    # Strip trailing semicolons (we'll only execute one statement anyway)
    return raw.strip().rstrip(";").strip()


# Block these keywords anywhere outside string literals
FORBIDDEN_KEYWORDS = (
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
)


def is_safe_select(sql):
    """
    Return True only if the SQL is a single SELECT statement with no
    destructive keywords. This is a guardrail — combine with a read-only
    DB connection for defense in depth.
    """
    if not sql:
        return False

    upper = sql.upper().strip()

    # Must start with SELECT (or WITH ... SELECT)
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False

    # Reject multiple statements (defense against stacked queries)
    if ";" in sql:
        return False

    # Reject forbidden keywords as standalone tokens
    # (so 'SELECT' embedded in column names like 'created_at' is fine)
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False

    return True


def question_to_sql(user_input):
    """Ask the LLM to generate SQL from a natural-language question."""
    prompt = f"""You are an SQL expert. Given this SQLite table:

movies(show_id, type, title, director, cast, country, date_added,
       release_year, rating, duration, listed_in, description)

- type: 'Movie' or 'TV Show'
- release_year: integer year
- rating: text like 'PG-13', 'TV-MA', 'R'
- listed_in: comma-separated genres like 'Dramas, International Movies'
- duration: text like '90 min' or '2 Seasons'

Rules:
- Only generate SELECT statements. Never DROP, DELETE, UPDATE, INSERT.
- Use a single statement; do not chain queries with semicolons.
- Return ONLY the SQL with no backticks, no markdown, no commentary.

Question: {user_input}
SQL:"""

    response = _get_llm().invoke(prompt)
    return _sanitize_sql(response.content)


def run_sql(query):
    """
    Execute a SELECT-only query against a read-only DB connection.
    Read-only mode is enforced via the sqlite URI 'mode=ro' parameter.
    """
    # URI form lets us open the DB in read-only mode
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        conn.close()


def handle_sql_query(user_input):
    """End-to-end: question -> SQL -> safety check -> execute -> format."""
    try:
        sql_query = question_to_sql(user_input)
        print("SQL QUERY:", sql_query)

        # GUARDRAIL: refuse anything that isn't a safe SELECT
        if not is_safe_select(sql_query):
            print("REJECTED: unsafe or non-SELECT query")
            return ("I can only run read-only SELECT queries on this database. "
                    "Please rephrase your question.")

        result = run_sql(sql_query)

        if not result:
            return "No data found"

        # Single scalar result (e.g. COUNT) -> return as plain number
        if len(result) == 1 and len(result[0]) == 1:
            return str(result[0][0])

        # Otherwise format rows as comma-separated values
        formatted = []
        for row in result:
            formatted.append(", ".join(str(col) for col in row))
        return "\n".join(formatted)

    except sqlite3.Error as e:
        print("SQL ERROR:", e)
        # Don't leak DB internals to the user
        return "The query failed to execute. Please try rephrasing."
    except Exception as e:
        print("UNEXPECTED ERROR in handle_sql_query:", e)
        return "Something went wrong processing your query."
