import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "data", "movies.db")


def question_to_sql(user_input):
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = f"""You are an SQL expert. Given this SQLite table:

movies(show_id, type, title, director, cast, country, date_added,
       release_year, rating, duration, listed_in, description)

- type: 'Movie' or 'TV Show'
- release_year: integer year
- rating: text like 'PG-13', 'TV-MA', 'R'
- listed_in: comma-separated genres like 'Dramas, International Movies'
- duration: text like '90 min' or '2 Seasons'

Write ONLY a valid SQLite SQL query to answer: {user_input}
Return ONLY the SQL, no backticks, end with semicolon."""

    response = llm.invoke(prompt)
    return response.content.strip()


def run_sql(query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result


def handle_sql_query(user_input):
    sql_query = question_to_sql(user_input)
    print("SQL QUERY:", sql_query)

    try:
        result = run_sql(sql_query)

        if not result:
            return "No data found"

        if len(result) == 1 and len(result[0]) == 1:
            return str(result[0][0])

        formatted = []
        for row in result:
            formatted.append(", ".join(str(col) for col in row))

        return "\n".join(formatted)

    except Exception as e:
        print("SQL ERROR:", e)
        return f"SQL Error: {str(e)}"