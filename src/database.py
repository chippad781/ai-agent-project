import pandas as pd
import sqlite3

def setup():
    connection = sqlite3.connect(r"data\movies.db")
    df = pd.read_csv(r"data\cleaned_data.csv")
    df.to_sql("movies", connection, if_exists="replace")
    connection.close()
    print("Database setup complete")

def run_query(query):
    connection = sqlite3.connect(r"data\movies.db")
    df =  pd.read_sql_query(query, connection)
    connection.close()
    return df

if __name__ == "__main__":
    setup()

query = input("Enter your query: ")
run_query(query)
print(run_query(query))