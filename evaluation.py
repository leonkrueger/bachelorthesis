import mysql.connector
import json
import os

from utils.sql_handler import SQLHandler
from system.insert_query_handler import handle_insert_query

# Create a database connection
conn = mysql.connector.connect(
    user="user",
    password="Start123",
    host="bachelorthesis_leon_krueger_mysql",
    database="db",
)

sql_handler = SQLHandler(conn)

with open(os.path.join("/mounted_evaluation", "errors.txt"), "w") as errors_file:
    with open(os.path.join("evaluation", "evaluation_input.sql")) as input_file:
        queries = input_file.read().split(";")
        queries_length = float(len(queries))
        for index, query in enumerate(queries):
            query = query.strip()
            print(f"{100 * index / queries_length}%")
            try:
                handle_insert_query(query, sql_handler)
            except Exception as e:
                print(f"Error while executing query: {query}")
                errors_file.write(f"Error {e} while executing query: {query}\n")

    results = {}

    with open(os.path.join("evaluation", "queries.sql")) as queries_file:
        for query in queries_file.readlines():
            try:
                output = sql_handler.execute_query(query)[0]
                results[query] = output
            except Exception as e:
                print(f"Error while executing query: {query}")
                errors_file.write(f"Error while executing query: {query}\n")

    with open(os.path.join("/mounted_evaluation", "results.json"), "w") as results_file:
        json.dump(results, results_file)

# Close the database connection
conn.close()
