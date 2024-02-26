import mysql.connector
import json
import os

from utils.io.sql_handler import SQLHandler
from system.insert_query_handler import handle_insert_query
from system.table_manager import TableManager
from utils.models.name_predictor import NamePredictor

# Create a database connection
conn = mysql.connector.connect(
    user="user",
    password="Start123",
    host="bachelorthesis_leon_krueger_mysql",
    database="db",
)

HF_API_TOKEN = open("HF_TOKEN", "r").read().strip()

sql_handler = SQLHandler(conn)
table_manager = TableManager(sql_handler)
name_predictor = NamePredictor(HF_API_TOKEN)

with open(os.path.join("/app", "mounted_evaluation", "errors.txt"), "w") as errors_file:
    with open(os.path.join("evaluation", "evaluation_input.sql")) as input_file:
        queries = input_file.read().split(";")
        queries_length = float(len(queries))
        for index, query in enumerate(queries):
            query = query.strip()
            if query == "":
                continue

            print(f"{100 * index / queries_length}%")
            try:
                handle_insert_query(query, sql_handler, table_manager, name_predictor)
            except Exception as e:
                print(f"Error while executing query: {query}")
                errors_file.write(f"Error {e} while executing query: {query}\n")

    results = {}

    for table in sql_handler.get_all_tables():
        query = f"SELECT * FROM {table};"
        try:
            output = sql_handler.execute_query(query)[0]
            results[query] = output
        except Exception as e:
            print(f"Error while executing query: {query}")
            errors_file.write(f"Error while executing query: {query}\n")

    with open(
        os.path.join("/app", "mounted_evaluation", "results.json"), "w"
    ) as results_file:
        json.dump(results, results_file)

# Close the database connection
conn.close()
