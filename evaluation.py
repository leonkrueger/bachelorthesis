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

# Switch if necessary
evaluation_folder = "bird"

errors_file_path = os.path.join("/app", "mounted_evaluation", "errors.txt")
errors_file = open(errors_file_path, "w", encoding="utf-8")

evaluation_folder = os.path.join("/app", "mounted_evaluation", evaluation_folder)


def save_results_and_clean_database(results_file_path: str) -> None:
    """Saves the database contents to a json-file"""
    results = {}

    for table in sql_handler.get_all_tables():
        query = f"SELECT * FROM {table};"
        try:
            output = sql_handler.execute_query(query)[0]
            output = [
                [
                    value if type(value) in [int, float, str] else str(value)
                    for value in row
                ]
                for row in output
            ]
            results[query] = output
        except Exception as e:
            print(f"Error while executing query: {query}")
            errors_file.write(f"Error while executing query: {query}\n")

    with open(
        results_file_path,
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(results, results_file)
    os.chmod(results_file_path, 0o777)

    sql_handler.reset_database()
    print(f"Saved results to {results_file_path}.")


def run_gold_standard(folder: str) -> None:
    """Runs the gold standard queries"""
    # Return if experiment was already run
    results_file_path = os.path.join(folder, "gold_standard_results.json")
    with open(results_file_path, encoding="utf-8") as results_file:
        if results_file.read().strip() != "":
            return

    with open(
        os.path.join(folder, "gold_standard_input.sql"), encoding="utf-8"
    ) as inserts_file:
        queries = inserts_file.read().split(";")

    for query in queries:
        query = query.strip()
        if query == "":
            continue
        try:
            sql_handler.execute_query(query)
        except Exception as e:
            print(f"Error while executing query: {query}")
            errors_file.write(f"Error {e} while executing query: {query}\n")

    save_results_and_clean_database(results_file_path)


def run_experiment(folder: str) -> None:
    """Runs the queries of one experiment"""
    for path in os.listdir(folder):
        inserts_file_path = os.path.join(folder, path)
        if os.path.isdir(inserts_file_path) or not inserts_file_path.endswith(".sql"):
            continue

        # Return if experiment was already run
        results_file_path = inserts_file_path[:-4].replace("input", "results") + ".json"
        with open(results_file_path, encoding="utf-8") as results_file:
            if results_file.read().strip() != "":
                return

        with open(inserts_file_path, encoding="utf-8") as input_file:
            queries = input_file.read().split(";")
        for query in queries:
            query = query.strip()
            if query == "":
                continue
            try:
                handle_insert_query(query, sql_handler, table_manager, name_predictor)
            except Exception as e:
                print(f"Error while executing query: {query}")
                errors_file.write(f"Error {e} while executing query: {query}\n")

        save_results_and_clean_database(results_file_path)


def run_experiments_for_database(folder: str) -> None:
    """Runs all experiments for a database"""
    run_gold_standard(folder)

    for path in os.listdir(folder):
        experiment_folder = os.path.join(folder, path)
        if not os.path.isdir(experiment_folder):
            continue

        run_experiment(experiment_folder)


for path in os.listdir(evaluation_folder):
    subfolder = os.path.join(evaluation_folder, path)
    if not os.path.isdir(subfolder):
        continue

    run_experiments_for_database(subfolder)

errors_file.close()
os.chmod(errors_file_path, 0o777)

# Close the database connection
conn.close()
