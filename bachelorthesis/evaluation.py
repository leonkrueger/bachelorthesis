import json
import os

import mysql.connector
from system.insert_query_handler import InsertQueryHandler
from system.sql_handler import SQLHandler
from system.strategies.heuristic.heuristic_strategy import HeuristicStrategy
from system.strategies.heuristic.name_predictor import NamePredictor
from system.strategies.llama2.llama2_model import LLama2Model, LLama2ModelType
from system.strategies.openai.openai_model import OpenAIModel
from system.table_manager import TableManager

# Create a database connection
conn = mysql.connector.connect(
    user="user",
    password="Start123",
    host="bachelorthesis_leon_krueger_mysql",
    database="db",
)

sql_handler = SQLHandler(conn)
table_manager = TableManager(sql_handler)

strategies = {
    "Llama2_finetuned": lambda: None,
    "Llama2": lambda: LLama2Model(
        LLama2ModelType.NON_FINE_TUNED_LOCAL,
        huggingface_api_token=os.getenv("HF_API_TOKEN"),
    ),
    "GPT4": lambda: None,  # OpenAIModel(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG_ID")),
    "Heuristics": lambda: None,  # HeuristicStrategy(NamePredictor(os.getenv("HF_API_TOKEN"))),
}

# Switch if necessary
evaluation_folder = "data"

errors_file_path = os.path.join("/app", "evaluation", "errors.txt")
errors_file = open(errors_file_path, "w", encoding="utf-8")

evaluation_folder = os.path.join("/app", "evaluation", evaluation_folder)

sql_handler.reset_database()


def save_results_and_clean_database(results_file_path: str) -> None:
    """Saves the database contents to a json-file"""
    results = {}
    print(f"Started evaluating {results_file_path}.")

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
        queries = inserts_file.read().split(";\n")

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

        for strategy_name, strategy_factory in strategies.items():
            strategy = strategy_factory()
            if strategy is None:
                continue

            results_file_path = os.path.join(
                folder, strategy_name, path[:-4].replace("input", "results") + ".json"
            )

            # Continue if experiment was already run
            with open(results_file_path, encoding="utf-8") as results_file:
                if results_file.read().strip() != "":
                    continue

            insert_query_handler = InsertQueryHandler(
                sql_handler, table_manager, strategy
            )

            with open(inserts_file_path, encoding="utf-8") as input_file:
                queries = input_file.read().split(";\n")
            for query in queries:
                query = query.strip()
                if query == "":
                    continue
                try:
                    insert_query_handler.handle_insert_query(query)
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
