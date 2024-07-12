import json
import os
import traceback

from system.databases.python_database import PythonDatabase
from system.insert_query_handler import InsertQueryHandler
from system.strategies.heuristic.heuristic_strategy import HeuristicStrategy
from system.strategies.heuristic.name_predictor import NamePredictor
from system.strategies.llama3.llama3_model import Llama3Model, Llama3ModelType
from system.strategies.openai.openai_strategy import OpenAIStrategy
from system.table_manager import TableManager
from system.utils.utils import load_env_variables

load_env_variables()

database = PythonDatabase()
table_manager = TableManager(database)

strategies = {
    "Llama3_finetuned": None,
    "Llama3": None,  # Llama3Model(
    #     Llama3ModelType.NON_FINE_TUNED_LOCAL,
    # ),
    "GPT4": None,  # OpenAIModel(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG_ID")),
    "Heuristics": None,  # HeuristicStrategy(),
    "missing_tables_300": None,  # Llama3Model(
    #     Llama3ModelType.FINE_TUNED,
    #     os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)),
    #         "fine_tuning",
    #         "output",
    #         "missing_tables_300",
    #     ),
    # ),
    "missing_tables_0": Llama3Model(
        Llama3ModelType.NON_FINE_TUNED,
    ),
}

# Switch if necessary
evaluation_folder = "data"

# Depends on if the script is run in Docker or as plain python
# evaluation_base_folder = os.path.join("/app", "evaluation")
evaluation_base_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "evaluation",
    "bachelorthesis",
)


evaluation_folder = os.path.join(evaluation_base_folder, evaluation_folder)

errors_file_path = os.path.join(evaluation_base_folder, "errors.txt")
errors_file = open(errors_file_path, "w", encoding="utf-8")

database.reset_database()


def save_results_and_clean_database(results_file_path: str) -> None:
    """Saves the database contents to a json-file"""
    results = {}
    for table in database.get_all_tables():
        query = f"SELECT * FROM {table};"
        try:
            output = database.select_all_data(table)
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

    database.reset_database()
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
            database.execute_query(query)
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

        for strategy_name, strategy in strategies.items():
            if strategy is None:
                continue

            results_file_path = os.path.join(
                folder, strategy_name, path[:-4].replace("input", "results") + ".json"
            )

            if not os.path.exists(results_file_path):
                continue

            # Continue if experiment was already run
            with open(results_file_path, encoding="utf-8") as results_file:
                if results_file.read().strip() != "":
                    continue

            insert_query_handler = InsertQueryHandler(database, table_manager, strategy)

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
                    errors_file.write(f"Error while executing query: {query}\n")
                    errors_file.write(traceback.format_exc() + "\n\n")

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
    if not os.path.isdir(subfolder) or path == "evaluation":
        continue

    run_experiments_for_database(subfolder)

errors_file.close()
os.chmod(errors_file_path, 0o777)

# Close the database connection
database.close()
