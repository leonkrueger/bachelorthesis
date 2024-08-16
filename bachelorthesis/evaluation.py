import json
import logging
import os
import traceback

from system.utils.utils import configure_logger, load_env_variables

load_env_variables()

from system.databases.python_database import PythonDatabase
from system.insert_query_handler import InsertQueryHandler
from system.strategies.heuristic.heuristic_strategy import (
    HeuristicStrategy,
    MatchingAlgorithm,
)
from system.strategies.heuristic.synonym_generator import WordnetSynonymGenerator
from system.strategies.llama3.llama3_model import Llama3Model
from system.strategies.llama3.llama3_strategy import Llama3Strategy
from system.strategies.openai.openai_strategy import OpenAIStrategy

database = PythonDatabase()

llm = Llama3Model()

strategies = {
    # "Llama3_finetuned": Llama3Strategy(
    #     llm, "missing_tables_12000_1_csv", "missing_columns_12000_1_own"
    # ),
    # "Llama3_not_finetuned": Llama3Strategy(llm),
    # "GPT3_5": OpenAIStrategy(),
    "Heuristic_exact": HeuristicStrategy(MatchingAlgorithm.EXACT_MATCH, llm),
    "Heuristic_fuzzy": HeuristicStrategy(MatchingAlgorithm.FUZZY_MATCH, llm),
    "Heuristic_synonyms": HeuristicStrategy(
        MatchingAlgorithm.FUZZY_MATCH_SYNONYMS, llm, WordnetSynonymGenerator()
    ),
}

# Switch if necessary
evaluation_folder = "data"

evaluation_base_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    *os.environ["EVALUATION_BASE_DIR_RELATIVE"].split("/"),
)


evaluation_folder = os.path.join(evaluation_base_folder, evaluation_folder)

logging_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "logs.txt"
)
configure_logger(logging_path)
logger = logging.getLogger(__name__)

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
            logger.error(f"Error while executing query: {query}\n")

    with open(
        results_file_path,
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(results, results_file)
    os.chmod(results_file_path, 0o777)

    database.reset_database()


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
            logger.error(f"Error {e} while executing query: {query}\n")

    save_results_and_clean_database(results_file_path)
    logger.info(f"Gold Standard executed: {folder}")


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

            insert_query_handler = InsertQueryHandler(database, strategy)

            with open(inserts_file_path, encoding="utf-8") as input_file:
                queries = input_file.read().split(";\n")
            for query in queries:
                query = query.strip()
                if query == "":
                    continue
                try:
                    insert_query_handler.handle_insert_query(query)
                except Exception as e:
                    logger.error(
                        f"Error while executing query: {query}\n"
                        + traceback.format_exc()
                        + "\n\n"
                    )

            save_results_and_clean_database(results_file_path)
            logger.info(f"Experiment executed: {results_file_path}")


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

# Close the database connection
database.close()
