import json
import logging
import os
import traceback

from system.utils.utils import (
    configure_logger,
    get_finetuned_model_dir,
    load_env_variables,
)
from tqdm import tqdm

load_env_variables()

from system.databases.python_database import PythonDatabase
from system.insert_handler import InsertHandler
from system.strategies.heuristic.heuristic_strategy import (
    HeuristicStrategy,
    MatchingAlgorithm,
)
from system.strategies.heuristic.synonym_generator import WordnetSynonymGenerator
from system.strategies.llama3.llama3_strategy import Llama3Strategy
from system.strategies.openai.openai_strategy import OpenAIStrategy

database = PythonDatabase()

strategies = {
    # "Llama3_finetuned": Llama3Strategy(
    #     get_finetuned_model_dir("missing_tables_12000_1_csv"),
    #     get_finetuned_model_dir("missing_columns_12000_1_own"),
    #     2,
    # ),
    "Llama3_finetuned_all_scenarios": Llama3Strategy(
        get_finetuned_model_dir("missing_tables_12000_1_csv_columns_deleted"),
        get_finetuned_model_dir("missing_columns_12000_1_own_data_collator"),
        2,
    ),
    # "Llama3_not_finetuned": Llama3Strategy(max_column_mapping_retries=2),
    # "GPT3_5": OpenAIStrategy(max_column_mapping_retries=1),
    # "GPT4o": OpenAIStrategy("gpt-4o-2024-05-13", 1),
    # "GPT4o_mini": OpenAIStrategy("gpt-4o-mini-2024-07-18", 1),
    # "Heuristic_exact": HeuristicStrategy(MatchingAlgorithm.EXACT_MATCH),
    # "Heuristic_fuzzy": HeuristicStrategy(MatchingAlgorithm.FUZZY_MATCH),
    # "Heuristic_synonyms": HeuristicStrategy(
    #     MatchingAlgorithm.FUZZY_MATCH_SYNONYMS, WordnetSynonymGenerator()
    # ),
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
    os.path.dirname(os.path.realpath(__file__)), "..", "logs_finetuned.txt"
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
    """Runs the gold standard inserts"""
    # Return if experiment was already run
    results_file_path = os.path.join(folder, "gold_standard_results.json")
    with open(results_file_path, encoding="utf-8") as results_file:
        if results_file.read().strip() != "":
            return

    with open(
        os.path.join(folder, "gold_standard_input.sql"), encoding="utf-8"
    ) as inserts_file:
        inserts = inserts_file.read().split(";\n")

    for insert in inserts:
        insert = insert.strip()
        if insert == "":
            continue
        try:
            database.execute(insert)
        except Exception as e:
            logger.error(f"Error {e} while executing insert: {insert}\n")

    save_results_and_clean_database(results_file_path)
    logger.info(f"Gold Standard executed: {folder}")


def run_experiment(folder: str) -> None:
    """Runs the inserts of one experiment"""
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

            insert_handler = InsertHandler(database, strategy)

            with open(inserts_file_path, encoding="utf-8") as input_file:
                inserts = input_file.read().split(";\n")
            for insert in tqdm(inserts):
                insert = insert.strip()
                if insert == "":
                    continue
                try:
                    insert_handler.handle_insert(insert)
                except Exception as e:
                    logger.error(
                        f"Error while executing insert: {insert}\n"
                        + traceback.format_exc()
                        + "\n\n"
                    )

            save_results_and_clean_database(results_file_path)
            if hasattr(strategy, "total_costs"):
                logger.info(
                    f"Experiment executed: {results_file_path} with costs: {strategy.total_costs}"
                )
                strategy.total_costs = 0.0
            else:
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
