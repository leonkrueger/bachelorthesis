import sys
import traceback

from system.utils.utils import get_finetuned_model_dir, load_env_variables

load_env_variables()

from system.databases.python_database import PythonDatabase
from system.strategies.heuristic.heuristic_strategy import (
    HeuristicStrategy,
    MatchingAlgorithm,
)
from system.strategies.heuristic.synonym_generator import WordnetSynonymGenerator
from system.strategies.llama3.llama3_strategy import Llama3Strategy
from system.strategies.openai.openai_strategy import OpenAIStrategy
from tabulate import tabulate

from bachelorthesis.system.insert_handler import InsertHandler

strategy_argument = sys.argv[1] if len(sys.argv) > 1 else "llama3_finetuned"

match strategy_argument:
    case "heuristic_exact":
        strategy = HeuristicStrategy(MatchingAlgorithm.EXACT_MATCH)
    case "heuristic_fuzzy":
        strategy = HeuristicStrategy(MatchingAlgorithm.FUZZY_MATCH)
    case "heuristic_synonyms":
        strategy = HeuristicStrategy(
            MatchingAlgorithm.FUZZY_MATCH_SYNONYMS, WordnetSynonymGenerator()
        )
    case "llama3_finetuned":
        strategy = Llama3Strategy(
            get_finetuned_model_dir("missing_tables_12000_1_csv"),
            get_finetuned_model_dir("missing_columns_12000_1_own"),
        )
    case "llama3_not_finetuned":
        strategy = Llama3Strategy()
    case "gpt":
        strategy = OpenAIStrategy()
    case _:
        print("Strategy not supported!")
        exit(1)

database = PythonDatabase()
insert_handler = InsertHandler(database, strategy)

# Get user input
user_input = input("SQL-Statement:\n").strip()
while user_input != "exit" and user_input != "exit;":
    try:
        if user_input.lower().startswith("insert"):
            tabulate(
                insert_handler.handle_insert(user_input),
                tablefmt="psql",
            )
        elif user_input.lower() == "reset;":
            database.reset_database()
        else:
            print(tabulate(database.execute(user_input), tablefmt="psql"))
    except Exception as e:
        print(traceback.format_exc())

    user_input = input("SQL-Statement:\n").strip()

# Close the database connection
database.close()
