import sys
import traceback

from system.utils.utils import load_env_variables

load_env_variables()

from system.databases.mysql_database import MySQLDatbase
from system.databases.python_database import PythonDatabase
from system.insert_query_handler import InsertQueryHandler
from system.strategies.heuristic.heuristic_strategy import (
    HeuristicStrategy,
    MatchingAlgorithm,
)
from system.strategies.heuristic.name_predictor import NamePredictor
from system.strategies.heuristic.synonym_generator import WordnetSynonymGenerator
from system.strategies.llama3.llama3_strategy import Llama3Strategy
from system.strategies.openai.openai_strategy import OpenAIStrategy
from tabulate import tabulate

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
            "missing_tables_12000_1_csv", "missing_columns_12000_1_own"
        )
    case "llama3_not_finetuned":
        strategy = Llama3Strategy()
    case "gpt":
        strategy = OpenAIStrategy()
    case _:
        print("Strategy not supported!")
        exit(1)

database = PythonDatabase()
insert_query_handler = InsertQueryHandler(database, strategy)

# Get user input
user_input = input("Query:\n").strip()
while user_input != "exit" and user_input != "exit;":
    try:
        if user_input.lower().startswith("insert"):
            tabulate(
                insert_query_handler.handle_insert_query(user_input),
                tablefmt="psql",
            )
        elif user_input.lower() == "reset;":
            database.reset_database()
        else:
            print(tabulate(database.execute_query(user_input), tablefmt="psql"))
    except Exception as e:
        print(traceback.format_exc())

    user_input = input("Query:\n").strip()

# Close the database connection
database.close()
