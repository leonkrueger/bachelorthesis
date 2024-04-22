import os
import traceback

from system.databases.mysql_database import MySQLDatbase
from system.insert_query_handler import InsertQueryHandler
from system.strategies.heuristic.heuristic_strategy import HeuristicStrategy
from system.strategies.heuristic.name_predictor import NamePredictor
from system.strategies.openai.openai_model import OpenAIModel
from system.table_manager import TableManager
from tabulate import tabulate

database = MySQLDatbase()
table_manager = TableManager(database)

strategy = HeuristicStrategy(NamePredictor(os.getenv("HF_API_TOKEN")))
# strategy = OpenAIModel(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG_ID"))

insert_query_handler = InsertQueryHandler(database, table_manager, strategy)

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
