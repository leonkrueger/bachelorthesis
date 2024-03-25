import mysql.connector
import traceback
import os

from tabulate import tabulate

from system.strategies.heuristic.heuristic_strategy import HeuristicStrategy
from system.strategies.heuristic.name_predictor import NamePredictor
from system.sql_handler import SQLHandler
from system.strategies.openai.openai_model import OpenAIModel
from system.insert_query_handler import handle_insert_query
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

strategy = HeuristicStrategy(NamePredictor(os.getenv("HF_API_TOKEN")))
# strategy = OpenAIModel(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG_ID"))

# Get user input
user_input = input("Query:\n").strip()
while user_input != "exit":
    try:
        if user_input.lower().startswith("insert"):
            tabulate(
                handle_insert_query(user_input, sql_handler, table_manager, strategy),
                tablefmt="psql",
            )
        else:
            print(tabulate(sql_handler.execute_query(user_input)[0], tablefmt="psql"))
    except Exception as e:
        print(traceback.format_exc())

    user_input = input("Query:\n").strip()

# Close the database connection
conn.close()
