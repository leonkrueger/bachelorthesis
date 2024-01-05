import mysql.connector
import traceback

from tabulate import tabulate
from utils.sql_handler import SQLHandler
from system.insert_query_handler import handle_insert_query

# Create a database connection
conn = mysql.connector.connect(
    user="user",
    password="Start123",
    host="bachelorthesis_leon_krueger_mysql",
    database="db",
)

sql_handler = SQLHandler(conn)

# Get user input
user_input = input("Query:\n").strip()
while user_input != "exit":
    try:
        if user_input.lower().startswith("insert"):
            tabulate(handle_insert_query(user_input, sql_handler), tablefmt="psql")
        else:
            print(tabulate(sql_handler.execute_query(user_input)[0], tablefmt="psql"))
    except Exception as e:
        print(traceback.format_exc())

    user_input = input("Query:\n").strip()

# Close the database connection
conn.close()
