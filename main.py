import mysql.connector
from tabulate import tabulate
import traceback
from sql_handler import SQLHandler

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
        print(tabulate(sql_handler.execute_query(user_input)[0], tablefmt="psql"))
    except Exception as e:
        print(traceback.format_exc())

    user_input = input("Query:\n").strip()

# Close the database connection
conn.close()