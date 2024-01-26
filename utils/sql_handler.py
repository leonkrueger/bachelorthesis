from mysql.connector import MySQLConnection


class SQLHandler:
    def __init__(self, conn: MySQLConnection) -> None:
        self.conn = conn

    def execute_query(
        self, query: str, params: tuple[str, ...] = ()
    ) -> tuple[list[tuple[str | int | float, ...]], int]:
        # Executes the query with the given parameters on the database
        # Returns the result of the query and the value of the automatically incremented id
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        output = cursor.fetchall()
        auto_increment_id = cursor.lastrowid
        self.conn.commit()
        cursor.close()
        return output, auto_increment_id

    def get_all_tables(self) -> list[str]:
        # Returns all table names of the database
        query = "SHOW TABLES;"
        output = self.execute_query(query)
        tables = [table[0] for table in output[0]]
        return tables

    def get_all_columns(self, table_name: str) -> list[str]:
        # Returns all column names of the given table
        query = f"SHOW COLUMNS FROM {table_name};"
        output = self.execute_query(query)
        cols = [col[0] for col in output[0]]
        return cols

    def create_table(
        self, table_name: str, column_names: list[str], column_types: list[str]
    ) -> None:
        # Creates the specified table
        query = f"CREATE TABLE {table_name} ({', '.join([f'{column[0]} {column[1]}' for column in zip(column_names, column_types)])});"
        self.execute_query(query)

    def create_column(
        self, table_name: str, column_name: str, column_type: str
    ) -> None:
        # Creates the specified column
        query = f"ALTER TABLE {table_name} ADD {column_name} {column_type};"
        self.execute_query(query)
