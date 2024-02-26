from mysql.connector import MySQLConnection

from utils.enums.table_origin import TableOrigin


class SQLHandler:
    def __init__(self, conn: MySQLConnection) -> None:
        self.conn = conn
        # All tables used for the internal database management
        self.internal_tables = {
            "table_registry": f"""CREATE TABLE IF NOT EXISTS table_registry(
                name VARCHAR(255),
                name_origin ENUM({", ".join([f"'{table_origin.value}'" for table_origin in TableOrigin])}));"""
        }

    def execute_query(
        self, query: str, params: tuple[str, ...] = ()
    ) -> tuple[list[tuple[str | int | float, ...]], int]:
        # Executes the query with the given parameters on the database
        # Returns the result of the query and the value of the automatically incremented id
        cursor = self.conn.cursor()
        # print(query)
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
        tables = [
            table[0]
            for table in output[0]
            if table[0] not in self.internal_tables.keys()
        ]
        return tables

    def get_all_columns(self, table_name: str) -> list[tuple[str, str]]:
        # Returns all column names and types of the given table
        query = f"SHOW COLUMNS FROM {table_name};"
        output = self.execute_query(query)
        cols = [(col[0], col[1]) for col in output[0]]
        return cols

    def create_table(
        self, table_name: str, column_names: list[str], column_types: list[str]
    ) -> None:
        # Creates the specified table
        query = f"CREATE TABLE {table_name} ({', '.join([f'{column[0]} {column[1]}' for column in zip(column_names, column_types)])});"
        self.execute_query(query)

    def create_internal_tables(self) -> None:
        # Creates all tables used for internal database management
        for query in self.internal_tables.values():
            self.execute_query(query)

    def create_column(
        self, table_name: str, column_name: str, column_type: str
    ) -> None:
        # Creates the specified column
        query = f"ALTER TABLE {table_name} ADD {column_name} {column_type};"
        self.execute_query(query)

    def remove_table(self, table_name: str) -> None:
        """Removes the specified table"""
        query = f"DROP TABLE {table_name};"
        self.execute_query(query)

    def reset_database(self) -> None:
        """Removes all custom tables in the database and empties the internal ones"""
        for table in self.get_all_tables():
            self.remove_table(table)

        for table in self.internal_tables.keys():
            query = f"DELETE FROM {table};"
            self.execute_query(query)
