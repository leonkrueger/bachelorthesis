from typing import Any, Dict, List, Tuple

import mysql.connector

from .database import Database


class MySQLDatbase(Database):
    def __init__(self) -> None:
        self.conn = mysql.connector.connect(
            user="user",
            password="Start123",
            host="bachelorthesis_leon_krueger_mysql",
            database="db",
        )

    def execute(
        self, statement: str, params: Tuple[str, ...] = ()
    ) -> List[Tuple[Any, ...]]:
        cursor = self.conn.cursor()
        cursor.execute(statement, params)
        output = cursor.fetchall()
        self.conn.commit()
        cursor.close()
        return output

    def select_all_data(self, table_name: str) -> List[Tuple[Any, ...]]:
        statement = f"SELECT * FROM {table_name};"
        return self.execute(statement)

    def get_all_tables(self) -> List[str]:
        statement = "SHOW TABLES;"
        output = self.execute(statement)
        return [table[0] for table in output]

    def get_all_columns(self, table_name: str) -> List[Tuple[str, str]]:
        statement = f"SHOW COLUMNS FROM {table_name};"
        output = self.execute(statement)
        cols = [col[0:2] for col in output]
        return cols

    def get_example_rows(self, table_name: str) -> List[Tuple[Any, ...]]:
        statement = f"SELECT * FROM {table_name} LIMIT 3;"
        return self.execute(statement)

    def get_database_state(self) -> Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]:
        return {
            table: (
                [column[0] for column in self.get_all_columns(table)],
                self.get_example_rows(table),
            )
            for table in self.get_all_tables()
        }

    def create_table(
        self, table_name: str, column_names: List[str], column_types: List[str]
    ) -> None:
        statement = f"CREATE TABLE {table_name} ({', '.join([f'{column[0]} {column[1]}' for column in zip(column_names, column_types)])});"
        self.execute(statement)

    def create_column(
        self, table_name: str, column_name: str, column_type: str
    ) -> None:
        statement = f"ALTER TABLE {table_name} ADD {column_name} {column_type};"
        self.execute(statement)

    def remove_table(self, table_name: str) -> None:
        statement = f"DROP TABLE {table_name};"
        self.execute(statement)

    def reset_database(self) -> None:
        for table in self.get_all_tables():
            self.remove_table(table)

    def close(self) -> None:
        self.conn.close()
