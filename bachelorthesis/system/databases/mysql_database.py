from typing import Any, Dict, List, Tuple

import mysql.connector

from ..data.table_origin import TableOrigin
from .database import Database


class MySQLDatbase(Database):
    def __init__(self) -> None:
        self.conn = mysql.connector.connect(
            user="user",
            password="Start123",
            host="bachelorthesis_leon_krueger_mysql",
            database="db",
        )

        # All tables used for the internal database management
        self.internal_tables = {
            "table_registry": f"""CREATE TABLE IF NOT EXISTS table_registry(
                name VARCHAR(1023),
                name_origin ENUM({", ".join([f"'{table_origin.value}'" for table_origin in TableOrigin])}));"""
        }

    def execute_query(
        self, query: str, params: Tuple[str, ...] = ()
    ) -> List[Tuple[Any, ...]]:
        cursor = self.conn.cursor()
        # print(query)
        cursor.execute(query, params)
        output = cursor.fetchall()
        self.conn.commit()
        cursor.close()
        return output

    def select_all_data(self, table_name: str) -> List[Tuple[Any, ...]]:
        query = f"SELECT * FROM {table_name};"
        return self.execute_query(query)

    def get_all_tables(self) -> List[str]:
        query = "SHOW TABLES;"
        output = self.execute_query(query)
        return [
            table[0] for table in output if table[0] not in self.internal_tables.keys()
        ]

    def get_all_columns(self, table_name: str) -> List[Tuple[str, str]]:
        query = f"SHOW COLUMNS FROM {table_name};"
        output = self.execute_query(query)
        cols = [col[0:2] for col in output]
        return cols

    def get_database_state(self) -> Dict[str, List[str]]:
        return {
            table: [column[0] for column in self.get_all_columns(table)]
            for table in self.get_all_tables()
        }

    def create_table(
        self, table_name: str, column_names: List[str], column_types: List[str]
    ) -> None:
        query = f"CREATE TABLE {table_name} ({', '.join([f'{column[0]} {column[1]}' for column in zip(column_names, column_types)])});"
        self.execute_query(query)

    def create_internal_tables(self) -> None:
        for query in self.internal_tables.values():
            self.execute_query(query)

    def create_column(
        self, table_name: str, column_name: str, column_type: str
    ) -> None:
        query = f"ALTER TABLE {table_name} ADD {column_name} {column_type};"
        self.execute_query(query)

    def remove_table(self, table_name: str) -> None:
        query = f"DROP TABLE {table_name};"
        self.execute_query(query)

    def reset_database(self) -> None:
        for table in self.get_all_tables():
            self.remove_table(table)

        for table in self.internal_tables.keys():
            query = f"DELETE FROM {table};"
            self.execute_query(query)

    def close(self) -> None:
        self.conn.close()
