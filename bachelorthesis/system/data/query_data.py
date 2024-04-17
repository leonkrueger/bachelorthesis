from .table_origin import TableOrigin
from typing import List, Dict


class QueryData:
    query: str
    database_state: Dict[str, List[str]]

    table: str | None = None
    table_origin: TableOrigin | None = None

    columns: List[str] | None = None
    column_types: List[str] | None = None

    values: List[List[str]] | None = None

    def __init__(self, query: str, database_state: Dict[str, List[str]]):
        self.query = query
        self.database_state = database_state
