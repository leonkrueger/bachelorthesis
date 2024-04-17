from .table_origin import TableOrigin
from typing import List, Dict, Union


class QueryData:
    query: str
    database_state: Dict[str, List[str]]

    table: Union[str, None] = None
    table_origin: Union[TableOrigin, None] = None

    columns: Union[List[str], None] = None
    column_types: Union[List[str], None] = None

    values: Union[List[List[str]], None] = None

    def __init__(self, query: str, database_state: Dict[str, List[str]]):
        self.query = query
        self.database_state = database_state
