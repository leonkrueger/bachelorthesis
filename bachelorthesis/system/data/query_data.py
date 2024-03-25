from .table_origin import TableOrigin


class QueryData:
    query: str
    database_state: dict[str, list[str]]

    table: str | None = None
    table_origin: TableOrigin | None = None

    columns: list[str] | None = None
    column_types: list[str] | None = None

    values: list[list[str]] | None = None

    def __init__(self, query: str, database_state: dict[str, list[str]]):
        self.query = query
        self.database_state = database_state
