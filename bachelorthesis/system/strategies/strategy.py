from typing import Dict, List

from ..data.query_data import QueryData


class Strategy:
    def predict_table_name(self, query_data: QueryData) -> str:
        """Returns the name of the predicted table that the query should be executed on.
        This can also be a table that does not yet exist."""
        raise NotImplementedError()

    def predict_column_mapping(self, query_data: QueryData) -> List[str]:
        """Returns a dict containing a mapping of columns names used in the query to the column names
        of the table specified in query_data.
        If there is no corresponding column in the database table, it maps to a suggested name for a new column.
        """
        raise NotImplementedError()
