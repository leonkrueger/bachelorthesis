from ..data.query_data import QueryData


class Strategy:
    def predict_table_name(self, query_data: QueryData) -> str:
        """Returns the name of the predicted table that the query should be executed on.
        This can also be a table that does not yet exist."""
        raise NotImplementedError()
