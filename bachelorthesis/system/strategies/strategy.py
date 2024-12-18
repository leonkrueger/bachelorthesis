from ..data.insert_data import InsertData


class Strategy:
    def predict_table_name(self, insert_data: InsertData) -> str:
        """Returns the name of the predicted table that the insert should be executed on.
        This can also be a table that does not yet exist."""
        raise NotImplementedError()

    def predict_column_mapping(self, insert_data: InsertData) -> list[str]:
        """Returns a dict containing a mapping of columns names used in the insert to the column names
        of the table specified in insert_data.
        If there is no corresponding column in the database table, it maps to a suggested name for a new column.
        """
        raise NotImplementedError()
