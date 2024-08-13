from enum import Enum
from typing import Any, Dict, List, Tuple

from thefuzz import fuzz, process

from ...data.query_data import QueryData
from ..strategy import Strategy
from .name_predictor import NamePredictor
from .synonym_generator import SynonymGenerator


class MatchingAlgorithm(Enum):
    EXACT_MATCH = lambda query_name, db_name, synonym_generator: (
        1 if query_name.lower() == db_name.lower() else 0
    )
    FUZZY_MATCH = lambda query_name, db_name, synonym_generator: fuzz.ratio(
        db_name, query_name
    )
    FUZZY_MATCH_SYNONYMS = (
        lambda query_name, db_name, synonym_generator: process.extract(
            db_name,
            synonym_generator.get_synonyms(query_name),
            scorer=fuzz.ratio,
            limit=1,
        )[0]
    )


class HeuristicStrategy(Strategy):
    MINIMAL_SIMILARITY = 60
    MINIMAL_COLUMNS_FOUND_RATIO = 0.5

    # Is filled when the table name is predicted and resetted after the last prediction step for a query
    # First value is the name of the table, second is the column mapping
    saved_column_mapping: Tuple[str, Dict[str, str]] = None

    def __init__(
        self,
        matching_algorithm: MatchingAlgorithm,
        synonym_generator: SynonymGenerator = None,
    ) -> None:
        super().__init__()

        self.matching_algorithm = matching_algorithm
        self.name_predictor = NamePredictor()
        self.synonym_generator = synonym_generator

    def predict_table_name(self, query_data: QueryData) -> str:
        if query_data.table:
            best_score = 0
            best_table = None

            # Find a table in the database whose name is close to any found synonym
            for db_table in query_data.database_state.keys():
                score = self.matching_algorithm(
                    query_data.table, db_table, self.synonym_generator
                )

                if score >= self.MINIMAL_SIMILARITY and score > best_score:
                    best_score = score
                    best_table = db_table

            if best_table:
                return best_table
        else:
            # Use the table with the best column mapping if the table was not specified.
            # Only use it, if at least 50% of the columns were mapped to a column of the table.
            db_table, column_mapping, column_found_ratio = (
                self.get_column_mapping_for_best_table(query_data)
            )
            if column_found_ratio >= self.MINIMAL_COLUMNS_FOUND_RATIO:
                return db_table

        # If no match was found, predict a new table name
        return self.name_predictor.predict_table_name(query_data)

    def predict_column_mapping(self, query_data: QueryData) -> List[str]:
        # If we already computed the table mapping in the table name prediction step, we use this
        if (
            self.saved_column_mapping
            and self.saved_column_mapping[0] == query_data.table
        ):
            return self.saved_column_mapping[1]
        # Otherwise we reset the variable
        self.saved_column_mapping = None

        # If the columns were not specified in the query, predict new ones based on the values
        if query_data.columns:
            query_columns = query_data.columns
        else:
            query_columns = [
                self.name_predictor.predict_column_name(query_data, value)
                for value in query_data.values[0]
            ]

        # If the table exists use its columns for the prediction
        # Else use an empty list as there are no columns in the table yet
        database_columns = (
            query_data.database_state[query_data.table][0]
            if query_data.table in query_data.database_state.keys()
            else []
        )
        # Compute the column mapping
        column_mapping = self.get_column_mapping_for_table(
            query_columns, database_columns
        )

        # Create ordered list of predicted columns with all names
        predicted_columns = []
        for query_column in query_columns:
            if query_column in column_mapping.keys():
                predicted_columns.append(column_mapping[query_column])
            else:
                predicted_columns.append(query_column)

        return predicted_columns

    def get_column_mapping_for_best_table(
        self, query_data: QueryData
    ) -> Tuple[str, Dict[str, str]]:
        """Computes the table with the best ratio of mapped columns.
        Returns the name of the table, the column mapping and the ratio of mapped columns.
        """
        # If the columns were not specified in the query, predict new ones based on the values
        if query_data.columns:
            query_columns = query_data.columns
        else:
            query_columns = [
                self.name_predictor.predict_column_name(query_data, value)
                for value in query_data.values[0]
            ]

        best_column_found_ratio = 0.0
        best_table = None
        best_column_mapping = None

        # Computes the table with the best ratio of mapped columns
        for table, table_data in query_data.database_state.items():
            column_mapping_for_table = self.get_column_mapping_for_table(
                query_columns, table_data[0]
            )
            if (
                ratio := len(column_mapping_for_table) / len(query_columns)
            ) > best_column_found_ratio:
                best_column_found_ratio = ratio
                best_table = table
                best_column_mapping = column_mapping_for_table

        # Include not mapped columns so that they can be added to the table
        for column in query_columns:
            if column not in best_column_mapping.keys():
                best_column_mapping[column] = column

        # Save the result so that it can be potentially used later
        self.saved_column_mapping = (best_table, best_column_mapping)
        return (best_table, best_column_mapping, best_column_found_ratio)

    def get_column_mapping_for_table(
        self, query_columns: list[str], table_columns: list[str]
    ) -> Dict[str, str]:
        """Computes a mapping of query_columns to table_columns."""
        column_mapping = []

        # Find all potentially good mappings
        for query_column in query_columns:
            for table_column in table_columns:
                score = self.matching_algorithm(
                    query_column, table_column, self.synonym_generator
                )

                # If a synonym was found that was close to the name of the column in the database
                # add it as a potential mapping
                if score >= self.MINIMAL_SIMILARITY:
                    column_mapping.append((query_column, table_column, score))

        # Sort all found mappings descending by the score
        column_mapping.sort(key=lambda match: match[2], reverse=True)

        # Filter all found mappings, so that no columns in the query or database occur more than once
        # Required as we only allow 1:1 mappings of columns
        filtered_column_mapping = []
        seen_query_columns = set()
        seen_table_columns = set()

        for match in column_mapping:
            if (
                match[0] not in seen_query_columns
                and match[1] not in seen_table_columns
            ):
                filtered_column_mapping.append(match)
                seen_query_columns.add(match[0])
                seen_table_columns.add(match[1])

        return filtered_column_mapping
